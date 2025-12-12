import torch
from dino_diff.models.denoiser import SanaFeatureDenoiser
from dino_diff.models.text_perceiver import LatentEncoder
from dino_diff.models.dino_sampling import DinoSampler
from dino_diff.models.text_embedder import T5TextEmbedder
from dino_diff.models.ella_perceiver import ELLA
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import wandb
from collections import defaultdict
from safetensors.torch import load_file
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
from pathlib import Path
from transformers import AutoImageProcessor, AutoTokenizer
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import torchvision

wandb.init(entity="toilaluan", project="dino-diff")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_PRETRAINED = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_BREAK_AT_LAYER = 5
TEXT_PRETRAINED = "google/flan-t5-xl"
DTYPE = torch.bfloat16

IMG_MEAN = torch.tensor([
    0.485,
    0.456,
    0.406
], device="cuda")
IMG_STD = torch.tensor([
    0.229,
    0.224,
    0.225
], device="cuda")

# ============================================================================
# Tokenization and Dataset preprocessing functions
# ============================================================================

def get_tokenizer():
    """Get the T5 tokenizer."""
    return AutoTokenizer.from_pretrained(TEXT_PRETRAINED)


def pretokenize_batch(batch, tokenizer):
    """
    Tokenize a batch of captions. Used with dataset.map().

    Args:
        batch: Batch from dataset with 'txt' column
        tokenizer: HuggingFace tokenizer

    Returns:
        Batch with added tokenized fields
    """
    captions = batch['txt']

    tokenized = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",
    )

    return {
        **batch,
        "input_ids": tokenized["input_ids"].tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
    }


def pretokenize_dataset(dataset, num_proc=8):
    """
    Pretokenize the entire dataset using dataset.map() for parallelization.

    Args:
        dataset: HuggingFace dataset
        num_proc: Number of processes for parallel tokenization

    Returns:
        Dataset with tokenized fields added
    """
    print(f"Pretokenizing dataset with {num_proc} processes...")
    tokenizer = get_tokenizer()

    tokenize_fn = partial(pretokenize_batch, tokenizer=tokenizer)

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        desc="Tokenizing captions",
        remove_columns=[],
    )

    print(f"Tokenization complete!")
    return tokenized_dataset


class OnTheFlyDataset(Dataset):
    """
    Dataset that processes images and text on-the-fly during training.
    No caching - computes everything in real-time.
    """

    def __init__(self, dataset, image_processor):
        """
        Args:
            dataset: Pretokenized HuggingFace dataset
            image_processor: DINO image processor
        """
        self.dataset = dataset
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get image
        image = item['jpg']

        # Handle different image formats
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image)).convert('RGB')

        # Process image
        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }


# Custom collate function
def collate_fn(batch):
    """Custom collate function to properly batch samples."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


# ============================================================================
# Dataset Loading and Initialization
# ============================================================================

print("Loading dataset...")
dataset = load_dataset(
    "BLIP3o/BLIP3o-Pretrain-Long-Caption",
    split="train",
    data_files=[f"sa_00000{x}.tar" for x in range(10)]
)

print(f"Dataset loaded: {len(dataset)} samples")

# Pretokenize dataset
dataset = pretokenize_dataset(dataset, num_proc=8)

# Initialize image processor
image_processor = AutoImageProcessor.from_pretrained(DINO_PRETRAINED)

# Create on-the-fly dataset
full_dataset = OnTheFlyDataset(dataset, image_processor)

# Split into train and validation
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# DataLoaders
BATCH_SIZE = 64
train_dataloader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=16,
    collate_fn=collate_fn,
    pin_memory=True,
    prefetch_factor=4
)
val_dataloader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=16,
    collate_fn=collate_fn,
    pin_memory=True,
    prefetch_factor=4
)

# ============================================================================
# Initialize Models
# ============================================================================

print("Initializing models...")

# Text embedder for computing text embeddings on-the-fly
text_embedder = T5TextEmbedder(pretrained_path=TEXT_PRETRAINED).to(DTYPE).eval().to(DEVICE)

# ELLA latent encoder
latent_encoder = ELLA().to(DEVICE).eval()
latent_encoder.load_state_dict(load_file("ella-sd1.5-tsc-t5xl.safetensors"))

# DINO sampler
dino_sampler = (
    DinoSampler.from_pretrained(DINO_PRETRAINED, break_at_layer=DINO_BREAK_AT_LAYER)
    .to(DTYPE)
    .eval()
    .to(DEVICE)
)

# Denoiser model (trainable)
denoiser = SanaFeatureDenoiser(
    dim=dino_sampler.config.hidden_size,
    num_attention_heads=16,
    attention_head_dim=128,
    num_layers=12,
    caption_channels=768,
).to(DEVICE)

# Compile models for faster inference
denoiser = torch.compile(denoiser)
dino_sampler = torch.compile(dino_sampler, fullgraph=True, dynamic=False)
text_embedder = torch.compile(text_embedder, fullgraph=True, dynamic=False)

print("Models initialized!")
print(f"DINO Sampler: {DINO_PRETRAINED}")
print(f"Text Embedder: {TEXT_PRETRAINED}")
print(f"Denoiser layers: {denoiser}")
print(f"Processing will be done on-the-fly without caching")

# Trainable parameters
trainable_params = [*denoiser.parameters()]

# Optimizer
opt = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.0)

# Learning rate scheduler with warmup
warmup_steps = 100
min_lr_ratio = 1e-6 / 1e-4  # 0.01

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return min_lr_ratio + (1 - min_lr_ratio) * (current_step / warmup_steps)
    return 1.0

scheduler = LambdaLR(opt, lr_lambda)

# Mixed precision scaler
scaler = GradScaler()

# wandb.watch(denoiser, log="gradients", log_freq=50)

# Define bins for timesteps
BIN_BOUNDARIES = torch.tensor([0.33, 0.66], device=DEVICE)
NUM_BINS = 3

def compute_losses(clean_inter, clean_final, noise_inter, noise_final, refined_inter, output):

    final_noise_level = F.mse_loss(noise_final, clean_final, reduction='none').mean(dim=[1, 2])
    inter_noise_level = F.mse_loss(noise_inter, clean_inter, reduction='none').mean(dim=[1, 2])

    final_pred_loss = F.mse_loss(output, clean_final, reduction='none').mean(dim=[1, 2])
    inter_pred_loss = F.mse_loss(refined_inter, clean_inter, reduction='none').mean(dim=[1, 2])

    normalized_final = final_pred_loss / (final_noise_level + 1e-8)  # Avoid division by zero
    normalized_inter = inter_pred_loss / (inter_noise_level + 1e-8)

    true_loss_per_sample = 1.0 * final_pred_loss

    final_loss = final_pred_loss.mean()
    inter_loss = inter_pred_loss.mean()
    true_loss = true_loss_per_sample.mean()

    return true_loss, final_loss, inter_loss, true_loss_per_sample, normalized_final.mean(), normalized_inter.mean()

# Processing function for one batch (shared for train/val)
def process_batch(batch, timesteps=None, train_mode=True):
    batch_size = batch["pixel_values"].shape[0]

    if timesteps is None:
        timesteps = torch.rand((batch_size), device=DEVICE)

    # Get inputs from batch
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    pixel_values = batch["pixel_values"].to(DEVICE)

    # Compute text embeddings on-the-fly
    with torch.no_grad():
        text_embeds = text_embedder(input_ids, attention_mask)

    # Compute clean DINO features on-the-fly
    with torch.no_grad():
        clean_features, _ = dino_sampler.forward_features(pixel_values)
        clean_inter = clean_features[DINO_BREAK_AT_LAYER]
        clean_final = clean_features[-1]

    # Add noise to pixels
    noise = torch.rand_like(pixel_values)
    noise = torchvision.transforms.functional.normalize(noise, IMG_MEAN, IMG_STD)
    noised_pixels = (1 - timesteps)[:, None, None, None] * pixel_values + timesteps[:, None, None, None] * noise

    # Get perceivers from ELLA
    with torch.no_grad():
        perceivers = latent_encoder(text_embeds, attention_mask, timesteps * 1000)

    # Forward pass through noised features
    noised_features, noised_pe = dino_sampler.forward_features(noised_pixels)
    x = noised_features[DINO_BREAK_AT_LAYER]

    # Denoise
    n_regs = 1+dino_sampler.config.num_register_tokens
    registers = clean_final[:, :n_regs, :]
    x = x[:, n_regs:, :]
    refined_x = denoiser(x, perceivers, timesteps)
    r_refined_x = torch.cat([registers, refined_x], dim=1)
    output = dino_sampler(r_refined_x, noised_pe)
    noised_inter = noised_features[DINO_BREAK_AT_LAYER][:, n_regs:, :]
    noised_final = noised_features[-1][:, n_regs:, :]
    clean_inter = clean_inter[:, n_regs:, :]
    clean_final = clean_final[:, n_regs:, :]
    # Compute losses
    true_loss, final_loss, inter_loss, true_loss_per_sample, normalized_final, normalized_inter = compute_losses(
        clean_inter, clean_final, noised_inter, noised_final, refined_x, output[:, n_regs:, :]
    )

    bin_idx = torch.bucketize(timesteps, BIN_BOUNDARIES)

    return true_loss, final_loss, inter_loss, true_loss_per_sample, bin_idx, normalized_final, normalized_inter

# Training function for one batch
def train_one_batch(batch):
    true_loss, final_loss, inter_loss, true_loss_per_sample, bin_idx, normalized_final, normalized_inter = process_batch(
        batch, train_mode=True
    )
    return true_loss, final_loss, inter_loss, true_loss_per_sample, bin_idx, normalized_final, normalized_inter

# Validation function
def validate(model, dataloader):
    model.eval()
    val_true_loss = 0.0
    val_final_loss = 0.0
    val_inter_loss = 0.0
    val_normalized_final = 0.0
    val_normalized_inter = 0.0
    bin_losses = defaultdict(list)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            with autocast(dtype=torch.bfloat16):
                true_loss, final_loss, inter_loss, true_loss_per_sample, bin_idx, normalized_final, normalized_inter = process_batch(
                    batch, train_mode=False
                )
            
            val_true_loss += true_loss.item()
            val_final_loss += final_loss.item()
            val_inter_loss += inter_loss.item()
            val_normalized_final += normalized_final.item()
            val_normalized_inter += normalized_inter.item()
            
            for i in range(NUM_BINS):
                mask = (bin_idx == i)
                if mask.any():
                    bin_losses[i].append(true_loss_per_sample[mask].mean().item())
            
            progress_bar.set_postfix(
                {
                    "True Loss": f"{true_loss.item():.4f}",
                    "Final Loss": f"{final_loss.item():.4f}",
                    "Inter Loss": f"{inter_loss.item():.4f}",
                }
            )
    
    avg_val_true_loss = val_true_loss / len(dataloader)
    avg_val_final_loss = val_final_loss / len(dataloader)
    avg_val_inter_loss = val_inter_loss / len(dataloader)
    avg_norm_final = val_normalized_final / len(dataloader)
    avg_norm_inter = val_normalized_inter / len(dataloader)

    bin_avgs = {}
    for i in range(NUM_BINS):
        if bin_losses[i]:
            bin_avgs[f"val_loss_bin_{i}"] = sum(bin_losses[i]) / len(bin_losses[i])
        else:
            bin_avgs[f"val_loss_bin_{i}"] = 0.0
    
    model.train()
    return avg_val_true_loss, avg_val_final_loss, avg_val_inter_loss, bin_avgs, avg_norm_final, avg_norm_inter

# Training loop
NUM_EPOCHS = 10000
step = 0
denoiser.train()

for epoch in range(NUM_EPOCHS):
    epoch_true_loss = 0.0
    epoch_final_loss = 0.0
    epoch_inter_loss = 0.0
    train_bin_losses = defaultdict(list)
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for batch in progress_bar:
        opt.zero_grad()
        
        with autocast(dtype=torch.bfloat16):
            true_loss, final_loss, inter_loss, true_loss_per_sample, bin_idx, avg_norm_final, avg_norm_inter = train_one_batch(
                batch
            )
        
        scaler.scale(true_loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        
        epoch_true_loss += true_loss.item()
        epoch_final_loss += final_loss.item()
        epoch_inter_loss += inter_loss.item()
        step += 1
        
        for i in range(NUM_BINS):
            mask = (bin_idx == i)
            if mask.any():
                train_bin_losses[i].append(true_loss_per_sample[mask].mean().item())
        
        progress_bar.set_postfix(
            {
                "True Loss": f"{true_loss.item():.4f}",
                "Final Loss": f"{final_loss.item():.4f}",
                "Inter Loss": f"{inter_loss.item():.4f}",
            }
        )
        
        wandb.log(
            {
                "train/true_loss": true_loss.item(),
                "train/final_loss": final_loss.item(),
                "train/inter_loss": inter_loss.item(),
                "train/avg_norm_final": avg_norm_final.item(),
                "train/avg_norm_inter": avg_norm_inter.item(),
                "step": step,
            }
        )
    
    avg_epoch_true_loss = epoch_true_loss / len(train_dataloader)
    avg_epoch_final_loss = epoch_final_loss / len(train_dataloader)
    avg_epoch_inter_loss = epoch_inter_loss / len(train_dataloader)
    
    logger.info(f"Epoch {epoch + 1} completed. Avg Train True Loss: {avg_epoch_true_loss:.4f}, "
                f"Avg Final Loss: {avg_epoch_final_loss:.4f}, Avg Inter Loss: {avg_epoch_inter_loss:.4f}")
    
    train_bin_avgs = {}
    for i in range(NUM_BINS):
        if train_bin_losses[i]:
            avg = sum(train_bin_losses[i]) / len(train_bin_losses[i])
            train_bin_avgs[f"train_loss_bin_{i}"] = avg
            logger.info(f"Train Loss Bin {i} (ts {0.33*i}-{0.33*(i+1)}): {avg:.4f}")
        else:
            train_bin_avgs[f"train_loss_bin_{i}"] = 0.0
    
    wandb.log({
        "train/avg_train_true_loss": avg_epoch_true_loss,
        "train/avg_train_final_loss": avg_epoch_final_loss,
        "train/avg_train_inter_loss": avg_epoch_inter_loss,
        "epoch": epoch + 1,
        **train_bin_avgs
    })
    
    # Validation
    avg_val_true_loss, avg_val_final_loss, avg_val_inter_loss, val_bin_avgs, avg_norm_final, avg_norm_inter = validate(denoiser, val_dataloader)
    
    logger.info(f"Validation: Avg True Loss: {avg_val_true_loss:.4f}, "
                f"Avg Final Loss: {avg_val_final_loss:.4f}, Avg Inter Loss: {avg_val_inter_loss:.4f}, Avg Norm Final: {avg_norm_final:.4f}, Avg Norm Inter: {avg_norm_inter:.4f}")
    
    for key, value in val_bin_avgs.items():
        logger.info(f"{key.capitalize()}: {value:.4f}")
    
    wandb.log({
        "avg_val_true_loss": avg_val_true_loss,
        "avg_val_final_loss": avg_val_final_loss,
        "avg_val_inter_loss": avg_val_inter_loss,
        "avg_norm_final": avg_norm_final,
        "avg_norm_inter": avg_norm_inter,
        **val_bin_avgs
    })

    # Save models (optional)
    torch.save(denoiser.state_dict(), f"denoiser-{epoch}.pth")
    # torch.save(latent_encoder.state_dict(), "latent_encoder-{epoch}.pth")
    logger.info("Training completed. Models saved.")

