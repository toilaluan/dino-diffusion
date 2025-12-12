import torch
from dino_diff.models.denoiser import SanaFeatureDenoiser
from dino_diff.models.text_perceiver import LatentEncoder
from dino_diff.models.dino_sampling import DinoSampler
from dino_diff.models.text_embedder import T5TextEmbedder
from dino_diff.models.ella_perceiver import ELLA
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import wandb
from collections import defaultdict
import torch.utils.checkpoint
from safetensors.torch import load_file
import torch.nn.functional as F
import numpy as np
import os
from glob import glob

wandb.init(entity="toilaluan", project="dino-diff")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_PRETRAINED = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_BREAK_AT_LAYER = 6


class CachedNumpyDataset(Dataset):
    """Custom Dataset that loads cached tensors from .npy files."""

    def __init__(self, text_cache_dir, dino_cache_dir, preprocessed_dir):
        """
        Args:
            text_cache_dir: Directory containing text embeddings and inputs
            dino_cache_dir: Directory containing DINO features
            preprocessed_dir: Directory containing original preprocessed data (for pixel_values)
        """
        self.text_cache_dir = text_cache_dir
        self.dino_cache_dir = dino_cache_dir

        # Load the preprocessed dataset to get pixel_values
        self.preprocessed_ds = load_from_disk(preprocessed_dir)

        # Get list of cached files to determine dataset size
        text_embed_files = sorted(glob(os.path.join(text_cache_dir, "text_embeds", "*.npy")))
        self.num_samples = len(text_embed_files)

        logger.info(f"Initialized CachedNumpyDataset with {self.num_samples} samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Load a single sample from cached .npy files."""
        # Construct file paths
        text_embed_path = os.path.join(self.text_cache_dir, "text_embeds", f"{idx:06d}.npy")
        text_input_path = os.path.join(self.text_cache_dir, "text_inputs", f"{idx:06d}.npy")
        clean_inter_path = os.path.join(self.dino_cache_dir, "clean_inter", f"{idx:06d}.npy")
        clean_final_path = os.path.join(self.dino_cache_dir, "clean_final", f"{idx:06d}.npy")

        # Load numpy arrays
        text_embeds = torch.from_numpy(np.load(text_embed_path))
        text_inputs_dict = np.load(text_input_path, allow_pickle=True).item()
        clean_inter = torch.from_numpy(np.load(clean_inter_path))
        clean_final = torch.from_numpy(np.load(clean_final_path))

        # Get pixel_values from preprocessed dataset
        pixel_values = self.preprocessed_ds[idx]["pixel_values"]

        # Convert text_inputs dict to proper format
        text_inputs = {
            "input_ids": torch.from_numpy(text_inputs_dict["input_ids"]),
            "attention_mask": torch.from_numpy(text_inputs_dict["attention_mask"])
        }

        return {
            "text_embeds": text_embeds,
            "text_inputs": text_inputs,
            "pixel_values": pixel_values,
            "clean_inter": clean_inter,
            "clean_final": clean_final
        }


# Custom collate function to handle nested dict structure
def collate_fn(batch):
    """Custom collate function to properly batch samples."""
    text_embeds = torch.stack([item["text_embeds"] for item in batch])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    clean_inter = torch.stack([item["clean_inter"] for item in batch])
    clean_final = torch.stack([item["clean_final"] for item in batch])

    # Stack text_inputs
    text_inputs = {
        "input_ids": torch.stack([item["text_inputs"]["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["text_inputs"]["attention_mask"] for item in batch])
    }

    return {
        "text_embeds": text_embeds,
        "text_inputs": text_inputs,
        "pixel_values": pixel_values,
        "clean_inter": clean_inter,
        "clean_final": clean_final
    }


# Load cached datasets using custom Dataset class
DS = CachedNumpyDataset(
    text_cache_dir="cache/cache_text",
    dino_cache_dir="cache/cache_dino",
    preprocessed_dir="data/preprocessed"
)

# Split into train and validation
train_size = int(0.8 * len(DS))
val_size = len(DS) - train_size
train_ds, val_ds = random_split(DS, [train_size, val_size])

# DataLoaders with custom collate function
BATCH_SIZE = 32
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, collate_fn=collate_fn)
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, collate_fn=collate_fn)

# Initialize models
latent_encoder = ELLA().to(DEVICE).eval()
latent_encoder.load_state_dict(load_file("ella-sd1.5-tsc-t5xl.safetensors"))

dino_sampler = (
    DinoSampler.from_pretrained(DINO_PRETRAINED, break_at_layer=DINO_BREAK_AT_LAYER)
    .eval()
    .to(DEVICE)
)

denoiser = SanaFeatureDenoiser(
    dim=dino_sampler.config.hidden_size,
    num_attention_heads=16,
    attention_head_dim=64,
    num_layers=12,
    num_cross_attention_heads=16,
    cross_attention_head_dim=64,
    caption_channels=768,
).to(DEVICE)
denoiser = torch.compile(denoiser)

print(dino_sampler)
print(denoiser)

# Trainable parameters
trainable_params = [*denoiser.parameters()]

# Optimizer
opt = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.0)

# Mixed precision scaler
scaler = GradScaler()

wandb.watch(denoiser, log="gradients", log_freq=50)

# Define bins for timesteps
BIN_BOUNDARIES = torch.tensor([0.33, 0.66], device=DEVICE)
NUM_BINS = 3

def compute_losses(clean_inter, clean_final, noised_features, refined_x, output, break_layer):
    final_targets = clean_final
    inter_targets = clean_inter
    x = noised_features[break_layer]

    # Assume shapes: (B, seq_len, dim)
    # Compute per-sample mean losses
    final_noise_level = F.mse_loss(noised_features[-1], final_targets, reduction='none').mean(dim=[1, 2])
    inter_noise_level = F.mse_loss(x, inter_targets, reduction='none').mean(dim=[1, 2])

    final_pred_loss = F.mse_loss(output, final_targets, reduction='none').mean(dim=[1, 2])
    inter_pred_loss = F.mse_loss(refined_x, inter_targets, reduction='none').mean(dim=[1, 2])

    normalized_final = final_pred_loss / (final_noise_level + 1e-8)  # Avoid division by zero
    normalized_inter = inter_pred_loss / (inter_noise_level + 1e-8)

    true_loss_per_sample = 0.5 * final_pred_loss + 0.5 * inter_pred_loss

    final_loss = normalized_final.mean()
    inter_loss = normalized_inter.mean()
    true_loss = true_loss_per_sample.mean()

    return true_loss, final_loss, inter_loss, true_loss_per_sample, normalized_final, normalized_inter

# Processing function for one batch (shared for train/val)
def process_batch(batch, timesteps=None, train_mode=True):
    if timesteps is None:
        timesteps = torch.rand((BATCH_SIZE), device=DEVICE)
    
    text_embeds = batch["text_embeds"].to(DEVICE)
    attention_mask = batch["text_inputs"]["attention_mask"].squeeze(1).to(DEVICE)
    pixel_values = batch["pixel_values"].squeeze(1).to(DEVICE)
    clean_inter = batch["clean_inter"].to(DEVICE)
    clean_final = batch["clean_final"].to(DEVICE)
    
    noise = torch.randn_like(pixel_values)
    noised_pixels = (1 - timesteps)[:, None, None, None] * pixel_values + timesteps[:, None, None, None] * noise
    
    with torch.no_grad():
        perceivers = latent_encoder(text_embeds, attention_mask, timesteps * 1000)
    
    noised_features, noised_pe = dino_sampler.forward_features(noised_pixels)
    
    x = noised_features[DINO_BREAK_AT_LAYER]
    
    refined_x = denoiser(x, perceivers, timesteps)
    
    output = dino_sampler(refined_x, noised_pe)
    
    true_loss, final_loss, inter_loss, true_loss_per_sample, normalized_final, normalized_inter = compute_losses(
        clean_inter, clean_final, noised_features, refined_x, output, DINO_BREAK_AT_LAYER
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
            true_loss, final_loss, inter_loss, true_loss_per_sample, bin_idx, _, _= train_one_batch(
                batch
            )
        
        scaler.scale(true_loss).backward()
        scaler.step(opt)
        scaler.update()
        
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
                "train_true_loss": true_loss.item(),
                "train_final_loss": final_loss.item(),
                "train_inter_loss": inter_loss.item(),
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
        "avg_train_true_loss": avg_epoch_true_loss,
        "avg_train_final_loss": avg_epoch_final_loss,
        "avg_train_inter_loss": avg_epoch_inter_loss,
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
torch.save(denoiser.state_dict(), "denoiser.pth")
torch.save(latent_encoder.state_dict(), "latent_encoder.pth")
logger.info("Training completed. Models saved.")