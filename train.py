import torch
from dino_diff.models.denoiser import SanaFeatureDenoiser
from dino_diff.models.text_perceiver import LatentEncoder
from dino_diff.models.dino_sampling import DinoSampler
from dino_diff.models.text_embedder import T5TextEmbedder
from datasets import load_dataset
from transformers import AutoImageProcessor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_PRETRAINED = "facebook/dinov3-vits16-pretrain-lvd1689m"

TEXT_EMBEDDER = T5TextEmbedder().eval().to(DEVICE)

TEXT_HIDDEN_SIZE = TEXT_EMBEDDER.get_hidden_size()
TEXT_TOKENIZER = TEXT_EMBEDDER.tokenizer
PROCESSOR = AutoImageProcessor.from_pretrained(DINO_PRETRAINED)

DINO_BREAK_AT_LAYER = 6

# Load dataset (using a small subset for demonstration)
DS = load_dataset(
    "BLIP3o/BLIP3o-Pretrain-Long-Caption",
    data_files=[f"sa_00000{x}.tar" for x in range(2)],
    split="train",
).select(range(5000))

# Preprocess dataset
DS = DS.map(
    lambda x: {
        "text_inputs": TEXT_TOKENIZER(
            x["txt"],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ),
        "pixel_values": PROCESSOR(images=x["jpg"], return_tensors="pt").pixel_values,
    },
    num_proc=12,
)

DS = DS.with_format("torch")


# DataLoader
BATCH_SIZE = 8
train_dataloader = DataLoader(DS, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Initialize models
latent_encoder = LatentEncoder(
    txt_in_size=TEXT_HIDDEN_SIZE,
    num_layers=12,
    hidden_size=1024,
    num_attention_heads=16,
    mlp_ratio=3,
    num_latents=256,
).to(DEVICE)

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
    caption_channels=1024,
).to(DEVICE)

print(dino_sampler)

# Trainable parameters
trainable_params = [*denoiser.parameters(), *latent_encoder.parameters()]

# Optimizer
opt = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.0)

# Mixed precision scaler
scaler = torch.GradScaler()


# Training function for one batch
def train_one_batch(input_ids, attention_mask, pixel_values):
    timesteps = torch.rand((input_ids.shape[0]), device=DEVICE) * 0.5
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    pixel_values = pixel_values.to(DEVICE)
    noise = torch.randn_like(pixel_values)
    sqrt_timesteps = torch.sqrt(timesteps)[:, None, None, None]
    sqrt_one_minus_timesteps = torch.sqrt(1 - timesteps)[:, None, None, None]
    noised_pixels = sqrt_one_minus_timesteps * pixel_values + sqrt_timesteps * noise
    with torch.no_grad():
        text_embeds = TEXT_EMBEDDER(input_ids, attention_mask)

    perceivers = latent_encoder(text_embeds, attention_mask, timesteps)

    clean_features, clean_pe = dino_sampler.forward_features(pixel_values)
    noised_features, noised_pe = dino_sampler.forward_features(noised_pixels)

    x = noised_features[DINO_BREAK_AT_LAYER]

    refined_x = denoiser(x, perceivers, timesteps)

    output = dino_sampler(refined_x, noised_pe)

    loss = F.mse_loss(output, clean_features[-1]) / F.mse_loss(
        noised_features[-1], clean_features[-1]
    )
    intermediate_loss = F.mse_loss(
        refined_x, clean_features[DINO_BREAK_AT_LAYER]
    ) / F.mse_loss(x, clean_features[DINO_BREAK_AT_LAYER])

    return loss, intermediate_loss


# Training loop
NUM_EPOCHS = 10000
step = 0

denoiser.train()
latent_encoder.train()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for batch in progress_bar:
        input_ids = batch["text_inputs"]["input_ids"].squeeze(1)
        attention_mask = batch["text_inputs"]["attention_mask"].squeeze(1)
        pixel_values = batch["pixel_values"].squeeze(1)
        opt.zero_grad()
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            loss, intermediate_loss = train_one_batch(
                input_ids, attention_mask, pixel_values
            )

        true_loss = loss * 0.1 + intermediate_loss
        scaler.scale(true_loss).backward()
        scaler.step(opt)
        scaler.update()

        epoch_loss += true_loss.item()
        step += 1

        progress_bar.set_postfix(
            {
                "Final Loss": f"{loss.item():.4f}",
                "Inter Loss": f"{intermediate_loss.item():.4f}",
            }
        )

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    logging.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

# Save models (optional)
torch.save(denoiser.state_dict(), "denoiser.pth")
torch.save(latent_encoder.state_dict(), "latent_encoder.pth")
logging.info("Training completed. Models saved.")
