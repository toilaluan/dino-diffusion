from transformers import DINOv3ViTModel, AutoImageProcessor
import torch.nn as nn
import torch
from typing import Optional


class DinoSampler(DINOv3ViTModel):
    def __init__(self, *args, **kwargs):
        break_at_layer = kwargs.pop("break_at_layer")
        super().__init__(*args, **kwargs)
        self.break_at_layer = break_at_layer

    @torch.no_grad
    def forward_features(
        self,
        pixel_values: torch.Tensor,
    ):
        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=None)
        position_embeddings = self.rope_embeddings(pixel_values)

        all_hidden_states = []
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = None
            hidden_states = layer_module(
                hidden_states,
                attention_mask=layer_head_mask,
                position_embeddings=position_embeddings,
            )
            all_hidden_states.append(hidden_states)

        return all_hidden_states, position_embeddings

    def forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor):
        for layer in self.layer[self.break_at_layer+1:]:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        return hidden_states


if __name__ == "__main__":
    from PIL import Image

    device = "cuda"
    image = Image.open("image.png").convert("RGB")
    pretrained_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_name)

    inputs = processor(images=image, return_tensors="pt").to(device)
    pixels = inputs.pixel_values
    model = DinoSampler.from_pretrained(
        pretrained_name, device_map="auto", break_at_layer=2
    )
    clean_features, pos_embs = model.forward_features(pixels)

    noise = torch.randn_like(pixels)

    for t in [0.9, 0.8, 0.5, 0.25, 0.1, 0.05]:
        noised_pixels = t * noise + (1 - t) * pixels

        noised_features, pos_embs = model.forward_features(noised_pixels)

        diff = clean_features[-1] - noised_features[-1]

        print(t, diff.abs().mean())
