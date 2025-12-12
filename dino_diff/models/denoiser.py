import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from typing import Optional


class SanaFeatureDenoiser(QwenImageTransformer2DModel):
    def __init__(
        self,
        dim: int = 1024,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        caption_channels: int = 2304,
    ):
        inner_dim = num_attention_heads * attention_head_dim
        super().__init__(
            num_attention_heads=num_attention_heads, # inner_dim = head * h_dim
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            joint_attention_dim=caption_channels,
            in_channels=dim
        )
        self.patch_embed = None
        self.proj_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
    ):
        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        B, L, _ = encoder_hidden_states.shape
        HW = hidden_states.shape[1]
        H = int(HW**0.5)
        assert H * H == HW

        temb = self.time_text_embed(timestep, hidden_states)

        image_rotary_emb = self.pos_embed((1, H, H), [L], device=hidden_states.device)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    None,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return output


if __name__ == "__main__":
    model = SanaFeatureDenoiser(
        dim=1024,
        num_attention_heads=24,
        attention_head_dim=128,
        num_layers=2,
        caption_channels=1024,
    )

    print(model)

    x = torch.zeros((1, 196, 1024))
    y = torch.zeros((1, 256, 1024))
    timestep = torch.tensor([1000])

    out = model(x, y, timestep)

    print(out.shape)
