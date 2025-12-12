import torch
import torch.nn as nn
from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel
from typing import Optional


class SanaFeatureDenoiser(SanaTransformer2DModel):
    def __init__(
        self,
        dim: int = 1024,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
    ):
        inner_dim = num_attention_heads * attention_head_dim
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
            cross_attention_dim=inner_dim,
            caption_channels=caption_channels,
            mlp_ratio=mlp_ratio,
        )
        self.x_projection = nn.Linear(dim, inner_dim)
        self.patch_embed = None
        self.proj_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ):
        batch_size, hw, _ = hidden_states.shape
        timestep, embedded_timestep = self.time_embed(
            timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        hidden_states = self.x_projection(hidden_states)

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, hidden_states.shape[-1]
        )

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)


        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                height=1,
                width=hw,
            )

        hidden_states = self.norm_out(
            hidden_states, embedded_timestep, self.scale_shift_table
        )
        hidden_states = self.proj_out(hidden_states)

        return hidden_states


if __name__ == "__main__":
    model = SanaFeatureDenoiser(
        dim=1024,
        num_attention_heads=4,
        attention_head_dim=384,
        num_layers=2,
        num_cross_attention_heads=4,
        caption_channels=1024,
        mlp_ratio=2.5,
    )

    print(model)

    x = torch.zeros((1, 196, 1024))
    y = torch.zeros((1, 256, 1024))
    timestep = torch.tensor([1000])

    out = model(x, y, timestep)

    print(out.shape)
