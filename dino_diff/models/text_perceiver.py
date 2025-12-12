import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class NormModulate(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.modulate = nn.Linear(hidden_size, hidden_size * 3)

    def forward(self, hidden_states, t_emb):
        scale, shift, gate = self.modulate(t_emb).chunk(3, dim=-1)
        return self.norm(hidden_states) * (1 + scale[:, None, :]) + shift[
            :, None, :
        ], gate


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.head_size = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.to_qkv = nn.Linear(hidden_size, self.head_size * num_attention_heads * 3)

        self.norm_q = nn.LayerNorm(self.head_size)
        self.norm_k = nn.LayerNorm(self.head_size)

        self.to_out = nn.Linear(self.head_size * num_attention_heads, hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        B, L, D = hidden_states.shape
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

        qkv = self.to_qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.num_attention_heads, -1).transpose(1, 2).contiguous()
        k = k.view(B, L, self.num_attention_heads, -1).transpose(1, 2).contiguous()
        v = v.view(B, L, self.num_attention_heads, -1).transpose(1, 2).contiguous()

        q = self.norm_q(q)
        k = self.norm_k(k)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        out = self.to_out(out)

        return out


class Block(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, mlp_ratio: int = 4):
        super().__init__()
        self.nm_msa = NormModulate(hidden_size)
        self.nm_mlp = NormModulate(hidden_size)

        self.attn = Attention(hidden_size, num_attention_heads)

        self.fc1 = nn.Linear(hidden_size, hidden_size * mlp_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size * mlp_ratio, hidden_size)

    def forward(self, hidden_states, attention_mask, t_emb):
        residual = hidden_states

        hidden_states, gate_msa = self.nm_msa(hidden_states, t_emb)
        hidden_states = self.attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states * gate_msa[:, None, :]

        residual = hidden_states

        hidden_states, gate_mlp = self.nm_mlp(hidden_states, t_emb)
        hidden_states = self.fc2(self.act(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states * gate_mlp[:, None, :]

        return hidden_states


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor):
        timesteps_proj = self.time_proj(timesteps)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class LatentEncoder(nn.Module):
    def __init__(
        self,
        txt_in_size: int,
        num_layers: int = 2,
        hidden_size: int = 512,
        num_attention_heads: int = 4,
        mlp_ratio: int = 2,
        num_latents: int = 256,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.time_embedder = TimestepProjEmbeddings(hidden_size)
        self.blocks = nn.ModuleList(
            [
                Block(hidden_size, num_attention_heads, mlp_ratio)
                for i in range(num_layers)
            ]
        )
        self.latents = nn.Parameter(
            torch.rand((num_latents, hidden_size)) * hidden_size**-0.5
        )
        latents_mask = torch.ones((num_latents,))
        self.register_buffer("latents_mask", latents_mask)
        self.txt_norm = nn.LayerNorm(txt_in_size)
        self.txt_in = nn.Linear(txt_in_size, hidden_size)

        self.norm_out = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        t_emb = self.time_embedder(timesteps)
        encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
        B = encoder_hidden_states.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        latents_mask = self.latents_mask.unsqueeze(0).expand(B, -1)
        hidden_states = torch.cat([latents, encoder_hidden_states], dim=1)
        mask = torch.cat([latents_mask, attention_mask], dim=1)

        for block in self.blocks:
            hidden_states = block(hidden_states, mask, t_emb)

        hidden_states = hidden_states[:, : self.num_latents, :]
        hidden_states = self.proj_out(self.norm_out(hidden_states))

        return hidden_states


if __name__ == "__main__":
    model = LatentEncoder(
        txt_in_size=768,
        num_layers=2,
        hidden_size=512,
        num_attention_heads=4,
        mlp_ratio=2,
        num_latents=256,
    )
    text_emb = torch.randn(2, 10, 768)  # B=2, seq=10, dim=768
    mask = torch.ones(2, 10)  # All attend
    timesteps = torch.tensor([100, 200])
    out = model(text_emb, mask, timesteps)
    print(out.shape)  # Should be (2, 256, 512)
