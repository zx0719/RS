import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, latent: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(latent)
        kv = self.norm_kv(x)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        latent = latent + attn_out
        latent = latent + self.ffn(self.norm_ffn(latent))
        return latent


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim: int,
        num_latents: int = 32,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        latents = self.latents.expand(bsz, -1, -1)
        for block in self.blocks:
            latents = block(latents, x)
        return self.norm(latents)
