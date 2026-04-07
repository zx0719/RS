import torch
import torch.nn as nn


class MSTokenProjector(nn.Module):
    def __init__(self, in_channels: int = 1024, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, ms_feat: torch.Tensor) -> torch.Tensor:
        x = ms_feat.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = self.norm(x)
        return x


class SARTokenProjector(nn.Module):
    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 512,
        num_tokens: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, num_tokens * out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, sar_feat: torch.Tensor) -> torch.Tensor:
        bsz = sar_feat.size(0)
        x = sar_feat.squeeze(1)
        x = self.mlp(x)
        x = x.view(bsz, self.num_tokens, self.out_dim)
        x = self.norm(x)
        return x
