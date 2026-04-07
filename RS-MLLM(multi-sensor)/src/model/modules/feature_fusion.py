import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        llm_hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.to_llm = nn.Linear(dim, llm_hidden_size)

    def forward(self, ms_tokens: torch.Tensor, sar_tokens: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ms_tokens, sar_tokens], dim=1)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.to_llm(x)
        return x
