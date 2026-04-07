from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TokenLinearProjector(nn.Module):
    """
    逐 token 线性映射：(B, N, C) -> (B, N, H)
    用于将预提取的 SARCLIP patch token 特征映射到 LLM embedding 空间。
    """

    def __init__(self, in_dim: int, llm_hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.llm_hidden_size = int(llm_hidden_size)

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
        )

        nn.init.xavier_uniform_(self.net[0].weight, gain=0.5)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) -> (B, N, H)
        return self.net(x)
