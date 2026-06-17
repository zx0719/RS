from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return self.net(x)


class BridgeGuidedProjector(nn.Module):
    """
    Bridge-guided projector: SARCLIP (768d) -> Qwen3 (2560d)

    架构：
      z_bridge   = W_init @ x          # closed-form 线性桥（从线性度实验导出）
      z_residual = MLP(x)               # 残差路径（学习局部结构）
      alpha      = sigmoid(gate(x))     # 可学习门控
      out        = alpha * z_bridge + (1-alpha) * z_residual

    训练策略：
      Stage A（对齐）: 冻结 bridge，只训 residual + gate
      Stage B（精调）: 全部解冻，bridge 用 0.1x lr

    用法：
      proj = BridgeGuidedProjector.from_bridge_ckpt(
          bridge_ckpt_path='path/to/bridge_W_sarclip_qwen.pt',
          dim_hidden=1024,
      )
    """

    def __init__(
        self,
        dim_sar: int = 768,
        dim_qwen: int = 2560,
        dim_hidden: int = 1024,
        bridge_ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.dim_sar = dim_sar
        self.dim_qwen = dim_qwen

        # ── Bridge path ──
        self.bridge = nn.Linear(dim_sar, dim_qwen, bias=True)

        # ── Residual path ──
        self.residual = nn.Sequential(
            nn.Linear(dim_sar, dim_hidden),
            nn.GELU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_qwen),
        )

        # ── Gate ──
        self.gate_net = nn.Sequential(
            nn.Linear(dim_sar, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

        if bridge_ckpt_path is not None:
            self.load_bridge(bridge_ckpt_path)

    def _init_weights(self) -> None:
        # residual: small init so bridge dominates at start
        for m in self.residual.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        # gate: init to 0.5 (equal mix before bridge is loaded)
        for m in self.gate_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def load_bridge(self, ckpt_path: str) -> None:
        """Load W matrix exported by linearity experiment."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        W        = ckpt["W_bridge"].float()   # (768, 2560)
        src_mean = ckpt["src_mean"].float()   # (768,)
        tgt_mean = ckpt["tgt_mean"].float()   # (2560,)

        # Scale W so fp16 matmul won't overflow:
        # fp16 max ≈ 65504; worst-case row norm * input_norm ≈ abs_max * sqrt(768)
        # We want abs_max(W) * sqrt(768) < 200 (safe margin)
        abs_max = W.abs().max().item()
        safe_limit = 200.0 / (abs_max * (768 ** 0.5) + 1e-6)
        if safe_limit < 1.0:
            scale = safe_limit
            W = W * scale
            tgt_mean = tgt_mean * scale
            print(f"[BridgeGuidedProjector] Scaled W by {scale:.4f} for fp16 safety")

        with torch.no_grad():
            self.bridge.weight.copy_(W.T)                    # (2560, 768)
            self.bridge.bias.copy_(tgt_mean - W.T @ src_mean)
        eff_rank = ckpt.get("effective_rank", "?")
        print(f"[BridgeGuidedProjector] Loaded W from {ckpt_path}  "
              f"shape={tuple(W.shape)}  effective_rank={eff_rank}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, 768) or (B, 768)
        returns: same leading dims, last dim = dim_qwen
        """
        if x.dim() == 3:
            B, N, _ = x.shape
            out = self._fwd(x.reshape(B * N, self.dim_sar))
            return out.reshape(B, N, self.dim_qwen)
        return self._fwd(x)

    def _fwd(self, x: torch.Tensor) -> torch.Tensor:
        # bridge runs in fp32 to avoid fp16 overflow (W has large values)
        x32 = x.float()
        z_bridge   = self.bridge(x32).to(x.dtype)
        z_residual = self.residual(x)
        alpha      = torch.sigmoid(self.gate_net(x))   # (B, 1)
        return alpha * z_bridge + (1.0 - alpha) * z_residual

    def freeze_bridge(self) -> None:
        for p in self.bridge.parameters():
            p.requires_grad_(False)

    def unfreeze_bridge(self) -> None:
        for p in self.bridge.parameters():
            p.requires_grad_(True)

    @classmethod
    def from_bridge_ckpt(
        cls,
        bridge_ckpt_path: str,
        dim_hidden: int = 1024,
    ) -> "BridgeGuidedProjector":
        return cls(bridge_ckpt_path=bridge_ckpt_path, dim_hidden=dim_hidden)
