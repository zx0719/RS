"""SARCLIP + Qwen3-VL pt 模式训练包

架构：预提取 SAR .pt 特征 → TokenLinearProjector → 冻结的 Qwen3-VL 语言模型
仅训练 projector，Qwen3-VL 全程冻结。
"""

__all__ = [
    "TokenLinearProjector",
    "SarQwenVLForCausalLM",
]
