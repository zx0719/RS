from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

try:
    from transformers import AutoTokenizer, AutoConfig
    from transformers import Qwen3VLForConditionalGeneration
except Exception as e:
    raise RuntimeError(
        "需要较新的 transformers，并包含 Qwen3-VL。\n"
        "建议：pip install -U transformers accelerate safetensors"
    ) from e

from sarclip_module import TokenLinearProjector


def infer_qwen3_vl_text_hidden_size(
    qwen_path: str,
    trust_remote_code: bool = True,
) -> int:
    cfg = AutoConfig.from_pretrained(qwen_path, trust_remote_code=trust_remote_code)

    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)

    if hasattr(cfg, "to_dict"):
        d = cfg.to_dict()
        text_cfg = d.get("text_config", None)
        if isinstance(text_cfg, dict) and "hidden_size" in text_cfg:
            return int(text_cfg["hidden_size"])

    raise AttributeError(
        f"无法从 Qwen3-VL config 中读取 text_config.hidden_size, config={type(cfg)}"
    )


class SarQwenVLForCausalLM(nn.Module):
    """
    将预提取 SAR patch 特征通过 projector 映射到 Qwen3-VL 语言 token embedding 空间。

    - 不使用 Qwen3-VL 自带视觉塔
    - Qwen3-VL 全程冻结，只训练 projector
    - 训练入口：train_pt.py 中的 forward_pt()
    """

    def __init__(
        self,
        qwen_path: str,
        projector: TokenLinearProjector,
        image_token: str = "<sar>",
        torch_dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device] = "cpu",
        trust_remote_code: bool = True,
        low_cpu_mem_usage: bool = True,
        device_map: Union[str, dict, None] = "cpu",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.image_token = image_token
        self.projector = projector

        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_path,
            trust_remote_code=trust_remote_code,
        )

        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.image_token]}
            )

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_path,
            torch_dtype=torch_dtype,
            # dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
        )

        # 扩 tokenizer（增加 <sar> token 后需要 resize embedding）
        self.vl_model.resize_token_embeddings(len(self.tokenizer))

        # 取语言模块
        if hasattr(self.vl_model, "language_model"):
            self.llm_causal = self.vl_model.language_model
        elif hasattr(self.vl_model, "model") and hasattr(self.vl_model.model, "language_model"):
            self.llm_causal = self.vl_model.model.language_model
        else:
            raise AttributeError(
                "无法在 Qwen3VLForConditionalGeneration 中找到 language_model。"
            )

        # 取 language backbone（支持 get_input_embeddings 的那层）
        if hasattr(self.llm_causal, "get_input_embeddings"):
            self.llm_backbone = self.llm_causal
        elif hasattr(self.llm_causal, "model") and hasattr(self.llm_causal.model, "get_input_embeddings"):
            self.llm_backbone = self.llm_causal.model
        else:
            raise AttributeError(
                "无法从 language_model 中定位可用的 language backbone。"
            )

        # 取 lm_head
        if hasattr(self.llm_causal, "lm_head"):
            self.lm_head = self.llm_causal.lm_head
        elif hasattr(self.vl_model, "lm_head"):
            self.lm_head = self.vl_model.lm_head
        else:
            raise AttributeError("无法找到 lm_head。")

        emb = self.llm_backbone.get_input_embeddings()
        self.llm_hidden_size = int(emb.weight.shape[1])

        # 预计算并缓存 embedding 均值量级（frozen，永不变化）
        # 避免每步 forward_pt 里做 .item() 触发 GPU→CPU 同步屏障
        with torch.no_grad():
            self._emb_scale: float = float(emb.weight.abs().mean().item())

        # 冻结整个 VL 模型，只训练 projector
        self.vl_model.eval()
        for p in self.vl_model.parameters():
            p.requires_grad = False

        self.sar_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        # projector 放主卡（cuda:0）
        self.main_device = torch.device(device)
        self.projector.to(self.main_device)

        # 语言模型输入 embedding 所在设备及 dtype（由 device_map 决定）
        self.lm_input_device = self.llm_backbone.get_input_embeddings().weight.device
        self.lm_emb_dtype    = self.llm_backbone.get_input_embeddings().weight.dtype

        if gradient_checkpointing:
            if hasattr(self.llm_causal, "gradient_checkpointing_enable"):
                self.llm_causal.gradient_checkpointing_enable()
                print("[INFO] gradient checkpointing enabled for language_model")

    @property
    def device(self) -> torch.device:
        return self.main_device

    def _check_projector_dim(self, sar_token_embeds: torch.Tensor) -> None:
        if sar_token_embeds.shape[-1] != self.llm_hidden_size:
            raise ValueError(
                f"projector 输出维度与 language backbone embedding 维度不匹配: "
                f"{sar_token_embeds.shape[-1]} vs {self.llm_hidden_size}"
            )

    # def _inject_sar_embeds(
    #     self,
    #     input_ids: torch.Tensor,
    #     sar_token_embeds: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     input_ids: (B, L)
    #     sar_token_embeds: (B, T, H)
    #     return: inputs_embeds (B, L, H)

    #     用 masked_scatter（非原位）替代原先的 in-place []= 赋值：
    #     - 原先对有 grad_fn 的 text_embeds 做 in-place 修改，
    #       在 gradient_checkpointing 重算时可能触发 autograd 版本号冲突；
    #     - masked_scatter 生成新 tensor，梯度正确反传至 sar_token_embeds（→ projector），
    #       同时消除 batch 维的 Python 循环。
    #     """
    #     _, t, _ = sar_token_embeds.shape

    #     sar_mask = (input_ids == self.sar_token_id)  # (B, L) bool
    #     counts   = sar_mask.sum(dim=1)               # (B,)
    #     if not (counts == t).all():
    #         bad_idx = (counts != t).nonzero(as_tuple=False).flatten().tolist()
    #         raise ValueError(
    #             f"<sar> token 数量不匹配（期望每样本 {t} 个）:\n"
    #             f"  样本索引 {bad_idx} 的实际数量 = {counts[bad_idx].tolist()}\n"
    #             f"  请检查 collate_fn 中 prompt 的构建逻辑。"
    #         )

    #     text_embeds = self.llm_backbone.get_input_embeddings()(input_ids)  # (B, L, H)
    #     mask_3d     = sar_mask.unsqueeze(-1).expand_as(text_embeds)        # (B, L, H)

    #     # masked_scatter：按 mask 的行优先 True 位置，依次填入 sar_token_embeds 的值
    #     # 要求 sar 位置在序列中按样本顺序排列（与 sar_token_embeds 的 batch 轴一致），
    #     # collate_fn 将所有 <sar> token 放在 prompt 最前面，保证了这一顺序。
    #     return text_embeds.masked_scatter(mask_3d, sar_token_embeds.reshape(-1))


    def _inject_sar_embeds(self, input_ids: torch.Tensor, sar_token_embeds: torch.Tensor) -> torch.Tensor:
        """
        input_ids:        [B, L]
        sar_token_embeds: [B, N, H]
        """
        text_embeds = self.llm_backbone.get_input_embeddings()(input_ids)

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids shape 非法: {tuple(input_ids.shape)}")

        if sar_token_embeds.dim() != 3:
            raise ValueError(f"sar_token_embeds shape 非法: {tuple(sar_token_embeds.shape)}")

        bsz, seq_len = input_ids.shape
        b2, num_img_tokens, hidden = sar_token_embeds.shape

        if bsz != b2:
            raise ValueError(
                f"batch size 不匹配: input_ids batch={bsz}, sar_token_embeds batch={b2}"
            )

        if hidden != text_embeds.size(-1):
            raise ValueError(
                f"hidden size 不匹配: sar={hidden}, text_embeds={text_embeds.size(-1)}"
            )

        # 找到 <sar> token 的位置
        mask = (input_ids == self.sar_token_id)   # [B, L]
        counts = mask.sum(dim=1)                    # [B]

        # 每个样本都应该恰好有 num_img_tokens 个 <sar>
        bad = (counts != num_img_tokens)
        if bad.any():
            bad_idx = torch.nonzero(bad, as_tuple=False).view(-1).tolist()
            detail = ", ".join(
                [f"sample{i}: count={int(counts[i].item())}, need={num_img_tokens}" for i in bad_idx]
            )
            raise ValueError(f"<sar> token 数量与视觉 token 数量不匹配: {detail}")

        # dtype / device 对齐：这是你这次报错的关键修复
        if sar_token_embeds.dtype != text_embeds.dtype:
            sar_token_embeds = sar_token_embeds.to(text_embeds.dtype)

        if sar_token_embeds.device != text_embeds.device:
            sar_token_embeds = sar_token_embeds.to(text_embeds.device)

        mask_3d = mask.unsqueeze(-1).expand(-1, -1, text_embeds.size(-1))  # [B, L, H]

        out = text_embeds.masked_scatter(mask_3d, sar_token_embeds.reshape(-1))
        return out

    def train(self, mode: bool = True) -> "SarQwenVLForCausalLM":
        super().train(mode)
        # vl_model 始终保持 eval，防止 dropout 等被激活
        self.vl_model.eval()
        return self
