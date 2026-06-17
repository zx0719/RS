from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

try:
    from transformers import AutoConfig, AutoTokenizer
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        # transformers < 4.52 doesn't have Qwen3-VL; fall back to Qwen2.5-VL
        from transformers import Qwen2_5_VLForConditionalGeneration as Qwen3VLForConditionalGeneration  # noqa: N812
except Exception as e:
    raise RuntimeError(
        "需要 transformers >= 4.49（含 Qwen2.5-VL）或 >= 4.52（含 Qwen3-VL）。\n"
        "建议：pip install -U transformers accelerate safetensors"
    ) from e

from src.model.sarclip_module import TokenLinearProjector


def infer_qwen3_vl_text_hidden_size(
    qwen_path: str,
    trust_remote_code: bool = True,
) -> int:
    cfg = AutoConfig.from_pretrained(qwen_path, trust_remote_code=trust_remote_code)

    # Qwen3-VL: hidden_size lives under text_config
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)

    if hasattr(cfg, "to_dict"):
        cfg_dict = cfg.to_dict()
        text_cfg = cfg_dict.get("text_config")
        if isinstance(text_cfg, dict) and "hidden_size" in text_cfg:
            return int(text_cfg["hidden_size"])

    # Qwen2.5-VL: hidden_size is a top-level attribute
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)

    raise AttributeError(
        f"无法从 VL config 中读取 hidden_size, config={type(cfg)}"
    )


class SarQwenVLForCausalLM(nn.Module):
    """
    将预提取 SAR patch 特征通过 projector 映射到 Qwen3-VL 语言 token embedding 空间。
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

        # flash_attention_2 需要单独安装；eager 在 bf16+gradient_checkpointing 下最稳定
        _attn_impl = "eager"
        try:
            import flash_attn  # noqa: F401
            _attn_impl = "flash_attention_2"
        except ImportError:
            pass

        self.vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_path,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            attn_implementation=_attn_impl,
        )
        print(f"[INFO] attn_implementation = {_attn_impl}")
        if device_map is None:
            self.vl_model.to(device)
        old_vocab = self.vl_model.get_input_embeddings().weight.shape[0]
        self.vl_model.resize_token_embeddings(len(self.tokenizer))
        # resize_token_embeddings 用 torch.empty 初始化新行，bf16 下可能含 NaN
        # 将新增行清零以保证有限值
        new_vocab = self.vl_model.get_input_embeddings().weight.shape[0]
        if new_vocab > old_vocab:
            with torch.no_grad():
                self.vl_model.get_input_embeddings().weight[old_vocab:].zero_()

        if hasattr(self.vl_model, "language_model"):
            # Qwen3-VL layout: vl_model.language_model is the causal LM
            self.llm_causal = self.vl_model.language_model
        elif hasattr(self.vl_model, "model") and hasattr(self.vl_model.model, "language_model"):
            self.llm_causal = self.vl_model.model.language_model
        elif hasattr(self.vl_model, "model"):
            # Qwen2.5-VL layout: vl_model.model is the backbone; treat vl_model as causal LM
            self.llm_causal = self.vl_model
        else:
            raise AttributeError("无法在 VL 模型中找到 language_model 或 model。")

        if hasattr(self.llm_causal, "get_input_embeddings"):
            self.llm_backbone = self.llm_causal
        elif hasattr(self.llm_causal, "model") and hasattr(self.llm_causal.model, "get_input_embeddings"):
            self.llm_backbone = self.llm_causal.model
        else:
            raise AttributeError("无法从 language_model 中定位可用的 language backbone。")

        if hasattr(self.llm_causal, "lm_head"):
            self.lm_head = self.llm_causal.lm_head
        elif hasattr(self.vl_model, "lm_head"):
            self.lm_head = self.vl_model.lm_head
        else:
            raise AttributeError("无法找到 lm_head。")

        emb = self.llm_backbone.get_input_embeddings()
        self.llm_hidden_size = int(emb.weight.shape[1])
        with torch.no_grad():
            self._emb_scale = float(emb.weight.abs().mean().item())

        self.vl_model.eval()
        for p in self.vl_model.parameters():
            p.requires_grad = False

        self.sar_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.main_device = torch.device(device)
        self.projector.to(self.main_device)
        # lm_input_device / lm_emb_dtype 用 property 动态读取，
        # 避免 DeepSpeed 接管后 embedding 已移到 GPU 但缓存值还是 cpu
        self._lm_emb_dtype_cache = self.llm_backbone.get_input_embeddings().weight.dtype

        if gradient_checkpointing and hasattr(self.llm_causal, "gradient_checkpointing_enable"):
            self.llm_causal.gradient_checkpointing_enable()
            print("[INFO] gradient checkpointing enabled for language_model")

    @property
    def lm_input_device(self) -> torch.device:
        return self.llm_backbone.get_input_embeddings().weight.device

    @property
    def lm_emb_dtype(self) -> torch.dtype:
        return self.llm_backbone.get_input_embeddings().weight.dtype

    @property
    def device(self) -> torch.device:
        return self.main_device

    def _check_projector_dim(self, sar_token_embeds: torch.Tensor) -> None:
        if sar_token_embeds.shape[-1] != self.llm_hidden_size:
            raise ValueError(
                "projector 输出维度与 language backbone embedding 维度不匹配: "
                f"{sar_token_embeds.shape[-1]} vs {self.llm_hidden_size}"
            )

    def _inject_sar_embeds(self, input_ids: torch.Tensor, sar_token_embeds: torch.Tensor) -> torch.Tensor:
        text_embeds = self.llm_backbone.get_input_embeddings()(input_ids)

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids shape 非法: {tuple(input_ids.shape)}")
        if sar_token_embeds.dim() != 3:
            raise ValueError(f"sar_token_embeds shape 非法: {tuple(sar_token_embeds.shape)}")

        bsz, _ = input_ids.shape
        b2, num_img_tokens, hidden = sar_token_embeds.shape
        if bsz != b2:
            raise ValueError(
                f"batch size 不匹配: input_ids batch={bsz}, sar_token_embeds batch={b2}"
            )
        if hidden != text_embeds.size(-1):
            raise ValueError(
                f"hidden size 不匹配: sar={hidden}, text_embeds={text_embeds.size(-1)}"
            )

        mask = input_ids == self.sar_token_id
        counts = mask.sum(dim=1)
        bad = counts != num_img_tokens
        if bad.any():
            bad_idx = torch.nonzero(bad, as_tuple=False).view(-1).tolist()
            detail = ", ".join(
                f"sample{i}: count={int(counts[i].item())}, need={num_img_tokens}"
                for i in bad_idx
            )
            raise ValueError(f"<sar> token 数量与视觉 token 数量不匹配: {detail}")

        if sar_token_embeds.dtype != text_embeds.dtype:
            sar_token_embeds = sar_token_embeds.to(text_embeds.dtype)
        if sar_token_embeds.device != text_embeds.device:
            sar_token_embeds = sar_token_embeds.to(text_embeds.device)

        mask_3d = mask.unsqueeze(-1).expand(-1, -1, text_embeds.size(-1))
        return text_embeds.masked_scatter(mask_3d, sar_token_embeds.reshape(-1))

    def train(self, mode: bool = True) -> "SarQwenVLForCausalLM":
        super().train(mode)
        self.vl_model.eval()
        return self
