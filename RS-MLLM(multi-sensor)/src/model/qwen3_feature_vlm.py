from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None

from src.model.modules.feature_fusion import FeatureFusion
from src.model.modules.modality_projectors import MSTokenProjector, SARTokenProjector
from src.model.modules.perceiver_resampler import PerceiverResampler


class Qwen3FeatureVLM(nn.Module):
    """
    Stage1 版本：
    - 主干使用 Qwen3-VL
    - 不走原生 RGB 视觉塔
    - 将 MS/SAR 离线特征桥接为伪视觉 tokens
    - 用特殊 token <rs_patch> 占位，再替换成融合后的模态 tokens
    """

    def __init__(
        self,
        qwen_name_or_path: str,
        image_token: str = "<rs_patch>",
        bridge_dim: int = 512,
        num_ms_queries: int = 32,
        num_sar_tokens: int = 4,
        num_resampler_layers: int = 2,
        num_fusion_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_llm: bool = True,
        trust_remote_code: bool = True,
    ):
        super().__init__()

        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Current transformers does not expose Qwen3VLForConditionalGeneration. "
                "Please upgrade transformers or verify your local install."
            )

        self.image_token = image_token

        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_name_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [image_token]}
            )

        self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_name_or_path,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Qwen3-VL 语言隐藏维度
        if hasattr(self.llm.config, "text_config") and hasattr(self.llm.config.text_config, "hidden_size"):
            self.llm_hidden_size = int(self.llm.config.text_config.hidden_size)
        elif hasattr(self.llm.config, "hidden_size"):
            self.llm_hidden_size = int(self.llm.config.hidden_size)
        else:
            raise RuntimeError("Cannot infer hidden size from Qwen3-VL config")

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)

        self.ms_projector = MSTokenProjector(
            in_channels=1024,
            out_dim=bridge_dim,
            dropout=dropout,
        )
        self.ms_resampler = PerceiverResampler(
            dim=bridge_dim,
            num_latents=num_ms_queries,
            depth=num_resampler_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.sar_projector = SARTokenProjector(
            in_dim=512,
            out_dim=bridge_dim,
            num_tokens=num_sar_tokens,
            dropout=dropout,
        )
        self.fusion = FeatureFusion(
            dim=bridge_dim,
            llm_hidden_size=self.llm_hidden_size,
            num_layers=num_fusion_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

    def trainable_parameters(self):
        modules = [
            self.ms_projector,
            self.ms_resampler,
            self.sar_projector,
            self.fusion,
        ]
        for module in modules:
            for p in module.parameters():
                if p.requires_grad:
                    yield p

    def encode_modalities(
        self,
        ms_feat: torch.Tensor,
        sar_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ms_tokens = self.ms_projector(ms_feat)         # [B, 1024, D]
        ms_tokens = self.ms_resampler(ms_tokens)       # [B, Nms, D]
        sar_tokens = self.sar_projector(sar_feat)      # [B, Nsar, D]

        fused_tokens = self.fusion(ms_tokens, sar_tokens)  # [B, Nms+Nsar, Hllm]

        ms_global = ms_tokens.mean(dim=1)
        sar_global = sar_tokens.mean(dim=1)
        return fused_tokens, ms_global, sar_global

    def _tokenize_prompt_and_target(
        self,
        prompt_texts: List[str],
        target_texts: List[str],
    ) -> Dict[str, List[List[int]]]:
        full_ids_list = []
        labels_list = []

        for prompt_text, target_text in zip(prompt_texts, target_texts):
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
            eos_id = [self.tokenizer.eos_token_id]

            full_ids = prompt_ids + target_ids + eos_id
            labels = [-100] * len(prompt_ids) + target_ids + eos_id

            full_ids_list.append(full_ids)
            labels_list.append(labels)

        return {
            "full_ids_list": full_ids_list,
            "labels_list": labels_list,
        }

    def _build_inputs_embeds_and_labels(
        self,
        full_ids_list: List[List[int]],
        labels_list: List[List[int]],
        fused_tokens: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_layer = self.llm.get_input_embeddings()
        emb_dtype = emb_layer.weight.dtype

        batch_embeds = []
        batch_labels = []
        batch_attn = []

        for i, (full_ids, labels) in enumerate(zip(full_ids_list, labels_list)):
            input_ids = torch.tensor(full_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            text_embeds = emb_layer(input_ids)  # [T, H]
            modal_tokens = fused_tokens[i].to(device=device, dtype=emb_dtype)  # [M, H]

            pos = (input_ids == self.image_token_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                raise ValueError(
                    f"Prompt must contain image token {self.image_token}, but not found."
                )
            image_pos = int(pos[0].item())

            left_emb = text_embeds[:image_pos]
            right_emb = text_embeds[image_pos + 1 :]
            merged_emb = torch.cat([left_emb, modal_tokens, right_emb], dim=0)

            left_lab = labels_t[:image_pos]
            right_lab = labels_t[image_pos + 1 :]
            modal_lab = torch.full(
                (modal_tokens.size(0),),
                -100,
                dtype=torch.long,
                device=device,
            )
            merged_lab = torch.cat([left_lab, modal_lab, right_lab], dim=0)

            merged_attn = torch.ones(merged_emb.size(0), dtype=torch.long, device=device)

            batch_embeds.append(merged_emb)
            batch_labels.append(merged_lab)
            batch_attn.append(merged_attn)

        max_len = max(x.size(0) for x in batch_embeds)
        hidden_size = batch_embeds[0].size(-1)

        padded_embeds = []
        padded_labels = []
        padded_attn = []

        for embeds, labels, attn in zip(batch_embeds, batch_labels, batch_attn):
            pad_len = max_len - embeds.size(0)

            if pad_len > 0:
                pad_embeds = torch.zeros(
                    pad_len,
                    hidden_size,
                    dtype=embeds.dtype,
                    device=device,
                )
                pad_labels = torch.full(
                    (pad_len,),
                    -100,
                    dtype=torch.long,
                    device=device,
                )
                pad_attn = torch.zeros(
                    pad_len,
                    dtype=torch.long,
                    device=device,
                )

                embeds = torch.cat([embeds, pad_embeds], dim=0)
                labels = torch.cat([labels, pad_labels], dim=0)
                attn = torch.cat([attn, pad_attn], dim=0)

            padded_embeds.append(embeds)
            padded_labels.append(labels)
            padded_attn.append(attn)

        inputs_embeds = torch.stack(padded_embeds, dim=0)
        labels = torch.stack(padded_labels, dim=0)
        attention_mask = torch.stack(padded_attn, dim=0)

        return inputs_embeds, attention_mask, labels

    def forward(
        self,
        ms_feat: torch.Tensor,
        sar_feat: torch.Tensor,
        prompt_text: List[str],
        target_text: List[str],
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        ms_feat = ms_feat.to(device)
        sar_feat = sar_feat.to(device)

        fused_tokens, ms_global, sar_global = self.encode_modalities(ms_feat, sar_feat)

        tokenized = self._tokenize_prompt_and_target(prompt_text, target_text)
        inputs_embeds, attention_mask, labels = self._build_inputs_embeds_and_labels(
            full_ids_list=tokenized["full_ids_list"],
            labels_list=tokenized["labels_list"],
            fused_tokens=fused_tokens,
            device=device,
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return {
            "loss": outputs.loss,
            "lm_loss": outputs.loss,
            "logits": outputs.logits,
            "labels": labels,
            "attention_mask": attention_mask,
            "ms_global": ms_global,
            "sar_global": sar_global,
        }
