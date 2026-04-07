"""
mixed_dataset_pt.py
===================
直接加载预提取的 SARCLIP .pt 特征文件进行训练，跳过图像读取和 encoder 推理。

核心变化（对比 mixed_dataset.py）：
  - Dataset.__getitem__ 返回 "sar_feat": Tensor(195, 768)，而非 "image": PIL
  - collate_fn 中 batch["sar_feats"] = stack of (195,768) → (B,195,768)
  - 不再需要 image_processor 参数

json 格式要求（在原 caption json 基础上新增 pt_path 字段）：
  {
    "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    "images": ["..."],
    "pt_path": "/mnt/.../xxx.pt"   ← get_pt*.py 写入的字段
  }
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────

@dataclass
class PtCaptionItem:
    image_id:   str
    pt_path:    str
    prompt:     str
    caption:    str
    user_role:  str = "user"
    assistant_role: str = "assistant"


class PtCaptionDataset(Dataset):
    """
    直接从 _pt.json 里读取预提取特征路径，加载 .pt 文件。

    参数：
      root         数据集根目录（用于构建 json 绝对路径）
      pt_json      相对 root 的 _pt.json 路径
      max_samples  调试用截断
    """

    def __init__(
        self,
        root: str,
        pt_json: str,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root     = Path(root)
        self.json_path = self.root / pt_json

        if not self.json_path.exists():
            raise FileNotFoundError(f"pt_json 不存在: {self.json_path}")

        self.items: List[PtCaptionItem] = []
        self._build_items()

        if max_samples is not None:
            self.items = self.items[: int(max_samples)]

        if len(self.items) == 0:
            raise RuntimeError(
                f"PtCaptionDataset: items=0，请检查 json 和 pt_path 字段。\n"
                f"json={self.json_path}"
            )

    def _build_items(self) -> None:
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"JSON 顶层应为 list，当前: {type(data)}")

        n_missing_pt = 0
        for sample in data:
            if not isinstance(sample, dict):
                continue

            pt_path = sample.get("pt_path", "")
            if not pt_path or not Path(pt_path).exists():
                n_missing_pt += 1
                continue

            messages = sample.get("messages", [])
            if not isinstance(messages, list) or len(messages) < 2:
                continue

            user_msg      = messages[0]
            assistant_msg = messages[1]
            if not isinstance(user_msg, dict) or not isinstance(assistant_msg, dict):
                continue

            user_role      = str(user_msg.get("role", "")).strip().lower()
            assistant_role = str(assistant_msg.get("role", "")).strip().lower()
            if user_role != "user" or assistant_role != "assistant":
                continue

            prompt  = str(user_msg.get("content", "")).strip()
            caption = str(assistant_msg.get("content", "")).strip()
            if not prompt or not caption:
                continue

            images   = sample.get("images", [])
            image_id = Path(images[0]).stem if images else Path(pt_path).stem

            self.items.append(PtCaptionItem(
                image_id       = image_id,
                pt_path        = pt_path,
                prompt         = prompt,
                caption        = caption,
                user_role      = user_role,
                assistant_role = assistant_role,
            ))

        if n_missing_pt > 0:
            print(f"[WARN] PtCaptionDataset: {n_missing_pt} 条 pt_path 缺失或文件不存在，已跳过")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        try:
            feat = torch.load(it.pt_path, map_location="cpu", weights_only=True)
            finite_mask = torch.isfinite(feat)
            if not finite_mask.all():
                return {
                    "_bad_sample": True,
                    "_bad_reason": (
                        f"non-finite feat: finite_ratio="
                        f"{finite_mask.float().mean().item():.6f}, "
                        f"abs_max={torch.nan_to_num(feat).abs().max().item():.6e}"
                    ),
                    "pt_path": it.pt_path,
                    "image_id": it.image_id,
                }

            abs_max = feat.abs().max().item()
            if abs_max > 1e4:
                return {
                    "_bad_sample": True,
                    "_bad_reason": f"feat abs_max too large: {abs_max:.6e}",
                    "pt_path": it.pt_path,
                    "image_id": it.image_id,
                }
            # feat: (195, 768) float32
        except Exception as e:
            return {
                "_bad_sample": True,
                "_bad_reason": str(e),
                "pt_path":     it.pt_path,
                "image_id":    it.image_id,
            }

        return {
            "sar_feat":      feat,           # (195, 768)
            "image_id":      it.image_id,
            "prompt":        it.prompt,
            "caption":       it.caption,
            "pt_path":       it.pt_path,
            "user_role":     it.user_role,
            "assistant_role":it.assistant_role,
        }


# ─────────────────────────────────────────────────────────────
# 2. MultiPtCaptionDataset（加权混训，对应 MultiCaptionDataset）
# ─────────────────────────────────────────────────────────────

class MultiPtCaptionDataset(Dataset):
    """按权重从多个 PtCaptionDataset 中采样，接口与 MultiCaptionDataset 一致。"""

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        weights:  Dict[str, float],
        epoch_length: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if not datasets:
            raise ValueError("datasets 不能为空")

        self.seed = int(seed)
        datasets  = {k: v for k, v in datasets.items() if len(v) > 0}
        if not datasets:
            raise RuntimeError("所有 datasets 都为空")

        active = [k for k in datasets if float(weights.get(k, 0)) > 0]
        if not active:
            raise ValueError("至least 一个数据集权重 > 0")

        total_w      = sum(float(weights[k]) for k in active)
        self.datasets = {k: datasets[k] for k in active}
        self.names    = active
        self.probs    = [float(weights[k]) / total_w for k in active]

        if epoch_length is None:
            epoch_length = max(len(ds) for ds in self.datasets.values())
        self.epoch_length = int(epoch_length)

        print("[INFO] MultiPtCaptionDataset loaded:")
        for name in self.names:
            print(f"  - {name}: size={len(self.datasets[name])}, prob={float(weights[name])/total_w:.4f}")

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng     = random.Random(self.seed + idx)
        ds_name = rng.choices(self.names, weights=self.probs, k=1)[0]
        ds      = self.datasets[ds_name]

        rng2 = random.Random(self.seed * 1000003 + idx)
        j    = rng2.randrange(len(ds))

        item = dict(ds[j])
        item["_source"] = ds_name
        return item


# ─────────────────────────────────────────────────────────────
# 3. collate_fn
# ─────────────────────────────────────────────────────────────

def build_pt_collate_fn(
    tokenizer,
    image_token:      str = "<sar>",
    num_image_tokens: int = 195,
    max_length:       int = 512,
    role_prefix_map:  Optional[Dict[str, str]] = None,
) -> Callable:
    """
    返回一个 collate_fn。

    batch 输出：
      sar_feats       (B, N, C)   预提取特征，直接送 projector
      input_ids       (B, L)
      attention_mask  (B, L)
      labels          (B, L)
      prompts / captions / sources  List[str]
    """
    image_tokens    = " ".join([image_token] * num_image_tokens)
    role_prefix_map = role_prefix_map or {
        "user": "User", "assistant": "Assistant", "system": "System",
    }

    def _strip_image_tag(text: str) -> str:
        return str(text).strip().replace("<image>", "").strip()

    def _role_prefix(role: str) -> str:
        role = str(role).strip().lower()
        return role_prefix_map.get(role, role.capitalize() or "User")

    def _collate(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        good  = [b for b in batch if not b.get("_bad_sample", False)]
        bads  = [b for b in batch if b.get("_bad_sample", False)]

        if bads:
            print(f"[WARN] dropped {len(bads)} bad .pt samples")
            for x in bads[:3]:
                print("[BAD PT]", x.get("pt_path", "NA"), x.get("_bad_reason", ""))

        if not good:
            return None

        batch = good

        # ── 特征 stack ──────────────────────────────────────────
        sar_feats = torch.stack([b["sar_feat"] for b in batch], dim=0)  # (B, 195, 768)

        # ── 文本构建 ────────────────────────────────────────────
        prompts_raw     = [_strip_image_tag(b["prompt"])           for b in batch]
        captions        = [str(b["caption"]).strip()               for b in batch]
        user_roles      = [str(b.get("user_role",      "user"))    for b in batch]
        assistant_roles = [str(b.get("assistant_role", "assistant")) for b in batch]
        sources         = [str(b.get("_source", "unknown"))        for b in batch]
        image_ids = [str(b.get("image_id", "")) for b in batch]
        pt_paths  = [str(b.get("pt_path", "")) for b in batch]

        prompts    = [
            f"{image_tokens}\n{_role_prefix(ur)}: {p}\n{_role_prefix(ar)}: "
            for p, ur, ar in zip(prompts_raw, user_roles, assistant_roles)
        ]
        full_texts = [p + c for p, c in zip(prompts, captions)]

        tok_full = tokenizer(
            full_texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        tok_prompt = tokenizer(
            prompts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )

        input_ids      = tok_full["input_ids"]
        attention_mask = tok_full["attention_mask"]
        labels         = input_ids.clone()
        prompt_lens    = tok_prompt["attention_mask"].sum(dim=1)

        for i, pl in enumerate(prompt_lens.tolist()):
            labels[i, :pl] = -100
        labels[attention_mask == 0] = -100

        sample_valid = (labels != -100).sum(dim=1) > 0
        if not sample_valid.any():
            print("[WARN] all samples in batch have no valid target tokens")
            return None

        dropped = int((~sample_valid).sum().item())
        if dropped > 0:
            print(f"[WARN] dropped {dropped} samples with no valid target tokens")

        keep = sample_valid.nonzero(as_tuple=False).flatten().tolist()
        sar_feats      = sar_feats[sample_valid]
        input_ids      = input_ids[sample_valid]
        attention_mask = attention_mask[sample_valid]
        labels         = labels[sample_valid]
        prompts        = [prompts[i]        for i in keep]
        captions       = [captions[i]       for i in keep]
        sources        = [sources[i]        for i in keep]

        return {
            "sar_feats":      sar_feats,       # (B, 195, 768)  ← 取代 sar_images
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "prompts":        prompts,
            "captions":       captions,
            "sources":        sources,
            "image_ids":      image_ids,
            "pt_paths":       pt_paths,
        }

    return _collate
