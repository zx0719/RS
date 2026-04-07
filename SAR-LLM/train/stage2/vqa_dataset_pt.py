"""
vqa_dataset_pt.py
=================
Stage2 VQA 数据集：支持单轮和多轮对话，直接加载预提取的 SARCLIP .pt 特征。

支持两种 JSON 格式：
  1. messages 格式（SARTEXT train/test）：
     {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
      "images": [...], "pt_path": "..."}

  2. conversations 格式（SAR-VQA_conv）：
     {"id": "...", "image": "...", "pt_path": "...",
      "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────

@dataclass
class VqaItem:
    image_id: str
    pt_path:  str
    # 对话轮次：[(user_text, assistant_text), ...]
    turns:    List[tuple]


class PtVqaDataset(Dataset):
    """
    加载预提取特征的 VQA 数据集，支持单轮和多轮对话。

    参数：
      root         数据集根目录（用于构建 json 绝对路径）
      pt_json      相对 root 的 json 路径
      max_samples  调试用截断
    """

    def __init__(
        self,
        root: str,
        pt_json: str,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root      = Path(root)
        self.json_path = self.root / pt_json

        if not self.json_path.exists():
            raise FileNotFoundError(f"pt_json 不存在: {self.json_path}")

        self.items: List[VqaItem] = []
        self._build_items()

        if max_samples is not None:
            self.items = self.items[: int(max_samples)]

        if len(self.items) == 0:
            raise RuntimeError(
                f"PtVqaDataset: items=0，请检查 json 和 pt_path 字段。\n"
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

            turns = self._parse_turns(sample)
            if not turns:
                continue

            # image_id
            images = sample.get("images", [])
            image  = sample.get("image", "")
            if images:
                image_id = Path(images[0]).stem
            elif image:
                image_id = Path(image).stem
            else:
                image_id = Path(pt_path).stem

            self.items.append(VqaItem(
                image_id = image_id,
                pt_path  = pt_path,
                turns    = turns,
            ))

        if n_missing_pt > 0:
            print(f"[WARN] PtVqaDataset: {n_missing_pt} 条 pt_path 缺失或文件不存在，已跳过")

    def _parse_turns(self, sample: dict) -> List[tuple]:
        """解析对话轮次，返回 [(user_text, assistant_text), ...]"""
        turns = []

        # messages 格式
        if "messages" in sample:
            msgs = sample["messages"]
            if not isinstance(msgs, list):
                return []
            # 按 user/assistant 配对
            i = 0
            while i < len(msgs) - 1:
                u = msgs[i]
                a = msgs[i + 1]
                if (isinstance(u, dict) and isinstance(a, dict)
                        and str(u.get("role", "")).lower() == "user"
                        and str(a.get("role", "")).lower() == "assistant"):
                    user_text = str(u.get("content", "")).strip().replace("<image>", "").strip()
                    asst_text = str(a.get("content", "")).strip()
                    if user_text and asst_text:
                        turns.append((user_text, asst_text))
                    i += 2
                else:
                    i += 1
            return turns

        # conversations 格式（human/gpt）
        if "conversations" in sample:
            convs = sample["conversations"]
            if not isinstance(convs, list):
                return []
            i = 0
            while i < len(convs) - 1:
                h = convs[i]
                g = convs[i + 1]
                if (isinstance(h, dict) and isinstance(g, dict)
                        and str(h.get("from", "")).lower() in ("human", "user")
                        and str(g.get("from", "")).lower() in ("gpt", "assistant")):
                    user_text = str(h.get("value", "")).strip().replace("<image>", "").strip()
                    asst_text = str(g.get("value", "")).strip()
                    if user_text and asst_text:
                        turns.append((user_text, asst_text))
                    i += 2
                else:
                    i += 1
            return turns

        return []

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
                        f"{finite_mask.float().mean().item():.6f}"
                    ),
                    "pt_path":  it.pt_path,
                    "image_id": it.image_id,
                }
            if feat.abs().max().item() > 1e4:
                return {
                    "_bad_sample": True,
                    "_bad_reason": f"feat abs_max too large: {feat.abs().max().item():.6e}",
                    "pt_path":  it.pt_path,
                    "image_id": it.image_id,
                }
        except Exception as e:
            return {
                "_bad_sample": True,
                "_bad_reason": str(e),
                "pt_path":     it.pt_path,
                "image_id":    it.image_id,
            }

        return {
            "sar_feat": feat,           # (195, 768)
            "image_id": it.image_id,
            "pt_path":  it.pt_path,
            "turns":    it.turns,       # [(user, assistant), ...]
        }


# ─────────────────────────────────────────────────────────────
# 2. MultiPtVqaDataset（加权混训）
# ─────────────────────────────────────────────────────────────

class MultiPtVqaDataset(Dataset):
    """按权重从多个 PtVqaDataset 中采样。"""

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("datasets 不能为空")

        self.seed     = int(seed)
        self.datasets = {k: v for k, v in datasets.items() if len(v) > 0}
        self.names    = list(self.datasets.keys())

        total = sum(len(ds) for ds in self.datasets.values())
        self.epoch_length = total

        print("[INFO] MultiPtVqaDataset loaded:")
        for name in self.names:
            print(f"  - {name}: size={len(self.datasets[name])}")

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 按数据集大小比例采样
        rng = random.Random(self.seed + idx)
        sizes = [len(self.datasets[n]) for n in self.names]
        ds_name = rng.choices(self.names, weights=sizes, k=1)[0]
        ds = self.datasets[ds_name]

        rng2 = random.Random(self.seed * 1000003 + idx)
        j    = rng2.randrange(len(ds))

        item = dict(ds[j])
        item["_source"] = ds_name
        return item


# ─────────────────────────────────────────────────────────────
# 3. collate_fn
# ─────────────────────────────────────────────────────────────

def build_vqa_collate_fn(
    tokenizer,
    image_token:      str = "<sar>",
    num_image_tokens: int = 195,
    max_length:       int = 1024,
) -> Callable:
    """
    VQA collate_fn：支持多轮对话。

    对于多轮对话，将所有轮次拼接成一个序列：
      <sar_tokens> User: Q1 Assistant: A1 User: Q2 Assistant: A2 ...
    只对 Assistant 回答部分计算 loss（User 部分 label=-100）。
    """
    image_tokens_str = " ".join([image_token] * num_image_tokens)

    def _build_text(turns: List[tuple], with_image: bool = True) -> tuple[str, str]:
        """
        构建完整文本和 prompt（不含最后一个 assistant 回答）。
        返回 (full_text, prompt_text)
        """
        parts_full   = []
        parts_prompt = []

        for i, (user_text, asst_text) in enumerate(turns):
            if i == 0 and with_image:
                user_part = f"{image_tokens_str}\nUser: {user_text}\nAssistant: "
            else:
                user_part = f"User: {user_text}\nAssistant: "
            parts_full.append(user_part + asst_text)
            parts_prompt.append(user_part)

        full_text   = "\n".join(parts_full)
        prompt_text = "\n".join(parts_prompt)
        return full_text, prompt_text

    def _collate(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        good = [b for b in batch if not b.get("_bad_sample", False)]
        bads = [b for b in batch if b.get("_bad_sample", False)]

        if bads:
            print(f"[WARN] dropped {len(bads)} bad .pt samples")
            for x in bads[:3]:
                print("[BAD PT]", x.get("pt_path", "NA"), x.get("_bad_reason", ""))

        if not good:
            return None

        sar_feats  = torch.stack([b["sar_feat"] for b in good], dim=0)  # (B, 195, 768)
        image_ids  = [str(b.get("image_id", "")) for b in good]
        pt_paths   = [str(b.get("pt_path",  "")) for b in good]
        sources    = [str(b.get("_source",  "unknown")) for b in good]

        full_texts   = []
        prompt_texts = []
        for b in good:
            turns = b.get("turns", [])
            if not turns:
                turns = [(b.get("prompt", ""), b.get("caption", ""))]
            full, prompt = _build_text(turns)
            full_texts.append(full)
            prompt_texts.append(prompt)

        tok_full = tokenizer(
            full_texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        tok_prompt = tokenizer(
            prompt_texts, padding=True, truncation=True,
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
        return {
            "sar_feats":      sar_feats[sample_valid],
            "input_ids":      input_ids[sample_valid],
            "attention_mask": attention_mask[sample_valid],
            "labels":         labels[sample_valid],
            "image_ids":      [image_ids[i] for i in keep],
            "pt_paths":       [pt_paths[i]  for i in keep],
            "sources":        [sources[i]   for i in keep],
        }

    return _collate


def build_vqa_inference_collate(
    tokenizer,
    image_token:      str = "<sar>",
    num_image_tokens: int = 195,
    max_length:       int = 1024,
) -> Callable:
    """
    推理专用 collate_fn：只 tokenize prompt（不含最后一轮 assistant 回答），
    同时返回 ground truth 用于指标计算。
    """
    image_tokens_str = " ".join([image_token] * num_image_tokens)

    def _collate(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        good = [b for b in batch if not b.get("_bad_sample", False)]
        if not good:
            return None

        sar_feats   = torch.stack([b["sar_feat"] for b in good], dim=0)
        prompts     = []
        references  = []

        for b in good:
            turns = b.get("turns", [])
            if not turns:
                turns = [(b.get("prompt", ""), b.get("caption", ""))]

            # 用所有轮次的 prompt 作为输入，最后一轮的 assistant 作为 reference
            parts = []
            for i, (user_text, asst_text) in enumerate(turns):
                if i == 0:
                    parts.append(f"{image_tokens_str}\nUser: {user_text}\nAssistant: ")
                else:
                    parts.append(f"User: {user_text}\nAssistant: ")
                if i < len(turns) - 1:
                    parts[-1] = parts[-1] + asst_text + "\n"

            prompts.append("".join(parts))
            references.append(turns[-1][1])  # 最后一轮 assistant 回答

        tok = tokenizer(
            prompts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return {
            "sar_feats":      sar_feats,
            "input_ids":      tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "references":     references,
        }

    return _collate
