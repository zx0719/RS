from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def norm_split(x) -> str:
    x = str(x).lower()
    return "val" if x == "validation" else x


def normalize_label_string(label_str) -> List[str]:
    if label_str is None:
        return []
    if isinstance(label_str, list):
        return [str(x).strip() for x in label_str if str(x).strip()]
    if isinstance(label_str, np.ndarray):
        return [str(x).strip() for x in label_str.tolist() if str(x).strip()]
    s = str(label_str).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple, np.ndarray)):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def labels_to_sentence(labels: List[str], lowercase: bool = True) -> str:
    labels = [x.strip() for x in labels if x and str(x).strip()]
    if lowercase:
        labels = [x.lower() for x in labels]
    if len(labels) == 0:
        content = "unknown land-cover categories"
    elif len(labels) == 1:
        content = labels[0]
    elif len(labels) == 2:
        content = f"{labels[0]} and {labels[1]}"
    else:
        content = ", ".join(labels[:-1]) + f", and {labels[-1]}"
    return f"This remote sensing patch contains {content}."


@dataclass
class PairedSample:
    patch_id_ms: str
    s1_name: str
    split_norm: str
    ms_output_file: str
    sar_feature_path: str
    labels_raw: str
    label_text: str


class BENStage1FeatureDataset(Dataset):
    def __init__(
        self,
        ms_mapping_csv: str,
        sar_mapping_csv: str,
        metadata_path: str,
        split: str,
        image_token: str = "<rs_patch>",
        prompt_template: str = "Describe the land-cover content of this remote sensing patch.",
        lowercase_labels: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.ms_mapping_csv = ms_mapping_csv
        self.sar_mapping_csv = sar_mapping_csv
        self.metadata_path = metadata_path
        self.split = norm_split(split)
        self.image_token = image_token
        self.prompt_template = prompt_template
        self.lowercase_labels = lowercase_labels
        self.max_samples = max_samples

        self.samples = self._build_samples()

    def _build_samples(self) -> List[PairedSample]:
        ms = pd.read_csv(self.ms_mapping_csv)
        sar = pd.read_csv(self.sar_mapping_csv)
        meta = pd.read_parquet(self.metadata_path)

        ms["split_norm"] = ms["split"].map(norm_split)
        sar["split_norm"] = sar["split"].map(norm_split)
        meta["split_norm"] = meta["split"].map(norm_split)

        ms = ms[ms["split_norm"] == self.split].copy()
        sar = sar[sar["split_norm"] == self.split].copy()
        meta = meta[meta["split_norm"] == self.split].copy()

        sar = sar.sort_values(["patch_id", "split_norm", "feature_path"]).copy()
        sar = sar.drop_duplicates(subset=["patch_id", "split_norm"], keep="first")

        ms2 = ms[["patch_id", "split_norm", "output_file", "labels_raw"]].copy()
        meta2 = meta[["patch_id", "s1_name", "split_norm", "labels"]].copy()
        sar2 = sar[["patch_id", "split_norm", "feature_path"]].copy()

        tmp = ms2.merge(meta2, on=["patch_id", "split_norm"], how="inner")
        paired = tmp.merge(
            sar2,
            left_on=["s1_name", "split_norm"],
            right_on=["patch_id", "split_norm"],
            how="inner",
            suffixes=("_ms", "_sar"),
        )

        out: List[PairedSample] = []
        for _, row in paired.iterrows():
            labels_from_meta = normalize_label_string(row["labels"])
            labels_from_ms = normalize_label_string(row["labels_raw"])
            labels = labels_from_meta if labels_from_meta else labels_from_ms
            label_text = labels_to_sentence(labels, lowercase=self.lowercase_labels)

            out.append(
                PairedSample(
                    patch_id_ms=str(row["patch_id_ms"]),
                    s1_name=str(row["s1_name"]),
                    split_norm=str(row["split_norm"]),
                    ms_output_file=str(row["output_file"]),
                    sar_feature_path=str(row["feature_path"]),
                    labels_raw=str(row["labels_raw"]),
                    label_text=label_text,
                )
            )

        if self.max_samples is not None:
            out = out[: self.max_samples]

        if len(out) == 0:
            raise RuntimeError(f"No samples found for split={self.split}")

        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        ms_feat = torch.load(sample.ms_output_file, map_location="cpu")
        if not isinstance(ms_feat, torch.Tensor):
            raise TypeError(f"MS feature must be torch.Tensor, got {type(ms_feat)}")
        ms_feat = ms_feat.float()

        sar_feat = np.load(sample.sar_feature_path)
        if not isinstance(sar_feat, np.ndarray):
            raise TypeError(f"SAR feature must be np.ndarray, got {type(sar_feat)}")
        sar_feat = torch.from_numpy(sar_feat).float()

        prompt_text = f"{self.image_token}\n{self.prompt_template}"

        return {
            "sample_id": sample.patch_id_ms,
            "s1_name": sample.s1_name,
            "split": sample.split_norm,
            "ms_feat": ms_feat,
            "sar_feat": sar_feat,
            "prompt_text": prompt_text,
            "target_text": sample.label_text,
            "label_text": sample.label_text,
        }


def ben_stage1_collate_fn(batch: List[Dict]) -> Dict:
    ms_feat = torch.stack([x["ms_feat"] for x in batch], dim=0)
    sar_feat = torch.stack([x["sar_feat"] for x in batch], dim=0)

    return {
        "sample_id": [x["sample_id"] for x in batch],
        "s1_name": [x["s1_name"] for x in batch],
        "split": [x["split"] for x in batch],
        "ms_feat": ms_feat,
        "sar_feat": sar_feat,
        "prompt_text": [x["prompt_text"] for x in batch],
        "target_text": [x["target_text"] for x in batch],
        "label_text": [x["label_text"] for x in batch],
    }
