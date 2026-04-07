from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pandas as pd
except Exception:
    pd = None

import rasterio
from rasterio.enums import Resampling


S2_ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S2_RGB_BANDS = ["B04", "B03", "B02"]
S2_MS_10M_20M_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S2_MS_WITH_60M_BANDS = S2_ALL_BANDS
S1_BANDS = ["VV", "VH"]


BEN19_LABELS = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]


CLC43_TO_BEN19 = {
    "Continuous urban fabric": "Urban fabric",
    "Discontinuous urban fabric": "Urban fabric",
    "Industrial or commercial units": "Industrial or commercial units",
    "Non-irrigated arable land": "Arable land",
    "Permanently irrigated land": "Arable land",
    "Rice fields": "Arable land",
    "Vineyards": "Permanent crops",
    "Fruit trees and berry plantations": "Permanent crops",
    "Olive groves": "Permanent crops",
    "Annual crops associated with permanent crops": "Permanent crops",
    "Pastures": "Pastures",
    "Complex cultivation patterns": "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas": "Agro-forestry areas",
    "Broad-leaved forest": "Broad-leaved forest",
    "Coniferous forest": "Coniferous forest",
    "Mixed forest": "Mixed forest",
    "Natural grassland": "Natural grassland and sparsely vegetated areas",
    "Sparsely vegetated areas": "Natural grassland and sparsely vegetated areas",
    "Moors and heathland": "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation": "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub": "Transitional woodland, shrub",
    "Beaches, dunes, sands": "Beaches, dunes, sands",
    "Inland marshes": "Inland wetlands",
    "Peatbogs": "Inland wetlands",
    "Salt marshes": "Coastal wetlands",
    "Salines": "Coastal wetlands",
    "Water courses": "Inland waters",
    "Water bodies": "Inland waters",
    "Coastal lagoons": "Marine waters",
    "Estuaries": "Marine waters",
    "Sea and ocean": "Marine waters",
}


class BENMMRGBMSSARDataset(Dataset):
    """
    BEN-MM / BigEarthNet 常用读取方式：
    1) rgb  : S2 的 B04/B03/B02
    2) ms   : S2 多光谱（默认 10m+20m 共 10 通道）
    3) sar  : S1 的 VV/VH

    推荐目录：
      root/
        BigEarthNet-S2/
        BigEarthNet-S1/
        metadata.parquet

    返回 sample 示例：
      {
        "patch_id": str,
        "rgb": Tensor[3,H,W],      # 如果 enable_rgb=True
        "ms": Tensor[C,H,W],       # 如果 enable_ms=True
        "sar": Tensor[2,H,W],      # 如果 enable_sar=True
        "labels": Tensor[19],      # 如果有元数据
        "label_names": list[str],
      }
    """

    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        enable_rgb: bool = True,
        enable_ms: bool = True,
        enable_sar: bool = True,
        ms_bands: Sequence[str] = S2_MS_10M_20M_BANDS,
        target_size: int = 120,
        use_metadata: bool = True,
        metadata_file: str = "metadata.parquet",
        s2_scale: float = 10000.0,
        sar_clip: Optional[Sequence[float]] = (-25.0, 5.0),
        return_label_names: bool = True,
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.s2_root = self.root / "BigEarthNet-S2"
        self.s1_root = self.root / "BigEarthNet-S1"
        self.enable_rgb = enable_rgb
        self.enable_ms = enable_ms
        self.enable_sar = enable_sar
        self.ms_bands = list(ms_bands)
        self.target_size = target_size
        self.use_metadata = use_metadata
        self.metadata_path = self.root / metadata_file
        self.s2_scale = s2_scale
        self.sar_clip = tuple(sar_clip) if sar_clip is not None else None
        self.return_label_names = return_label_names
        self.transform = transform
        self.split = split.lower() if split is not None else None

        if not (self.enable_rgb or self.enable_ms or self.enable_sar):
            raise ValueError("enable_rgb / enable_ms / enable_sar 至少要开一个")
        if (self.enable_rgb or self.enable_ms) and not self.s2_root.exists():
            raise FileNotFoundError(f"缺少 S2 目录: {self.s2_root}")
        if self.enable_sar and not self.s1_root.exists():
            raise FileNotFoundError(f"缺少 S1 目录: {self.s1_root}")

        self.metadata = self._load_metadata() if self.use_metadata else None
        self.s2_map = self._scan_s2_patch_dirs() if (self.enable_rgb or self.enable_ms) else {}
        self.s1_map = self._scan_s1_patch_dirs() if self.enable_sar else {}
        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError("没有找到任何样本，请检查 root / split / metadata")

    def _load_metadata(self):
        if not self.metadata_path.exists():
            return None
        if pd is None:
            raise ImportError("读取 metadata.parquet 需要 pandas + pyarrow")
        df = pd.read_parquet(self.metadata_path)
        cols = {c.lower(): c for c in df.columns}

        def pick(*names):
            for n in names:
                if n.lower() in cols:
                    return cols[n.lower()]
            for c in df.columns:
                lc = c.lower()
                for n in names:
                    if n.lower() in lc:
                        return c
            return None

        patch_col = pick("patch_id", "patch_name", "name")
        split_col = pick("split")
        s1_col = pick("s1_name", "patch_name_s1", "s1_patch_name")
        labels19_col = pick("labels", "labels_19", "bigearthnet_19_labels", "new_labels")
        labels43_col = pick("original_labels", "labels_43", "bigearthnet_43_labels")

        # 避免 original_labels 因为包含 "labels" 子串而被误识别成 19 类标签列
        if labels19_col is not None and labels43_col is not None and labels19_col == labels43_col:
            labels19_col = None

        if patch_col is None:
            raise RuntimeError("metadata.parquet 里没识别到 patch_id/patch_name 列")

        if self.split is not None and split_col is not None:
            df = df[df[split_col].astype(str).str.lower() == self.split].copy()

        meta = {}
        for _, row in df.iterrows():
            patch_id = str(row[patch_col])
            entry = {"patch_id": patch_id, "s1_name": None, "label_names": []}
            if s1_col is not None and row.get(s1_col) is not None:
                entry["s1_name"] = str(row[s1_col])

            labels = None
            if labels19_col is not None:
                labels = row[labels19_col]
            if (labels is None or (isinstance(labels, float) and np.isnan(labels))) and labels43_col is not None:
                labels = row[labels43_col]

            label_names = self._normalize_labels(labels)
            if labels43_col is not None and labels19_col is None:
                label_names = self._map_43_to_19(label_names)
            entry["label_names"] = label_names
            meta[patch_id] = entry
        return meta

    def _normalize_labels(self, labels) -> List[str]:
        if labels is None:
            return []
        if isinstance(labels, list):
            return [str(x) for x in labels]
        if isinstance(labels, tuple):
            return [str(x) for x in labels]
        if isinstance(labels, np.ndarray):
            return [str(x) for x in labels.tolist()]
        if isinstance(labels, str):
            s = labels.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    obj = json.loads(s.replace("'", '"'))
                    if isinstance(obj, list):
                        return [str(x) for x in obj]
                except Exception:
                    pass
            if ";" in s:
                return [x.strip() for x in s.split(";") if x.strip()]
            if "," in s:
                return [x.strip() for x in s.split(",") if x.strip()]
            return [s]
        return [str(labels)]

    def _map_43_to_19(self, labels43: List[str]) -> List[str]:
        out = []
        for x in labels43:
            y = CLC43_TO_BEN19.get(x)
            if y is not None and y not in out:
                out.append(y)
        return out

    def _scan_s2_patch_dirs(self) -> Dict[str, Path]:
        mapping = {}
        for tile_dir in self.s2_root.iterdir():
            if not tile_dir.is_dir():
                continue
            for patch_dir in tile_dir.iterdir():
                if not patch_dir.is_dir():
                    continue
                pid = patch_dir.name
                probe = patch_dir / f"{pid}_B04.tif"
                if probe.exists():
                    mapping[pid] = patch_dir
        return mapping

    def _scan_s1_patch_dirs(self) -> Dict[str, Path]:
        mapping = {}
        for scene_dir in self.s1_root.iterdir():
            if not scene_dir.is_dir():
                continue
            for patch_dir in scene_dir.iterdir():
                if not patch_dir.is_dir():
                    continue
                pid = patch_dir.name
                probe_vv = patch_dir / f"{pid}_VV.tif"
                probe_vh = patch_dir / f"{pid}_VH.tif"
                if probe_vv.exists() and probe_vh.exists():
                    mapping[pid] = patch_dir
        return mapping

    def _build_samples(self):
        samples = []
        if self.metadata is not None:
            for patch_id, meta in self.metadata.items():
                if (self.enable_rgb or self.enable_ms) and patch_id not in self.s2_map:
                    continue
                s1_name = meta.get("s1_name")
                if self.enable_sar:
                    if s1_name is None or s1_name not in self.s1_map:
                        continue
                samples.append({
                    "patch_id": patch_id,
                    "s1_name": s1_name,
                    "label_names": meta.get("label_names", []),
                })
        else:
            # 没 metadata 时，只按 S2 扫。适合你当前只有目录结构、先把读取跑通的场景。
            for patch_id in sorted(self.s2_map.keys()):
                samples.append({
                    "patch_id": patch_id,
                    "s1_name": None,
                    "label_names": [],
                })
        return samples

    def _read_tif(self, path: Path, out_size: int) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read(1, out_shape=(out_size, out_size), resampling=Resampling.bilinear)
        return arr.astype(np.float32)

    def _read_s2_stack(self, patch_id: str, bands: Sequence[str]) -> torch.Tensor:
        patch_dir = self.s2_map[patch_id]
        arrs = []
        for b in bands:
            tif = patch_dir / f"{patch_id}_{b}.tif"
            x = self._read_tif(tif, self.target_size)
            x = x / self.s2_scale
            arrs.append(x)
        return torch.from_numpy(np.stack(arrs, axis=0)).float()

    def _read_s1_stack(self, s1_name: str) -> torch.Tensor:
        patch_dir = self.s1_map[s1_name]
        arrs = []
        for b in S1_BANDS:
            tif = patch_dir / f"{s1_name}_{b}.tif"
            x = self._read_tif(tif, self.target_size)
            if self.sar_clip is not None:
                lo, hi = self.sar_clip
                x = np.clip(x, lo, hi)
                x = (x - lo) / (hi - lo + 1e-6)
            arrs.append(x)
        return torch.from_numpy(np.stack(arrs, axis=0)).float()

    def _labels_to_multihot(self, label_names: Sequence[str]) -> torch.Tensor:
        y = torch.zeros(len(BEN19_LABELS), dtype=torch.float32)
        for name in label_names:
            if name in BEN19_LABELS:
                y[BEN19_LABELS.index(name)] = 1.0
        return y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        patch_id = item["patch_id"]
        s1_name = item["s1_name"]
        label_names = item["label_names"]

        out = {
            "patch_id": patch_id,
            "labels": self._labels_to_multihot(label_names),
        }
        if self.return_label_names:
            out["label_names"] = label_names

        if self.enable_rgb:
            out["rgb"] = self._read_s2_stack(patch_id, S2_RGB_BANDS)
        if self.enable_ms:
            out["ms"] = self._read_s2_stack(patch_id, self.ms_bands)
        if self.enable_sar:
            out["sar"] = self._read_s1_stack(s1_name)
            out["s1_name"] = s1_name

        if self.transform is not None:
            out = self.transform(out)
        return out


if __name__ == "__main__":
    ds = BENMMRGBMSSARDataset(
        root="/mnt/data/mm_data/ben-mm",
        split=None,
        enable_rgb=True,
        enable_ms=True,
        enable_sar=False,
        use_metadata=False,
    )
    x = ds[0]
    print("patch_id:", x["patch_id"])
    if "rgb" in x:
        print("rgb:", tuple(x["rgb"].shape))
    if "ms" in x:
        print("ms:", tuple(x["ms"].shape))
    if "sar" in x:
        print("sar:", tuple(x["sar"].shape))
    print("labels:", tuple(x["labels"].shape))
