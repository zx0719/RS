#!/usr/bin/env python3
"""
sample_negative_scenes.py — 从公开数据中抽样负样本（无港口/机场的图像）。

用于 Gate 场景分类器训练的 "none" 类别。
策略：从 SARDet_100K 中抽样纯海面/陆地图，排除标注了 harbor/bridge 的图像。

Usage
-----
    python scripts/sample_negative_scenes.py \
        --public-data /mnt/data/mm_data/SAR/SARDet_100K/data/Images/ \
        --output offline_training_pack/data/negative_samples/ \
        --count 2000
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sample negative scene images")
    parser.add_argument("--public-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images that do NOT have harbor/bridge annotations
    blacklist_ids = set()
    ann_root = Path("/mnt/data/mm_data/SAR/SARDet_100K/data/Annotations")

    for split in ("train", "val"):
        ann_file = ann_root / f"{split}.json"
        if not ann_file.exists():
            continue
        data = json.load(open(ann_file))
        for ann in data["annotations"]:
            # category_id 4=bridge, 5=harbor — exclude images with these
            if ann["category_id"] in (4, 5):
                blacklist_ids.add(ann["image_id"])

    # Collect candidate images
    candidates = []
    img_root = Path(args.public_data)
    for split in ("train", "val"):
        img_dir = img_root / split
        if not img_dir.is_dir():
            continue
        # Parse COCO JSON for images not in blacklist
        ann_file = ann_root / f"{split}.json"
        if ann_file.exists():
            data = json.load(open(ann_file))
            for img_info in data["images"]:
                if img_info["id"] not in blacklist_ids:
                    fpath = img_dir / img_info["file_name"]
                    if fpath.exists():
                        candidates.append(fpath)

    # Sample
    if len(candidates) > args.count:
        candidates = random.sample(candidates, args.count)

    print(f"Sampling {len(candidates)} negative images from {len(candidates)} candidates")

    for i, src in enumerate(candidates):
        dst = out_dir / f"neg_{i:04d}{src.suffix}"
        shutil.copy2(src, dst)

    print(f"Done. {len(candidates)} images saved to {out_dir}")


if __name__ == "__main__":
    main()
