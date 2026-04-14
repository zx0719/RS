#!/usr/bin/env python3
"""
Prepare SSDD dataset for YOLOv8-OBB training.

Converts the SSDD RBox VOC-style annotations to YOLO-OBB format and
generates a dataset.yaml ready for use with Ultralytics YOLOv8.

Usage
-----
    python data/prepare_ssdd.py \\
        --ssdd-root /mnt/data/mm_data/SAR/dection/SSDD/Official-SSDD-OPEN/RBox_SSDD/voc_style \\
        --output-dir /home/zhuxiang/RS/SAR/experiments/data/ssdd_yolo_obb \\
        [--val-ratio 0.15] \\
        [--seed 42]

Output layout
-------------
    {output-dir}/
        images/
            train/   (jpg copies)
            val/
            test/
        labels/
            train/   (YOLO-OBB .txt)
            val/
            test/
        dataset.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root or from data/ subdirectory
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from modules.detector.dataset_utils import (
    convert_ssdd_to_yolo_obb,
    generate_ssdd_yaml,
    split_ssdd_train_val,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SSDD RBox VOC dataset to YOLO-OBB format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ssdd-root",
        type=str,
        default="/mnt/data/mm_data/SAR/dection/SSDD/Official-SSDD-OPEN/RBox_SSDD/voc_style",
        help="Root of the SSDD VOC-style dataset (contains Annotations/, JPEGImages_train/, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/zhuxiang/RS/SAR/experiments/data/ssdd_yolo_obb",
        help="Destination root for the YOLO-OBB dataset.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of training images to use for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ssdd_root  = Path(args.ssdd_root)
    output_dir = Path(args.output_dir)

    if not ssdd_root.exists():
        sys.exit(f"ERROR: SSDD root not found: {ssdd_root}")

    print(f"SSDD root  : {ssdd_root}")
    print(f"Output dir : {output_dir}")
    print(f"Val ratio  : {args.val_ratio}")
    print(f"Seed       : {args.seed}")
    print()

    # 1. Build train / val split from the training annotations
    print("[1/4] Splitting train → train + val ...")
    train_stems, val_stems = split_ssdd_train_val(
        ssdd_root=ssdd_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"      train: {len(train_stems)} images,  val: {len(val_stems)} images")

    # 2. Convert train split
    print("[2/4] Converting train split ...")
    train_stats = convert_ssdd_to_yolo_obb(
        ssdd_root=ssdd_root,
        output_root=output_dir,
        split="train",
        file_stems=train_stems,
    )

    # 3. Convert val split
    print("[3/4] Converting val split ...")
    val_stats = convert_ssdd_to_yolo_obb(
        ssdd_root=ssdd_root,
        output_root=output_dir,
        split="val",
        file_stems=val_stems,
    )

    # 4. Convert test split
    # SSDD test images live in JPEGImages_test/; annotations share Annotations/.
    # We use all XMLs whose corresponding test images exist.
    print("[4/4] Converting test split ...")
    ann_dir = ssdd_root / "Annotations"
    test_img_dir = ssdd_root / "JPEGImages_test"
    if test_img_dir.exists():
        # Collect stems that have a matching test image
        test_stems = []
        for xml_path in sorted(ann_dir.glob("*.xml")):
            stem = xml_path.stem
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                if (test_img_dir / f"{stem}{ext}").exists():
                    test_stems.append(stem)
                    break
        test_stats = convert_ssdd_to_yolo_obb(
            ssdd_root=ssdd_root,
            output_root=output_dir,
            split="test",
            file_stems=test_stems,
        )
    else:
        print(f"      WARNING: {test_img_dir} not found — skipping test split")
        test_stats = {"n_images": 0, "n_objects": 0, "output_dir": str(output_dir)}

    # 5. Generate YAML
    yaml_path = generate_ssdd_yaml(output_root=output_dir)

    # Summary table
    print()
    print(f"{'Split':<10} {'Images':>8} {'Objects':>10}")
    print("-" * 30)
    print(f"{'train':<10} {train_stats['n_images']:>8} {train_stats['n_objects']:>10}")
    print(f"{'val':<10} {val_stats['n_images']:>8} {val_stats['n_objects']:>10}")
    print(f"{'test':<10} {test_stats['n_images']:>8} {test_stats['n_objects']:>10}")
    print()
    print(f"Dataset YAML : {yaml_path}")
    print("Done.")


if __name__ == "__main__":
    main()
