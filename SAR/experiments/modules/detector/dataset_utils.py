"""
dataset_utils.py — DOTA-format OBB annotation → YOLO-OBB format converter

DOTA annotation format (one .txt per image):
    x1 y1 x2 y2 x3 y3 x4 y4 category difficulty

YOLO-OBB format (one .txt per image, normalised to [0,1]):
    class_index cx cy w h angle   (angle in radians, Ultralytics convention)

Usage
-----
    # Convert a full DOTA dataset split
    python dataset_utils.py \\
        --dota_dir /data/DOTA/train \\
        --output_dir /data/yolo_obb/train \\
        --img_width 1024 --img_height 1024

    # Or import and call programmatically
    from modules.detector.dataset_utils import convert_dota_to_yolo_obb
    convert_dota_to_yolo_obb(
        dota_label_dir="labels_dota/train",
        yolo_label_dir="labels_yolo/train",
        img_width=4000,
        img_height=4000,
        class_names=["carrier", "destroyer", ...],
    )
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class OBBEntry(NamedTuple):
    """Parsed DOTA annotation line."""
    points: np.ndarray   # shape (4, 2), pixel coordinates
    category: str
    difficulty: int


class YoloOBBEntry(NamedTuple):
    """YOLO-OBB label line (normalised)."""
    class_index: int
    cx: float
    cy: float
    w: float
    h: float
    angle_rad: float


# ---------------------------------------------------------------------------
# DOTA class name → our Evidence Package class code (and YOLO index)
# Edit to match your specific DOTA subset or custom dataset.
# ---------------------------------------------------------------------------

DEFAULT_DOTA_TO_CODE: dict[str, str] = {
    # DOTA ship categories → our codes
    "ship":            "other_vessel",
    "large-vehicle":   "other_vessel",
    "small-vehicle":   "other_vessel",
    "carrier":         "carrier",
    "destroyer":       "destroyer",
    "frigate":         "frigate",
    "replenishment":   "replenishment",
    "amphibious":      "amphibious",
    # Aircraft
    "plane":           "other_aircraft",
    "helicopter":      "helicopter",
    "fighter":         "fighter",
    "bomber":          "bomber",
    "transport-plane": "transport",
    "early-warning":   "aew",
}

# Canonical class order — must match class_map.py and dataset.yaml
CANONICAL_CLASS_ORDER: list[str] = [
    "carrier",
    "destroyer",
    "frigate",
    "replenishment",
    "amphibious",
    "other_vessel",
    "fighter",
    "bomber",
    "transport",
    "aew",
    "helicopter",
    "other_aircraft",
]


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def _quad_to_xywh_angle(points: np.ndarray) -> tuple[float, float, float, float, float]:
    """Convert a quadrilateral (4×2 pixel array) to (cx, cy, w, h, angle_rad).

    Uses the minimum-area enclosing rectangle approach via OpenCV-compatible
    math (no cv2 dependency).

    Returns
    -------
    cx, cy:     Centre of the OBB in pixels.
    w, h:       Width (long axis) and height of the OBB in pixels.
    angle_rad:  Rotation angle in radians (OpenCV convention: long axis from
                the positive x-axis, in range [-π/2, π/2]).
    """
    # Reorder points to consistent winding
    pts = points.astype(float)

    # Compute centre
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())

    # Find the longest edge to determine the primary axis
    edge_lengths = [
        np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)
    ]
    longest_idx = int(np.argmax(edge_lengths))
    p0 = pts[longest_idx]
    p1 = pts[(longest_idx + 1) % 4]

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    angle_rad = float(math.atan2(dy, dx))

    # Project all points onto the two axes to get w and h
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    local_x = (pts[:, 0] - cx) * cos_a + (pts[:, 1] - cy) * sin_a
    local_y = -(pts[:, 0] - cx) * sin_a + (pts[:, 1] - cy) * cos_a

    w = float(local_x.max() - local_x.min())
    h = float(local_y.max() - local_y.min())

    # Ensure w >= h (long axis as width, matching Ultralytics convention)
    if h > w:
        w, h = h, w
        angle_rad += math.pi / 2.0

    # Normalise angle to [-π/2, π/2]
    while angle_rad > math.pi / 2.0:
        angle_rad -= math.pi
    while angle_rad < -math.pi / 2.0:
        angle_rad += math.pi

    return cx, cy, w, h, angle_rad


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_dota_line(line: str) -> OBBEntry | None:
    """Parse a single DOTA annotation line.

    Expected format:
        x1 y1 x2 y2 x3 y3 x4 y4 category difficulty

    Returns None for header lines or malformed lines.
    """
    line = line.strip()
    if not line or line.startswith("imagesource") or line.startswith("gsd"):
        return None

    parts = line.split()
    if len(parts) < 9:
        return None

    try:
        coords = np.array([float(p) for p in parts[:8]], dtype=float).reshape(4, 2)
        category = parts[8].lower()
        difficulty = int(parts[9]) if len(parts) >= 10 else 0
    except (ValueError, IndexError):
        return None

    return OBBEntry(points=coords, category=category, difficulty=difficulty)


def dota_entry_to_yolo(
    entry: OBBEntry,
    img_width: int,
    img_height: int,
    class_names: list[str],
    dota_to_code: dict[str, str] | None = None,
    skip_difficult: bool = False,
) -> YoloOBBEntry | None:
    """Convert a parsed DOTA entry to a normalised YOLO-OBB entry.

    Returns None if the category is unmapped or if skip_difficult is True
    and the entry is marked difficult (difficulty == 2).
    """
    if skip_difficult and entry.difficulty == 2:
        return None

    mapping = dota_to_code or DEFAULT_DOTA_TO_CODE
    code = mapping.get(entry.category)
    if code is None:
        return None

    try:
        class_index = class_names.index(code)
    except ValueError:
        return None

    cx, cy, w, h, angle_rad = _quad_to_xywh_angle(entry.points)

    # Normalise to [0, 1]
    return YoloOBBEntry(
        class_index=class_index,
        cx=cx / img_width,
        cy=cy / img_height,
        w=w / img_width,
        h=h / img_height,
        angle_rad=angle_rad,
    )


# ---------------------------------------------------------------------------
# File-level conversion
# ---------------------------------------------------------------------------

def convert_dota_label_file(
    src: Path,
    dst: Path,
    img_width: int,
    img_height: int,
    class_names: list[str],
    dota_to_code: dict[str, str] | None = None,
    skip_difficult: bool = False,
) -> int:
    """Convert a single DOTA label file to YOLO-OBB format.

    Returns the number of annotations written.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            entry = parse_dota_line(line)
            if entry is None:
                continue
            yolo_entry = dota_entry_to_yolo(
                entry, img_width, img_height, class_names, dota_to_code, skip_difficult
            )
            if yolo_entry is None:
                continue
            f_out.write(
                f"{yolo_entry.class_index} "
                f"{yolo_entry.cx:.6f} {yolo_entry.cy:.6f} "
                f"{yolo_entry.w:.6f} {yolo_entry.h:.6f} "
                f"{yolo_entry.angle_rad:.6f}\n"
            )
            written += 1

    return written


def convert_dota_to_yolo_obb(
    dota_label_dir: str | Path,
    yolo_label_dir: str | Path,
    img_width: int,
    img_height: int,
    class_names: list[str] | None = None,
    dota_to_code: dict[str, str] | None = None,
    skip_difficult: bool = False,
) -> dict[str, int]:
    """Convert an entire directory of DOTA label files to YOLO-OBB format.

    Parameters
    ----------
    dota_label_dir:  Directory containing DOTA .txt annotation files.
    yolo_label_dir:  Output directory for YOLO-OBB .txt files.
    img_width:       Image width in pixels (assumed uniform; use per-file if not).
    img_height:      Image height in pixels.
    class_names:     Ordered list of class code strings. Defaults to CANONICAL_CLASS_ORDER.
    dota_to_code:    Custom DOTA category → Evidence code mapping.
    skip_difficult:  Whether to skip annotations with difficulty == 2.

    Returns
    -------
    dict mapping filename stem → number of annotations written.
    """
    src_dir = Path(dota_label_dir)
    dst_dir = Path(yolo_label_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = CANONICAL_CLASS_ORDER

    stats: dict[str, int] = {}
    label_files = sorted(src_dir.glob("*.txt"))

    if not label_files:
        print(f"WARNING: No .txt files found in {src_dir}", file=sys.stderr)

    for src_file in label_files:
        dst_file = dst_dir / src_file.name
        count = convert_dota_label_file(
            src=src_file,
            dst=dst_file,
            img_width=img_width,
            img_height=img_height,
            class_names=class_names,
            dota_to_code=dota_to_code,
            skip_difficult=skip_difficult,
        )
        stats[src_file.stem] = count

    total = sum(stats.values())
    print(
        f"Converted {len(stats)} label files → {total} annotations "
        f"written to {dst_dir}"
    )
    return stats


# ---------------------------------------------------------------------------
# dataset.yaml generator
# ---------------------------------------------------------------------------

def generate_dataset_yaml(
    output_path: str | Path,
    train_images: str,
    val_images: str,
    test_images: str | None = None,
    class_names: list[str] | None = None,
) -> None:
    """Write a YOLO-OBB dataset.yaml file.

    Parameters
    ----------
    output_path:   Where to write the YAML file.
    train_images:  Path to training images directory.
    val_images:    Path to validation images directory.
    test_images:   Optional path to test images directory.
    class_names:   Ordered class name list (default: CANONICAL_CLASS_ORDER).
    """
    if class_names is None:
        class_names = CANONICAL_CLASS_ORDER

    lines = [
        f"train: {train_images}",
        f"val:   {val_images}",
    ]
    if test_images:
        lines.append(f"test:  {test_images}")

    lines += [
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for name in class_names:
        lines.append(f"  - {name}")

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"dataset.yaml written to {output_path}")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def count_annotations(label_dir: str | Path, num_classes: int) -> dict[str, int]:
    """Count per-class annotation frequencies in a YOLO label directory.

    Returns a dict mapping class index (str) → count.
    """
    label_dir = Path(label_dir)
    counts: dict[int, int] = {i: 0 for i in range(num_classes)}

    for label_file in label_dir.glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    cls_idx = int(parts[0])
                    counts[cls_idx] = counts.get(cls_idx, 0) + 1
                except (ValueError, IndexError):
                    pass

    return {str(k): v for k, v in sorted(counts.items())}


# ---------------------------------------------------------------------------
# SSDD RBox VOC → YOLO-OBB conversion
# ---------------------------------------------------------------------------

def _parse_ssdd_xml(xml_path: Path) -> tuple[int, int, list[dict]]:
    """Parse a single SSDD RBox VOC XML file.

    Returns
    -------
    img_width, img_height, list of annotation dicts with keys:
        cx, cy, w, h, theta_deg
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    objects = []
    for obj in root.iter("object"):
        rb = obj.find("rotated_bndbox")
        if rb is None:
            continue
        objects.append({
            "cx":        float(rb.findtext("rotated_bbox_cx")),
            "cy":        float(rb.findtext("rotated_bbox_cy")),
            "w":         float(rb.findtext("rotated_bbox_w")),
            "h":         float(rb.findtext("rotated_bbox_h")),
            "theta_deg": float(rb.findtext("rotated_bbox_theta")),
        })
    return img_w, img_h, objects


def _rbox_to_corners(cx: float, cy: float, w: float, h: float, theta_deg: float) -> list[float]:
    """Convert rotated box (cx, cy, w, h, theta_deg) to 4 corner coordinates.

    Returns [x1, y1, x2, y2, x3, y3, x4, y4] in pixel coordinates.
    theta_deg follows OpenCV convention: positive = clockwise.
    """
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    hw, hh = w / 2.0, h / 2.0
    # Four corners relative to center before rotation
    offsets = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    corners: list[float] = []
    for dx, dy in offsets:
        x = cx + dx * cos_t - dy * sin_t
        y = cy + dx * sin_t + dy * cos_t
        corners.extend([x, y])
    return corners


def convert_ssdd_to_yolo_obb(
    ssdd_root: str | Path,
    output_root: str | Path,
    split: str = "train",
    file_stems: list[str] | None = None,
) -> dict[str, int]:
    """Convert SSDD RBox VOC format to YOLO-OBB format for one split.

    YOLO-OBB label line format (Ultralytics OBB standard)::

        <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>

    Where (x1..x4, y1..y4) are the four corner coordinates of the rotated box,
    normalised by image width/height to [0, 1].  This 9-column format is what
    ``yolo obb train`` expects; the previous 6-column (cx cy w h angle) format
    is NOT compatible with the Ultralytics trainer.

    Parameters
    ----------
    ssdd_root:    Root of the SSDD VOC-style dataset (contains Annotations/).
    output_root:  Destination root.  Labels go to labels/{split}/, images to
                  images/{split}/.
    split:        Split name, e.g. "train", "val", "test".
    file_stems:   Optional explicit list of stems to process.  When None,
                  all XMLs in Annotations/ are used.

    Returns
    -------
    dict with keys n_images, n_objects, output_dir.
    """
    ssdd_root = Path(ssdd_root)
    output_root = Path(output_root)

    # Determine annotation and image source directories per split.
    # Prefer the split-specific Annotations_train / Annotations_test dirs when
    # they exist; fall back to the combined Annotations/ directory.
    if split == "test":
        ann_dir = ssdd_root / "Annotations_test"
        if not ann_dir.exists():
            ann_dir = ssdd_root / "Annotations"
        img_src_dir = ssdd_root / "JPEGImages_test"
    else:
        ann_dir = ssdd_root / "Annotations_train"
        if not ann_dir.exists():
            ann_dir = ssdd_root / "Annotations"
        img_src_dir = ssdd_root / "JPEGImages_train"

    # Also accept the flat JPEGImages/ directory as a fallback for images
    img_src_dir_fallback = ssdd_root / "JPEGImages"

    label_out_dir = output_root / "labels" / split
    image_out_dir = output_root / "images" / split
    label_out_dir.mkdir(parents=True, exist_ok=True)
    image_out_dir.mkdir(parents=True, exist_ok=True)

    if file_stems is not None:
        xml_files = [ann_dir / f"{stem}.xml" for stem in file_stems]
    else:
        xml_files = sorted(ann_dir.glob("*.xml"))

    n_images = 0
    n_objects = 0

    for xml_path in xml_files:
        if not xml_path.exists():
            print(f"WARNING: XML not found: {xml_path}", file=sys.stderr)
            continue

        stem = xml_path.stem
        img_w, img_h, objs = _parse_ssdd_xml(xml_path)

        # Write YOLO-OBB label (9 columns: class x1 y1 x2 y2 x3 y3 x4 y4)
        label_path = label_out_dir / f"{stem}.txt"
        with label_path.open("w", encoding="utf-8") as f_out:
            for obj in objs:
                corners = _rbox_to_corners(
                    obj["cx"], obj["cy"], obj["w"], obj["h"], obj["theta_deg"]
                )
                # Normalise x coords by img_w, y coords by img_h
                norm = [
                    corners[i] / img_w if i % 2 == 0 else corners[i] / img_h
                    for i in range(8)
                ]
                coords_str = " ".join(f"{v:.6f}" for v in norm)
                f_out.write(f"0 {coords_str}\n")

        n_objects += len(objs)

        # Copy image (try primary dir, then fallback to flat JPEGImages/)
        copied = False
        search_dirs = [img_src_dir]
        if img_src_dir_fallback.exists() and img_src_dir_fallback != img_src_dir:
            search_dirs.append(img_src_dir_fallback)
        for search_dir in search_dirs:
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                src_img = search_dir / f"{stem}{ext}"
                if src_img.exists():
                    shutil.copy2(src_img, image_out_dir / src_img.name)
                    copied = True
                    break
            if copied:
                break
        if not copied:
            print(f"WARNING: No image found for {stem} in {img_src_dir}", file=sys.stderr)

        n_images += 1

    return {
        "n_images":   n_images,
        "n_objects":  n_objects,
        "output_dir": str(output_root),
    }


def split_ssdd_train_val(
    ssdd_root: str | Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split SSDD training stems into train and val subsets.

    SSDD only ships with train/test splits.  This function carves a val set
    from the training annotation files so that YOLOv8 training can measure
    validation metrics during training.

    Parameters
    ----------
    ssdd_root:  Root of the SSDD VOC-style dataset.
    val_ratio:  Fraction of training stems to reserve for validation.
    seed:       Random seed for reproducible splits.

    Returns
    -------
    (train_stems, val_stems) — lists of filename stems (no extension).
    """
    import random

    ssdd_root = Path(ssdd_root)
    # Prefer Annotations_train/ (928 training images) over the combined
    # Annotations/ directory (1160 total) so test images are not leaked into
    # the train/val split.
    ann_train_dir = ssdd_root / "Annotations_train"
    if ann_train_dir.exists():
        ann_dir = ann_train_dir
    else:
        # Fall back to ImageSets/Main/train.txt if available
        train_list = ssdd_root / "ImageSets" / "Main" / "train.txt"
        if train_list.exists():
            all_stems = sorted(train_list.read_text().splitlines())
            rng = random.Random(seed)
            rng.shuffle(all_stems)
            n_val = max(1, int(len(all_stems) * val_ratio))
            return all_stems[n_val:], all_stems[:n_val]
        ann_dir = ssdd_root / "Annotations"

    all_stems = sorted(p.stem for p in ann_dir.glob("*.xml"))

    rng = random.Random(seed)
    rng.shuffle(all_stems)

    n_val = max(1, int(len(all_stems) * val_ratio))
    val_stems   = all_stems[:n_val]
    train_stems = all_stems[n_val:]

    return train_stems, val_stems


def generate_ssdd_yaml(
    output_root: str | Path,
    yaml_path: str | Path | None = None,
) -> Path:
    """Write a YOLO dataset YAML file for the converted SSDD-OBB dataset.

    Parameters
    ----------
    output_root:  Root directory of the converted dataset
                  (the same value passed to convert_ssdd_to_yolo_obb).
    yaml_path:    Where to write the YAML.  Defaults to
                  {output_root}/dataset.yaml.

    Returns
    -------
    Path of the written YAML file.
    """
    output_root = Path(output_root).resolve()
    if yaml_path is None:
        yaml_path = output_root / "dataset.yaml"
    yaml_path = Path(yaml_path)

    content = (
        f"path: {output_root}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['ship']\n"
    )

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(content, encoding="utf-8")
    print(f"SSDD dataset YAML written to {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DOTA-format OBB annotations to YOLO-OBB format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dota_dir",
        type=str,
        required=True,
        help="Directory containing DOTA .txt annotation files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for YOLO-OBB .txt files.",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=1024,
        help="Uniform image width in pixels.",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=1024,
        help="Uniform image height in pixels.",
    )
    parser.add_argument(
        "--skip_difficult",
        action="store_true",
        help="Skip annotations marked as difficult (difficulty == 2).",
    )
    parser.add_argument(
        "--gen_yaml",
        action="store_true",
        help="Also generate a dataset.yaml in the parent of --output_dir.",
    )
    parser.add_argument(
        "--train_images",
        type=str,
        default="images/train",
        help="(For --gen_yaml) Path to training images.",
    )
    parser.add_argument(
        "--val_images",
        type=str,
        default="images/val",
        help="(For --gen_yaml) Path to validation images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    stats = convert_dota_to_yolo_obb(
        dota_label_dir=args.dota_dir,
        yolo_label_dir=args.output_dir,
        img_width=args.img_width,
        img_height=args.img_height,
        skip_difficult=args.skip_difficult,
    )

    if args.gen_yaml:
        yaml_path = Path(args.output_dir).parent / "dataset.yaml"
        generate_dataset_yaml(
            output_path=yaml_path,
            train_images=args.train_images,
            val_images=args.val_images,
        )

    # Print summary
    total = sum(stats.values())
    print(f"\nSummary: {len(stats)} files, {total} annotations total")
    count_map = count_annotations(args.output_dir, len(CANONICAL_CLASS_ORDER))
    print("Per-class counts:")
    for idx_str, cnt in count_map.items():
        name = CANONICAL_CLASS_ORDER[int(idx_str)] if int(idx_str) < len(CANONICAL_CLASS_ORDER) else "unknown"
        print(f"  [{idx_str}] {name}: {cnt}")
