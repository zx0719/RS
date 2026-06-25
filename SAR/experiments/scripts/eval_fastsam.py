#!/usr/bin/env python3
"""Evaluate FastSAM segmentation against polygon annotations.

Expected labels:
  - harbor  -> class_id 0
  - airport -> class_id 22

This script does not train FastSAM. It runs zero-shot inference and reports IoU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.segmentation import FastSAMSegmenter

HARBOR_CLASS_ID = 0
AIRPORT_CLASS_ID = 22
DEBUG_MAX_EXAMPLES = 20


def _extract_class_id(item: dict[str, Any]) -> int | None:
    raw = item.get("class_id", item.get("class__id"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _normalize_polygon_points(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        for key in ("points", "polygon", "vertices", "contour", "segmentation"):
            pts = _normalize_polygon_points(raw.get(key))
            if pts is not None:
                return pts
        return None

    if isinstance(raw, list):
        if not raw:
            return None

        if all(isinstance(v, (int, float)) for v in raw) and len(raw) >= 6 and len(raw) % 2 == 0:
            pts = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
            return pts

        if all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in raw):
            pts = np.asarray([[float(v[0]), float(v[1])] for v in raw], dtype=np.float32)
            return pts

        for item in raw:
            pts = _normalize_polygon_points(item)
            if pts is not None:
                return pts
    return None


def _build_gt_mask(polygons: list[dict[str, Any]], image_shape: tuple[int, int]) -> tuple[np.ndarray, str] | None:
    h, w = image_shape
    harbor_mask = np.zeros((h, w), dtype=np.uint8)
    airport_mask = np.zeros((h, w), dtype=np.uint8)

    for item in polygons:
        if not isinstance(item, dict):
            continue
        class_id = _extract_class_id(item)
        if class_id not in (HARBOR_CLASS_ID, AIRPORT_CLASS_ID):
            continue

        pts = None
        for key in ("polygon", "points", "vertices", "contour", "segmentation"):
            pts = _normalize_polygon_points(item.get(key))
            if pts is not None:
                break
        if pts is None or len(pts) < 3:
            continue

        pts_int = np.round(pts).astype(np.int32)
        target_mask = harbor_mask if class_id == HARBOR_CLASS_ID else airport_mask
        cv2.fillPoly(target_mask, [pts_int], 1)

    harbor_area = int(harbor_mask.sum())
    airport_area = int(airport_mask.sum())
    if harbor_area == 0 and airport_area == 0:
        return None
    if airport_area >= harbor_area:
        return airport_mask, "airport"
    return harbor_mask, "harbor"


def _find_image(image_root: Path, stem: str) -> Path | None:
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
        candidate = image_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _polygon_dir_from_data_root(data_root: Path) -> Path:
    return data_root / "segmentation" / "train" / "polygons"


def _default_image_root(data_root: Path) -> Path:
    candidates = [
        data_root / "segmentation" / "train" / "images",
        data_root / "segmentation" / "images",
        data_root / "images",
        data_root,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _mask_from_pred_polygon(polygon: list[list[float]], image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if not polygon:
        return mask
    pts = np.asarray(polygon, dtype=np.float32)
    if len(pts) < 3:
        return mask
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float(inter) / float(union)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FastSAM against polygon annotations")
    parser.add_argument("--data-root", required=True, help="Dataset root containing segmentation/train/polygons")
    parser.add_argument("--image-root", default=None, help="Directory containing source scene images")
    parser.add_argument("--model-path", default="FastSAM-s.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of polygon files for quick checks")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    polygon_dir = _polygon_dir_from_data_root(data_root)
    image_root = Path(args.image_root) if args.image_root else _default_image_root(data_root)

    if not polygon_dir.is_dir():
        raise FileNotFoundError(f"Polygon directory not found: {polygon_dir}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    segmenter = FastSAMSegmenter(
        model_path=args.model_path,
        device=args.device,
        score_thresh=args.score_thresh,
    )

    polygon_files = sorted(polygon_dir.glob("*.json"))
    if args.limit > 0:
        polygon_files = polygon_files[:args.limit]

    results: list[float] = []
    per_scene: dict[str, list[float]] = {"harbor": [], "airport": []}
    debug_lines: list[str] = []
    skipped_lines: list[str] = []
    missing_images = 0
    skipped_empty_gt = 0

    for poly_path in polygon_files:
        with open(poly_path, "r", encoding="utf-8") as f:
            polygons = json.load(f)
        if isinstance(polygons, dict):
            polygons = [polygons]

        image_path = _find_image(image_root, poly_path.stem)
        if image_path is None:
            missing_images += 1
            if len(skipped_lines) < DEBUG_MAX_EXAMPLES:
                skipped_lines.append(
                    f"reason=missing_image poly={poly_path.name} image_stem={poly_path.stem}"
                )
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            missing_images += 1
            if len(skipped_lines) < DEBUG_MAX_EXAMPLES:
                skipped_lines.append(
                    f"reason=unreadable_image poly={poly_path.name} image={image_path}"
                )
            continue

        gt = _build_gt_mask(polygons, image.shape)
        if gt is None:
            skipped_empty_gt += 1
            if len(skipped_lines) < DEBUG_MAX_EXAMPLES:
                class_ids = []
                for item in polygons:
                    if isinstance(item, dict):
                        class_ids.append(item.get("class_id", item.get("class__id")))
                skipped_lines.append(
                    f"reason=empty_gt poly={poly_path.name} image={image_path.name} "
                    f"class_ids={class_ids[:10]}"
                )
            continue
        gt_mask, scene_type = gt

        pred = segmenter.segment(image_path, prompt=scene_type)
        pred_mask = _mask_from_pred_polygon(pred.get("mask_polygon", []), image.shape)
        iou = _compute_iou(pred_mask, gt_mask)
        results.append(iou)
        per_scene[scene_type].append(iou)

        if len(debug_lines) < DEBUG_MAX_EXAMPLES:
            debug_lines.append(
                f"scene={scene_type} image={image_path.name} poly={poly_path.name} "
                f"gt_area_px={int(gt_mask.sum())} pred_area_px={int(pred_mask.sum())} "
                f"success={pred.get('success', False)} iou={iou:.4f}"
            )

    print(f"[FastSAM][Eval] polygon_dir={polygon_dir}")
    print(f"[FastSAM][Eval] image_root={image_root}")
    print(f"[FastSAM][Eval] total_polygons={len(polygon_files)}")
    print(f"[FastSAM][Eval] evaluated={len(results)} missing_images={missing_images} skipped_empty_gt={skipped_empty_gt}")
    if results:
        print(f"[FastSAM][Eval] mIoU={np.mean(results):.4f}")
    for scene_type in ("harbor", "airport"):
        vals = per_scene[scene_type]
        if vals:
            print(f"[FastSAM][Eval] {scene_type}_mIoU={np.mean(vals):.4f} n={len(vals)}")
    print(f"[FastSAM][DEBUG] Sample eval records (first {len(debug_lines)}):")
    for line in debug_lines:
        print(f"  {line}")
    print(f"[FastSAM][DEBUG] Sample skipped records (first {len(skipped_lines)}):")
    for line in skipped_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
