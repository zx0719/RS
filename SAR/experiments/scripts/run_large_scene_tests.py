#!/usr/bin/env python3
"""
Run tiled large-scene SAR inference on the fixed MSAR-1.0 manifest.

The script references remote source images in-place, writes one output
directory per case, and deliberately avoids full-resolution visualization.
Each case emits:
  - <case_name>_evidence.json
  - <case_name>_overview.jpg
  - <case_name>_run.log
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFile

from modules.report.collab_config import collaborative_generator_kwargs, resolve_collaborative_config, resolve_vlm_config

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import rasterio  # type: ignore[import-untyped]
    from rasterio.enums import Resampling  # type: ignore[import-untyped]
    from rasterio.windows import Window  # type: ignore[import-untyped]

    _RASTERIO_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal envs
    _RASTERIO_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "tests" / "fixtures" / "large_scene_manifest.json"
DEFAULT_WEIGHTS = "/mnt/data/zhuxiang/SAR_experiments/runs/v4-multiclass/weights/best.pt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "large_scene"

LOGGER = logging.getLogger("large_scene_tests")


@dataclass(frozen=True)
class LargeSceneCase:
    name: str
    path: Path
    relative_path: str
    scene_category: str
    region_name: str
    region_type: str
    usage: str
    expected_width: int
    expected_height: int
    expected_size_mb: int | None
    suites: tuple[str, ...]


@dataclass(frozen=True)
class ImageMeta:
    width: int
    height: int
    bands: int | None
    dtype: str | None


class WindowReader:
    """Read image windows without materialising full-resolution arrays."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._dataset: Any | None = None
        self._pil_image: Image.Image | None = None
        self._use_rasterio = False

        if _RASTERIO_AVAILABLE:
            try:
                self._dataset = rasterio.open(path)
                self._use_rasterio = True
            except Exception as exc:
                LOGGER.warning("rasterio could not open %s: %s; falling back to PIL", path, exc)

        if not self._use_rasterio:
            self._pil_image = Image.open(path)

    @property
    def meta(self) -> ImageMeta:
        if self._use_rasterio and self._dataset is not None:
            return ImageMeta(
                width=int(self._dataset.width),
                height=int(self._dataset.height),
                bands=int(self._dataset.count),
                dtype=str(self._dataset.dtypes[0]) if self._dataset.dtypes else None,
            )
        assert self._pil_image is not None
        bands = len(self._pil_image.getbands()) if self._pil_image.getbands() else 1
        return ImageMeta(
            width=int(self._pil_image.width),
            height=int(self._pil_image.height),
            bands=bands,
            dtype=str(self._pil_image.mode),
        )

    def read_window(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        if self._use_rasterio and self._dataset is not None:
            arr = self._dataset.read(window=Window(x, y, width, height))
            if arr.ndim == 3:
                if arr.shape[0] == 1:
                    return arr[0]
                return np.moveaxis(arr[:3], 0, -1)
            return arr

        assert self._pil_image is not None
        crop = self._pil_image.crop((x, y, x + width, y + height))
        return np.asarray(crop)

    def read_overview(self, max_side: int) -> np.ndarray:
        meta = self.meta
        scale = min(max_side / max(meta.width, meta.height), 1.0)
        out_width = max(1, int(round(meta.width * scale)))
        out_height = max(1, int(round(meta.height * scale)))

        if self._use_rasterio and self._dataset is not None:
            arr = self._dataset.read(
                out_shape=(self._dataset.count, out_height, out_width),
                resampling=Resampling.bilinear,
            )
            if arr.ndim == 3:
                if arr.shape[0] == 1:
                    return arr[0]
                return np.moveaxis(arr[:3], 0, -1)
            return arr

        assert self._pil_image is not None
        img = self._pil_image.copy()
        img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        return np.asarray(img)

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
        if self._pil_image is not None:
            self._pil_image.close()

    def __enter__(self) -> "WindowReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if "root" not in manifest or "cases" not in manifest:
        raise ValueError(f"Invalid large-scene manifest: {path}")
    return manifest


def select_cases(
    manifest: dict[str, Any],
    suite: str,
    names: Iterable[str] | None = None,
) -> list[LargeSceneCase]:
    root = Path(manifest["root"])
    requested = set(names or [])
    cases: list[LargeSceneCase] = []

    for raw in manifest["cases"]:
        raw_suites = tuple(raw.get("suites", ()))
        if suite not in raw_suites:
            continue
        if requested and raw["name"] not in requested:
            continue
        cases.append(
            LargeSceneCase(
                name=raw["name"],
                path=root / raw["relative_path"],
                relative_path=raw["relative_path"],
                scene_category=raw["scene_category"],
                region_name=raw["region_name"],
                region_type=raw["region_type"],
                usage=raw["usage"],
                expected_width=int(raw["expected_width"]),
                expected_height=int(raw["expected_height"]),
                expected_size_mb=raw.get("expected_size_mb"),
                suites=raw_suites,
            )
        )

    if requested:
        found = {case.name for case in cases}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Requested case(s) are not in suite '{suite}': {missing}")
    if not cases:
        raise ValueError(f"No large-scene cases selected for suite '{suite}'")
    return cases


def tile_grid(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and smaller than tile_size")

    stride = tile_size - overlap

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, length - tile_size + 1, stride))
        last = length - tile_size
        if values[-1] != last:
            values.append(last)
        return values

    windows: list[tuple[int, int, int, int]] = []
    for y in starts(height):
        for x in starts(width):
            windows.append((x, y, min(tile_size, width - x), min(tile_size, height - y)))
    return windows


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.ndim == 3 and arr.shape[2] == 2:
        arr = arr[:, :, :1]

    if arr.dtype != np.uint8:
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr.reshape(-1)
        if finite.size == 0:
            scaled = np.zeros(arr.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(finite, (1, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.min(finite))
                hi = float(np.max(finite))
            if hi <= lo:
                scaled = np.zeros(arr.shape, dtype=np.uint8)
            else:
                scaled = np.clip((arr.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        arr = scaled

    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    return arr[:, :, :3].astype(np.uint8, copy=False)


def shift_object_to_global(
    obj: dict[str, Any],
    offset_x: int,
    offset_y: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    shifted = copy.deepcopy(obj)
    pixel = shifted.setdefault("geometry", {}).setdefault("pixel", {})

    if "center_x" in pixel:
        pixel["center_x"] = round(float(pixel["center_x"]) + offset_x, 2)
    if "center_y" in pixel:
        pixel["center_y"] = round(float(pixel["center_y"]) + offset_y, 2)

    if isinstance(pixel.get("polygon"), list):
        polygon: list[list[float]] = []
        for point in pixel["polygon"]:
            if len(point) < 2:
                continue
            polygon.append([
                round(float(point[0]) + offset_x, 2),
                round(float(point[1]) + offset_y, 2),
            ])
        pixel["polygon"] = polygon

    if isinstance(pixel.get("bbox_axis_aligned"), list) and len(pixel["bbox_axis_aligned"]) == 4:
        x1, y1, x2, y2 = pixel["bbox_axis_aligned"]
        pixel["bbox_axis_aligned"] = [
            round(float(x1) + offset_x, 2),
            round(float(y1) + offset_y, 2),
            round(float(x2) + offset_x, 2),
            round(float(y2) + offset_y, 2),
        ]

    return clamp_object_to_bounds(shifted, image_width, image_height)


def clamp_object_to_bounds(obj: dict[str, Any], image_width: int, image_height: int) -> dict[str, Any]:
    pixel = obj.setdefault("geometry", {}).setdefault("pixel", {})
    max_x = float(max(image_width - 1, 0))
    max_y = float(max(image_height - 1, 0))

    if "center_x" in pixel:
        pixel["center_x"] = round(min(max(float(pixel["center_x"]), 0.0), max_x), 2)
    if "center_y" in pixel:
        pixel["center_y"] = round(min(max(float(pixel["center_y"]), 0.0), max_y), 2)

    polygon = pixel.get("polygon")
    if isinstance(polygon, list) and polygon:
        clipped: list[list[float]] = []
        for point in polygon:
            if len(point) < 2:
                continue
            clipped.append([
                round(min(max(float(point[0]), 0.0), max_x), 2),
                round(min(max(float(point[1]), 0.0), max_y), 2),
            ])
        pixel["polygon"] = clipped
        if clipped:
            xs = [p[0] for p in clipped]
            ys = [p[1] for p in clipped]
            pixel["bbox_axis_aligned"] = [
                round(min(xs), 2),
                round(min(ys), 2),
                round(max(xs), 2),
                round(max(ys), 2),
            ]
    elif isinstance(pixel.get("bbox_axis_aligned"), list) and len(pixel["bbox_axis_aligned"]) == 4:
        x1, y1, x2, y2 = [float(v) for v in pixel["bbox_axis_aligned"]]
        pixel["bbox_axis_aligned"] = [
            round(min(max(min(x1, x2), 0.0), max_x), 2),
            round(min(max(min(y1, y2), 0.0), max_y), 2),
            round(min(max(max(x1, x2), 0.0), max_x), 2),
            round(min(max(max(y1, y2), 0.0), max_y), 2),
        ]

    return obj


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def nms_objects(objects: list[dict[str, Any]], iou_thresh: float) -> list[dict[str, Any]]:
    ordered = sorted(
        objects,
        key=lambda o: float(o.get("score", {}).get("confidence", 0.0)),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []

    for obj in ordered:
        box = obj.get("geometry", {}).get("pixel", {}).get("bbox_axis_aligned")
        code = obj.get("class", {}).get("code")
        if not isinstance(box, list) or len(box) != 4:
            kept.append(obj)
            continue

        duplicate = False
        for prev in kept:
            prev_box = prev.get("geometry", {}).get("pixel", {}).get("bbox_axis_aligned")
            prev_code = prev.get("class", {}).get("code")
            if prev_code != code or not isinstance(prev_box, list) or len(prev_box) != 4:
                continue
            if bbox_iou([float(v) for v in box], [float(v) for v in prev_box]) >= iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(obj)

    return kept


def validate_object_bounds(objects: list[dict[str, Any]], image_width: int, image_height: int) -> None:
    max_x = float(max(image_width - 1, 0))
    max_y = float(max(image_height - 1, 0))
    errors: list[str] = []

    for obj in objects:
        obj_id = obj.get("object_id", "<unknown>")
        pixel = obj.get("geometry", {}).get("pixel", {})
        for key, limit in (("center_x", max_x), ("center_y", max_y)):
            if key in pixel and not (0.0 <= float(pixel[key]) <= limit):
                errors.append(f"{obj_id}.{key}={pixel[key]} outside [0,{limit}]")
        polygon = pixel.get("polygon")
        if isinstance(polygon, list):
            for idx, point in enumerate(polygon):
                if len(point) < 2:
                    errors.append(f"{obj_id}.polygon[{idx}] is invalid: {point}")
                    continue
                x, y = float(point[0]), float(point[1])
                if not (0.0 <= x <= max_x and 0.0 <= y <= max_y):
                    errors.append(f"{obj_id}.polygon[{idx}]={point} outside image bounds")
        box = pixel.get("bbox_axis_aligned")
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = [float(v) for v in box]
            if not (0.0 <= x1 <= x2 <= max_x and 0.0 <= y1 <= y2 <= max_y):
                errors.append(f"{obj_id}.bbox_axis_aligned={box} outside image bounds")

    if errors:
        raise ValueError("Object coordinate bounds validation failed: " + "; ".join(errors[:10]))


def duplicate_pair_count(objects: list[dict[str, Any]], iou_thresh: float = 0.95) -> int:
    count = 0
    for i, obj in enumerate(objects):
        box = obj.get("geometry", {}).get("pixel", {}).get("bbox_axis_aligned")
        code = obj.get("class", {}).get("code")
        if not isinstance(box, list) or len(box) != 4:
            continue
        for other in objects[i + 1:]:
            other_box = other.get("geometry", {}).get("pixel", {}).get("bbox_axis_aligned")
            if other.get("class", {}).get("code") != code:
                continue
            if isinstance(other_box, list) and len(other_box) == 4:
                if bbox_iou([float(v) for v in box], [float(v) for v in other_box]) >= iou_thresh:
                    count += 1
    return count


def build_detector(weights: Path, score_thresh: float, device: str) -> Any:
    sys.path.insert(0, str(PROJECT_ROOT))
    from modules.detector import DetectorTool
    from modules.detector.class_map import CLASS_MAP

    return DetectorTool(
        model_path=str(weights),
        class_map=CLASS_MAP,
        score_thresh=score_thresh,
        device=device,
    )


def run_tiled_detection(
    reader: WindowReader,
    detector: Any,
    case: LargeSceneCase,
    tile_size: int,
    overlap: int,
    temp_dir: Path,
    nms_iou: float,
    max_tiles: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = reader.meta
    windows = tile_grid(meta.width, meta.height, tile_size=tile_size, overlap=overlap)
    if max_tiles is not None:
        windows = windows[:max_tiles]

    raw_objects: list[dict[str, Any]] = []
    started = time.time()

    for idx, (x, y, width, height) in enumerate(windows, start=1):
        LOGGER.info(
            "[%s] tile %d/%d x=%d y=%d w=%d h=%d",
            case.name, idx, len(windows), x, y, width, height,
        )
        tile_rgb = to_uint8_rgb(reader.read_window(x, y, width, height))
        tile_path = temp_dir / f"{case.name}_tile_{idx:04d}_x{x}_y{y}.jpg"
        Image.fromarray(tile_rgb).save(tile_path, quality=92)

        tile_objects, _vis = detector.detect(str(tile_path), save_vis=False)
        for obj in tile_objects:
            raw_objects.append(shift_object_to_global(obj, x, y, meta.width, meta.height))

    deduped = nms_objects(raw_objects, iou_thresh=nms_iou)
    validate_object_bounds(deduped, meta.width, meta.height)

    run_stats = {
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_count": len(windows),
        "raw_object_count": len(raw_objects),
        "deduped_object_count": len(deduped),
        "removed_by_nms": len(raw_objects) - len(deduped),
        "high_iou_duplicate_pairs_after_nms": duplicate_pair_count(deduped),
        "elapsed_sec": round(time.time() - started, 2),
    }
    return deduped, run_stats


def create_overview(
    reader: WindowReader,
    objects: list[dict[str, Any]],
    output_path: Path,
    max_side: int,
) -> None:
    meta = reader.meta
    overview_rgb = to_uint8_rgb(reader.read_overview(max_side=max_side))
    image = Image.fromarray(overview_rgb)
    draw = ImageDraw.Draw(image)
    scale_x = image.width / meta.width
    scale_y = image.height / meta.height

    colors = {
        "ship": (0, 210, 90),
        "aircraft": (250, 180, 40),
        "ground": (245, 95, 40),
        "infrastructure": (40, 135, 230),
        "unknown": (220, 220, 220),
    }
    counts: dict[str, int] = {}
    for obj in objects:
        cls = obj.get("class", {})
        name = cls.get("name_cn") or cls.get("code") or "unknown"
        counts[name] = counts.get(name, 0) + 1
        color = colors.get(cls.get("super_class", "unknown"), colors["unknown"])
        polygon = obj.get("geometry", {}).get("pixel", {}).get("polygon")
        if isinstance(polygon, list) and len(polygon) >= 3:
            pts = [(float(x) * scale_x, float(y) * scale_y) for x, y in polygon]
            draw.line(pts + [pts[0]], fill=color, width=2)
        else:
            box = obj.get("geometry", {}).get("pixel", {}).get("bbox_axis_aligned")
            if isinstance(box, list) and len(box) == 4:
                x1, y1, x2, y2 = [float(v) for v in box]
                draw.rectangle(
                    (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                    outline=color,
                    width=2,
                )

    summary = f"{meta.width}x{meta.height} | objects: {len(objects)}"
    draw.rectangle((8, 8, min(image.width - 8, 520), 42), fill=(0, 0, 0))
    draw.text((16, 16), summary, fill=(255, 255, 255))

    if counts:
        y = 50
        for name, count in sorted(counts.items()):
            text = f"{name}: {count}"
            draw.rectangle((8, y, min(image.width - 8, 220), y + 24), fill=(0, 0, 0))
            draw.text((16, y + 4), text, fill=(255, 255, 255))
            y += 26

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=90)


def build_evidence(
    case: LargeSceneCase,
    objects: list[dict[str, Any]],
    overview_path: Path,
    run_stats: dict[str, Any],
    vlm_url: str | None = None,
    env_file: str | None = None,
) -> dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from modules.evidence import EvidenceBuilder
    from modules.eval import QualityGate
    from modules.report.collaborative import CollaborativeReportGenerator
    from modules.report.large_scene import attach_large_scene_metadata

    mission = {
        "region_name": case.region_name,
        "region_type": case.region_type,
        "priority": "HIGH",
        "user_prompt": f"MSAR-1.0 large scene validation: {case.name}",
    }
    package = EvidenceBuilder().build(str(case.path), mission, objects)
    package.setdefault("attachments", {})["overview_image"] = {
        "uri": f"file://{overview_path}",
        "role": "large_scene_overview",
    }
    package["trace"]["large_scene_test"] = {
        "case_name": case.name,
        "relative_path": case.relative_path,
        "scene_category": case.scene_category,
        "usage": case.usage,
        "run_stats": run_stats,
    }
    vlm_config = resolve_vlm_config(env_file=env_file, vlm_url=vlm_url)
    if vlm_config.vlm_base_url:
        from modules.report.vlm_describer import VLMDescriber
        from modules.report.vlm_trace import attach_vlm_scene_description
        describer = VLMDescriber(
            base_url=vlm_config.vlm_base_url,
            model_name=vlm_config.vlm_model_name or "qwen3-vl-4b",
            api_key=vlm_config.api_key or "EMPTY",
            timeout=vlm_config.timeout,
        )
        scene_desc = describer.describe(str(overview_path))
        if scene_desc:
            attach_vlm_scene_description(
                package,
                description=scene_desc,
                describer=describer,
                image_path=str(overview_path),
            )
        elif vlm_config.require_vlm:
            raise RuntimeError(f"VLM scene description is required but empty for {case.name}.")
    elif vlm_config.require_vlm:
        raise RuntimeError("SAR_REQUIRE_VLM=1 but SAR_VLM_URL/--vlm-url is empty.")
    package = attach_large_scene_metadata(package)

    collab_config = resolve_collaborative_config(
        cache_dir=str(PROJECT_ROOT / "output" / ".large_scene_cache"),
        env_file=env_file,
    )
    package = CollaborativeReportGenerator(
        **collaborative_generator_kwargs(collab_config),
    ).generate(package)
    package = QualityGate().evaluate(package)
    return package


def configure_logging(log_path: Path | None = None) -> logging.Handler | None:
    LOGGER.setLevel(logging.INFO)
    if not LOGGER.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        LOGGER.addHandler(stream)

    if log_path is None:
        return None

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(file_handler)
    return file_handler


def check_case_file(case: LargeSceneCase) -> ImageMeta:
    if not case.path.exists():
        raise FileNotFoundError(f"Large-scene source image not found: {case.path}")
    with WindowReader(case.path) as reader:
        meta = reader.meta
    if meta.width != case.expected_width or meta.height != case.expected_height:
        raise ValueError(
            f"{case.name}: expected {case.expected_width}x{case.expected_height}, "
            f"got {meta.width}x{meta.height}"
        )
    return meta


def run_case(args: argparse.Namespace, case: LargeSceneCase, detector: Any | None) -> dict[str, Any]:
    case_dir = Path(args.output_root) / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / f"{case.name}_run.log"
    file_handler = configure_logging(log_path)

    try:
        LOGGER.info("case=%s path=%s", case.name, case.path)
        meta = check_case_file(case)
        file_size_mb = round(case.path.stat().st_size / (1024 * 1024), 2)
        LOGGER.info(
            "image meta width=%d height=%d bands=%s dtype=%s size_mb=%.2f",
            meta.width, meta.height, meta.bands, meta.dtype, file_size_mb,
        )

        if args.dry_run:
            return {
                "case_name": case.name,
                "path": str(case.path),
                "width": meta.width,
                "height": meta.height,
                "bands": meta.bands,
                "dtype": meta.dtype,
                "size_mb": file_size_mb,
                "status": "DRY_RUN_OK",
            }

        if detector is None:
            raise RuntimeError("detector is required unless --dry-run is set")

        tile_root = Path(tempfile.mkdtemp(prefix=f"{case.name}_tiles_", dir=str(case_dir)))
        try:
            with WindowReader(case.path) as reader:
                max_tiles = args.max_tiles if args.max_tiles > 0 else None
                objects, run_stats = run_tiled_detection(
                    reader=reader,
                    detector=detector,
                    case=case,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    temp_dir=tile_root,
                    nms_iou=args.nms_iou,
                    max_tiles=max_tiles,
                )
                if len(objects) > args.max_objects:
                    raise RuntimeError(
                        f"{case.name}: {len(objects)} objects after NMS exceeds "
                        f"--max-objects={args.max_objects}"
                    )

                overview_path = case_dir / f"{case.name}_overview.jpg"
                create_overview(reader, objects, overview_path, max_side=args.overview_size)

            package = build_evidence(case, objects, overview_path, run_stats, vlm_url=args.vlm_url, env_file=args.env_file)
            evidence_path = case_dir / f"{case.name}_evidence.json"
            evidence_path.write_text(
                json.dumps(package, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            summary = {
                "case_name": case.name,
                "path": str(case.path),
                "width": meta.width,
                "height": meta.height,
                "size_mb": file_size_mb,
                "objects": len(objects),
                "evidence_path": str(evidence_path),
                "overview_path": str(overview_path),
                "log_path": str(log_path),
                "run_stats": run_stats,
                "quality_review_required": package.get("quality", {})
                .get("review_gate", {})
                .get("needs_human_review"),
                "status": "OK",
            }
            (case_dir / f"{case.name}_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            LOGGER.info("case complete: %s", json.dumps(summary, ensure_ascii=False))
            return summary
        finally:
            if args.keep_tiles:
                LOGGER.info("kept temporary tiles at %s", tile_root)
            else:
                shutil.rmtree(tile_root, ignore_errors=True)
    finally:
        if file_handler is not None:
            LOGGER.removeHandler(file_handler)
            file_handler.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MSAR-1.0 large-scene tiled detection smoke/full tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--case", action="append", default=[], help="Run only this case name; repeatable.")
    parser.add_argument("--weights", type=Path, default=Path(DEFAULT_WEIGHTS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--score-thresh", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=1536)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--overview-size", type=int, default=2000)
    parser.add_argument("--vlm-url", default=None, help="Optional VLM endpoint for scene-level description")
    parser.add_argument("--vlm-model", default=None, help="Optional VLM model name")
    parser.add_argument("--env-file", default=None, help="Optional .env file for shared model settings")
    parser.add_argument("--max-objects", type=int, default=5000)
    parser.add_argument("--max-tiles", type=int, default=0, help="Debug only; 0 means process all tiles.")
    parser.add_argument("--dry-run", action="store_true", help="Check manifest files and dimensions without inference.")
    parser.add_argument("--keep-tiles", action="store_true", help="Keep temporary JPEG tiles for debugging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    manifest = load_manifest(args.manifest)
    cases = select_cases(manifest, suite=args.suite, names=args.case)
    args.output_root.mkdir(parents=True, exist_ok=True)

    detector = None
    if not args.dry_run:
        if not args.weights.exists():
            raise FileNotFoundError(f"Detector weights not found: {args.weights}")
        detector = build_detector(args.weights, score_thresh=args.score_thresh, device=args.device)

    summaries: list[dict[str, Any]] = []
    for case in cases:
        summaries.append(run_case(args, case, detector))

    summary_path = args.output_root / f"{args.suite}_run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "suite": args.suite,
                "manifest": str(args.manifest),
                "weights": None if args.dry_run else str(args.weights),
                "dry_run": bool(args.dry_run),
                "cases": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("summary written to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
