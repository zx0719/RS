"""
tiling.py — 大图滑动窗口切片推理工具

提供切片网格生成、坐标映射、跨 tile NMS 去重等功能，
供 DetectorTool._detect_tiled() 调用。

核心函数：
    tile_grid()              生成滑动窗口坐标列表
    read_image_as_array()    读取图像为 uint8 RGB numpy 数组（支持 GeoTIFF）
    shift_object_to_global() 将 tile 内坐标映射回原图
    clamp_object_to_bounds() 将坐标裁剪到图像边界内
    nms_objects()            基于 bbox IoU 的跨 tile NMS 去重
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 切片网格
# ---------------------------------------------------------------------------

def tile_grid(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """生成滑动窗口坐标列表。

    Parameters
    ----------
    width, height : 原图尺寸（像素）
    tile_size     : 切片边长（像素）
    overlap       : 相邻切片重叠像素数

    Returns
    -------
    list of (x, y, tile_w, tile_h) — 每个切片的左上角坐标和实际尺寸
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and < tile_size")

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


# ---------------------------------------------------------------------------
# 图像读取
# ---------------------------------------------------------------------------

def read_image_as_array(image_path: str | Path) -> np.ndarray:
    """读取图像为 uint8 RGB numpy 数组。

    支持 JPEG/PNG/BMP 及 GeoTIFF（单波段/多波段）。
    浮点/16bit 图像做 1%~99% 百分位拉伸后转 uint8。

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8
    """
    from PIL import Image

    try:
        pil = Image.open(str(image_path))
        arr = np.asarray(pil)
    except Exception:
        # GeoTIFF fallback via rasterio
        try:
            import rasterio
            with rasterio.open(str(image_path)) as src:
                arr = src.read(1)  # 第一波段
        except Exception as e:
            raise ValueError(f"无法读取图像: {image_path}") from e

    return _to_uint8_rgb(arr)


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """将任意 dtype/通道数的 numpy 数组转为 uint8 RGB (H,W,3)。"""
    arr = np.asarray(arr)

    # 多波段取前3通道，单通道扩展为3通道
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.ndim == 3 and arr.shape[2] == 2:
        arr = arr[:, :, :1]

    # 非 uint8 做百分位拉伸
    if arr.dtype != np.uint8:
        flat = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr.reshape(-1)
        if flat.size == 0:
            arr = np.zeros(arr.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(flat, (1, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.min(flat)), float(np.max(flat))
            if hi <= lo:
                arr = np.zeros(arr.shape, dtype=np.uint8)
            else:
                arr = np.clip(
                    (arr.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255
                ).astype(np.uint8)

    # 确保 3 通道
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    else:
        arr = arr[:, :, :3]

    return arr.astype(np.uint8, copy=False)


# ---------------------------------------------------------------------------
# 坐标映射
# ---------------------------------------------------------------------------

def shift_object_to_global(
    obj: dict[str, Any],
    offset_x: int,
    offset_y: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """将 tile 内检测坐标映射回原图坐标系，并裁剪到图像边界。

    Parameters
    ----------
    obj           : DetectorTool.detect() 返回的单个 object dict
    offset_x/y    : tile 左上角在原图中的像素偏移
    image_width/height : 原图尺寸（用于边界裁剪）
    """
    shifted = copy.deepcopy(obj)
    pixel = shifted.setdefault("geometry", {}).setdefault("pixel", {})

    if "center_x" in pixel:
        pixel["center_x"] = round(float(pixel["center_x"]) + offset_x, 2)
    if "center_y" in pixel:
        pixel["center_y"] = round(float(pixel["center_y"]) + offset_y, 2)

    if isinstance(pixel.get("polygon"), list):
        pixel["polygon"] = [
            [round(float(p[0]) + offset_x, 2), round(float(p[1]) + offset_y, 2)]
            for p in pixel["polygon"] if len(p) >= 2
        ]

    if isinstance(pixel.get("bbox_axis_aligned"), list) and len(pixel["bbox_axis_aligned"]) == 4:
        x1, y1, x2, y2 = pixel["bbox_axis_aligned"]
        pixel["bbox_axis_aligned"] = [
            round(float(x1) + offset_x, 2),
            round(float(y1) + offset_y, 2),
            round(float(x2) + offset_x, 2),
            round(float(y2) + offset_y, 2),
        ]

    return clamp_object_to_bounds(shifted, image_width, image_height)


def clamp_object_to_bounds(
    obj: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """将 object 的所有像素坐标裁剪到 [0, image_width-1] × [0, image_height-1]。"""
    pixel = obj.setdefault("geometry", {}).setdefault("pixel", {})
    max_x = float(max(image_width - 1, 0))
    max_y = float(max(image_height - 1, 0))

    for key, limit in (("center_x", max_x), ("center_y", max_y)):
        if key in pixel:
            pixel[key] = round(min(max(float(pixel[key]), 0.0), limit), 2)

    polygon = pixel.get("polygon")
    if isinstance(polygon, list) and polygon:
        clipped = [
            [round(min(max(float(p[0]), 0.0), max_x), 2),
             round(min(max(float(p[1]), 0.0), max_y), 2)]
            for p in polygon if len(p) >= 2
        ]
        pixel["polygon"] = clipped
        if clipped:
            xs = [p[0] for p in clipped]
            ys = [p[1] for p in clipped]
            pixel["bbox_axis_aligned"] = [
                round(min(xs), 2), round(min(ys), 2),
                round(max(xs), 2), round(max(ys), 2),
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


# ---------------------------------------------------------------------------
# 跨 tile NMS
# ---------------------------------------------------------------------------

def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """计算两个轴对齐矩形框的 IoU。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def nms_objects(
    objects: list[dict[str, Any]],
    iou_thresh: float = 0.5,
) -> list[dict[str, Any]]:
    """对跨 tile 的检测结果做 NMS 去重。

    按置信度降序排列，同类别框 IoU ≥ iou_thresh 时保留高置信度框。
    无 bbox_axis_aligned 的框直接保留（不参与 NMS）。
    """
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

        duplicate = any(
            prev.get("class", {}).get("code") == code
            and isinstance(prev.get("geometry", {}).get("pixel", {}).get("bbox_axis_aligned"), list)
            and _bbox_iou(
                [float(v) for v in box],
                [float(v) for v in prev["geometry"]["pixel"]["bbox_axis_aligned"]],
            ) >= iou_thresh
            for prev in kept
        )
        if not duplicate:
            kept.append(obj)

    return kept
