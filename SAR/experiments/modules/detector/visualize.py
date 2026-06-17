"""
visualize.py — OBB 检测结果可视化

在原图上绘制旋转框、类别标签、置信度，输出标注图。
中文标签通过 PIL + NotoSansCJK 字体渲染，避免 cv2 乱码。

用法（独立）：
    from modules.detector.visualize import draw_detections
    annotated = draw_detections(image_path, objects)
    annotated.save("output.jpg")

用法（通过 DetectorTool）：
    objects = tool.detect(image_path, save_vis=True, vis_dir="output/vis")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 中文字体路径（按优先级查找） ────────────────────────────────────────────
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]
_FONT_PATH = next((p for p in _FONT_CANDIDATES if Path(p).exists()), None)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if _FONT_PATH:
        try:
            return ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            pass
    return ImageFont.load_default()


# 每个 super_class 对应的 RGB 颜色（PIL 用 RGB）
_COLORS: dict[str, tuple[int, int, int]] = {
    "ship":           (0,   200,  50),
    "aircraft":       (255, 180,  50),
    "ground":         (255, 100,   0),
    "infrastructure": (0,   100, 200),
    "unknown":        (180, 180, 180),
}

_THICKNESS: dict[str, int] = {
    "CRITICAL": 3,
    "HIGH":     2,
    "MEDIUM":   2,
    "LOW":      1,
}


def _get_color(obj: dict) -> tuple[int, int, int]:
    sc = obj.get("class", {}).get("super_class", "unknown")
    return _COLORS.get(sc, _COLORS["unknown"])


def _get_thickness(obj: dict) -> int:
    pri = obj.get("class", {}).get("priority", "LOW")
    return _THICKNESS.get(pri, 2)


def draw_detections(
    image_path: str | Path,
    objects: list[dict],
    score_thresh: float = 0.0,
    show_conf: bool = True,
    font_size: int = 16,
) -> np.ndarray:
    """在图像上绘制 OBB 检测结果，返回 BGR numpy 数组。

    Parameters
    ----------
    image_path:   原始图像路径（支持 GeoTIFF、JPG、PNG、BMP）
    objects:      DetectorTool.detect() 返回的 objects 列表
    score_thresh: 低于此置信度的框不绘制（默认全绘）
    show_conf:    是否在标签中显示置信度
    font_size:    中文字体大小（像素）
    """
    # ── 读图 ────────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        try:
            pil_img = Image.open(str(image_path)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"无法读取图像: {image_path}") from e

    # 转 PIL RGB 用于文字渲染
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(font_size)
    font_small = _load_font(max(font_size - 2, 12))

    for obj in objects:
        conf = obj.get("score", {}).get("confidence", 0.0)
        if conf < score_thresh:
            continue

        polygon = obj.get("geometry", {}).get("pixel", {}).get("polygon")
        if not polygon or len(polygon) < 3:
            continue

        pts = [(int(p[0]), int(p[1])) for p in polygon]
        color = _get_color(obj)
        thickness = _get_thickness(obj)

        # 绘制旋转框
        draw.polygon(pts, outline=color + (255,) if len(color) == 3 else color)
        # 加粗：多画几次
        for _ in range(thickness - 1):
            draw.polygon(pts, outline=color)

        # 标签
        name_cn = obj.get("class", {}).get("name_cn", "?")
        label = f"{name_cn} {conf:.2f}" if show_conf else name_cn

        # 标签位置：框最高点上方
        top_pt = min(pts, key=lambda p: p[1])
        tx, ty = top_pt[0], top_pt[1] - font_size - 4

        # 计算文字尺寸
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 背景矩形
        pad = 3
        draw.rectangle(
            [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
            fill=color,
        )
        # 白色文字
        draw.text((tx, ty), label, font=font, fill=(255, 255, 255))

    # ── 右下角统计信息 ───────────────────────────────────────────────────────
    counts: dict[str, int] = {}
    for obj in objects:
        if obj.get("score", {}).get("confidence", 0.0) >= score_thresh:
            name = obj.get("class", {}).get("name_cn", "?")
            counts[name] = counts.get(name, 0) + 1

    if counts:
        w, h = pil_img.size
        lines = [f"共 {sum(counts.values())} 个目标"] + \
                [f"{name}: {cnt}" for name, cnt in sorted(counts.items())]
        pad = 6
        line_h = font_size + pad
        box_h = line_h * len(lines) + pad * 2
        box_w = max(
            draw.textbbox((0, 0), l, font=font_small)[2]
            for l in lines
        ) + pad * 4

        bx0, by0 = w - box_w - 6, h - box_h - 6
        draw.rectangle([bx0, by0, w - 6, h - 6], fill=(30, 30, 30, 200))
        for i, line in enumerate(lines):
            draw.text(
                (bx0 + pad, by0 + pad + i * line_h),
                line, font=font_small, fill=(220, 220, 220),
            )

    # 转回 BGR numpy
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def save_annotated(
    image_path: str | Path,
    objects: list[dict],
    output_dir: str | Path,
    suffix: str = "_annotated",
    **kwargs,
) -> Path:
    """绘制并保存标注图，返回输出路径。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(image_path)
    out_path = out_dir / (src.stem + suffix + ".jpg")

    annotated = draw_detections(image_path, objects, **kwargs)
    cv2.imwrite(str(out_path), annotated)
    return out_path

