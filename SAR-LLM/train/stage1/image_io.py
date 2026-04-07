from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_sar_as_rgb(path: str) -> Image.Image:
    """读取 SAR 图像，并保证输出为 3 通道 RGB（灰度复制 3 份）。

    数据里 SAR 可能是：
    - 1 通道 jpg/png：直接复制为 3 通道；
    - 3 通道 jpg/png：内部已经是灰度 copy（3 通道相同），无需额外处理；
    - 2 通道 tif：取第 2 通道 (index=1) 作为灰度，再复制为 3 通道；
    - 其他 tif：尽量鲁棒地选一个通道转灰度。

    返回：
        PIL.Image，mode=RGB
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in {".jpg", ".jpeg", ".png"}:
        img = Image.open(path)
        if img.mode == "RGB":
            return img
        # 1 通道 / RGBA / 其他情况：统一转灰度再复制
        gray = img.convert("L")
        return Image.merge("RGB", (gray, gray, gray))

    if ext in {".tif", ".tiff"}:
        arr = _read_tif(path)
        gray = _select_gray_from_tif(arr)
        gray8 = _to_uint8(gray, path=path)
        pil_l = Image.fromarray(gray8, mode="L")
        return Image.merge("RGB", (pil_l, pil_l, pil_l))

    # 兜底：交给 PIL 读，再转灰度复制
    img = Image.open(path).convert("L")
    return Image.merge("RGB", (img, img, img))


def _read_tif(path: str) -> np.ndarray:
    """读取 tif，返回 numpy 数组。

    优先 tifffile；如果环境里没有，再尝试 imageio。
    """
    try:
        import tifffile  # type: ignore

        return tifffile.imread(path)
    except Exception:
        try:
            import imageio.v2 as imageio  # type: ignore

            return imageio.imread(path)
        except Exception as e:
            raise RuntimeError(
                "读取 .tif 需要 tifffile 或 imageio。建议：pip install tifffile imageio"
            ) from e


def _select_gray_from_tif(arr: np.ndarray) -> np.ndarray:
    """从 tif 的多通道数组中挑一个灰度通道。

    规则：
    - (H, W)：直接用；
    - (H, W, 2)：取第二通道 arr[:, :, 1]；
    - (2, H, W)：取第二通道 arr[1]；
    - 其他 (H, W, C)：取第 0 通道；
    - 其他 (C, H, W)：取第 0 通道；
    """
    a = np.asarray(arr)

    if a.ndim == 2:
        return a

    if a.ndim != 3:
        raise ValueError(f"Unsupported tif ndim={a.ndim}, shape={a.shape}")

    # 尽量判断 channel 维在哪
    if a.shape[2] in {2, 3, 4, 13}:
        # 常见格式：HWC
        if a.shape[2] == 2:
            return a[:, :, 1]
        return a[:, :, 0]

    if a.shape[0] in {2, 3, 4, 13}:
        # 常见格式：CHW
        if a.shape[0] == 2:
            return a[1, :, :]
        return a[0, :, :]

    # 实在判断不了：默认取最后一维第 0 通道
    return a[:, :, 0]

class BadSarImageError(RuntimeError):
    pass


def _to_uint8(gray: np.ndarray, path: str | None = None) -> np.ndarray:
    g = np.asarray(gray).astype(np.float32, copy=False)

    finite_mask = np.isfinite(g)
    finite_ratio = float(finite_mask.mean())

    if finite_ratio == 0.0:
        raise BadSarImageError(
            f"all pixels are non-finite"
            + (f" | path={path}" if path is not None else "")
        )

    if finite_ratio < 0.1:
        raise BadSarImageError(
            f"too few finite pixels: finite_ratio={finite_ratio:.6f}"
            + (f" | path={path}" if path is not None else "")
        )

    valid = g[finite_mask]
    lo, hi = np.percentile(valid, [1, 99])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
        if hi <= lo:
            hi = lo + 1e-6

    g = np.where(finite_mask, g, lo)

    x = (g - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return (x * 255.0).astype(np.uint8)

# def _to_uint8(gray: np.ndarray) -> np.ndarray:
#     g = np.asarray(gray)

#     # 统一转 float32 处理
#     g = g.astype(np.float32, copy=False)

#     # 先把非有限值处理掉
#     finite_mask = np.isfinite(g)
#     if not finite_mask.any():
#         # 整张图全坏，直接返回全零，至少不会把训练炸掉
#         return np.zeros(g.shape, dtype=np.uint8)

#     valid = g[finite_mask]
#     lo, hi = np.percentile(valid, [1, 99])

#     if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#         lo, hi = float(valid.min()), float(valid.max())
#         if hi <= lo:
#             hi = lo + 1e-6

#     # 把无效值先填成 lo
#     g = np.where(finite_mask, g, lo)

#     x = (g - lo) / (hi - lo)
#     x = np.clip(x, 0.0, 1.0)
#     x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

#     return (x * 255.0).astype(np.uint8)

