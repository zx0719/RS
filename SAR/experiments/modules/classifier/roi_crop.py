"""
roi_crop.py — OBB ROI extraction for fine-grained classification.

Extracts axis-aligned chips from SAR images at YOLO OBB detection locations,
padding and resizing to a fixed chip size for classifier input.

Usage
-----
    from modules.classifier.roi_crop import crop_obb_roi, batch_crop_obb_roi

    chip = crop_obb_roi(image, cx=100, cy=200, w=80, h=20, angle_deg=-15)
    chips = batch_crop_obb_roi(image, objects, chip_size=128)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def crop_obb_roi(
    image: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
    chip_size: int = 128,
    margin: float = 0.2,
) -> np.ndarray:
    """Crop and axis-align a single OBB ROI from a SAR image.

    Parameters
    ----------
    image     : Source image as numpy array (H, W) or (H, W, C).
    cx, cy    : OBB centre in pixel coordinates.
    w, h      : OBB width and height (long axis first).
    angle_deg : OBB rotation angle in degrees (positive = counter-clockwise).
    chip_size : Output chip side length in pixels (square).
    margin    : Extra margin ratio added around the OBB before cropping
                (0.2 = 20% on each side).

    Returns
    -------
    Chip as numpy array (chip_size, chip_size). Single-channel grayscale.
    """
    if not _CV2_AVAILABLE:
        raise ImportError("opencv-python is required: pip install opencv-python")

    # -- 1. Pad the OBB dimensions -------------------------------------------
    padded_w = w * (1.0 + 2.0 * margin)
    padded_h = h * (1.0 + 2.0 * margin)
    half_diag = np.sqrt(padded_w**2 + padded_h**2) / 2.0

    # -- 2. Axis-aligned bounding box around the padded OBB ------------------
    x_min = int(max(0, cx - half_diag))
    y_min = int(max(0, cy - half_diag))
    x_max = int(min(image.shape[1], cx + half_diag))
    y_max = int(min(image.shape[0], cy + half_diag))

    if x_max <= x_min or y_max <= y_min:
        # Degenerate: return blank chip
        return np.zeros((chip_size, chip_size), dtype=np.float32)

    # -- 3. Crop -------------------------------------------------------------
    if image.ndim == 3:
        crop = image[y_min:y_max, x_min:x_max, :].astype(np.float32)
        if crop.shape[2] == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        crop = image[y_min:y_max, x_min:x_max].astype(np.float32)

    # -- 4. Rotation matrix to axis-align the OBB ---------------------------
    angle_rad = np.deg2rad(angle_deg)
    rot_mat = cv2.getRotationMatrix2D(
        (crop.shape[1] / 2.0, crop.shape[0] / 2.0),
        angle_deg,
        1.0,
    )
    rotated = cv2.warpAffine(
        crop, rot_mat, (crop.shape[1], crop.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # -- 5. Resize to fixed chip_size ----------------------------------------
    resized = cv2.resize(rotated, (chip_size, chip_size), interpolation=cv2.INTER_LINEAR)

    # -- 6. Normalise to [0, 1] ----------------------------------------------
    max_val = resized.max()
    if max_val > 0:
        resized = resized / max_val

    return resized.astype(np.float32)


def batch_crop_obb_roi(
    image_uri: str | Path,
    objects: list[dict[str, Any]],
    chip_size: int = 128,
    margin: float = 0.2,
) -> list[np.ndarray]:
    """Batch-extract OBB ROIs for a list of detection objects.

    Parameters
    ----------
    image_uri : Path to the source SAR image.
    objects   : List of Evidence Package object dicts, each containing
                ``geometry.pixel`` with cx, cy, w, h, angle_deg.
    chip_size : Output chip size.
    margin    : Extra margin ratio around each OBB.

    Returns
    -------
    List of numpy arrays (chip_size, chip_size) in the same order as objects.
    """
    import cv2 as _cv2

    image = _cv2.imread(str(image_uri), _cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_uri}")

    chips: list[np.ndarray] = []
    for obj in objects:
        pixel = obj.get("geometry", {}).get("pixel", {})
        cx = float(pixel.get("center_x", 0))
        cy = float(pixel.get("center_y", 0))
        w = float(pixel.get("width", 0))
        h = float(pixel.get("height", 0))
        angle = float(pixel.get("angle_deg", 0))

        chip = crop_obb_roi(image, cx, cy, w, h, angle, chip_size=chip_size, margin=margin)
        chips.append(chip)

    return chips
