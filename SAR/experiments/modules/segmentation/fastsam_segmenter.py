"""
fastsam_segmenter.py — M3: FastSAM zero-shot harbor/airport area segmentation.

Triggered by the Gate (SceneGate) when harbor or airport is detected.
Produces a segmentation mask and computes area in km^2 using GSD metadata.

Usage
-----
    from modules.segmentation import FastSAMSegmenter

    seg = FastSAMSegmenter()
    result = seg.segment("scene.tif", prompt="harbor", gsd_m=1.0)
    # {"mask_polygon": [...], "mask_area_km2": 0.456, ...}
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from ultralytics import FastSAM as _UltraFastSAM
    _FASTSAM_AVAILABLE = True
except ImportError:
    _FASTSAM_AVAILABLE = False
    _UltraFastSAM = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default FastSAM model bundled inside the offline pack.
DEFAULT_FASTSAM_MODEL: str = "FastSAM-s.pt"

# Minimum area threshold (km^2) to accept a segmentation as valid
MIN_AREA_KM2: float = 0.01

# Minimum mask confidence
MIN_MASK_CONFIDENCE: float = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simplify_contour(contour: np.ndarray, epsilon_factor: float = 0.005) -> list[list[float]]:
    """Simplify a contour polygon using Douglas-Peucker."""
    peri = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * peri
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.squeeze(1).astype(float).tolist()


def _mask_area_px(mask: np.ndarray) -> int:
    """Count non-zero pixels in a binary mask."""
    return int(np.count_nonzero(mask))


def _mask_area_km2(area_px: int, gsd_x_m: float, gsd_y_m: float) -> float:
    """Convert pixel area to km^2 using ground sample distance."""
    return round(float(area_px) * gsd_x_m * gsd_y_m / 1e6, 6)


def _resolve_fastsam_model_path(model_path: str | Path) -> str:
    """Resolve a bundled FastSAM checkpoint without allowing network download."""
    requested = Path(model_path)
    model_name = requested.name

    search_roots = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else None,
    ]
    search_roots = [root for root in search_roots if root is not None]

    candidates = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        for root in search_roots:
            candidates.append(root / requested)
            candidates.append(root / "assets" / "models" / model_name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "FastSAM checkpoint not found locally. "
        "Expected a bundled file such as assets/models/FastSAM-s.pt."
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FastSAMSegmenter:
    """FastSAM zero-shot segmentation for harbor/airport area estimation.

    Parameters
    ----------
    model_path  : Path or name of FastSAM checkpoint (e.g. "FastSAM-s.pt").
    device      : PyTorch device string.
    score_thresh: Minimum confidence for mask acceptance.
    """

    SEGMENTER_VERSION: str = "fastsam-v1.0"

    def __init__(
        self,
        model_path: str = DEFAULT_FASTSAM_MODEL,
        device: str = "cuda:0",
        score_thresh: float = 0.5,
    ) -> None:
        if not _FASTSAM_AVAILABLE:
            raise ImportError(
                "ultralytics is required for FastSAM. "
                "Please run: pip install ultralytics"
            )
        if not _CV2_AVAILABLE:
            raise ImportError("opencv-python is required: pip install opencv-python")

        self.model_path = _resolve_fastsam_model_path(model_path)
        self.device = device
        self.score_thresh = score_thresh

        self._model: Any = _UltraFastSAM(self.model_path)  # type: ignore[call-arg]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(
        self,
        image_uri: str | Path,
        prompt: str = "harbor",
        gsd_x_m: float | None = None,
        gsd_y_m: float | None = None,
        retinex: bool = True,
    ) -> dict[str, Any]:
        """Segment harbor or airport area and compute area.

        Parameters
        ----------
        image_uri : Path to the SAR image.
        prompt    : "harbor" or "airport" — determines which mask to extract.
        gsd_x_m   : Ground sample distance (m/px) in x direction.
        gsd_y_m   : Ground sample distance (m/px) in y direction.
        retinex   : If True, preprocess SAR image with CLAHE for contrast.

        Returns
        -------
        dict:
            mask_polygon     : Simplified polygon of the largest mask [[x,y],...]
            mask_area_px     : Mask area in pixels
            mask_area_km2    : Area in km^2 (None if GSD unavailable)
            mask_confidence  : Mean confidence of the mask
            scene_type       : "harbor" | "airport"
            segmentation_uri : None (saved externally if needed)
            success          : bool
        """
        image_path = str(image_uri)

        # 1. Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_uri}")

        # 2. Preprocess: CLAHE contrast enhancement for SAR
        if retinex:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

        # Convert to 3-channel for FastSAM (expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 3. Run FastSAM with "everything" mode (no prompt = segment all)
        results = self._model(
            source=img_rgb,
            device=self.device,
            retina_masks=True,
            conf=self.score_thresh,
            iou=0.7,
            verbose=False,
        )

        # 4. Extract masks from results
        masks = self._extract_masks(results)
        if not masks:
            return self._empty_result(prompt)

        # 5. Select the largest valid mask
        best = self._select_best_mask(masks)
        if best is None:
            return self._empty_result(prompt)

        # 6. Compute area
        area_px = _mask_area_px(best["mask"])
        area_km2 = None
        if gsd_x_m is not None and gsd_y_m is not None:
            area_km2 = _mask_area_km2(area_px, gsd_x_m, gsd_y_m)

        # 7. Extract contour polygon
        contours, _ = cv2.findContours(
            best["mask"].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return self._empty_result(prompt)

        largest_contour = max(contours, key=cv2.contourArea)
        polygon = _simplify_contour(largest_contour)

        # 8. Validate minimum area
        if area_km2 is not None and area_km2 < MIN_AREA_KM2:
            return self._empty_result(prompt)

        return {
            "mask_polygon": polygon,
            "mask_area_px": area_px,
            "mask_area_km2": area_km2,
            "mask_confidence": round(float(best["confidence"]), 4),
            "scene_type": prompt,
            "segmentation_uri": None,
            "model_version": self.SEGMENTER_VERSION,
            "success": True,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_masks(self, results: list[Any]) -> list[dict[str, Any]]:
        """Extract mask data from FastSAM results."""
        masks_out: list[dict[str, Any]] = []
        for r in results:
            if r.masks is None:
                continue
            mask_data = r.masks.data  # [N, H, W] tensor
            if mask_data is None or mask_data.shape[0] == 0:
                continue
            confs = r.masks.conf if hasattr(r.masks, "conf") else None
            for i in range(mask_data.shape[0]):
                m = mask_data[i].cpu().numpy()
                conf = float(confs[i]) if confs is not None else 1.0
                if conf >= MIN_MASK_CONFIDENCE:
                    masks_out.append({"mask": m, "confidence": conf})
        return masks_out

    def _select_best_mask(
        self, masks: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Select the mask with the largest area (most likely the facility)."""
        if not masks:
            return None
        # Sort by area descending, pick largest
        masks_sorted = sorted(masks, key=lambda m: _mask_area_px(m["mask"]), reverse=True)
        return masks_sorted[0]

    @staticmethod
    def _empty_result(prompt: str) -> dict[str, Any]:
        return {
            "mask_polygon": [],
            "mask_area_px": 0,
            "mask_area_km2": None,
            "mask_confidence": 0.0,
            "scene_type": prompt,
            "segmentation_uri": None,
            "model_version": FastSAMSegmenter.SEGMENTER_VERSION,
            "success": False,
        }
