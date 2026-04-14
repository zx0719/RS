"""
detector.py — M2 Detection module (YOLOv8-OBB)

Wraps Ultralytics YOLOv8-OBB inference and produces Evidence Package
`objects[]` entries conforming to schema v1.0.

Usage
-----
    from modules.detector import DetectorTool

    tool = DetectorTool(model_path="runs/detect/weights/best.pt")
    objects = tool.detect("scene.tif")
"""

from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .class_map import get_class_descriptor

# ---------------------------------------------------------------------------
# Conditional ultralytics import — module is still usable without it (e.g.
# during unit tests that mock the model), but will raise at runtime.
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    warnings.warn(
        "ultralytics is not installed. "
        "DetectorTool cannot run inference until ultralytics is installed: "
        "pip install ultralytics",
        ImportWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_object_id() -> str:
    """Return a unique object ID in the format `obj-XXXXXX`."""
    return f"obj-{uuid.uuid4().hex[:6].upper()}"


def _polygon_from_obb(cx: float, cy: float, w: float, h: float, angle_deg: float) -> list[list[float]]:
    """Compute the four corner points of an OBB given its centre, axes, and rotation.

    Parameters
    ----------
    cx, cy:     Centre in pixel coordinates.
    w, h:       Width and height of the box (long axis first by convention).
    angle_deg:  Rotation angle in degrees (positive = counter-clockwise).

    Returns
    -------
    Four [x, y] corner points in order: TL, TR, BR, BL (relative to unrotated box).
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    hw = w / 2.0
    hh = h / 2.0

    # Local corners (before rotation): TL, TR, BR, BL
    corners_local = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ], dtype=float)

    # Rotation matrix
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
    corners_world = (rot @ corners_local.T).T + np.array([cx, cy])

    return corners_world.tolist()


def _bbox_axis_aligned(polygon: list[list[float]]) -> list[float]:
    """Compute axis-aligned bounding box [x_min, y_min, x_max, y_max] from a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _calibrate_confidence(raw: float) -> float:
    """Placeholder confidence calibration: multiply by 0.97."""
    return round(float(raw) * 0.97, 6)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DetectorTool:
    """YOLOv8-OBB detector that returns Evidence Package `objects[]` entries.

    Parameters
    ----------
    model_path:   Path to the YOLOv8-OBB weights file (`.pt`).
    score_thresh: Minimum confidence threshold for detections (0–1).
    nms_thresh:   IoU threshold used during Non-Maximum Suppression.
    device:       PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    """

    DETECTOR_VERSION: str = "yolov8-obb-shipair-v1.3"

    def __init__(
        self,
        model_path: str | Path,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.5,
        device: str = "cuda:0",
    ) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. "
                "Please run: pip install ultralytics"
            )

        self.model_path = Path(model_path)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.device = device

        self._model: Any = YOLO(str(self.model_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image_uri: str | Path) -> list[dict]:
        """Run inference on a single image and return Evidence Package objects.

        Parameters
        ----------
        image_uri:
            Path (or URI) to the image file. Supports any format readable by
            OpenCV / PIL, including GeoTIFF (pixel data only; geo-referencing
            is handled downstream by the geo-pipeline-engineer).

        Returns
        -------
        List of ``objects[]`` dicts conforming to Evidence Package schema v1.0.
        Each dict is ready to be inserted into ``evidence_package["objects"]``.
        """
        image_path = str(image_uri)

        results = self._model.predict(
            source=image_path,
            conf=self.score_thresh,
            iou=self.nms_thresh,
            device=self.device,
            verbose=False,
        )

        objects: list[dict] = []

        for result in results:
            obb = result.obb  # OBBBoxes instance (may be None if no detections)
            if obb is None or len(obb) == 0:
                continue

            # Extract tensors as numpy arrays
            # xywhr: [cx, cy, w, h, angle_rad] — Ultralytics stores angle in radians
            xywhr = obb.xywhr.cpu().numpy()           # (N, 5)
            confs = obb.conf.cpu().numpy()             # (N,)
            class_ids = obb.cls.cpu().numpy().astype(int)  # (N,)

            for i in range(len(xywhr)):
                cx, cy, w, h, angle_rad = xywhr[i]
                angle_deg = float(np.rad2deg(angle_rad))
                confidence = float(confs[i])
                cls_id = int(class_ids[i])

                polygon = _polygon_from_obb(
                    float(cx), float(cy), float(w), float(h), angle_deg
                )
                bbox_aa = _bbox_axis_aligned(polygon)
                class_desc = get_class_descriptor(cls_id)

                obj: dict = {
                    "object_id": _generate_object_id(),
                    "source_module": "detector_yolov8_obb_v1",
                    "status": "VALID",
                    "class": {
                        "code": class_desc["code"],
                        "name_cn": class_desc["name_cn"],
                        "super_class": class_desc["super_class"],
                        "priority": class_desc["priority"],
                    },
                    "score": {
                        "confidence": round(confidence, 6),
                        "calibrated_confidence": _calibrate_confidence(confidence),
                    },
                    "geometry": {
                        "pixel": {
                            "center_x": round(float(cx), 2),
                            "center_y": round(float(cy), 2),
                            "width": round(float(w), 2),
                            "height": round(float(h), 2),
                            "angle_deg": round(angle_deg, 2),
                            "polygon": [[round(v, 2) for v in pt] for pt in polygon],
                            "bbox_axis_aligned": [round(v, 2) for v in bbox_aa],
                        }
                        # geometry.geo is intentionally absent — filled by geo-pipeline-engineer
                    },
                    "evidence": {
                        "crop_uri": None,
                        "detector_version": self.DETECTOR_VERSION,
                    },
                    "audit": {
                        "review_status": "UNREVIEWED",
                        "review_comment": "",
                        "reviewer": "",
                    },
                }

                objects.append(obj)

        return objects

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def detect_batch(self, image_uris: list[str | Path]) -> dict[str, list[dict]]:
        """Run detection on multiple images.

        Returns
        -------
        Mapping of ``image_uri`` (str) → list of objects.
        """
        return {str(uri): self.detect(uri) for uri in image_uris}

    def __repr__(self) -> str:
        return (
            f"DetectorTool(model={self.model_path.name!r}, "
            f"score_thresh={self.score_thresh}, nms_thresh={self.nms_thresh}, "
            f"device={self.device!r})"
        )
