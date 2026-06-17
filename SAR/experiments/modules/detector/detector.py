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

from .class_map import CLASS_MAP, get_class_descriptor, ClassDescriptor

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
    model_path    : Path to the YOLOv8-OBB weights file (`.pt`).
    score_thresh  : Minimum confidence threshold for detections (0–1).
    nms_thresh    : IoU threshold used during Non-Maximum Suppression.
    device        : PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    tile_size     : Tile side length in pixels for large-image tiling (default 640).
    tile_overlap  : Overlap between adjacent tiles in pixels (default 128, ~20%).
    tile_threshold: Images larger than this (max side) trigger tiled inference (default 1280).
    nms_iou       : IoU threshold for cross-tile NMS deduplication (default 0.5).
    """

    DETECTOR_VERSION: str = "yolov8-obb-shipair-v1.3"

    def __init__(
        self,
        model_path: str | Path,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.5,
        device: str = "cuda:0",
        class_map: dict[int, str] | None = None,
        tile_size: int = 640,
        tile_overlap: int = 128,
        tile_threshold: int = 1280,
        nms_iou: float = 0.5,
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
        self._class_map = class_map
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_threshold = tile_threshold
        self.nms_iou = nms_iou

        self._model: Any = YOLO(str(self.model_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image_uri: str | Path,
        save_vis: bool = False,
        vis_dir: str | Path | None = None,
    ) -> tuple[list[dict], str | None]:
        """Run inference on a single image and return Evidence Package objects.

        For images whose longest side exceeds ``tile_threshold``, tiled
        inference is used automatically: the image is split into overlapping
        tiles, each tile is inferred independently, coordinates are mapped
        back to the original image space, and cross-tile duplicates are
        removed with NMS.

        Parameters
        ----------
        image_uri : Path to the image file (GeoTIFF, JPEG, PNG, BMP).
        save_vis  : If True, save an annotated image with OBB boxes drawn.
        vis_dir   : Directory for the annotated image (defaults to image dir).

        Returns
        -------
        (objects, vis_path) — objects list and path to annotated image (or None).
        """
        from PIL import Image as _PILImage

        image_path = str(image_uri)

        try:
            with _PILImage.open(image_path) as _im:
                img_w, img_h = _im.size
        except Exception:
            img_w, img_h = 0, 0

        if img_w > 0 and max(img_w, img_h) > self.tile_threshold:
            print(
                f"[tiling] 图像 {img_w}×{img_h} > {self.tile_threshold}，"
                f"启用切片推理 (tile={self.tile_size}, overlap={self.tile_overlap})"
            )
            objects = self._detect_tiled(image_path, img_w, img_h)
        else:
            objects = self._detect_single(image_path)

        vis_path: str | None = None
        if save_vis:
            from .visualize import save_annotated
            _vis_dir = vis_dir if vis_dir is not None else Path(image_uri).parent
            out = save_annotated(image_uri, objects, output_dir=_vis_dir)
            vis_path = str(out)
            print(f"标注图保存在: {out}")

        return objects, vis_path

    # ------------------------------------------------------------------
    # Internal inference methods
    # ------------------------------------------------------------------

    def _detect_single(self, image_path: str) -> list[dict]:
        """Run inference on a single image (no tiling). Returns objects[]."""
        results = self._model.predict(
            source=image_path,
            conf=self.score_thresh,
            iou=self.nms_thresh,
            device=self.device,
            verbose=False,
        )

        objects: list[dict] = []

        for result in results:
            obb = result.obb
            if obb is None or len(obb) == 0:
                continue

            xywhr = obb.xywhr.cpu().numpy()
            confs = obb.conf.cpu().numpy()
            class_ids = obb.cls.cpu().numpy().astype(int)

            for i in range(len(xywhr)):
                cx, cy, w, h, angle_rad = xywhr[i]
                angle_deg = float(np.rad2deg(angle_rad))
                confidence = float(confs[i])
                cls_id = int(class_ids[i])

                polygon = _polygon_from_obb(
                    float(cx), float(cy), float(w), float(h), angle_deg
                )
                bbox_aa = _bbox_axis_aligned(polygon)

                if self._class_map is not None:
                    from .class_map import UNKNOWN_CLASS
                    entry = self._class_map.get(cls_id, UNKNOWN_CLASS)
                    if isinstance(entry, dict):
                        class_desc = entry
                    else:
                        class_desc = next(
                            (v for v in CLASS_MAP.values() if v["code"] == entry),
                            UNKNOWN_CLASS,
                        )
                else:
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

    def _detect_tiled(self, image_path: str, img_w: int, img_h: int) -> list[dict]:
        """Tiled inference for large images. Splits into overlapping tiles,
        runs _detect_single on each, maps coordinates back, then NMS."""
        import tempfile
        from PIL import Image as _PILImage
        from .tiling import tile_grid, read_image_as_array, shift_object_to_global, nms_objects

        img_arr = read_image_as_array(image_path)
        windows = tile_grid(img_w, img_h, self.tile_size, self.tile_overlap)
        print(f"[tiling] 共 {len(windows)} 个 tile")

        raw_objects: list[dict] = []

        with tempfile.TemporaryDirectory() as tmp:
            for idx, (x, y, tw, th) in enumerate(windows):
                tile_arr = img_arr[y:y + th, x:x + tw]
                tile_path = str(Path(tmp) / f"tile_{idx:04d}.jpg")
                _PILImage.fromarray(tile_arr).save(tile_path, quality=92)

                tile_objs = self._detect_single(tile_path)
                for obj in tile_objs:
                    raw_objects.append(
                        shift_object_to_global(obj, x, y, img_w, img_h)
                    )

        deduped = nms_objects(raw_objects, iou_thresh=self.nms_iou)
        print(
            f"[tiling] 原始检测 {len(raw_objects)} 个，NMS后保留 {len(deduped)} 个"
        )
        return deduped

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def detect_batch(self, image_uris: list[str | Path]) -> dict[str, list[dict]]:
        """Run detection on multiple images.

        Returns
        -------
        Mapping of ``image_uri`` (str) → list of objects.
        """
        return {str(uri): self.detect(uri)[0] for uri in image_uris}

    def __repr__(self) -> str:
        return (
            f"DetectorTool(model={self.model_path.name!r}, "
            f"score_thresh={self.score_thresh}, nms_thresh={self.nms_thresh}, "
            f"device={self.device!r})"
        )
