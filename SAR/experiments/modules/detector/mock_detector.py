"""
mock_detector.py — Synthetic detector for testing

Produces reproducible fake detections without ultralytics or model weights.
Useful for unit tests and CI pipelines where real inference is not needed.
"""

from __future__ import annotations

import math
import random
import uuid

from .class_map import CLASS_MAP, ClassDescriptor

# ---------------------------------------------------------------------------
# Class code sets
# ---------------------------------------------------------------------------

_SHIP_CODES: list[str] = ["destroyer", "frigate", "carrier", "replenishment", "amphibious"]
_AIRCRAFT_CODES: list[str] = ["fighter", "transport", "helicopter"]

# Build lookup: code -> ClassDescriptor
_CODE_TO_DESCRIPTOR: dict[str, ClassDescriptor] = {
    desc["code"]: desc for desc in CLASS_MAP.values()
}


# ---------------------------------------------------------------------------
# Internal helpers (mirrors detector.py style)
# ---------------------------------------------------------------------------

def _generate_object_id() -> str:
    """Return a unique object ID in the format `obj-XXXXXX`."""
    return f"obj-{uuid.uuid4().hex[:6].upper()}"


def _polygon_from_obb(
    cx: float, cy: float, w: float, h: float, angle_deg: float
) -> list[list[float]]:
    """Compute the four corner points of an OBB given its centre, axes, and rotation.

    Returns four [x, y] corner points: TL, TR, BR, BL (relative to unrotated box).
    """
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    hw = w / 2.0
    hh = h / 2.0

    # Local corners (before rotation): TL, TR, BR, BL
    corners_local = [
        (-hw, -hh),
        ( hw, -hh),
        ( hw,  hh),
        (-hw,  hh),
    ]

    corners_world: list[list[float]] = []
    for lx, ly in corners_local:
        rx = cos_a * lx - sin_a * ly + cx
        ry = sin_a * lx + cos_a * ly + cy
        corners_world.append([round(rx, 2), round(ry, 2)])

    return corners_world


def _bbox_axis_aligned(polygon: list[list[float]]) -> list[float]:
    """Compute axis-aligned bounding box [x_min, y_min, x_max, y_max] from a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]


def _calibrate_confidence(raw: float) -> float:
    """Placeholder confidence calibration: multiply by 0.97."""
    return round(float(raw) * 0.97, 6)


# ---------------------------------------------------------------------------
# MockDetector
# ---------------------------------------------------------------------------

class MockDetector:
    """Synthetic detector for testing — no ultralytics/weights required.

    Parameters
    ----------
    n_ships : int
        Number of ship objects to generate (default 2)
    n_aircraft : int
        Number of aircraft objects to generate (default 1)
    seed : int
        Random seed for reproducibility (default 42)
    confidence_range : tuple[float, float]
        Min/max confidence range (default (0.75, 0.95))
    """

    DETECTOR_VERSION: str = "mock-v1"

    def __init__(
        self,
        n_ships: int = 2,
        n_aircraft: int = 1,
        seed: int = 42,
        confidence_range: tuple[float, float] = (0.75, 0.95),
    ) -> None:
        self.n_ships = n_ships
        self.n_aircraft = n_aircraft
        self.seed = seed
        self.confidence_range = confidence_range

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image_uri: str) -> list[dict]:
        """Return a list of synthetic objects[] matching Evidence Package schema.

        Parameters
        ----------
        image_uri : str
            Path or URI of the image. Used to vary coordinates per image via
            hashing, ensuring different images produce different coordinates.

        Returns
        -------
        list[dict]
            Each dict is a fully-populated Evidence Package ``objects[]`` entry.
        """
        rng = random.Random(self.seed + hash(image_uri) % 1000)
        objects: list[dict] = []

        # Generate ships
        for i in range(self.n_ships):
            code = _SHIP_CODES[i % len(_SHIP_CODES)]
            obj = self._make_object(rng, code, "ship")
            objects.append(obj)

        # Generate aircraft
        for i in range(self.n_aircraft):
            code = _AIRCRAFT_CODES[i % len(_AIRCRAFT_CODES)]
            obj = self._make_object(rng, code, "aircraft")
            objects.append(obj)

        return objects

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_object(self, rng: random.Random, code: str, super_class: str) -> dict:
        """Build a single synthetic Evidence Package object entry."""
        desc = _CODE_TO_DESCRIPTOR.get(code)
        if desc is None:
            # Fallback: construct a minimal descriptor
            desc = ClassDescriptor(
                code=code,
                name_cn=code,
                super_class=super_class,
                priority="LOW",
            )

        # Pixel coordinates — scatter around centre (1024, 1024)
        cx = round(1024.0 + rng.uniform(-300, 300), 2)
        cy = round(1024.0 + rng.uniform(-300, 300), 2)

        # OBB dimensions
        if super_class == "ship":
            w = round(rng.gauss(80, 20), 2)
            h = round(rng.gauss(18, 5), 2)
        else:  # aircraft
            w = round(rng.gauss(40, 10), 2)
            h = round(rng.gauss(12, 3), 2)

        # Ensure positive dimensions
        w = max(w, 5.0)
        h = max(h, 2.0)

        angle_deg = round(rng.uniform(-90, 90), 2)
        confidence = round(rng.uniform(self.confidence_range[0], self.confidence_range[1]), 6)

        polygon = _polygon_from_obb(cx, cy, w, h, angle_deg)
        bbox_aa = _bbox_axis_aligned(polygon)

        return {
            "object_id": _generate_object_id(),
            "source_module": "mock_detector_v1",
            "status": "VALID",
            "class": {
                "code": desc["code"],
                "name_cn": desc["name_cn"],
                "super_class": desc["super_class"],
                "priority": desc["priority"],
            },
            "score": {
                "confidence": confidence,
                "calibrated_confidence": _calibrate_confidence(confidence),
            },
            "geometry": {
                "pixel": {
                    "center_x": cx,
                    "center_y": cy,
                    "width": w,
                    "height": h,
                    "angle_deg": angle_deg,
                    "polygon": polygon,
                    "bbox_axis_aligned": bbox_aa,
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
