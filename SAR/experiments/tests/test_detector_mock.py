"""
test_detector_mock.py — Unit tests for MockDetector

Verifies that MockDetector produces schema-compliant, reproducible synthetic
detections without requiring ultralytics or model weights.
"""

from __future__ import annotations

import sys
import os

# Ensure the experiments package root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from modules.detector import MockDetector
from modules.detector.class_map import CLASS_MAP

# Collect all valid class codes from the class map
_ALL_VALID_CODES: set[str] = {desc["code"] for desc in CLASS_MAP.values()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mock_detector_returns_correct_count():
    """n_ships=2, n_aircraft=1 should produce exactly 3 objects."""
    detector = MockDetector(n_ships=2, n_aircraft=1, seed=42)
    objects = detector.detect("test_image.tif")
    assert len(objects) == 3


def test_mock_detector_schema_compliant():
    """Every object must have required Evidence Package schema fields."""
    detector = MockDetector(n_ships=2, n_aircraft=1, seed=42)
    objects = detector.detect("test_image.tif")

    required_fields = [
        "object_id",
        "source_module",
        "status",
        "class",
        "score",
        "geometry",
        "evidence",
        "audit",
    ]

    for obj in objects:
        for field in required_fields:
            assert field in obj, f"Missing top-level field: {field!r}"

        # class sub-fields
        assert "code" in obj["class"], "Missing class.code"
        assert "super_class" in obj["class"], "Missing class.super_class"
        assert "name_cn" in obj["class"], "Missing class.name_cn"
        assert "priority" in obj["class"], "Missing class.priority"

        # score sub-fields
        assert "confidence" in obj["score"], "Missing score.confidence"
        assert "calibrated_confidence" in obj["score"], "Missing score.calibrated_confidence"

        # geometry.pixel sub-fields
        pixel = obj["geometry"]["pixel"]
        assert "center_x" in pixel, "Missing geometry.pixel.center_x"
        assert "center_y" in pixel, "Missing geometry.pixel.center_y"
        assert "width" in pixel, "Missing geometry.pixel.width"
        assert "height" in pixel, "Missing geometry.pixel.height"
        assert "angle_deg" in pixel, "Missing geometry.pixel.angle_deg"
        assert "polygon" in pixel, "Missing geometry.pixel.polygon"
        assert "bbox_axis_aligned" in pixel, "Missing geometry.pixel.bbox_axis_aligned"

        # evidence sub-fields
        assert "crop_uri" in obj["evidence"], "Missing evidence.crop_uri"
        assert "detector_version" in obj["evidence"], "Missing evidence.detector_version"


def test_mock_detector_reproducible():
    """Same image_uri + seed must produce identical results across two calls."""
    detector = MockDetector(n_ships=2, n_aircraft=1, seed=42)
    image_uri = "s3://sar-bucket/scene_001.tif"

    result_a = detector.detect(image_uri)
    result_b = detector.detect(image_uri)

    assert len(result_a) == len(result_b)
    for obj_a, obj_b in zip(result_a, result_b):
        assert obj_a["class"]["code"] == obj_b["class"]["code"]
        assert obj_a["score"]["confidence"] == obj_b["score"]["confidence"]
        assert obj_a["geometry"]["pixel"]["center_x"] == obj_b["geometry"]["pixel"]["center_x"]
        assert obj_a["geometry"]["pixel"]["center_y"] == obj_b["geometry"]["pixel"]["center_y"]


def test_mock_detector_different_image_different_coords():
    """Different image URIs should produce different pixel coordinates."""
    detector = MockDetector(n_ships=1, n_aircraft=0, seed=42)

    result_a = detector.detect("scene_alpha.tif")
    result_b = detector.detect("scene_beta.tif")

    assert len(result_a) == 1 and len(result_b) == 1

    cx_a = result_a[0]["geometry"]["pixel"]["center_x"]
    cx_b = result_b[0]["geometry"]["pixel"]["center_x"]
    cy_a = result_a[0]["geometry"]["pixel"]["center_y"]
    cy_b = result_b[0]["geometry"]["pixel"]["center_y"]

    # At least one coordinate axis must differ
    assert (cx_a != cx_b) or (cy_a != cy_b), (
        "Expected different coordinates for different image URIs, "
        f"but got cx={cx_a}/{cx_b}, cy={cy_a}/{cy_b}"
    )


def test_mock_detector_polygon_has_4_points():
    """Each object's polygon must contain exactly 4 [x, y] pairs."""
    detector = MockDetector(n_ships=2, n_aircraft=1, seed=42)
    objects = detector.detect("test_image.tif")

    for obj in objects:
        polygon = obj["geometry"]["pixel"]["polygon"]
        assert len(polygon) == 4, f"Expected 4 polygon points, got {len(polygon)}"
        for pt in polygon:
            assert len(pt) == 2, f"Expected [x, y] pair, got {pt}"
            assert isinstance(pt[0], (int, float)), "Polygon x must be numeric"
            assert isinstance(pt[1], (int, float)), "Polygon y must be numeric"


def test_mock_detector_zero_objects():
    """n_ships=0, n_aircraft=0 must return an empty list."""
    detector = MockDetector(n_ships=0, n_aircraft=0, seed=42)
    objects = detector.detect("empty_image.tif")
    assert objects == [], f"Expected empty list, got {objects}"


def test_mock_detector_class_codes_valid():
    """All generated class codes must be present in CLASS_MAP."""
    detector = MockDetector(n_ships=5, n_aircraft=3, seed=0)
    objects = detector.detect("class_check.tif")

    for obj in objects:
        code = obj["class"]["code"]
        assert code in _ALL_VALID_CODES, (
            f"Class code {code!r} is not in CLASS_MAP. "
            f"Valid codes: {sorted(_ALL_VALID_CODES)}"
        )
