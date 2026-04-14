"""
test_integration.py — End-to-end stub test for the full Evidence Package pipeline.

This test simulates the M1→M2→M3→M4→M5 pipeline using lightweight mock
functions.  No real detector, geo, or report modules are imported; all
intermediate outputs are constructed in-place so the test suite remains
self-contained.

The test validates that QualityGate can be dropped into the end of any
pipeline that produces a well-formed Evidence Package.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from modules.eval import ConsistencyChecker, HallucinationDetector, QualityGate


# ---------------------------------------------------------------------------
# Mock pipeline helpers
# ---------------------------------------------------------------------------


def _mock_m1_preprocess(image_uri: str, area_name: str) -> dict:
    """M1: build the input + scene sub-blocks."""
    return {
        "schema_version": "1.0.0",
        "package_id": "sar-20260414-TEST01",
        "task_type": "intel_brief",
        "status": "PREPROCESSED",
        "input": {
            "image_uri": image_uri,
            "metadata": {
                "sensor": "GF-3",
                "resolution_m": 1.0,
                "acquisition_time": "2026-04-14T06:30:00Z",
                "area_name": area_name,
            },
        },
        "scene": {
            "scene_type": "harbor",
            "geo_bbox": {
                "min_lon": 120.25, "min_lat": 22.77,
                "max_lon": 120.28, "max_lat": 22.79,
            },
        },
        "objects": [],
        "statistics": {},
        "attachments": {},
        "report": {},
        "quality": {},
        "errors": [],
    }


def _mock_m2_detect(pkg: dict) -> dict:
    """M2: populate objects[] with a single destroyer detection."""
    pkg = copy.deepcopy(pkg)
    pkg["objects"] = [
        {
            "object_id": "obj-000001",
            "status": "VALID",
            "class": {
                "code": "destroyer",
                "name_cn": "驱逐舰",
                "super_class": "ship",
                "priority": "HIGH",
            },
            "score": {"confidence": 0.93, "calibrated_confidence": 0.90},
            "geometry": {
                "pixel": {
                    "center_x": 1032.4, "center_y": 812.7,
                    "width": 86.3, "height": 18.9, "angle_deg": -27.4,
                },
                "geo": {},
            },
            "attributes": {"heading_deg": 332.6, "estimated_length_m": 86.3, "is_near_pier": True},
            "audit": {"review_status": "UNREVIEWED"},
        }
    ]
    pkg["status"] = "DETECTED"
    return pkg


def _mock_m3_geolocate(pkg: dict) -> dict:
    """M3: fill geometry.geo for each object."""
    pkg = copy.deepcopy(pkg)
    for obj in pkg["objects"]:
        obj["geometry"]["geo"] = {"center_lon": 120.265000, "center_lat": 22.782000}
    pkg["status"] = "GEOLOCATED"
    return pkg


def _mock_m4_fuse(pkg: dict) -> dict:
    """M4: compute statistics from objects."""
    pkg = copy.deepcopy(pkg)
    valid_objects = [o for o in pkg["objects"] if o.get("status") == "VALID"]
    by_class: dict[str, int] = {}
    for obj in valid_objects:
        code = obj["class"]["code"]
        by_class[code] = by_class.get(code, 0) + 1
    pkg["statistics"] = {
        "totals": {"all_objects": len(valid_objects)},
        "by_class": by_class,
    }
    pkg["status"] = "READY_FOR_NLG"
    return pkg


def _mock_m5_report(pkg: dict) -> dict:
    """M5: generate a report body grounded in the statistics."""
    pkg = copy.deepcopy(pkg)
    totals = pkg["statistics"]["totals"]["all_objects"]
    by_class = pkg["statistics"]["by_class"]
    class_parts = "、".join(f"{v} 艘{k}" for k, v in by_class.items())
    body = (
        f"侦察区域：某港口。本次侦察共发现目标 {totals} 艘，其中{class_parts}，"
        "停泊于港口西侧码头，坐标约 120.265°E、22.782°N。"
    )
    pkg["report"] = {
        "title": "某港口SAR目标侦察通报",
        "date": "2026-04-14",
        "body": body,
    }
    pkg["status"] = "REPORT_DRAFTED"
    return pkg


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFullPipelineWithMockModules:
    """End-to-end pipeline tests using mock M1–M5 functions."""

    def test_full_pipeline_with_mock_modules(self) -> None:
        """Happy path: pipeline produces a consistent, hallucination-free package."""
        # Run mock pipeline
        pkg = _mock_m1_preprocess("s3://sar-bucket/test.tif", "某港口")
        pkg = _mock_m2_detect(pkg)
        pkg = _mock_m3_geolocate(pkg)
        pkg = _mock_m4_fuse(pkg)
        pkg = _mock_m5_report(pkg)

        # Evaluate
        gate = QualityGate()
        pkg = gate.evaluate(pkg)

        quality = pkg["quality"]

        # All consistency checks pass
        cc = quality["consistency_checks"]
        assert cc["count_consistent"] is True
        assert cc["class_consistent"] is True
        assert cc["coordinate_present"] is True
        assert cc["required_fields_present"] is True

        # No hallucinations
        nc = quality["nlg_checks"]
        assert nc["number_hallucination"] is False
        assert nc["class_hallucination"] is False
        assert nc["format_compliant"] is True

        # No human review needed
        rg = quality["review_gate"]
        assert rg["needs_human_review"] is False
        assert rg["reason"] == []

    def test_hallucinated_report_triggers_review(
        self, minimal_evidence_package: dict
    ) -> None:
        """A report body with wrong counts should trigger human review."""
        pkg = copy.deepcopy(minimal_evidence_package)
        pkg["report"]["body"] = (
            "侦察区域：某港口。本次侦察共发现目标 5 艘，其中航母 2 艘、驱逐舰 3 艘。"
        )

        gate = QualityGate()
        pkg = gate.evaluate(pkg)

        assert pkg["quality"]["review_gate"]["needs_human_review"] is True
        reasons = pkg["quality"]["review_gate"]["reason"]
        assert len(reasons) > 0

    def test_missing_geo_triggers_review(
        self, evidence_missing_geo: dict
    ) -> None:
        """Missing geo on VALID objects should trigger human review."""
        gate = QualityGate()
        pkg = gate.evaluate(evidence_missing_geo)

        assert pkg["quality"]["review_gate"]["needs_human_review"] is True
        assert any("geo" in r.lower() or "coordinate" in r.lower()
                   for r in pkg["quality"]["review_gate"]["reason"])

    def test_low_confidence_triggers_review(
        self, minimal_evidence_package: dict
    ) -> None:
        """Objects with calibrated_confidence below threshold trigger review."""
        pkg = copy.deepcopy(minimal_evidence_package)
        pkg["objects"][0]["score"]["calibrated_confidence"] = 0.45

        gate = QualityGate()
        pkg = gate.evaluate(pkg)

        assert pkg["quality"]["review_gate"]["needs_human_review"] is True
        assert any("Low-confidence" in r for r in pkg["quality"]["review_gate"]["reason"])

    def test_quality_gate_does_not_mutate_original_reference(
        self, minimal_evidence_package: dict
    ) -> None:
        """evaluate() should return the updated package (mutates in-place — verify quality key)."""
        pkg = copy.deepcopy(minimal_evidence_package)
        gate = QualityGate()
        returned = gate.evaluate(pkg)

        # evaluate mutates and returns the same object
        assert returned is pkg
        assert "consistency_checks" in returned["quality"]

    def test_all_eval_components_importable(self) -> None:
        """Smoke test: all three eval classes can be instantiated."""
        checker = ConsistencyChecker()
        detector = HallucinationDetector()
        gate = QualityGate()
        assert checker is not None
        assert detector is not None
        assert gate is not None
