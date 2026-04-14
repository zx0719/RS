"""
conftest.py — Shared pytest fixtures for the eval-qa test suite.

All fixtures return plain Python dicts that conform to Evidence Package v1.0
schema.  No real modules are imported; all data is hard-coded so tests run
without detector / geo / report dependencies.
"""

from __future__ import annotations

import copy

import pytest


# ---------------------------------------------------------------------------
# Base template (reused across fixtures)
# ---------------------------------------------------------------------------

_BASE_PACKAGE: dict = {
    "schema_version": "1.0.0",
    "package_id": "sar-20260414-000001",
    "task_type": "intel_brief",
    "status": "REPORT_DRAFTED",
    "input": {
        "image_uri": "s3://sar-bucket/scenes/20260414_harbor.tif",
        "metadata": {
            "sensor": "GF-3",
            "resolution_m": 1.0,
            "acquisition_time": "2026-04-14T06:30:00Z",
            "area_name": "某港口",
        },
    },
    "scene": {
        "scene_type": "harbor",
        "geo_bbox": {
            "min_lon": 120.25,
            "min_lat": 22.77,
            "max_lon": 120.28,
            "max_lat": 22.79,
        },
    },
    "objects": [
        {
            "object_id": "obj-000001",
            "status": "VALID",
            "class": {
                "code": "destroyer",
                "name_cn": "驱逐舰",
                "super_class": "ship",
                "priority": "HIGH",
            },
            "score": {
                "confidence": 0.92,
                "calibrated_confidence": 0.89,
            },
            "geometry": {
                "pixel": {
                    "center_x": 1032.4,
                    "center_y": 812.7,
                    "width": 86.3,
                    "height": 18.9,
                    "angle_deg": -27.4,
                },
                "geo": {
                    "center_lon": 120.265000,
                    "center_lat": 22.782000,
                },
            },
            "attributes": {
                "heading_deg": 332.6,
                "estimated_length_m": 86.3,
                "is_near_pier": True,
            },
            "audit": {"review_status": "UNREVIEWED"},
        }
    ],
    "statistics": {
        "totals": {"all_objects": 1},
        "by_class": {"destroyer": 1},
    },
    "attachments": {},
    "report": {
        "title": "某港口SAR目标侦察通报",
        "date": "2026-04-14",
        "body": (
            "侦察区域：某港口。本次侦察共发现目标 1 艘，其中驱逐舰 1 艘，"
            "停泊于港口西侧码头，坐标约 120.265°E、22.782°N。"
        ),
    },
    "quality": {},
    "errors": [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_evidence_package() -> dict:
    """Return a minimal, fully consistent Evidence Package (1 destroyer, harbor)."""
    return copy.deepcopy(_BASE_PACKAGE)


@pytest.fixture()
def evidence_with_hallucination() -> dict:
    """Return a package whose report.body mentions a wrong count (hallucination).

    The statistics say 1 destroyer, but the body claims 3 ships and mentions
    a carrier class that does not appear in statistics.
    """
    pkg = copy.deepcopy(_BASE_PACKAGE)
    pkg["report"]["body"] = (
        "侦察区域：某港口。本次侦察共发现目标 3 艘，其中航母 1 艘、驱逐舰 2 艘，"
        "停泊于港口西侧码头。"
    )
    return pkg


@pytest.fixture()
def evidence_missing_geo() -> dict:
    """Return a package where a VALID object is missing geo coordinates."""
    pkg = copy.deepcopy(_BASE_PACKAGE)
    # Remove geo sub-block from the only object
    pkg["objects"][0]["geometry"]["geo"] = {}
    return pkg


@pytest.fixture()
def evidence_count_mismatch() -> dict:
    """Return a package where statistics.totals.all_objects does not match objects."""
    pkg = copy.deepcopy(_BASE_PACKAGE)
    # Report says 2 but only 1 VALID object exists
    pkg["statistics"]["totals"]["all_objects"] = 2
    return pkg


@pytest.fixture()
def evidence_class_count_mismatch() -> dict:
    """Return a package where by_class count disagrees with actual objects."""
    pkg = copy.deepcopy(_BASE_PACKAGE)
    # Statistics claim 2 destroyers but only 1 exists
    pkg["statistics"]["by_class"]["destroyer"] = 2
    return pkg
