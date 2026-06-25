"""
builder.py — M4 Evidence Fusion Builder

Assembles a complete Evidence Package (schema v1.0) from:
  - image_uri   : path / URI of the source GeoTIFF
  - mission     : dict with region_name, region_type, priority, user_prompt
  - objects     : list of detector output dicts (geometry.pixel already filled)
  - scene       : optional scene-level dict (pre-populated by caller or None)

Statistics are computed PURELY from the objects[] list.  No LLM inference.
"""

from __future__ import annotations

import logging
import math
import random
import string
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

from modules.class_labels import get_class_name_cn, infer_super_class
from modules.geo.preprocess import ImagePreprocessor
from modules.geo.geolocalize import GeoLocalizer
from .schema_validator import validate_evidence_package

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0.0"
_TASK_TYPE = "intel_brief"

# Class codes that belong to super-class "ship" / "aircraft" (v3 23-class system)
_SHIP_CODES = frozenset({
    "military_auxiliary", "combat_ship", "liquid_cargo", "harbor_service",
    "bulk_carrier", "other_vessel", "survey_vessel", "amphibious",
    "container_ship", "engineering_vessel", "fishing_vessel", "tug_boat",
    "passenger_ship", "ro_ro_ship", "sailing_vessel", "research_vessel",
    "ship",
    # Legacy
    "carrier", "destroyer", "frigate", "replenishment",
})
_AIRCRAFT_CODES = frozenset({
    "combat_aircraft", "transport_aircraft", "combat_support_aircraft",
    "helicopter", "other_aircraft",
    "aircraft",
    # Legacy
    "fighter", "bomber", "transport", "aew",
})

_LOW_CONFIDENCE_THRESHOLD = 0.5
_REVIEW_CONFIDENCE_THRESHOLD = 0.4


class EvidenceBuilder:
    """M4 — Assemble and validate the full Evidence Package.

    Usage
    -----
    >>> builder = EvidenceBuilder()
    >>> package = builder.build(image_uri, mission, objects, scene=None)
    """

    def build(
        self,
        image_uri: str,
        mission: dict[str, Any],
        objects: list[dict[str, Any]],
        scene: Optional[dict[str, Any]] = None,
        segmentation: Optional[dict[str, Any]] = None,
        gate_result: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Assemble a complete Evidence Package.

        Parameters
        ----------
        image_uri:
            URI or path to the source GeoTIFF.
        mission:
            Dict containing ``region_name``, ``region_type``, ``priority``
            and optionally ``user_prompt``.
        objects:
            Detector output list.  Each element must have
            ``geometry.pixel`` filled.
        scene:
            Optional pre-populated scene block.  When None a minimal scene
            block is derived from the GeoLocalizer transform.
        segmentation:
            Optional M3 FastSAM output dict with keys:
            ``mask_polygon``, ``mask_area_km2``, ``scene_type``, ``success``.
        gate_result:
            Optional Gate classifier output dict with keys:
            ``scene_class``, ``confidence``, ``gate_decision``.

        Returns
        -------
        Full Evidence Package dict with ``status = "READY_FOR_NLG"``.

        Raises
        ------
        ValueError
            If ``statistics.totals.all_objects`` does not equal ``len(objects)``.
        """
        now_iso = _now_iso8601()

        # ── M1 Preprocessing ────────────────────────────────────────────────
        preprocessor = ImagePreprocessor()
        input_block = preprocessor.parse(image_uri, mission)

        # ── M3 Geolocalisation ──────────────────────────────────────────────
        localizer = GeoLocalizer(image_uri)
        geo_objects = localizer.localize_objects(objects)

        # ── Scene block ─────────────────────────────────────────────────────
        if scene is None:
            scene = _build_minimal_scene(localizer, mission)

        # ── Merge gate result into scene ────────────────────────────────────
        if gate_result is not None:
            scene["scene_type"] = gate_result.get("scene_class", "unknown")
            scene["scene_confidence"] = gate_result.get("confidence")
            scene["scene_source"] = gate_result.get("model_version", "unknown")

        # ── Spatial containment (ship in harbor? aircraft in airport?) ───────
        if segmentation is not None and segmentation.get("success"):
            geo_objects = _apply_spatial_containment(geo_objects, segmentation)
            # Attach segmentation info to scene
            scene.setdefault("segmentation", {})
            scene["segmentation"]["mask_area_km2"] = segmentation.get("mask_area_km2")
            scene["segmentation"]["segmentation_uri"] = segmentation.get("segmentation_uri")
            scene["segmentation"]["segmentation_version"] = segmentation.get("model_version")
            # Set environment flags
            scene.setdefault("environment", {})
            stype = segmentation.get("scene_type", "")
            scene["environment"]["is_harbor"] = (stype == "harbor")
            scene["environment"]["is_airport"] = (stype == "airport")

        # ── Statistics ──────────────────────────────────────────────────────
        statistics = _compute_statistics(geo_objects, localizer, segmentation)

        # ── Validate totals ─────────────────────────────────────────────────
        total_in_stats = statistics["totals"]["all_objects"]
        if total_in_stats != len(geo_objects):
            raise ValueError(
                f"statistics.totals.all_objects ({total_in_stats}) does not match "
                f"len(objects) ({len(geo_objects)}). This is a programming error."
            )

        # ── Assemble package ────────────────────────────────────────────────
        package: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "package_id": _generate_package_id(now_iso),
            "task_type": _TASK_TYPE,
            "status": "READY_FOR_NLG",
            "created_at": now_iso,
            "updated_at": now_iso,
            "trace": {
                "request_id": f"req-{_random_hex(4)}",
                "pipeline_run_id": f"pipe-{_random_hex(4)}",
                "operator": "system",
                "source_system": "sar-intel-pipeline",
            },
            "input": input_block,
            "scene": scene,
            "objects": geo_objects,
            "statistics": statistics,
            "attachments": {},
            "report": {},
            "quality": {},
            "errors": [],
        }

        # ── Schema validation ────────────────────────────────────────────────
        validate_evidence_package(package)

        logger.info(
            "Evidence Package built: %s | %d objects | status=%s",
            package["package_id"],
            len(geo_objects),
            package["status"],
        )

        return package


# ---------------------------------------------------------------------------
# Statistics computation (no LLM, no hardcoding — all from objects[])
# ---------------------------------------------------------------------------

def _compute_statistics(
    objects: list[dict[str, Any]],
    localizer: Optional[GeoLocalizer] = None,
    segmentation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Compute the full statistics block from *objects* programmatically."""
    n = len(objects)

    # ── Totals ───────────────────────────────────────────────────────────────
    ship_count = sum(
        1 for o in objects
        if o.get("class", {}).get("super_class") == "ship"
        or o.get("class", {}).get("code") in _SHIP_CODES
    )
    aircraft_count = sum(
        1 for o in objects
        if o.get("class", {}).get("super_class") == "aircraft"
        or o.get("class", {}).get("code") in _AIRCRAFT_CODES
    )

    totals: dict[str, Any] = {
        "all_objects": n,
        "ships": ship_count,
        "aircraft": aircraft_count,
    }

    # ── Containment counts ───────────────────────────────────────────────────
    in_harbor = sum(1 for o in objects if o.get("attributes", {}).get("in_harbor"))
    in_airport = sum(1 for o in objects if o.get("attributes", {}).get("in_airport"))
    if in_harbor > 0:
        totals["in_harbor"] = in_harbor
    if in_airport > 0:
        totals["in_airport"] = in_airport

    # ── by_class ─────────────────────────────────────────────────────────────
    class_counter: Counter[str] = Counter()
    for obj in objects:
        code = obj.get("class", {}).get("code", "unknown")
        class_counter[code] += 1

    by_class = [
        {
            "code": code,
            "name_cn": get_class_name_cn(code),
            "super_class": infer_super_class(code),
            "count": cnt,
        }
        for code, cnt in class_counter.most_common()
    ]

    # ── by_super_class ───────────────────────────────────────────────────────
    by_super_class: list[dict[str, Any]] = [
        {"code": "ship", "count": ship_count},
        {"code": "aircraft", "count": aircraft_count},
    ]
    # Add area class when segmentation is available
    if segmentation is not None and segmentation.get("success"):
        area_type = segmentation.get("scene_type", "")
        if area_type:
            by_super_class.append({"code": area_type, "count": 1,
                                   "area_km2": segmentation.get("mask_area_km2")})

    # ── Confidence summary ───────────────────────────────────────────────────
    confidences = [
        obj.get("score", {}).get("confidence")
        for obj in objects
        if obj.get("score", {}).get("confidence") is not None
    ]
    if confidences:
        mean_conf = round(sum(confidences) / len(confidences), 4)
        low_conf_count = sum(1 for c in confidences if c < _LOW_CONFIDENCE_THRESHOLD)
        review_required = sum(1 for c in confidences if c < _REVIEW_CONFIDENCE_THRESHOLD)
    else:
        mean_conf = None
        low_conf_count = 0
        review_required = 0

    confidence_summary = {
        "mean_confidence": mean_conf,
        "low_confidence_count": low_conf_count,
        "review_required_count": review_required,
    }

    # ── Spatial summary ──────────────────────────────────────────────────────
    spatial_summary = _compute_spatial_summary(objects, localizer)

    # ── Area (from segmentation) ─────────────────────────────────────────────
    result: dict[str, Any] = {
        "totals": totals,
        "by_class": by_class,
        "by_super_class": by_super_class,
        "spatial_summary": spatial_summary,
        "confidence_summary": confidence_summary,
    }
    if segmentation is not None and segmentation.get("success"):
        result["area"] = {
            "harbor_area_km2": (
                segmentation["mask_area_km2"]
                if segmentation.get("scene_type") == "harbor" else None
            ),
            "airport_area_km2": (
                segmentation["mask_area_km2"]
                if segmentation.get("scene_type") == "airport" else None
            ),
        }

    return result


def _compute_spatial_summary(
    objects: list[dict[str, Any]],
    localizer: Optional[GeoLocalizer],
) -> dict[str, Any]:
    """Compute nearest-neighbour distance and simple cluster count."""
    if len(objects) < 2:
        return {
            "distribution": "单目标" if len(objects) == 1 else "无目标",
            "cluster_count": min(len(objects), 1),
            "nearest_neighbor_mean_m": 0.0,
        }

    # Collect geo centres
    coords: list[tuple[float, float]] = []
    for obj in objects:
        geo = obj.get("geometry", {}).get("geo", {})
        if geo.get("center_lon") is not None and geo.get("center_lat") is not None:
            coords.append((geo["center_lon"], geo["center_lat"]))

    if len(coords) < 2:
        return {
            "distribution": "坐标不完整，无法计算分布",
            "cluster_count": 1,
            "nearest_neighbor_mean_m": 0.0,
        }

    # Nearest-neighbour mean (brute-force; n is small for SAR scenes)
    nn_distances: list[float] = []
    for i, (lon1, lat1) in enumerate(coords):
        best = math.inf
        for j, (lon2, lat2) in enumerate(coords):
            if i == j:
                continue
            d = _haversine_m(lon1, lat1, lon2, lat2)
            if d < best:
                best = d
        if best < math.inf:
            nn_distances.append(best)

    nn_mean = round(sum(nn_distances) / len(nn_distances), 1) if nn_distances else 0.0

    # Simple cluster count: targets within 500 m of each other belong to same cluster
    cluster_count = _simple_cluster_count(coords, threshold_m=500.0)

    # Qualitative distribution description
    if cluster_count == 1:
        distribution = "目标集中分布于同一区域"
    elif cluster_count <= 3:
        distribution = f"目标分为{cluster_count}个簇群分布"
    else:
        distribution = "目标分散分布于多个区域"

    return {
        "distribution": distribution,
        "cluster_count": cluster_count,
        "nearest_neighbor_mean_m": nn_mean,
    }


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _simple_cluster_count(
    coords: list[tuple[float, float]], threshold_m: float = 500.0
) -> int:
    """Union-Find cluster counting by distance threshold."""
    n = len(coords)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            d = _haversine_m(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if d <= threshold_m:
                union(i, j)

    return len({find(i) for i in range(n)})


# ---------------------------------------------------------------------------
# Spatial containment
# ---------------------------------------------------------------------------

def _apply_spatial_containment(
    objects: list[dict[str, Any]],
    segmentation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Tag objects whose centre falls inside the harbor/airport mask.

    Uses OpenCV pointPolygonTest for efficient point-in-polygon testing.
    """
    polygon = segmentation.get("mask_polygon", [])
    scene_type = segmentation.get("scene_type", "")
    if not polygon or len(polygon) < 3:
        return objects

    try:
        import cv2
        import numpy as np
        contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    except ImportError:
        return objects

    for obj in objects:
        pixel = obj.get("geometry", {}).get("pixel", {})
        cx = pixel.get("center_x")
        cy = pixel.get("center_y")
        if cx is None or cy is None:
            continue

        result = cv2.pointPolygonTest(contour, (float(cx), float(cy)), False)
        if result >= 0:  # inside or on edge
            attrs = obj.setdefault("attributes", {})
            if scene_type == "harbor":
                attrs["in_harbor"] = True
            elif scene_type == "airport":
                attrs["in_airport"] = True

    return objects


# ---------------------------------------------------------------------------
# Scene block helper
# ---------------------------------------------------------------------------

def _build_minimal_scene(
    localizer: GeoLocalizer,
    mission: dict[str, Any],
) -> dict[str, Any]:
    """Build a minimal scene block from the GeoLocalizer transform."""
    scene_type = mission.get("region_type", "unknown")
    scene_type_cn_map = {
        "harbor": "港口",
        "airport": "机场",
        "anchorage": "锚地",
        "shipyard": "船坞",
        "airbase": "空军基地",
        "coastal_area": "沿岸区域",
        "unknown": "未知",
    }
    scene: dict[str, Any] = {
        "scene_id": f"scene-{_random_hex(4)}",
        "scene_type": scene_type,
        "scene_type_cn": scene_type_cn_map.get(scene_type, "未知"),
        "scene_confidence": None,
        "geo_bounds": None,
        "image_center": None,
        "scale": {
            "gsd_m": localizer.gsd_x_m,
            "pixel_area_m2": (
                localizer.gsd_x_m * localizer.gsd_y_m
                if localizer.gsd_x_m and localizer.gsd_y_m
                else None
            ),
        },
        "environment": {
            "is_near_coast": scene_type in ("harbor", "anchorage", "coastal_area"),
            "is_airport": scene_type in ("airport", "airbase"),
            "is_harbor": scene_type == "harbor",
        },
    }
    return scene


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now_iso8601() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_package_id(iso_str: str) -> str:
    """Generate a package_id like ``sar-YYYYMMDD-XXXXXX``."""
    try:
        dt = datetime.fromisoformat(iso_str)
        date_part = dt.strftime("%Y%m%d")
    except ValueError:
        date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = _random_hex(3).upper()  # 6 hex chars
    return f"sar-{date_part}-{suffix}"


def _random_hex(n_bytes: int) -> str:
    """Return a random hex string of length 2*n_bytes."""
    import os
    return os.urandom(n_bytes).hex()
