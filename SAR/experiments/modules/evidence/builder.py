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

from modules.geo.preprocess import ImagePreprocessor
from modules.geo.geolocalize import GeoLocalizer
from .schema_validator import validate_evidence_package

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0.0"
_TASK_TYPE = "intel_brief"

# Class codes that belong to super-class "ship" / "aircraft"
_SHIP_CODES = frozenset({
    "carrier", "destroyer", "frigate", "replenishment",
    "amphibious", "other_vessel",
})
_AIRCRAFT_CODES = frozenset({
    "fighter", "bomber", "transport", "aew", "helicopter", "other_aircraft",
})

# Chinese names for class codes (mirrors class_map.py)
_CODE_TO_CN: dict[str, str] = {
    "carrier": "航空母舰",
    "destroyer": "驱逐舰",
    "frigate": "护卫舰",
    "replenishment": "补给舰",
    "amphibious": "两栖舰",
    "other_vessel": "其他舰船",
    "fighter": "战斗机",
    "bomber": "轰炸机",
    "transport": "运输机",
    "aew": "预警机",
    "helicopter": "直升机",
    "other_aircraft": "其他飞机",
}

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

        # ── Statistics ──────────────────────────────────────────────────────
        statistics = _compute_statistics(geo_objects, localizer)

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

    totals = {
        "all_objects": n,
        "ships": ship_count,
        "aircraft": aircraft_count,
    }

    # ── by_class ─────────────────────────────────────────────────────────────
    class_counter: Counter[str] = Counter()
    for obj in objects:
        code = obj.get("class", {}).get("code", "unknown")
        class_counter[code] += 1

    by_class = [
        {
            "code": code,
            "name_cn": _CODE_TO_CN.get(code, code),
            "count": cnt,
        }
        for code, cnt in class_counter.most_common()
    ]

    # ── by_super_class ───────────────────────────────────────────────────────
    by_super_class = [
        {"code": "ship", "count": ship_count},
        {"code": "aircraft", "count": aircraft_count},
    ]

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

    return {
        "totals": totals,
        "by_class": by_class,
        "by_super_class": by_super_class,
        "spatial_summary": spatial_summary,
        "confidence_summary": confidence_summary,
    }


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
