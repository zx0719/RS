"""
large_scene.py — Large-scene report input compression and routing helpers.

Stage 1 goals:
  - build compact Evidence Digest for large-scene generation
  - decide which generation tier should handle a case
  - expose lightweight token/cost profiling helpers for tests and scripts
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from modules.class_labels import (
    get_class_name_cn,
    get_super_class_name_cn,
    normalize_by_class_list,
)
from .collab_config import RouteThresholds, route_thresholds_from_env


ROUTE_TEMPLATE = "template"
ROUTE_SMALL = "small_llm"
ROUTE_LARGE = "large_llm"
ROUTE_LARGE_REFINE = "large_refine"


@dataclass(frozen=True)
class PromptProfile:
    prompt_chars: int
    prompt_lines: int
    digest_bytes: int
    route: str


def is_large_scene(evidence: dict[str, Any]) -> bool:
    trace = evidence.get("trace", {})
    if "large_scene_test" in trace:
        return True
    attachments = evidence.get("attachments", {})
    return any(
        item.get("role") in {"large_scene_overview", "large_scene_overview_as_target_distribution"}
        for item in attachments.values()
        if isinstance(item, dict)
    )


def build_evidence_digest(evidence: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, deterministic digest for large-scene report generation."""
    inp = evidence.get("input", {})
    metadata = inp.get("metadata", {})
    mission = inp.get("mission", {})
    scene = evidence.get("scene", {})
    statistics = evidence.get("statistics", {})
    confidence_summary = statistics.get("confidence_summary", {})
    spatial_summary = statistics.get("spatial_summary", {})
    by_class = normalize_by_class_list(statistics.get("by_class", []))
    trace = evidence.get("trace", {})
    large_scene_trace = trace.get("large_scene_test", {})
    objects = evidence.get("objects", [])

    representative_targets = [
        _representative_target(obj)
        for obj in objects[: min(len(objects), 6)]
    ]
    class_digest = [
        {
            "code": item["code"],
            "name_cn": item["name_cn"],
            "super_class": item.get("super_class"),
            "count": item.get("count", 0),
        }
        for item in by_class
    ]

    global_summary = {
        "satellite": metadata.get("satellite", "未知卫星"),
        "sensor": metadata.get("sensor", "SAR"),
        "acquisition_time": metadata.get("acquisition_time", ""),
        "region_name": mission.get("region_name", "未知区域"),
        "region_type": mission.get("region_type", "unknown"),
        "scene_type_cn": scene.get("scene_type_cn", ""),
        "scene_description": scene.get("scene_description", ""),
        "object_count": statistics.get("totals", {}).get("all_objects", len(objects)),
        "ships": statistics.get("totals", {}).get("ships", 0),
        "aircraft": statistics.get("totals", {}).get("aircraft", 0),
        "needs_caution": confidence_summary.get("review_required_count", 0) > 0,
    }

    cluster_summary = {
        "distribution": spatial_summary.get("distribution", "分布情况不详"),
        "cluster_count": spatial_summary.get("cluster_count", 0),
        "nearest_neighbor_mean_m": spatial_summary.get("nearest_neighbor_mean_m"),
        "run_stats": {
            "tile_count": large_scene_trace.get("run_stats", {}).get("tile_count"),
            "raw_object_count": large_scene_trace.get("run_stats", {}).get("raw_object_count"),
            "deduped_object_count": large_scene_trace.get("run_stats", {}).get("deduped_object_count"),
        },
    }

    quality_flags = {
        "review_required_count": confidence_summary.get("review_required_count", 0),
        "low_confidence_count": confidence_summary.get("low_confidence_count", 0),
        "class_name_normalized": all(_is_class_name_normalized(item) for item in class_digest),
        "coordinate_mode": _coordinate_mode(evidence),
    }

    digest = {
        "digest_version": "large-scene-v1",
        "scene_scope": "large_scene",
        "global_summary": global_summary,
        "class_digest": class_digest,
        "cluster_summary": cluster_summary,
        "representative_targets": representative_targets,
        "quality_flags": quality_flags,
        "trace": {
            "case_name": large_scene_trace.get("case_name"),
            "scene_category": large_scene_trace.get("scene_category"),
            "usage": large_scene_trace.get("usage"),
        },
    }
    digest["digest_id"] = compute_digest_id(digest)
    return digest


def choose_generation_route(
    evidence: dict[str, Any],
    digest: dict[str, Any] | None = None,
    thresholds: RouteThresholds | None = None,
) -> str:
    return explain_generation_route(evidence, digest=digest, thresholds=thresholds)["route"]


def explain_generation_route(
    evidence: dict[str, Any],
    digest: dict[str, Any] | None = None,
    thresholds: RouteThresholds | None = None,
) -> dict[str, Any]:
    """Choose generation tier for a case."""
    digest = digest or build_evidence_digest(evidence)
    thresholds = thresholds or route_thresholds_from_env()

    object_count = int(digest["global_summary"].get("object_count", 0))
    class_count = sum(1 for item in digest.get("class_digest", []) if item.get("count", 0) > 0)
    review_required = int(digest["quality_flags"].get("review_required_count", 0))
    low_confidence = int(digest["quality_flags"].get("low_confidence_count", 0))
    cluster_count = int(digest["cluster_summary"].get("cluster_count", 0))

    low_conf_ratio = (low_confidence / object_count) if object_count > 0 else 0.0
    normalized_cluster_ratio = (cluster_count / object_count) if object_count > 0 else 0.0
    single_class_scene = class_count <= 1

    metrics = {
        "object_count": object_count,
        "class_count": class_count,
        "review_required_count": review_required,
        "low_confidence_count": low_confidence,
        "low_confidence_ratio": round(low_conf_ratio, 4),
        "cluster_count": cluster_count,
        "cluster_ratio": round(normalized_cluster_ratio, 4),
        "single_class_scene": single_class_scene,
    }
    threshold_view = thresholds.__dict__

    large_model_reasons: list[str] = []
    if object_count >= thresholds.large_model_object_threshold:
        large_model_reasons.append("object_count")
    if class_count > thresholds.large_model_class_threshold:
        large_model_reasons.append("class_count")
    if review_required >= thresholds.large_model_review_threshold:
        large_model_reasons.append("review_required_count")
    if low_conf_ratio >= thresholds.large_model_low_conf_ratio:
        large_model_reasons.append("low_confidence_ratio")
    if (
        not single_class_scene
        and cluster_count >= thresholds.large_model_cluster_threshold
        and normalized_cluster_ratio >= 0.25
    ):
        large_model_reasons.append("cluster_complexity")
    if large_model_reasons:
        return {
            "route": ROUTE_LARGE,
            "reason": "complex_case_requires_large_model",
            "triggered_by": large_model_reasons,
            "metrics": metrics,
            "thresholds": threshold_view,
        }

    large_refine_reasons: list[str] = []
    if object_count >= thresholds.large_refine_object_threshold:
        large_refine_reasons.append("object_count")
    if class_count >= thresholds.large_refine_class_threshold:
        large_refine_reasons.append("class_count")
    if review_required >= thresholds.large_refine_review_threshold:
        large_refine_reasons.append("review_required_count")
    if len(large_refine_reasons) == 3:
        return {
            "route": ROUTE_LARGE_REFINE,
            "reason": "mid_complex_case_prefers_large_refine",
            "triggered_by": large_refine_reasons,
            "metrics": metrics,
            "thresholds": threshold_view,
        }

    if object_count <= 3 and class_count <= 1 and review_required == 0:
        return {
            "route": ROUTE_TEMPLATE,
            "reason": "simple_case_uses_template",
            "triggered_by": ["simple_case"],
            "metrics": metrics,
            "thresholds": threshold_view,
        }

    return {
        "route": ROUTE_SMALL,
        "reason": "default_small_llm",
        "triggered_by": ["default"],
        "metrics": metrics,
        "thresholds": threshold_view,
    }


def attach_large_scene_metadata(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of evidence enriched with digest + route metadata."""
    enriched = dict(evidence)
    statistics = dict(enriched.get("statistics", {}))
    if "by_class" in statistics and isinstance(statistics.get("by_class"), list):
        statistics["by_class"] = normalize_by_class_list(statistics["by_class"])
    enriched["statistics"] = statistics
    digest = build_evidence_digest(evidence)
    route_expl = explain_generation_route(evidence, digest)
    route = route_expl["route"]
    report_ctx = dict(enriched.get("report_context", {}))
    report_ctx["large_scene_digest"] = digest
    report_ctx["generation_route"] = route
    report_ctx["generation_route_decision"] = route_expl
    report_ctx["prompt_profile"] = profile_prompt_input(evidence, digest, route).__dict__
    enriched["report_context"] = report_ctx
    return enriched


def profile_prompt_input(
    evidence: dict[str, Any],
    digest: dict[str, Any] | None = None,
    route: str | None = None,
) -> PromptProfile:
    digest = digest or build_evidence_digest(evidence)
    route = route or choose_generation_route(evidence, digest)
    prompt_payload = json.dumps(digest, ensure_ascii=False, sort_keys=True)
    return PromptProfile(
        prompt_chars=len(prompt_payload),
        prompt_lines=prompt_payload.count("\n") + 1,
        digest_bytes=len(prompt_payload.encode("utf-8")),
        route=route,
    )


def compute_digest_id(digest: dict[str, Any]) -> str:
    payload = json.dumps(digest, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _representative_target(obj: dict[str, Any]) -> dict[str, Any]:
    cls = obj.get("class", {})
    pixel = obj.get("geometry", {}).get("pixel", {})
    return {
        "object_id": obj.get("object_id"),
        "class_name_cn": get_class_name_cn(
            cls.get("code"),
            cls.get("name_cn"),
            cls.get("super_class"),
        ),
        "target_type_cn": get_super_class_name_cn(cls.get("super_class")),
        "confidence": obj.get("score", {}).get("confidence"),
        "center_x": pixel.get("center_x"),
        "center_y": pixel.get("center_y"),
    }


def _coordinate_mode(evidence: dict[str, Any]) -> str:
    warnings = evidence.get("quality_warnings", [])
    for item in warnings:
        if item.get("code") == "NON_WGS84_COORDINATES_SUPPRESSED":
            return "suppressed_non_wgs84"
    return "wgs84_or_unknown"


def _is_class_name_normalized(item: dict[str, Any]) -> bool:
    expected = get_class_name_cn(
        item.get("code"),
        item.get("name_cn"),
        item.get("super_class"),
    )
    return item.get("name_cn") == expected
