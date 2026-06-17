from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_large_scene_tests.py"
SPEC = importlib.util.spec_from_file_location("run_large_scene_tests_stage1", SCRIPT_PATH)
large_scene_script = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = large_scene_script
SPEC.loader.exec_module(large_scene_script)

sys.path.insert(0, str(PROJECT_ROOT))

from modules.class_labels import get_class_name_cn, normalize_by_class_list
from modules.report.large_scene import (
    ROUTE_LARGE,
    ROUTE_LARGE_REFINE,
    ROUTE_SMALL,
    build_evidence_digest,
    choose_generation_route,
    explain_generation_route,
    profile_prompt_input,
)
from modules.report.prompt_templates import build_prompt_payload


def _load_large_scene_evidence(case_name: str) -> dict:
    path = (
        PROJECT_ROOT
        / "output"
        / "large_scene"
        / case_name
        / f"{case_name}_evidence.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_class_name_normalization_maps_generic_codes_to_cn() -> None:
    assert get_class_name_cn("ship", "ship", "ship") == "舰船"
    assert get_class_name_cn("aircraft", "aircraft", "aircraft") == "飞机"

    normalized = normalize_by_class_list(
        [{"code": "ship", "name_cn": "ship", "count": 47}]
    )
    assert normalized[0]["name_cn"] == "舰船"
    assert normalized[0]["super_class"] == "ship"


def test_large_scene_digest_and_prompt_payload_are_compact_and_routed() -> None:
    evidence = _load_large_scene_evidence("ship2_large_tiff")
    evidence["scene"]["scene_description"] = "图像显示锚地水域内存在多目标分布"

    digest = build_evidence_digest(evidence)
    assert digest["scene_scope"] == "large_scene"
    assert digest["global_summary"]["object_count"] == 47
    assert digest["class_digest"][0]["name_cn"] == "舰船"
    assert len(digest["representative_targets"]) <= 6

    route = choose_generation_route(evidence, digest)
    assert route == ROUTE_SMALL

    profile = profile_prompt_input(evidence, digest, route)
    assert profile.prompt_chars > 0
    assert profile.digest_bytes >= profile.prompt_chars

    payload = build_prompt_payload(evidence)
    assert payload["mode"] == "large_scene"
    assert payload["route"] == ROUTE_SMALL
    assert "<LARGE_SCENE_DIGEST>" in payload["user_prompt"]
    assert '"name_cn": "舰船"' in payload["user_prompt"]
    assert "图像显示锚地水域内存在多目标分布" in payload["user_prompt"]


def test_large_scene_prompt_payload_respects_forced_route_from_report_context() -> None:
    evidence = _load_large_scene_evidence("ship2_large_tiff")
    evidence["report_context"] = {"generation_route": ROUTE_LARGE_REFINE}
    payload = build_prompt_payload(evidence)
    assert payload["route"] == ROUTE_LARGE_REFINE


def test_large_scene_route_escalates_for_complex_case() -> None:
    evidence = _load_large_scene_evidence("ship2_large_tiff")
    complex_case = copy.deepcopy(evidence)
    complex_case["statistics"]["totals"]["all_objects"] = 160
    complex_case["statistics"]["totals"]["ships"] = 120
    complex_case["statistics"]["totals"]["aircraft"] = 20
    complex_case["statistics"]["by_class"] = [
        {"code": "ship", "name_cn": "ship", "count": 120},
        {"code": "aircraft", "name_cn": "aircraft", "count": 20},
        {"code": "tank", "name_cn": "tank", "count": 10},
        {"code": "bridge", "name_cn": "bridge", "count": 10},
    ]
    complex_case["statistics"]["spatial_summary"]["cluster_count"] = 9
    complex_case["statistics"]["confidence_summary"]["review_required_count"] = 12
    route = choose_generation_route(complex_case)
    assert route == ROUTE_LARGE


def test_large_scene_route_prefers_large_refine_for_mid_complex_case() -> None:
    evidence = _load_large_scene_evidence("ship2_large_tiff")
    mid_case = copy.deepcopy(evidence)
    mid_case["statistics"]["totals"]["all_objects"] = 30
    mid_case["statistics"]["totals"]["ships"] = 18
    mid_case["statistics"]["totals"]["aircraft"] = 6
    mid_case["statistics"]["by_class"] = [
        {"code": "ship", "name_cn": "ship", "count": 18},
        {"code": "aircraft", "name_cn": "aircraft", "count": 6},
        {"code": "tank", "name_cn": "tank", "count": 6},
    ]
    mid_case["statistics"]["spatial_summary"]["cluster_count"] = 6
    mid_case["statistics"]["confidence_summary"]["low_confidence_count"] = 5
    mid_case["statistics"]["confidence_summary"]["review_required_count"] = 4
    route = choose_generation_route(mid_case)
    assert route == ROUTE_LARGE_REFINE


def test_large_scene_route_thresholds_can_be_overridden_by_env() -> None:
    evidence = _load_large_scene_evidence("ship2_large_tiff")
    case = copy.deepcopy(evidence)
    case["statistics"]["totals"]["all_objects"] = 20
    case["statistics"]["totals"]["ships"] = 10
    case["statistics"]["totals"]["aircraft"] = 5
    case["statistics"]["by_class"] = [
        {"code": "ship", "name_cn": "ship", "count": 10},
        {"code": "aircraft", "name_cn": "aircraft", "count": 5},
        {"code": "tank", "name_cn": "tank", "count": 5},
    ]
    case["statistics"]["spatial_summary"]["cluster_count"] = 4
    case["statistics"]["confidence_summary"]["review_required_count"] = 2
    case["statistics"]["confidence_summary"]["low_confidence_count"] = 2
    with patch.dict(os.environ, {"SAR_ROUTE_LARGE_REFINE_OBJECT_THRESHOLD": "10", "SAR_ROUTE_LARGE_REFINE_REVIEW_THRESHOLD": "2"}):
        route = choose_generation_route(case)
    assert route == ROUTE_LARGE_REFINE


def test_large_scene_route_explanation_contains_reason_and_metrics() -> None:
    evidence = _load_large_scene_evidence("ship2_large_tiff")
    expl = explain_generation_route(evidence)
    assert expl["route"] == ROUTE_SMALL
    assert "reason" in expl
    assert "metrics" in expl
    assert "triggered_by" in expl


def test_run_large_scene_build_evidence_attaches_report_context() -> None:
    manifest = large_scene_script.load_manifest(
        PROJECT_ROOT / "tests" / "fixtures" / "large_scene_manifest.json"
    )
    case = large_scene_script.select_cases(manifest, suite="smoke", names=["ship2_large_tiff"])[0]
    package = _load_large_scene_evidence("ship2_large_tiff")
    objects = package["objects"]
    overview_path = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    run_stats = package["trace"]["large_scene_test"]["run_stats"]

    rebuilt = large_scene_script.build_evidence(case, objects, overview_path, run_stats)
    report_context = rebuilt.get("report_context", {})
    assert report_context["generation_route"] == ROUTE_SMALL
    assert report_context["large_scene_digest"]["global_summary"]["object_count"] == 47
    assert report_context["prompt_profile"]["route"] == ROUTE_SMALL
    assert rebuilt["report"]["body_sections"][0]["source"] == "template_v1"
