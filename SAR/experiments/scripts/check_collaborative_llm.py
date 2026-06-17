#!/usr/bin/env python3
"""
Quick health check and dry-run for collaborative small/large/VLM routing.

This checks:
  1. small model endpoint
  2. large model endpoint
  3. VLM endpoint
  4. small direct generation
  5. large direct generation
  6. small draft + large refine
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.report.collab_config import (
    collaborative_generator_kwargs,
    read_env_file,
    resolve_collaborative_config,
    resolve_vlm_config,
)
from modules.report.collaborative import CollaborativeReportGenerator
from modules.report.large_scene import ROUTE_LARGE, ROUTE_LARGE_REFINE
from modules.report.vlm_describer import VLMDescriber


def _env_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _cuda_status() -> dict[str, object]:
    status: dict[str, object] = {
        "import_ok": False,
        "cuda_available": False,
        "device_count": 0,
    }
    try:
        import torch  # type: ignore

        status["import_ok"] = True
        status["torch_version"] = getattr(torch, "__version__", None)
        status["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        status["cuda_available"] = bool(torch.cuda.is_available())
        status["device_count"] = int(torch.cuda.device_count()) if status["cuda_available"] else 0
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def _requires_local_gpu_probe(config: object) -> bool:
    if not getattr(config, "require_gpu", False):
        return False
    return bool(getattr(config, "small_model_path", None) or getattr(config, "large_model_path", None))


def _setting(
    cli_value: str | None,
    env_values: dict[str, str],
    key: str,
    default: str | None = None,
) -> str | None:
    if cli_value:
        return cli_value
    if env_values.get(key) not in (None, ""):
        return env_values[key]
    return os.getenv(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check collaborative small/large LLM setup.")
    parser.add_argument("--small-url", default=None)
    parser.add_argument("--small-model", default=None)
    parser.add_argument("--small-model-path", default=None)
    parser.add_argument("--large-url", default=None)
    parser.add_argument("--large-model", default=None)
    parser.add_argument("--large-model-path", default=None)
    parser.add_argument("--vlm-url", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--local-device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        default=None,
        help="Require CUDA visibility for local model checks.",
    )
    parser.add_argument(
        "--require-model-participation",
        action="store_true",
        default=None,
        help="Exit non-zero if generation falls back to template output.",
    )
    parser.add_argument(
        "--require-vlm",
        action="store_true",
        default=None,
        help="Exit non-zero if the VLM endpoint is missing or cannot produce a probe description.",
    )
    parser.add_argument("--env-file", default=None,
                        help="可选 .env 配置文件，用于加载协同模型参数")
    return parser.parse_args()


def _dummy_pkg() -> dict:
    return {
        "schema_version": "1.0.0",
        "package_id": "collab-check-001",
        "task_type": "intel_brief",
        "status": "READY_FOR_NLG",
        "created_at": "2026-05-25T00:00:00+00:00",
        "updated_at": "2026-05-25T00:00:00+00:00",
        "trace": {
            "pipeline_run_id": "pipe-collab-check",
            "large_scene_test": {
                "case_name": "collab-check",
                "relative_path": "dummy/path",
                "scene_category": "mixed",
                "usage": "health check",
                "run_stats": {"tile_count": 8, "raw_object_count": 30, "deduped_object_count": 30},
            },
        },
        "input": {
            "input_id": "input-check",
            "image": {"uri": "file:///tmp/test.tif", "file_name": "test.tif", "format": "GeoTIFF", "sha256": "x"},
            "metadata": {"satellite": "GF-3", "sensor": "SAR", "acquisition_time": "2026-05-25T00:00:00+00:00"},
            "mission": {"region_name": "测试区域", "region_type": "harbor", "priority": "HIGH"},
        },
        "scene": {"scene_type_cn": "港口"},
        "objects": [],
        "statistics": {
            "totals": {"all_objects": 30, "ships": 18, "aircraft": 6},
            "by_class": [
                {"code": "ship", "name_cn": "ship", "count": 18},
                {"code": "aircraft", "name_cn": "aircraft", "count": 6},
                {"code": "tank", "name_cn": "tank", "count": 6},
            ],
            "by_super_class": [],
            "spatial_summary": {"distribution": "目标分布于多个区域", "cluster_count": 6, "nearest_neighbor_mean_m": 320.0},
            "confidence_summary": {"mean_confidence": 0.71, "low_confidence_count": 6, "review_required_count": 4},
        },
        "attachments": {"overview_image": {"uri": "file:///tmp/overview.jpg", "role": "large_scene_overview"}},
        "report": {},
        "quality": {},
        "errors": [],
    }


def _print_result(name: str, result: dict) -> None:
    trace = result.get("report_context", {}).get("generation_trace", {})
    body = result.get("report", {}).get("body", "")
    print(f"\n[{name}]")
    print(json.dumps(trace, ensure_ascii=False, indent=2))
    print(f"body: {body[:160]}")


def _probe_vlm(describer: VLMDescriber) -> str:
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        Image.new("RGB", (64, 64), color=(0, 0, 0)).save(tmp_path)
        return describer.describe(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    args = parse_args()
    env_values = read_env_file(args.env_file)
    require_gpu = args.require_gpu
    if require_gpu is None:
        require_gpu = _env_bool(os.getenv("SAR_REQUIRE_GPU", env_values.get("SAR_REQUIRE_GPU")))
    require_vlm = args.require_vlm
    if require_vlm is None:
        require_vlm = _env_bool(os.getenv("SAR_REQUIRE_VLM", env_values.get("SAR_REQUIRE_VLM")))
    require_model_participation = args.require_model_participation
    if require_model_participation is None:
        allow_template_fallback_value = os.getenv(
            "SAR_ALLOW_TEMPLATE_FALLBACK",
            env_values.get("SAR_ALLOW_TEMPLATE_FALLBACK"),
        )
        require_model_participation = (
            False
            if allow_template_fallback_value is None
            else not _env_bool(allow_template_fallback_value)
        )
    config = resolve_collaborative_config(
        cache_dir=str(PROJECT_ROOT / "output" / ".collab_check_cache"),
        env_file=args.env_file,
        small_url=_setting(args.small_url, env_values, "SAR_SMALL_LLM_URL"),
        small_model=_setting(args.small_model, env_values, "SAR_SMALL_LLM_MODEL", "Qwen2.5-7B-Instruct"),
        small_model_path=_setting(args.small_model_path, env_values, "SAR_SMALL_LLM_PATH"),
        large_url=_setting(args.large_url, env_values, "SAR_LARGE_LLM_URL"),
        large_model=_setting(args.large_model, env_values, "SAR_LARGE_LLM_MODEL", "Qwen2.5-72B-Instruct"),
        large_model_path=_setting(args.large_model_path, env_values, "SAR_LARGE_LLM_PATH"),
        local_device=_setting(args.local_device, env_values, "SAR_LLM_DEVICE"),
        local_max_new_tokens=args.max_new_tokens,
        require_gpu=require_gpu,
        allow_template_fallback=not require_model_participation,
    )
    if _requires_local_gpu_probe(config):
        cuda_status = _cuda_status()
        if not cuda_status.get("cuda_available"):
            print("[health]")
            print(json.dumps(
                {
                    "llm": {
                        "require_gpu": config.require_gpu,
                        "small_model_path": config.small_model_path,
                        "large_model_path": config.large_model_path,
                        "healthy": False,
                        "reason": "cuda_not_visible",
                        "cuda": cuda_status,
                    },
                    "vlm": {
                        "model_name": _setting(args.vlm_model, env_values, "SAR_VLM_MODEL", "qwen3-vl-4b"),
                        "base_url": _setting(args.vlm_url, env_values, "SAR_VLM_URL"),
                        "require_vlm": require_vlm,
                    },
                    "probes_skipped": True,
                    "skip_reason": "strict_local_gpu_requires_cuda",
                },
                ensure_ascii=False,
                indent=2,
            ))
            print("ERROR: local GPU LLM health check requires CUDA, but CUDA is not visible.", file=sys.stderr)
            raise SystemExit(1)
    gen = CollaborativeReportGenerator(**collaborative_generator_kwargs(config))
    vlm_config = resolve_vlm_config(
        env_file=args.env_file,
        vlm_url=_setting(args.vlm_url, env_values, "SAR_VLM_URL"),
        vlm_model=_setting(args.vlm_model, env_values, "SAR_VLM_MODEL", "qwen3-vl-4b"),
        require_vlm=require_vlm,
    )
    vlm = VLMDescriber(
        base_url=vlm_config.vlm_base_url,
        model_name=vlm_config.vlm_model_name or "qwen3-vl-4b",
        api_key=vlm_config.api_key or "EMPTY",
        timeout=vlm_config.timeout,
    )

    print("[health]")
    vlm_healthy = vlm.health_check()
    print(json.dumps({
        "llm": gen.health_status(),
        "vlm": {
            "model_name": vlm_config.vlm_model_name,
            "base_url": vlm_config.vlm_base_url,
            "healthy": vlm_healthy,
            "require_vlm": vlm_config.require_vlm,
        },
    }, ensure_ascii=False, indent=2))

    if vlm_config.require_vlm:
        if not vlm_config.vlm_base_url:
            print("ERROR: VLM is required but SAR_VLM_URL/--vlm-url is empty.", file=sys.stderr)
            raise SystemExit(1)
        if not vlm_healthy:
            print("ERROR: VLM is required but the endpoint health check failed.", file=sys.stderr)
            raise SystemExit(1)

    base = _dummy_pkg()

    direct_large = json.loads(json.dumps(base))
    direct_large["report_context"] = {"generation_route": ROUTE_LARGE}
    try:
        direct_large_result = gen.generate(direct_large)
    except Exception as exc:
        print("\n[large_direct]")
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        if require_model_participation:
            raise SystemExit(1) from exc
        direct_large_result = None
    if direct_large_result is not None:
        if require_model_participation and not _has_model_participation(direct_large_result):
            print("ERROR: large_direct did not use a model backend.", file=sys.stderr)
            raise SystemExit(1)
        _print_result("large_direct", direct_large_result)

    refine_case = json.loads(json.dumps(base))
    refine_case["report_context"] = {"generation_route": ROUTE_LARGE_REFINE}
    try:
        refine_result = gen.generate(refine_case)
    except Exception as exc:
        print("\n[large_refine]")
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        if require_model_participation:
            raise SystemExit(1) from exc
        refine_result = None
    if refine_result is not None:
        if require_model_participation and not _has_model_participation(refine_result):
            print("ERROR: large_refine did not use a model backend.", file=sys.stderr)
            raise SystemExit(1)
        _print_result("large_refine", refine_result)

    if vlm_config.vlm_base_url:
        print("\n[vlm_probe]")
        vlm_probe = _probe_vlm(vlm)
        print(vlm_probe)
        if vlm_config.require_vlm and not vlm_probe.strip():
            print("ERROR: VLM is required but the probe returned an empty description.", file=sys.stderr)
            raise SystemExit(1)


def _has_model_participation(result: dict) -> bool:
    source = result.get("report", {}).get("body_sections", [{}])[0].get("source")
    return source in {
        "small_llm_v1",
        "small_llm_cache_v1",
        "large_llm_v1",
        "large_llm_cache_v1",
        "large_llm_refine_v1",
        "large_llm_refine_cache_v1",
    }


if __name__ == "__main__":
    main()
