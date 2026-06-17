#!/usr/bin/env python3
"""
Run the collaborative-output acceptance chain end to end.

This script is designed for real small/large model integration once endpoints
are available. It can also run in metadata-refresh mode for existing outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.report.collab_config import read_env_file, resolve_collaborative_config, resolve_vlm_config

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "large_scene_original_format"


def _env_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


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
    parser = argparse.ArgumentParser(description="Run collaborative acceptance chain.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "original_format_docx_manifest.json",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip evidence metadata refresh step.",
    )
    parser.add_argument(
        "--rebuild-reports",
        action="store_true",
        help="Regenerate large-scene evidence/docx before validation.",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip direct collaborative endpoint check.",
    )
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
        help="Require CUDA visibility for local model generation.",
    )
    parser.add_argument(
        "--require-local-gpu",
        action="store_true",
        default=None,
        help="Require validation evidence that every model output used a local GPU backend.",
    )
    parser.add_argument(
        "--require-vlm",
        action="store_true",
        default=None,
        help="Fail if VLM scene descriptions are missing.",
    )
    parser.add_argument(
        "--require-model-participation",
        action="store_true",
        default=None,
        help="Fail if report generation falls back to template output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "collaborative_acceptance_report.json",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env.collaborative.example",
        help="Optional env file to seed collaborative settings",
    )
    return parser.parse_args()


def _run(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def _new_acceptance_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"accept-{stamp}-{uuid4().hex[:8]}"


def main() -> None:
    args = parse_args()
    env_values = read_env_file(args.env_file)
    acceptance_run_id = os.getenv("SAR_ACCEPTANCE_RUN_ID") or _new_acceptance_run_id()
    require_gpu = args.require_gpu
    if require_gpu is None:
        require_gpu = _env_bool(os.getenv("SAR_REQUIRE_GPU", env_values.get("SAR_REQUIRE_GPU")))
    require_local_gpu = args.require_local_gpu
    if require_local_gpu is None:
        require_local_gpu = _env_bool(os.getenv("SAR_REQUIRE_LOCAL_GPU", env_values.get("SAR_REQUIRE_LOCAL_GPU")))
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
    collab_config = resolve_collaborative_config(
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
    vlm_config = resolve_vlm_config(
        env_file=args.env_file,
        vlm_url=_setting(args.vlm_url, env_values, "SAR_VLM_URL"),
        vlm_model=_setting(args.vlm_model, env_values, "SAR_VLM_MODEL", "qwen3-vl-4b"),
        require_vlm=require_vlm,
    )
    env = dict(os.environ)
    env.update(env_values)
    env["SAR_ACCEPTANCE_RUN_ID"] = acceptance_run_id
    if collab_config.small_base_url:
        env["SAR_SMALL_LLM_URL"] = collab_config.small_base_url
    if collab_config.small_model_name:
        env["SAR_SMALL_LLM_MODEL"] = collab_config.small_model_name
    if collab_config.small_model_path:
        env["SAR_SMALL_LLM_PATH"] = collab_config.small_model_path
    if collab_config.large_base_url:
        env["SAR_LARGE_LLM_URL"] = collab_config.large_base_url
    if collab_config.large_model_name:
        env["SAR_LARGE_LLM_MODEL"] = collab_config.large_model_name
    if collab_config.large_model_path:
        env["SAR_LARGE_LLM_PATH"] = collab_config.large_model_path
    if collab_config.local_device:
        env["SAR_LLM_DEVICE"] = collab_config.local_device
    if collab_config.local_max_new_tokens:
        env["SAR_LLM_MAX_NEW_TOKENS"] = str(collab_config.local_max_new_tokens)
    env["SAR_REQUIRE_GPU"] = "1" if collab_config.require_gpu else "0"
    env["SAR_REQUIRE_LOCAL_GPU"] = "1" if require_local_gpu else "0"
    env["SAR_ALLOW_TEMPLATE_FALLBACK"] = "0" if require_model_participation else "1"
    if vlm_config.vlm_base_url:
        env["SAR_VLM_URL"] = vlm_config.vlm_base_url
    if vlm_config.vlm_model_name:
        env["SAR_VLM_MODEL"] = vlm_config.vlm_model_name
    env["SAR_REQUIRE_VLM"] = "1" if vlm_config.require_vlm else "0"
    effective_vlm_url = vlm_config.vlm_base_url
    effective_vlm_model = vlm_config.vlm_model_name

    steps: list[dict[str, object]] = []
    aborted_after_step: str | None = None
    strict_mode = require_model_participation or collab_config.require_gpu or require_local_gpu or vlm_config.require_vlm
    effective_small_url = collab_config.small_base_url
    effective_large_url = collab_config.large_base_url
    effective_small_model = collab_config.small_model_name
    effective_large_model = collab_config.large_model_name
    effective_small_model_path = collab_config.small_model_path
    effective_large_model_path = collab_config.large_model_path

    if not args.skip_health_check and (
        effective_small_url
        or effective_large_url
        or effective_small_model_path
        or effective_large_model_path
        or effective_vlm_url
    ):
        code, output = _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "check_collaborative_llm.py"),
                *(["--small-url", effective_small_url] if effective_small_url else []),
                *(["--small-model", effective_small_model] if effective_small_model else []),
                *(["--small-model-path", effective_small_model_path] if effective_small_model_path else []),
                *(["--large-url", effective_large_url] if effective_large_url else []),
                *(["--large-model", effective_large_model] if effective_large_model else []),
                *(["--large-model-path", effective_large_model_path] if effective_large_model_path else []),
                *(["--vlm-url", effective_vlm_url] if effective_vlm_url else []),
                *(["--vlm-model", effective_vlm_model] if effective_vlm_model else []),
                *(["--local-device", collab_config.local_device] if collab_config.local_device else []),
                *(["--max-new-tokens", str(collab_config.local_max_new_tokens)] if collab_config.local_max_new_tokens else []),
                *(["--require-gpu"] if collab_config.require_gpu else []),
                *(["--require-vlm"] if vlm_config.require_vlm else []),
                *(["--require-model-participation"] if require_model_participation else []),
                ],
            env,
        )
        steps.append({"step": "health_check", "returncode": code, "output": output})
        if code != 0 and strict_mode:
            aborted_after_step = "health_check"

    if aborted_after_step is None and args.rebuild_reports:
        code, output = _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_original_large_scene_docx.py"),
                str(args.env_file),
                "--acceptance-run-id",
                acceptance_run_id,
                *(["--strict-release"] if strict_mode else []),
                *(["--require-model-participation"] if require_model_participation else []),
                *(["--require-gpu"] if collab_config.require_gpu else []),
                *(["--require-local-gpu"] if require_local_gpu else []),
                *(["--require-vlm"] if vlm_config.require_vlm else []),
                *(["--require-fresh-model-output"] if strict_mode else []),
            ],
            env,
        )
        steps.append({"step": "rebuild_reports", "returncode": code, "output": output})
        if code != 0:
            aborted_after_step = "rebuild_reports"

    if aborted_after_step is None and not args.skip_refresh:
        code, output = _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "refresh_large_scene_metadata.py"),
                "--manifest", str(args.manifest),
            ],
            env,
        )
        steps.append({"step": "refresh_metadata", "returncode": code, "output": output})
        if code != 0 and strict_mode:
            aborted_after_step = "refresh_metadata"

    if aborted_after_step is None:
        validation_steps = [
            ("release_manifest", "build_large_scene_release_manifest.py", "release_manifest.json"),
            ("docx_validation", "validate_docx_outputs.py", "docx_validation_report.json"),
            ("vlm_validation", "validate_vlm_outputs.py", "vlm_validation_report.json"),
            ("collaborative_validation", "validate_collaborative_outputs.py", "collaborative_validation_report.json"),
            ("release_readiness", "validate_release_readiness.py", "release_readiness_report.json"),
            ("route_summary", "summarize_collaborative_routes.py", "collaborative_route_summary.json"),
            ("threshold_sweep", "sweep_collaborative_thresholds.py", "collaborative_threshold_sweep.json"),
            ("threshold_recommendation", "recommend_collaborative_thresholds.py", "collaborative_threshold_recommendation.json"),
            ("recommended_env", "write_recommended_env.py", ".env.collaborative.recommended"),
        ]
        for step_name, script_name, out_name in validation_steps:
            code, output = _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / script_name),
                    *(["--manifest", str(args.manifest)] if script_name in {"validate_docx_outputs.py", "validate_vlm_outputs.py", "validate_release_readiness.py"} else []),
                    *(["--require-vlm"] if script_name in {"validate_vlm_outputs.py", "validate_release_readiness.py"} and vlm_config.require_vlm else []),
                    *(["--require-vlm-trace"] if script_name in {"validate_vlm_outputs.py", "validate_release_readiness.py"} and vlm_config.require_vlm else []),
                    *(["--require-model-participation"] if script_name in {"validate_collaborative_outputs.py", "validate_release_readiness.py"} and require_model_participation else []),
                    *(["--require-gpu"] if script_name in {"validate_collaborative_outputs.py", "validate_release_readiness.py"} and collab_config.require_gpu else []),
                    *(["--require-local-gpu"] if script_name in {"validate_collaborative_outputs.py", "validate_release_readiness.py"} and require_local_gpu else []),
                    *(["--require-acceptance-run-id", acceptance_run_id] if script_name in {"validate_collaborative_outputs.py", "validate_release_readiness.py"} and strict_mode else []),
                    *(["--require-fresh-model-output"] if script_name in {"validate_collaborative_outputs.py", "validate_release_readiness.py"} and strict_mode else []),
                    *(["--input-manifest", str(args.manifest)] if script_name == "build_large_scene_release_manifest.py" else []),
                    *(["--require-acceptance-run-id", acceptance_run_id] if script_name == "build_large_scene_release_manifest.py" and strict_mode else []),
                    *(["--input", str(args.manifest.parent / "collaborative_validation_report.json")] if script_name == "summarize_collaborative_routes.py" else []),
                    *(["--manifest", str(args.manifest)] if script_name == "sweep_collaborative_thresholds.py" else []),
                    *(["--input", str(args.manifest.parent / "collaborative_threshold_sweep.json")] if script_name == "recommend_collaborative_thresholds.py" else []),
                    *(["--recommendation", str(args.manifest.parent / "collaborative_threshold_recommendation.json"), "--base-env", str(args.env_file)] if script_name == "write_recommended_env.py" else []),
                    "--output",
                    str(args.manifest.parent / out_name),
                ],
                env,
            )
            steps.append({"step": step_name, "returncode": code, "output": output})
            if code != 0 and strict_mode and step_name in {
                "release_manifest",
                "docx_validation",
                "vlm_validation",
                "collaborative_validation",
                "release_readiness",
            }:
                aborted_after_step = step_name
                break

    report = {
        "manifest": str(args.manifest),
        "acceptance_run_id": acceptance_run_id,
        "small_url": effective_small_url,
        "small_model": effective_small_model,
        "small_model_path": effective_small_model_path,
        "large_url": effective_large_url,
        "large_model": effective_large_model,
        "large_model_path": effective_large_model_path,
        "vlm_url": effective_vlm_url,
        "vlm_model": effective_vlm_model,
        "rebuild_reports": args.rebuild_reports,
        "require_model_participation": require_model_participation,
        "require_gpu": collab_config.require_gpu,
        "require_local_gpu": require_local_gpu,
        "require_vlm": vlm_config.require_vlm,
        "aborted_after_step": aborted_after_step,
        "steps": steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    failed_steps = [step for step in steps if step.get("returncode") != 0]
    if failed_steps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
