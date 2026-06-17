#!/usr/bin/env python3
"""Preflight checks for strict GPU-only local small-model acceptance.

This script only checks configuration and CUDA visibility. It does not load a
language model, a vision model, or run inference.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.report.collab_config import read_env_file

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "large_scene_original_format"
TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def _setting(
    cli_value: str | Path | None,
    env_values: dict[str, str],
    key: str,
    default: str | None = None,
) -> str | None:
    if cli_value not in (None, ""):
        return str(cli_value)
    if env_values.get(key) not in (None, ""):
        return env_values[key]
    env_value = os.getenv(key)
    if env_value not in (None, ""):
        return env_value
    return default


def _bool_setting(
    cli_value: bool | None,
    env_values: dict[str, str],
    key: str,
    default: bool,
) -> bool:
    if cli_value is not None:
        return cli_value
    if key in env_values:
        return _env_bool(env_values.get(key), default)
    return _env_bool(os.getenv(key), default)


def _check(
    name: str,
    ok: bool,
    message: str,
    *,
    severity: str = "error",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "severity": severity,
        "message": message,
        "details": details or {},
    }


def _resolve_python(python_path: str) -> dict[str, Any]:
    if os.path.isabs(python_path):
        resolved = Path(python_path)
    else:
        found = shutil.which(python_path)
        resolved = Path(found) if found else Path(python_path)
    return {
        "input": python_path,
        "path": str(resolved),
        "exists": resolved.exists(),
        "executable": resolved.exists() and os.access(str(resolved), os.X_OK),
    }


def _run_nvidia_smi() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc), "returncode": None}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": f"timeout after {exc.timeout}s", "returncode": None}

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "gpus": lines,
        "stderr": proc.stderr.strip(),
    }


def _run_torch_cuda_check(python_path: str) -> dict[str, Any]:
    snippet = r"""
import json
import sys

result = {
    "ok": False,
    "import_ok": False,
    "cuda_available": False,
    "device_count": 0,
    "device_names": [],
}
try:
    import torch

    result["import_ok"] = True
    result["torch_version"] = getattr(torch, "__version__", None)
    result["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["device_count"] = int(torch.cuda.device_count()) if result["cuda_available"] else 0
    if result["cuda_available"]:
        result["device_names"] = [
            torch.cuda.get_device_name(index) for index in range(result["device_count"])
        ]
    result["ok"] = result["cuda_available"] and result["device_count"] > 0
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if result["ok"] else 3)
"""
    try:
        proc = subprocess.run(
            [python_path, "-c", snippet],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc), "returncode": None}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": f"timeout after {exc.timeout}s", "returncode": None}

    data: dict[str, Any]
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        data = {"ok": False, "parse_error": True, "stdout": proc.stdout.strip()}
    data["returncode"] = proc.returncode
    if proc.stderr.strip():
        data["stderr"] = proc.stderr.strip()
    return data


def _run_python_import_check(python_path: str, modules: list[str]) -> dict[str, Any]:
    snippet = r"""
import importlib
import json
import sys

modules = sys.argv[1:]
result = {"ok": True, "modules": {}}
for module_name in modules:
    try:
        module = importlib.import_module(module_name)
        result["modules"][module_name] = {
            "ok": True,
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        result["modules"][module_name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        result["ok"] = False

print(json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if result["ok"] else 3)
"""
    try:
        proc = subprocess.run(
            [python_path, "-c", snippet, *modules],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc), "returncode": None}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": f"timeout after {exc.timeout}s", "returncode": None}

    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        data = {"ok": False, "parse_error": True, "stdout": proc.stdout.strip()}
    data["returncode"] = proc.returncode
    if proc.stderr.strip():
        data["stderr"] = proc.stderr.strip()
    return data


def _check_vlm_health(base_url: str, api_key: str, timeout: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/models"
    req = urllib_request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
            return {
                "ok": resp.status == 200,
                "url": url,
                "status_code": resp.status,
                "body_preview": body[:512],
            }
    except urllib_error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        return {
            "ok": False,
            "url": url,
            "status_code": exc.code,
            "body_preview": body[:512],
        }
    except Exception as exc:
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}


def _parse_timeout(value: str | None, default: int = 5) -> int:
    try:
        return int(value or default)
    except ValueError:
        return default


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    env_file = Path(args.env_file) if args.env_file else None
    env_values = read_env_file(env_file)
    manifest = Path(args.manifest)
    model_path_value = _setting(args.model_path, env_values, "SAR_SMALL_LLM_PATH")
    model_path = Path(model_path_value).expanduser() if model_path_value else None
    python_value = str(args.python or sys.executable)
    python_status = _resolve_python(python_value)
    device = _setting(args.device, env_values, "SAR_LLM_DEVICE", "cuda:0") or ""
    require_gpu = _bool_setting(args.require_gpu, env_values, "SAR_REQUIRE_GPU", False)
    require_local_gpu = _bool_setting(
        args.require_local_gpu,
        env_values,
        "SAR_REQUIRE_LOCAL_GPU",
        False,
    )
    allow_template_fallback = _env_bool(
        _setting(None, env_values, "SAR_ALLOW_TEMPLATE_FALLBACK"),
        default=True,
    )
    require_vlm = _bool_setting(args.require_vlm, env_values, "SAR_REQUIRE_VLM", False)
    vlm_url = _setting(args.vlm_url, env_values, "SAR_VLM_URL")
    vlm_model = _setting(args.vlm_model, env_values, "SAR_VLM_MODEL", "qwen3-vl-4b")
    api_key = _setting(None, env_values, "SAR_LLM_API_KEY", "EMPTY") or "EMPTY"
    vlm_timeout = _parse_timeout(_setting(None, env_values, "SAR_VLM_TIMEOUT"), default=args.vlm_timeout)
    cache_dir = manifest.parent / ".report_cache"

    checks: list[dict[str, Any]] = []
    if env_file:
        checks.append(
            _check(
                "env_file",
                env_file.exists(),
                "Environment file was found." if env_file.exists() else "Environment file is missing.",
                severity="warning",
                details={"path": str(env_file)},
            )
        )

    checks.extend(
        [
            _check(
                "manifest_exists",
                manifest.exists(),
                "Large-scene manifest exists." if manifest.exists() else "Large-scene manifest is missing.",
                details={"path": str(manifest)},
            ),
            _check(
                "python_executable",
                bool(python_status["executable"]),
                "Configured Python is executable."
                if python_status["executable"]
                else "Configured Python is missing or not executable.",
                details=python_status,
            ),
            _check(
                "small_model_path_configured",
                model_path is not None,
                "Local small-model path is configured."
                if model_path is not None
                else "SAR_SMALL_LLM_PATH/--model-path is not configured.",
                details={"path": str(model_path) if model_path else None},
            ),
            _check(
                "small_model_path_exists",
                model_path is not None and model_path.is_dir(),
                "Local small-model directory exists."
                if model_path is not None and model_path.is_dir()
                else "Local small-model directory is missing.",
                details={"path": str(model_path) if model_path else None},
            ),
        ]
    )

    device_lower = device.strip().lower()
    device_is_cuda = "cuda" in device_lower and "cpu" not in device_lower
    checks.extend(
        [
            _check(
                "local_device_cuda",
                device_is_cuda,
                "Local model device is explicitly CUDA."
                if device_is_cuda
                else "Local model device must be CUDA, not CPU/auto.",
                details={"device": device},
            ),
            _check(
                "require_gpu_enabled",
                require_gpu,
                "SAR_REQUIRE_GPU is enabled."
                if require_gpu
                else "SAR_REQUIRE_GPU must be enabled for strict acceptance.",
                details={"require_gpu": require_gpu},
            ),
            _check(
                "require_local_gpu_enabled",
                require_local_gpu,
                "SAR_REQUIRE_LOCAL_GPU is enabled."
                if require_local_gpu
                else "SAR_REQUIRE_LOCAL_GPU must be enabled for strict acceptance.",
                details={"require_local_gpu": require_local_gpu},
            ),
            _check(
                "template_fallback_disabled",
                not allow_template_fallback,
                "Template fallback is disabled."
                if not allow_template_fallback
                else "SAR_ALLOW_TEMPLATE_FALLBACK must be 0 for strict acceptance.",
                details={"allow_template_fallback": allow_template_fallback},
            ),
        ]
    )

    nvidia_status = _run_nvidia_smi()
    checks.append(
        _check(
            "nvidia_smi",
            bool(nvidia_status.get("ok")),
            "nvidia-smi can see at least one GPU."
            if nvidia_status.get("ok")
            else "nvidia-smi failed; GPU is not visible in this shell.",
            details=nvidia_status,
        )
    )

    if python_status["executable"]:
        torch_status = _run_torch_cuda_check(str(python_status["path"]))
        import_status = _run_python_import_check(
            str(python_status["path"]),
            ["docx", "PIL", "openai", "torch", "transformers"],
        )
    else:
        torch_status = {"ok": False, "error": "python executable unavailable"}
        import_status = {"ok": False, "error": "python executable unavailable"}
    checks.append(
        _check(
            "torch_cuda",
            bool(torch_status.get("ok")),
            "Torch reports CUDA as available."
            if torch_status.get("ok")
            else "Torch cannot see CUDA in the configured Python process.",
            details=torch_status,
        )
    )
    checks.append(
        _check(
            "runtime_imports",
            bool(import_status.get("ok")),
            "Configured Python can import report runtime dependencies."
            if import_status.get("ok")
            else "Configured Python is missing one or more report runtime dependencies.",
            details=import_status,
        )
    )

    if require_vlm:
        checks.append(
            _check(
                "vlm_url_configured",
                bool(vlm_url),
                "VLM URL is configured." if vlm_url else "SAR_REQUIRE_VLM=1 but VLM URL is empty.",
                details={"vlm_url": vlm_url, "vlm_model": vlm_model},
            )
        )
        if vlm_url:
            vlm_health = _check_vlm_health(vlm_url, api_key, vlm_timeout)
            checks.append(
                _check(
                    "vlm_health",
                    bool(vlm_health.get("ok")),
                    "VLM /models endpoint is reachable."
                    if vlm_health.get("ok")
                    else "VLM /models endpoint is not reachable.",
                    details=vlm_health,
                )
            )
    else:
        checks.append(
            _check(
                "vlm_strict_mode",
                True,
                "VLM strict mode is disabled for this run.",
                severity="info",
                details={"require_vlm": False},
            )
        )

    checks.append(
        _check(
            "report_cache_status",
            True,
            "Report cache status captured.",
            severity="info",
            details={"path": str(cache_dir), "exists": cache_dir.exists()},
        )
    )

    failures = [
        check["name"]
        for check in checks
        if check.get("severity") == "error" and not check.get("ok")
    ]
    warnings = [
        check["name"]
        for check in checks
        if check.get("severity") == "warning" and not check.get("ok")
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "inference_loaded": False,
        "resolved": {
            "env_file": str(env_file) if env_file else None,
            "manifest": str(manifest),
            "python": python_status,
            "small_model_path": str(model_path) if model_path else None,
            "local_device": device,
            "require_gpu": require_gpu,
            "require_local_gpu": require_local_gpu,
            "allow_template_fallback": allow_template_fallback,
            "require_vlm": require_vlm,
            "vlm_url": vlm_url,
            "vlm_model": vlm_model,
        },
        "checks": checks,
        "summary": {
            "ok": not failures,
            "failure_count": len(failures),
            "warning_count": len(warnings),
            "failures": failures,
            "warnings": warnings,
        },
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight strict GPU acceptance settings.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env.collaborative.example",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "original_format_docx_manifest.json",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-gpu", action="store_true", default=None)
    parser.add_argument("--require-local-gpu", action="store_true", default=None)
    parser.add_argument("--require-vlm", action="store_true", default=None)
    parser.add_argument("--vlm-url", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--vlm-timeout", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "gpu_acceptance_preflight.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    write_report(report, args.output)
    print(args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
