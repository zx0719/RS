from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "preflight_gpu_acceptance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("preflight_gpu_acceptance", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_strict_env(
    tmp_path: Path,
    *,
    model_dir: Path,
    device: str = "cuda:0",
    template_fallback: str = "0",
    require_vlm: str = "0",
    vlm_url: str = "",
) -> Path:
    env_file = tmp_path / ".env.collab"
    env_file.write_text(
        "\n".join(
            [
                f"SAR_SMALL_LLM_PATH={model_dir}",
                "SAR_SMALL_LLM_MODEL=qwen3-4b",
                f"SAR_LLM_DEVICE={device}",
                "SAR_REQUIRE_GPU=1",
                "SAR_REQUIRE_LOCAL_GPU=1",
                f"SAR_ALLOW_TEMPLATE_FALLBACK={template_fallback}",
                f"SAR_REQUIRE_VLM={require_vlm}",
                f"SAR_VLM_URL={vlm_url}",
                "SAR_VLM_MODEL=qwen3-vl-4b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return env_file


def _base_args(module, tmp_path: Path, env_file: Path, manifest: Path):
    return module.parse_args(
        [
            "--env-file",
            str(env_file),
            "--manifest",
            str(manifest),
            "--python",
            sys.executable,
            "--output",
            str(tmp_path / "preflight.json"),
        ]
    )


def _mock_gpu_ok(monkeypatch, module) -> None:
    monkeypatch.setattr(
        module,
        "_run_nvidia_smi",
        lambda: {
            "ok": True,
            "returncode": 0,
            "gpus": ["NVIDIA Test GPU, 555.55, 24576 MiB"],
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        module,
        "_run_torch_cuda_check",
        lambda python_path: {
            "ok": True,
            "returncode": 0,
            "cuda_available": True,
            "device_count": 1,
            "device_names": ["NVIDIA Test GPU"],
        },
    )
    monkeypatch.setattr(
        module,
        "_run_python_import_check",
        lambda python_path, modules: {
            "ok": True,
            "returncode": 0,
            "modules": {module_name: {"ok": True} for module_name in modules},
        },
    )


def test_preflight_passes_with_mocked_gpu(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    model_dir = tmp_path / "qwen3-4b"
    model_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    env_file = _write_strict_env(tmp_path, model_dir=model_dir)
    _mock_gpu_ok(monkeypatch, module)

    report = module.build_report(_base_args(module, tmp_path, env_file, manifest))

    assert report["summary"]["ok"] is True
    assert report["inference_loaded"] is False
    assert report["summary"]["failures"] == []
    assert report["resolved"]["local_device"] == "cuda:0"


def test_preflight_rejects_template_fallback(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    model_dir = tmp_path / "qwen3-4b"
    model_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    env_file = _write_strict_env(tmp_path, model_dir=model_dir, template_fallback="1")
    _mock_gpu_ok(monkeypatch, module)

    report = module.build_report(_base_args(module, tmp_path, env_file, manifest))

    assert report["summary"]["ok"] is False
    assert "template_fallback_disabled" in report["summary"]["failures"]


def test_preflight_rejects_cpu_device(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    model_dir = tmp_path / "qwen3-4b"
    model_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    env_file = _write_strict_env(tmp_path, model_dir=model_dir, device="cpu")
    _mock_gpu_ok(monkeypatch, module)

    report = module.build_report(_base_args(module, tmp_path, env_file, manifest))

    assert report["summary"]["ok"] is False
    assert "local_device_cuda" in report["summary"]["failures"]


def test_preflight_rejects_required_vlm_without_url(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    model_dir = tmp_path / "qwen3-4b"
    model_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    env_file = _write_strict_env(tmp_path, model_dir=model_dir, require_vlm="1")
    _mock_gpu_ok(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "_check_vlm_health",
        lambda base_url, api_key, timeout: (_ for _ in ()).throw(AssertionError("unexpected VLM probe")),
    )

    report = module.build_report(_base_args(module, tmp_path, env_file, manifest))

    assert report["summary"]["ok"] is False
    assert "vlm_url_configured" in report["summary"]["failures"]


def test_preflight_checks_required_vlm_health(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    model_dir = tmp_path / "qwen3-4b"
    model_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    env_file = _write_strict_env(
        tmp_path,
        model_dir=model_dir,
        require_vlm="1",
        vlm_url="http://127.0.0.1:8101/v1",
    )
    _mock_gpu_ok(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "_check_vlm_health",
        lambda base_url, api_key, timeout: {
            "ok": True,
            "url": f"{base_url}/models",
            "status_code": 200,
        },
    )

    report = module.build_report(_base_args(module, tmp_path, env_file, manifest))

    assert report["summary"]["ok"] is True
    assert any(check["name"] == "vlm_health" and check["ok"] for check in report["checks"])


def test_preflight_rejects_missing_runtime_imports(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    model_dir = tmp_path / "qwen3-4b"
    model_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    env_file = _write_strict_env(tmp_path, model_dir=model_dir)
    _mock_gpu_ok(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "_run_python_import_check",
        lambda python_path, modules: {
            "ok": False,
            "returncode": 3,
            "modules": {"docx": {"ok": False, "error": "ModuleNotFoundError"}},
        },
    )

    report = module.build_report(_base_args(module, tmp_path, env_file, manifest))

    assert report["summary"]["ok"] is False
    assert "runtime_imports" in report["summary"]["failures"]
