from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_collaborative_acceptance_script_produces_report(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "run_collaborative_acceptance.py"
    manifest = PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    output = PROJECT_ROOT / "output" / "large_scene_original_format" / "collaborative_acceptance_report.json"
    sweep_script = PROJECT_ROOT / "scripts" / "sweep_collaborative_thresholds.py"
    sweep_output = PROJECT_ROOT / "output" / "large_scene_original_format" / "collaborative_threshold_sweep.json"
    env_file = tmp_path / ".env.collab"
    env_file.write_text(
        "SAR_REQUIRE_GPU=0\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=1\n"
        "SAR_REQUIRE_VLM=0\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(sweep_script), "--output", str(sweep_output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--env-file",
            str(env_file),
            "--skip-health-check",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["steps"]
    assert report["acceptance_run_id"].startswith("accept-")
    assert "vlm_model" in report
    assert report["require_vlm"] is False
    assert any(step["step"] == "vlm_validation" for step in report["steps"])
    assert any(step["step"] == "collaborative_validation" for step in report["steps"])
    assert any(step["step"] == "release_readiness" for step in report["steps"])
    assert any(step["step"] == "threshold_sweep" for step in report["steps"])
    assert any(step["step"] == "threshold_recommendation" for step in report["steps"])
    assert any(step["step"] == "recommended_env" for step in report["steps"])


def test_run_collaborative_acceptance_passes_local_model_settings(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "run_collaborative_acceptance.py"
    manifest = PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    output = tmp_path / "acceptance.json"
    env_file = tmp_path / ".env.collab"
    model_dir = tmp_path / "qwen3-4b"
    model_dir.mkdir()
    env_file.write_text(
        f"SAR_SMALL_LLM_PATH={model_dir}\n"
        "SAR_SMALL_LLM_MODEL=qwen3-4b\n"
        "SAR_LLM_DEVICE=cpu\n"
        "SAR_LLM_MAX_NEW_TOKENS=64\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--env-file",
            str(env_file),
            "--skip-health-check",
            "--skip-refresh",
            "--output",
            str(output),
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["small_model_path"] == str(model_dir)
    assert report["require_gpu"] is False


def test_run_collaborative_acceptance_reads_strict_flags_from_env_file(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "run_collaborative_acceptance.py"
    manifest = PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    output = tmp_path / "acceptance_strict_env.json"
    env_file = tmp_path / ".env.collab"
    env_file.write_text(
        "SAR_REQUIRE_GPU=1\n"
        "SAR_REQUIRE_LOCAL_GPU=1\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--env-file",
            str(env_file),
            "--skip-health-check",
            "--skip-refresh",
            "--output",
            str(output),
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["require_gpu"] is True
    assert report["require_local_gpu"] is True
    assert report["require_model_participation"] is True
    collab_step = next(step for step in report["steps"] if step["step"] == "collaborative_validation")
    assert collab_step["returncode"] == 1
    assert report["aborted_after_step"] == "collaborative_validation"
    assert not any(step["step"] == "release_readiness" for step in report["steps"])


def test_run_collaborative_acceptance_reads_model_names_from_env_file(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "run_collaborative_acceptance.py"
    manifest = PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    output = tmp_path / "acceptance_model_env.json"
    model_dir = tmp_path / "qwen3-4b"
    model_dir.mkdir()
    env_file = tmp_path / ".env.collab"
    env_file.write_text(
        f"SAR_SMALL_LLM_PATH={model_dir}\n"
        "SAR_SMALL_LLM_MODEL=qwen3-4b-env\n"
        "SAR_LARGE_LLM_MODEL=large-env\n"
        "SAR_VLM_MODEL=vlm-env\n"
        "SAR_REQUIRE_GPU=0\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=1\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--env-file",
            str(env_file),
            "--skip-health-check",
            "--skip-refresh",
            "--output",
            str(output),
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["small_model_path"] == str(model_dir)
    assert report["small_model"] == "qwen3-4b-env"
    assert report["large_model"] == "large-env"
    assert report["vlm_model"] == "vlm-env"


def test_run_collaborative_acceptance_process_env_overrides_empty_env_file_values(tmp_path: Path, monkeypatch) -> None:
    script = PROJECT_ROOT / "scripts" / "run_collaborative_acceptance.py"
    manifest = PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    output = tmp_path / "acceptance_process_env.json"
    model_dir = tmp_path / "qwen3-4b"
    model_dir.mkdir()
    env_file = tmp_path / ".env.collab"
    env_file.write_text(
        "SAR_SMALL_LLM_PATH=\n"
        "SAR_VLM_URL=\n"
        "SAR_VLM_MODEL=vlm-env\n"
        "SAR_REQUIRE_GPU=0\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=1\n"
        "SAR_REQUIRE_VLM=0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SAR_SMALL_LLM_PATH", str(model_dir))
    monkeypatch.setenv("SAR_VLM_URL", "http://vlm.from.process/v1")

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--env-file",
            str(env_file),
            "--skip-health-check",
            "--skip-refresh",
            "--output",
            str(output),
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["small_model_path"] == str(model_dir)
    assert report["vlm_url"] == "http://vlm.from.process/v1"


def test_run_collaborative_acceptance_can_require_vlm(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "run_collaborative_acceptance.py"
    manifest = PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    output = tmp_path / "acceptance_require_vlm.json"
    env_file = tmp_path / ".env.collab"
    env_file.write_text("SAR_REQUIRE_VLM=1\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--env-file",
            str(env_file),
            "--skip-health-check",
            "--skip-refresh",
            "--require-vlm",
            "--output",
            str(output),
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["require_vlm"] is True
    vlm_step = next(step for step in report["steps"] if step["step"] == "vlm_validation")
    if vlm_step["returncode"] == 1:
        assert report["aborted_after_step"] == "vlm_validation"
        assert not any(step["step"] == "release_readiness" for step in report["steps"])
    else:
        assert vlm_step["returncode"] == 0
        assert report["aborted_after_step"] in {"collaborative_validation", "release_readiness"}


def test_run_collaborative_acceptance_passes_strict_rebuild_flags(tmp_path: Path) -> None:
    import scripts.run_collaborative_acceptance as acceptance_script

    env_file = tmp_path / ".env.collab"
    env_file.write_text(
        "SAR_REQUIRE_GPU=1\n"
        "SAR_REQUIRE_LOCAL_GPU=1\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=0\n"
        "SAR_REQUIRE_VLM=1\n"
        "SAR_VLM_URL=http://vlm.local/v1\n",
        encoding="utf-8",
    )
    output = tmp_path / "acceptance.json"
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
        captured.append(cmd)
        if "check_collaborative_llm.py" in " ".join(cmd):
            return 0, "health ok"
        if "build_original_large_scene_docx.py" in " ".join(cmd):
            return 1, "strict rebuild blocked fake output"
        return 0, "ok"

    old_argv = sys.argv
    with patch.object(acceptance_script, "_run", side_effect=_fake_run):
        try:
            sys.argv = [
                "run_collaborative_acceptance.py",
                "--env-file",
                str(env_file),
                "--rebuild-reports",
                "--output",
                str(output),
            ]
            with pytest.raises(SystemExit):
                acceptance_script.main()
        finally:
            sys.argv = old_argv

    rebuild_cmd = next(cmd for cmd in captured if "build_original_large_scene_docx.py" in " ".join(cmd))
    assert "--strict-release" in rebuild_cmd
    assert "--require-model-participation" in rebuild_cmd
    assert "--require-gpu" in rebuild_cmd
    assert "--require-local-gpu" in rebuild_cmd
    assert "--require-vlm" in rebuild_cmd
    assert "--require-fresh-model-output" in rebuild_cmd

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["aborted_after_step"] == "rebuild_reports"
    assert not any(step["step"] == "collaborative_validation" for step in report["steps"])


def test_summarize_collaborative_routes_script_produces_summary() -> None:
    script = PROJECT_ROOT / "scripts" / "summarize_collaborative_routes.py"
    output = PROJECT_ROOT / "output" / "large_scene_original_format" / "collaborative_route_summary.json"

    subprocess.run(
        [sys.executable, str(script)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert "routes" in summary
    assert "sources" in summary


def test_gpu_small_model_acceptance_script_is_strict() -> None:
    script = PROJECT_ROOT / "scripts" / "run_gpu_small_model_acceptance.sh"
    content = script.read_text(encoding="utf-8")
    assert "preflight_gpu_acceptance.py" in content
    assert "gpu_acceptance_preflight.json" in content
    assert 'REQUIRE_VLM="${SAR_REQUIRE_VLM:-1}"' in content
    assert "nvidia-smi" in content
    assert "torch.cuda.is_available()" in content
    assert "rm -rf output/large_scene_original_format/.report_cache" in content
    assert "--require-model-participation" in content
    assert "--require-gpu" in content
    assert "--require-local-gpu" in content
    assert "--require-acceptance-run-id" in content
    assert "--require-fresh-model-output" in content
    assert "SAR_REQUIRE_VLM" in content
    assert "--require-vlm" in content
    assert "validate_release_readiness.py" in content
    assert "release_readiness_gpu_strict.json" in content


def test_collaborative_example_env_requires_local_gpu() -> None:
    env_file = PROJECT_ROOT / ".env.collaborative.example"
    content = env_file.read_text(encoding="utf-8")
    assert "SAR_REQUIRE_GPU=1" in content
    assert "SAR_REQUIRE_LOCAL_GPU=1" in content
    assert "SAR_ALLOW_TEMPLATE_FALLBACK=0" in content
    assert "SAR_REQUIRE_VLM=1" in content
