from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_check_collaborative_llm_accepts_vlm_args() -> None:
    script = PROJECT_ROOT / "scripts" / "check_collaborative_llm.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--small-url", "http://small.local/v1",
            "--large-url", "http://large.local/v1",
            "--vlm-url", "http://vlm.local/v1",
            "--small-model", "small-model",
            "--large-model", "large-model",
            "--vlm-model", "vlm-model",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode in (0, 1)


def test_check_collaborative_llm_can_run_with_vlm_only() -> None:
    script = PROJECT_ROOT / "scripts" / "check_collaborative_llm.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--vlm-url", "http://vlm.local/v1",
            "--vlm-model", "vlm-model",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode in (0, 1)


def test_check_collaborative_llm_accepts_local_model_args(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "check_collaborative_llm.py"
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--small-model-path",
            str(local_model),
            "--local-device",
            "cpu",
            "--max-new-tokens",
            "16",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode in (0, 1)
    assert "local_model_exists" in result.stdout


def test_check_collaborative_llm_can_require_gpu(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "check_collaborative_llm.py"
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--small-model-path",
            str(local_model),
            "--require-gpu",
            "--require-model-participation",
            "--max-new-tokens",
            "16",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode in (0, 1)
    assert '"require_gpu": true' in result.stdout


def test_check_collaborative_llm_can_require_vlm() -> None:
    script = PROJECT_ROOT / "scripts" / "check_collaborative_llm.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--require-vlm",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 1
    assert "VLM is required" in result.stderr
    assert '"require_vlm": true' in result.stdout


def test_check_collaborative_llm_reads_strict_flags_from_env_file(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "check_collaborative_llm.py"
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    env_file = tmp_path / ".env.collab"
    env_file.write_text(
        f"SAR_SMALL_LLM_PATH={local_model}\n"
        "SAR_REQUIRE_GPU=1\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--env-file",
            str(env_file),
            "--max-new-tokens",
            "16",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 1
    assert '"require_gpu": true' in result.stdout
    assert str(local_model) in result.stdout


def test_check_collaborative_llm_strict_gpu_exits_before_generator(tmp_path: Path, capsys) -> None:
    import scripts.check_collaborative_llm as check_script

    local_model = tmp_path / "local-model"
    local_model.mkdir()
    env_file = tmp_path / ".env.collab"
    env_file.write_text(
        f"SAR_SMALL_LLM_PATH={local_model}\n"
        "SAR_REQUIRE_GPU=1\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=0\n",
        encoding="utf-8",
    )

    old_argv = sys.argv
    with patch.object(
        check_script,
        "CollaborativeReportGenerator",
        side_effect=AssertionError("generator must not be constructed"),
    ), patch.object(
        check_script,
        "_cuda_status",
        return_value={"import_ok": True, "cuda_available": False, "device_count": 0},
    ):
        try:
            sys.argv = [
                "check_collaborative_llm.py",
                "--env-file",
                str(env_file),
            ]
            with pytest.raises(SystemExit) as exc:
                check_script.main()
        finally:
            sys.argv = old_argv

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "strict_local_gpu_requires_cuda" in captured.out
    assert "probes_skipped" in captured.out


def test_check_collaborative_llm_reads_model_settings_from_env_file(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "check_collaborative_llm.py"
    local_model = tmp_path / "qwen3-4b"
    local_model.mkdir()
    env_file = tmp_path / ".env.collab"
    env_file.write_text(
        f"SAR_SMALL_LLM_PATH={local_model}\n"
        "SAR_SMALL_LLM_MODEL=qwen3-4b-env\n"
        "SAR_LLM_DEVICE=cpu\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=1\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--env-file",
            str(env_file),
            "--max-new-tokens",
            "16",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode in (0, 1)
    assert str(local_model) in result.stdout
    assert "local_model_exists" in result.stdout
