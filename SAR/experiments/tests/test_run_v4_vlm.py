from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_v4_test_report as report_cli


def test_run_v4_can_read_vlm_url_from_env_file(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.jpg"
    image_path.write_text("dummy", encoding="utf-8")
    weights_path = tmp_path / "weights.pt"
    weights_path.write_text("dummy", encoding="utf-8")
    env_file = tmp_path / ".env.vlm"
    env_file.write_text("SAR_VLM_URL=http://vlm.from.env/v1\nSAR_VLM_MODEL=vlm-env-model\n", encoding="utf-8")

    original_output = report_cli.OUTPUT_DIR
    report_cli.OUTPUT_DIR = tmp_path / "out"

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url
            self.model_name = model_name

        def describe(self, _image: str) -> str:
            return "图像显示目标区域场景特征明显"

    with patch.object(report_cli, "run_detection", return_value=([], None)), \
         patch.object(report_cli, "assemble_word", return_value=str(tmp_path / "fake.docx")), \
         patch("modules.report.vlm_describer.VLMDescriber", _FakeVLM):
        try:
            sys.argv = [
                "run_v4_test_report.py",
                "--weights", str(weights_path),
                "--image", str(image_path),
                "--region", "测试区域",
                "--env-file", str(env_file),
                "--no-image",
            ]
            report_cli.main()
        finally:
            report_cli.OUTPUT_DIR = original_output

    evidence_files = list((tmp_path / "out").glob("*_evidence.json"))
    assert evidence_files
    evidence = json.loads(evidence_files[0].read_text(encoding="utf-8"))
    assert evidence["scene"]["scene_description"] == "图像显示目标区域场景特征明显"


def test_run_v4_require_vlm_fails_on_empty_description(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.jpg"
    image_path.write_text("dummy", encoding="utf-8")
    weights_path = tmp_path / "weights.pt"
    weights_path.write_text("dummy", encoding="utf-8")
    env_file = tmp_path / ".env.vlm"
    env_file.write_text(
        "SAR_VLM_URL=http://vlm.from.env/v1\n"
        "SAR_REQUIRE_VLM=1\n",
        encoding="utf-8",
    )

    original_output = report_cli.OUTPUT_DIR
    report_cli.OUTPUT_DIR = tmp_path / "out"

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url

        def describe(self, _image: str) -> str:
            return ""

    with patch.object(report_cli, "run_detection", return_value=([], None)), \
         patch.object(report_cli, "assemble_word", return_value=str(tmp_path / "fake.docx")), \
         patch("modules.report.vlm_describer.VLMDescriber", _FakeVLM):
        try:
            sys.argv = [
                "run_v4_test_report.py",
                "--weights", str(weights_path),
                "--image", str(image_path),
                "--region", "测试区域",
                "--env-file", str(env_file),
                "--no-image",
            ]
            try:
                report_cli.main()
            except RuntimeError as exc:
                assert "VLM returned an empty scene description" in str(exc)
            else:
                raise AssertionError("expected RuntimeError")
        finally:
            report_cli.OUTPUT_DIR = original_output


def test_run_v4_require_vlm_cli_fails_when_vlm_url_missing(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.jpg"
    image_path.write_text("dummy", encoding="utf-8")
    weights_path = tmp_path / "weights.pt"
    weights_path.write_text("dummy", encoding="utf-8")
    env_file = tmp_path / ".env.vlm"
    env_file.write_text("SAR_REQUIRE_VLM=0\n", encoding="utf-8")

    original_output = report_cli.OUTPUT_DIR
    report_cli.OUTPUT_DIR = tmp_path / "out"

    with patch.object(report_cli, "run_detection", return_value=([], None)), \
         patch.object(report_cli, "assemble_word", return_value=str(tmp_path / "fake.docx")):
        try:
            sys.argv = [
                "run_v4_test_report.py",
                "--weights", str(weights_path),
                "--image", str(image_path),
                "--region", "测试区域",
                "--env-file", str(env_file),
                "--require-vlm",
                "--no-image",
            ]
            try:
                report_cli.main()
            except RuntimeError as exc:
                assert "SAR_VLM_URL/--vlm-url is empty" in str(exc)
            else:
                raise AssertionError("expected RuntimeError")
        finally:
            report_cli.OUTPUT_DIR = original_output
