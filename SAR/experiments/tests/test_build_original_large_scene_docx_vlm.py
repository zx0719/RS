from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.build_original_large_scene_docx as build_docx_script


def test_build_original_large_scene_docx_can_inject_scene_description(tmp_path: Path) -> None:
    source_root = tmp_path / "large_scene"
    source_root.mkdir(parents=True)
    summary = {
        "cases": [
            {
                "case_name": "ship2_large_tiff",
            }
        ]
    }
    case_dir = source_root / "ship2_large_tiff"
    case_dir.mkdir()
    (source_root / "full_run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    src_evidence = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json"
    src_overview = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    shutil.copy2(src_evidence, case_dir / src_evidence.name)
    shutil.copy2(src_overview, case_dir / src_overview.name)

    output_root = tmp_path / "large_scene_original_format"

    class _FakeDocxAssembler:
        TEMPLATE_VERSION = "brief-template-v2"

        def assemble(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True, include_image=True):
            out = Path(output_dir) / "fake.docx"
            out.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(out)

    class _FakeGenerator:
        def generate(self, package: dict) -> dict:
            package["report"] = {
                "body": "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "template_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            package["status"] = "REPORT_DRAFTED"
            return package

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url

        def describe(self, _image: str) -> str:
            return "图像显示锚地水域内目标分布较为分散"

    old_source = build_docx_script.SOURCE_ROOT
    old_output = build_docx_script.OUTPUT_ROOT
    build_docx_script.SOURCE_ROOT = source_root
    build_docx_script.OUTPUT_ROOT = output_root

    env = dict(os.environ)
    env["SAR_VLM_URL"] = "http://vlm.local/v1"
    env["SAR_VLM_MODEL"] = "vlm-model"

    old_argv = sys.argv
    with patch.dict(os.environ, env, clear=False), \
         patch.object(build_docx_script, "DocxAssembler", _FakeDocxAssembler), \
         patch.object(build_docx_script, "CollaborativeReportGenerator", return_value=_FakeGenerator()), \
         patch.object(build_docx_script, "VLMDescriber", _FakeVLM):
        try:
            sys.argv = ["build_original_large_scene_docx.py"]
            build_docx_script.main()
        finally:
            sys.argv = old_argv
            build_docx_script.SOURCE_ROOT = old_source
            build_docx_script.OUTPUT_ROOT = old_output

    updated = json.loads((output_root / "ship2_large_tiff" / "ship2_large_tiff_original_format_evidence.json").read_text(encoding="utf-8"))
    assert updated["scene"]["scene_description"] == "图像显示锚地水域内目标分布较为分散"


def test_build_original_large_scene_docx_uses_process_vlm_url_when_env_file_value_is_empty(tmp_path: Path) -> None:
    source_root = tmp_path / "large_scene"
    source_root.mkdir(parents=True)
    summary = {"cases": [{"case_name": "ship2_large_tiff"}]}
    case_dir = source_root / "ship2_large_tiff"
    case_dir.mkdir()
    (source_root / "full_run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    src_evidence = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json"
    src_overview = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    shutil.copy2(src_evidence, case_dir / src_evidence.name)
    shutil.copy2(src_overview, case_dir / src_overview.name)

    output_root = tmp_path / "large_scene_original_format"
    env_file = tmp_path / ".env.collab"
    env_file.write_text("SAR_VLM_URL=\nSAR_REQUIRE_VLM=1\n", encoding="utf-8")
    seen: dict[str, object] = {}

    class _FakeDocxAssembler:
        TEMPLATE_VERSION = "brief-template-v2"

        def assemble(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True, include_image=True):
            out = Path(output_dir) / "fake.docx"
            out.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(out)

    class _FakeGenerator:
        def generate(self, package: dict) -> dict:
            package["report"] = {
                "body": "据GF-3卫星侦察，发现舰船1艘。图像显示港区内目标分布较密集。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            package["status"] = "REPORT_DRAFTED"
            return package

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            seen["base_url"] = base_url
            self.base_url = base_url
            self.model_name = model_name

        def describe(self, _image: str) -> str:
            return "图像显示港区内目标分布较密集"

    old_source = build_docx_script.SOURCE_ROOT
    old_output = build_docx_script.OUTPUT_ROOT
    build_docx_script.SOURCE_ROOT = source_root
    build_docx_script.OUTPUT_ROOT = output_root

    env = dict(os.environ)
    env["SAR_VLM_URL"] = "http://vlm.from.process/v1"

    old_argv = sys.argv
    with patch.dict(os.environ, env, clear=False), \
         patch.object(build_docx_script, "DocxAssembler", _FakeDocxAssembler), \
         patch.object(build_docx_script, "CollaborativeReportGenerator", return_value=_FakeGenerator()), \
         patch.object(build_docx_script, "VLMDescriber", _FakeVLM):
        try:
            sys.argv = ["build_original_large_scene_docx.py", str(env_file)]
            build_docx_script.main()
        finally:
            sys.argv = old_argv
            build_docx_script.SOURCE_ROOT = old_source
            build_docx_script.OUTPUT_ROOT = old_output

    assert seen["base_url"] == "http://vlm.from.process/v1"


def test_build_original_large_scene_docx_stamps_acceptance_run_id(tmp_path: Path) -> None:
    source_root = tmp_path / "large_scene"
    source_root.mkdir(parents=True)
    summary = {"cases": [{"case_name": "ship2_large_tiff"}]}
    case_dir = source_root / "ship2_large_tiff"
    case_dir.mkdir()
    (source_root / "full_run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    src_evidence = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json"
    src_overview = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    shutil.copy2(src_evidence, case_dir / src_evidence.name)
    shutil.copy2(src_overview, case_dir / src_overview.name)

    output_root = tmp_path / "large_scene_original_format"

    class _FakeDocxAssembler:
        TEMPLATE_VERSION = "brief-template-v2"

        def assemble(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True, include_image=True):
            out = Path(output_dir) / "fake.docx"
            out.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(out)

    class _FakeGenerator:
        def generate(self, package: dict) -> dict:
            package["report"] = {
                "body": "据GF-3卫星侦察，发现舰船1艘。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            package["status"] = "REPORT_DRAFTED"
            return package

    old_source = build_docx_script.SOURCE_ROOT
    old_output = build_docx_script.OUTPUT_ROOT
    build_docx_script.SOURCE_ROOT = source_root
    build_docx_script.OUTPUT_ROOT = output_root

    old_argv = sys.argv
    with patch.object(build_docx_script, "DocxAssembler", _FakeDocxAssembler), \
         patch.object(build_docx_script, "CollaborativeReportGenerator", return_value=_FakeGenerator()):
        try:
            sys.argv = [
                "build_original_large_scene_docx.py",
                "--acceptance-run-id",
                "accept-test-run",
            ]
            build_docx_script.main()
        finally:
            sys.argv = old_argv
            build_docx_script.SOURCE_ROOT = old_source
            build_docx_script.OUTPUT_ROOT = old_output

    manifest = json.loads((output_root / "original_format_docx_manifest.json").read_text(encoding="utf-8"))
    updated = json.loads((output_root / "ship2_large_tiff" / "ship2_large_tiff_original_format_evidence.json").read_text(encoding="utf-8"))
    assert manifest[0]["acceptance_run_id"] == "accept-test-run"
    assert updated["trace"]["acceptance_run_id"] == "accept-test-run"
    assert updated["report_context"]["acceptance_run_id"] == "accept-test-run"


def test_build_original_large_scene_docx_require_vlm_fails_on_empty_description(tmp_path: Path) -> None:
    source_root = tmp_path / "large_scene"
    source_root.mkdir(parents=True)
    summary = {"cases": [{"case_name": "ship2_large_tiff"}]}
    case_dir = source_root / "ship2_large_tiff"
    case_dir.mkdir()
    (source_root / "full_run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    src_evidence = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json"
    src_overview = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    shutil.copy2(src_evidence, case_dir / src_evidence.name)
    shutil.copy2(src_overview, case_dir / src_overview.name)

    output_root = tmp_path / "large_scene_original_format"

    class _FakeDocxAssembler:
        TEMPLATE_VERSION = "brief-template-v2"

        def assemble(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True, include_image=True):
            out = Path(output_dir) / "fake.docx"
            out.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(out)

    class _FakeGenerator:
        def generate(self, package: dict) -> dict:
            package["report"] = {
                "body": "据GF-3卫星侦察，发现舰船1艘。图像显示锚地目标分布较分散。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            package["status"] = "REPORT_DRAFTED"
            return package

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url

        def describe(self, _image: str) -> str:
            return ""

    old_source = build_docx_script.SOURCE_ROOT
    old_output = build_docx_script.OUTPUT_ROOT
    build_docx_script.SOURCE_ROOT = source_root
    build_docx_script.OUTPUT_ROOT = output_root

    env = dict(os.environ)
    env["SAR_VLM_URL"] = "http://vlm.local/v1"
    env["SAR_VLM_MODEL"] = "vlm-model"
    env["SAR_REQUIRE_VLM"] = "1"

    old_argv = sys.argv
    with patch.dict(os.environ, env, clear=False), \
         patch.object(build_docx_script, "DocxAssembler", _FakeDocxAssembler), \
         patch.object(build_docx_script, "CollaborativeReportGenerator", return_value=_FakeGenerator()), \
         patch.object(build_docx_script, "VLMDescriber", _FakeVLM):
        try:
            sys.argv = ["build_original_large_scene_docx.py"]
            with pytest.raises(RuntimeError, match="VLM scene description is required"):
                build_docx_script.main()
        finally:
            sys.argv = old_argv
            build_docx_script.SOURCE_ROOT = old_source
            build_docx_script.OUTPUT_ROOT = old_output


def test_build_original_large_scene_docx_strict_fails_before_docx_write(tmp_path: Path) -> None:
    source_root = tmp_path / "large_scene"
    source_root.mkdir(parents=True)
    summary = {"cases": [{"case_name": "ship2_large_tiff"}]}
    case_dir = source_root / "ship2_large_tiff"
    case_dir.mkdir()
    (source_root / "full_run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    src_evidence = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json"
    src_overview = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    shutil.copy2(src_evidence, case_dir / src_evidence.name)
    shutil.copy2(src_overview, case_dir / src_overview.name)

    output_root = tmp_path / "large_scene_original_format"
    assemble_calls: list[bool] = []

    class _FakeDocxAssembler:
        TEMPLATE_VERSION = "brief-template-v2"

        def assemble(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True, include_image=True):
            assemble_calls.append(True)
            out = Path(output_dir) / "fake.docx"
            out.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(out)

    class _FakeGenerator:
        def generate(self, package: dict) -> dict:
            package["report"] = {
                "body": "据GF-3卫星侦察，发现舰船1艘。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            package["report_context"] = {
                "generation_trace": {
                    "selected_source": "small_llm_v1",
                    "attempts": [{"role": "small_llm", "status": "used"}],
                }
            }
            package["status"] = "REPORT_DRAFTED"
            return package

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url
            self.model_name = model_name

        def describe(self, _image: str) -> str:
            return "图像显示锚地目标分布较分散"

    old_source = build_docx_script.SOURCE_ROOT
    old_output = build_docx_script.OUTPUT_ROOT
    build_docx_script.SOURCE_ROOT = source_root
    build_docx_script.OUTPUT_ROOT = output_root

    old_argv = sys.argv
    env = dict(os.environ)
    env["SAR_VLM_URL"] = "http://vlm.local/v1"
    env["SAR_VLM_MODEL"] = "vlm-model"
    with patch.dict(os.environ, env, clear=False), \
         patch.object(build_docx_script, "DocxAssembler", _FakeDocxAssembler), \
         patch.object(build_docx_script, "CollaborativeReportGenerator", return_value=_FakeGenerator()), \
         patch.object(build_docx_script, "VLMDescriber", _FakeVLM):
        try:
            sys.argv = [
                "build_original_large_scene_docx.py",
                "--strict-release",
                "--acceptance-run-id",
                "accept-strict",
            ]
            with pytest.raises(RuntimeError, match="Strict report validation failed before DOCX write"):
                build_docx_script.main()
        finally:
            sys.argv = old_argv
            build_docx_script.SOURCE_ROOT = old_source
            build_docx_script.OUTPUT_ROOT = old_output

    assert assemble_calls == []
    assert not (output_root / "ship2_large_tiff" / "fake.docx").exists()
    assert not (output_root / "ship2_large_tiff" / "ship2_large_tiff_overview_docx.png").exists()
    assert not (output_root / "original_format_docx_manifest.json").exists()


def test_build_original_large_scene_docx_strict_commits_only_after_full_batch_success(tmp_path: Path) -> None:
    source_root = tmp_path / "large_scene"
    source_root.mkdir(parents=True)
    cases = [{"case_name": "case_ok"}, {"case_name": "case_fail"}]
    (source_root / "full_run_summary.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")

    src_evidence = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json"
    src_overview = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    for case in cases:
        case_name = case["case_name"]
        case_dir = source_root / case_name
        case_dir.mkdir()
        evidence = json.loads(src_evidence.read_text(encoding="utf-8"))
        evidence.setdefault("trace", {}).setdefault("large_scene_test", {})["case_name"] = case_name
        (case_dir / f"{case_name}_evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        shutil.copy2(src_overview, case_dir / f"{case_name}_overview.jpg")

    output_root = tmp_path / "large_scene_original_format"
    assemble_calls: list[str] = []

    class _FakeDocxAssembler:
        TEMPLATE_VERSION = "brief-template-v2"

        def assemble(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True, include_image=True):
            case_name = evidence_package["trace"]["large_scene_test"]["case_name"]
            assemble_calls.append(case_name)
            out = Path(output_dir) / f"{case_name}.docx"
            out.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(out)

    class _FakeGenerator:
        def generate(self, package: dict) -> dict:
            case_name = package["trace"]["large_scene_test"]["case_name"]
            package["report"] = {
                "body": "据GF-3卫星侦察，发现舰船1艘。图像显示锚地目标分布较分散。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            trace = {
                "selected_source": "small_llm_v1",
                "available_backends": {},
                "attempts": [{"role": "small_llm", "status": "used"}],
            }
            if case_name == "case_ok":
                trace["selected_backend"] = {
                    "role": "small_llm",
                    "source": "small_llm_v1",
                    "mode": "local",
                    "model_path": "/models/qwen3-4b",
                    "require_gpu": True,
                    "cuda_available": True,
                    "local_gpu_verified": True,
                }
            package.setdefault("report_context", {})["generation_trace"] = trace
            package["status"] = "REPORT_DRAFTED"
            return package

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url
            self.model_name = model_name

        def describe(self, _image: str) -> str:
            return "图像显示锚地目标分布较分散"

    old_source = build_docx_script.SOURCE_ROOT
    old_output = build_docx_script.OUTPUT_ROOT
    build_docx_script.SOURCE_ROOT = source_root
    build_docx_script.OUTPUT_ROOT = output_root

    old_argv = sys.argv
    env = dict(os.environ)
    env["SAR_VLM_URL"] = "http://vlm.local/v1"
    env["SAR_VLM_MODEL"] = "vlm-model"
    with patch.dict(os.environ, env, clear=False), \
         patch.object(build_docx_script, "DocxAssembler", _FakeDocxAssembler), \
         patch.object(build_docx_script, "CollaborativeReportGenerator", return_value=_FakeGenerator()), \
         patch.object(build_docx_script, "VLMDescriber", _FakeVLM):
        try:
            sys.argv = [
                "build_original_large_scene_docx.py",
                "--strict-release",
                "--acceptance-run-id",
                "accept-batch",
            ]
            with pytest.raises(RuntimeError, match="case_fail: selected_backend is missing"):
                build_docx_script.main()
        finally:
            sys.argv = old_argv
            build_docx_script.SOURCE_ROOT = old_source
            build_docx_script.OUTPUT_ROOT = old_output

    assert assemble_calls == ["case_ok"]
    assert not (output_root / "case_ok" / "case_ok.docx").exists()
    assert not (output_root / "case_ok" / "case_ok_original_format_evidence.json").exists()
    assert not (output_root / "case_ok" / "case_ok_overview_docx.png").exists()
    assert not (output_root / "original_format_docx_manifest.json").exists()


def test_build_original_large_scene_docx_strict_success_rewrites_staging_paths(tmp_path: Path) -> None:
    source_root = tmp_path / "large_scene"
    source_root.mkdir(parents=True)
    summary = {"cases": [{"case_name": "case_ready"}]}
    case_dir = source_root / "case_ready"
    case_dir.mkdir()
    (source_root / "full_run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    src_evidence = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json"
    src_overview = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    shutil.copy2(src_evidence, case_dir / "case_ready_evidence.json")
    shutil.copy2(src_overview, case_dir / "case_ready_overview.jpg")

    output_root = tmp_path / "large_scene_original_format"

    class _FakeDocxAssembler:
        TEMPLATE_VERSION = "brief-template-v2"

        def assemble(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True, include_image=True):
            out = Path(output_dir) / "case_ready.docx"
            out.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(out)

    class _FakeGenerator:
        def generate(self, package: dict) -> dict:
            package["report"] = {
                "body": "据GF-3卫星侦察，发现舰船1艘。图像显示锚地目标分布较分散。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            package.setdefault("report_context", {})["generation_trace"] = {
                "selected_source": "small_llm_v1",
                "selected_backend": {
                    "role": "small_llm",
                    "source": "small_llm_v1",
                    "mode": "local",
                    "model_path": "/models/qwen3-4b",
                    "require_gpu": True,
                    "cuda_available": True,
                    "local_gpu_verified": True,
                },
                "available_backends": {},
                "attempts": [{"role": "small_llm", "status": "used"}],
            }
            package["status"] = "REPORT_DRAFTED"
            return package

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url
            self.model_name = model_name

        def describe(self, _image: str) -> str:
            return "图像显示锚地目标分布较分散"

    old_source = build_docx_script.SOURCE_ROOT
    old_output = build_docx_script.OUTPUT_ROOT
    build_docx_script.SOURCE_ROOT = source_root
    build_docx_script.OUTPUT_ROOT = output_root

    old_argv = sys.argv
    env = dict(os.environ)
    env["SAR_VLM_URL"] = "http://vlm.local/v1"
    env["SAR_VLM_MODEL"] = "vlm-model"
    with patch.dict(os.environ, env, clear=False), \
         patch.object(build_docx_script, "DocxAssembler", _FakeDocxAssembler), \
         patch.object(build_docx_script, "CollaborativeReportGenerator", return_value=_FakeGenerator()), \
         patch.object(build_docx_script, "VLMDescriber", _FakeVLM):
        try:
            sys.argv = [
                "build_original_large_scene_docx.py",
                "--strict-release",
                "--acceptance-run-id",
                "accept-ready",
            ]
            build_docx_script.main()
        finally:
            sys.argv = old_argv
            build_docx_script.SOURCE_ROOT = old_source
            build_docx_script.OUTPUT_ROOT = old_output

    manifest = json.loads((output_root / "original_format_docx_manifest.json").read_text(encoding="utf-8"))
    evidence_path = Path(manifest[0]["evidence_path"])
    updated = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert manifest[0]["docx_path"] == str(output_root / "case_ready" / "case_ready.docx")
    assert manifest[0]["evidence_path"] == str(output_root / "case_ready" / "case_ready_original_format_evidence.json")
    assert updated["attachments"]["annotated_image"]["uri"] == f"file://{output_root / 'case_ready' / 'case_ready_overview_docx.png'}"
    assert updated["report"]["docx"]["uri"] == f"file://{output_root / 'case_ready' / 'case_ready.docx'}"
    assert "/tmp/sar_strict_docx_" not in json.dumps(updated, ensure_ascii=False)


def test_build_original_large_scene_docx_strict_appends_vlm_description_when_model_omits_it(tmp_path: Path) -> None:
    source_root = tmp_path / "large_scene"
    source_root.mkdir(parents=True)
    summary = {"cases": [{"case_name": "case_ready"}]}
    case_dir = source_root / "case_ready"
    case_dir.mkdir()
    (source_root / "full_run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    src_evidence = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json"
    src_overview = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    shutil.copy2(src_evidence, case_dir / "case_ready_evidence.json")
    shutil.copy2(src_overview, case_dir / "case_ready_overview.jpg")

    output_root = tmp_path / "large_scene_original_format"

    class _FakeDocxAssembler:
        TEMPLATE_VERSION = "brief-template-v2"

        def assemble(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True, include_image=True):
            out = Path(output_dir) / "case_ready.docx"
            out.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(out)

    class _FakeGenerator:
        def generate(self, package: dict) -> dict:
            package["report"] = {
                "body": "据GF-3卫星侦察，发现舰船1艘。",
                "body_sections": [{"section_name": "summary", "content": "据GF-3卫星侦察，发现舰船1艘。", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            package.setdefault("report_context", {})["generation_trace"] = {
                "selected_source": "small_llm_v1",
                "selected_backend": {
                    "role": "small_llm",
                    "source": "small_llm_v1",
                    "mode": "local",
                    "model_path": "/models/qwen3-4b",
                    "require_gpu": True,
                    "cuda_available": True,
                    "local_gpu_verified": True,
                },
                "available_backends": {},
                "attempts": [{"role": "small_llm", "status": "used"}],
            }
            package["status"] = "REPORT_DRAFTED"
            return package

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url
            self.model_name = model_name

        def describe(self, _image: str) -> str:
            return "图像显示锚地目标分布较分散"

    old_source = build_docx_script.SOURCE_ROOT
    old_output = build_docx_script.OUTPUT_ROOT
    build_docx_script.SOURCE_ROOT = source_root
    build_docx_script.OUTPUT_ROOT = output_root

    old_argv = sys.argv
    env = dict(os.environ)
    env["SAR_VLM_URL"] = "http://vlm.local/v1"
    env["SAR_VLM_MODEL"] = "vlm-model"
    with patch.dict(os.environ, env, clear=False), \
         patch.object(build_docx_script, "DocxAssembler", _FakeDocxAssembler), \
         patch.object(build_docx_script, "CollaborativeReportGenerator", return_value=_FakeGenerator()), \
         patch.object(build_docx_script, "VLMDescriber", _FakeVLM):
        try:
            sys.argv = [
                "build_original_large_scene_docx.py",
                "--strict-release",
                "--acceptance-run-id",
                "accept-ready",
            ]
            build_docx_script.main()
        finally:
            sys.argv = old_argv
            build_docx_script.SOURCE_ROOT = old_source
            build_docx_script.OUTPUT_ROOT = old_output

    manifest = json.loads((output_root / "original_format_docx_manifest.json").read_text(encoding="utf-8"))
    evidence_path = Path(manifest[0]["evidence_path"])
    updated = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert "VLM场景补充显示，图像显示锚地目标分布较分散。" in updated["report"]["body"]
    assert updated["report"]["body_sections"][0]["content"] == updated["report"]["body"]


def test_build_original_large_scene_docx_fails_when_generator_returns_no_body(tmp_path: Path) -> None:
    source_root = tmp_path / "large_scene"
    source_root.mkdir(parents=True)
    summary = {"cases": [{"case_name": "ship2_large_tiff"}]}
    case_dir = source_root / "ship2_large_tiff"
    case_dir.mkdir()
    (source_root / "full_run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    src_evidence = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json"
    src_overview = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    shutil.copy2(src_evidence, case_dir / src_evidence.name)
    shutil.copy2(src_overview, case_dir / src_overview.name)

    output_root = tmp_path / "large_scene_original_format"

    class _FakeGenerator:
        def generate(self, package: dict) -> dict:
            package["report"] = {
                "body_sections": [{"section_name": "summary", "content": "", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            package["status"] = "REPORT_DRAFTED"
            return package

    old_source = build_docx_script.SOURCE_ROOT
    old_output = build_docx_script.OUTPUT_ROOT
    build_docx_script.SOURCE_ROOT = source_root
    build_docx_script.OUTPUT_ROOT = output_root

    old_argv = sys.argv
    with patch.object(build_docx_script, "CollaborativeReportGenerator", return_value=_FakeGenerator()):
        try:
            sys.argv = ["build_original_large_scene_docx.py"]
            with pytest.raises(RuntimeError, match="Report body generation failed"):
                build_docx_script.main()
        finally:
            sys.argv = old_argv
            build_docx_script.SOURCE_ROOT = old_source
            build_docx_script.OUTPUT_ROOT = old_output
