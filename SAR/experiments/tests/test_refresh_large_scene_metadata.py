from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.refresh_large_scene_metadata as refresh_metadata_script


def test_refresh_large_scene_metadata_script_updates_available_backends(tmp_path: Path) -> None:
    root = tmp_path / "experiments" / "output" / "large_scene_original_format" / "ship2_large_tiff"
    root.mkdir(parents=True)

    source_evidence = PROJECT_ROOT / "output" / "large_scene_original_format" / "ship2_large_tiff" / "ship2_large_tiff_original_format_evidence.json"
    copied_evidence = root / source_evidence.name
    shutil.copy2(source_evidence, copied_evidence)

    manifest_path = tmp_path / "experiments" / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "case_name": "ship2_large_tiff",
                    "docx_path": str(root / "fake.docx"),
                    "evidence_path": str(copied_evidence),
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    script = PROJECT_ROOT / "scripts" / "refresh_large_scene_metadata.py"
    env = dict(os.environ)
    env["SAR_SMALL_LLM_URL"] = "http://small.local/v1"
    env["SAR_SMALL_LLM_MODEL"] = "small-model"
    env["SAR_LARGE_LLM_URL"] = "http://large.local/v1"
    env["SAR_LARGE_LLM_MODEL"] = "large-model"

    subprocess.run(
        [sys.executable, str(script), "--manifest", str(manifest_path)],
        check=True,
        cwd=str(tmp_path / "experiments"),
        env=env,
    )

    updated = json.loads(copied_evidence.read_text(encoding="utf-8"))
    backends = updated["report_context"]["generation_trace"]["available_backends"]
    assert backends["small_llm"]["model_name"] == "small-model"
    assert backends["large_llm"]["model_name"] == "large-model"


def test_refresh_large_scene_metadata_can_inject_scene_description(tmp_path: Path) -> None:
    root = tmp_path / "experiments" / "output" / "large_scene_original_format" / "ship2_large_tiff"
    root.mkdir(parents=True)

    source_evidence = PROJECT_ROOT / "output" / "large_scene_original_format" / "ship2_large_tiff" / "ship2_large_tiff_original_format_evidence.json"
    copied_evidence = root / source_evidence.name
    shutil.copy2(source_evidence, copied_evidence)

    source_overview = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    local_overview = root / source_overview.name
    shutil.copy2(source_overview, local_overview)

    manifest_path = tmp_path / "experiments" / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "case_name": "ship2_large_tiff",
                    "docx_path": str(root / "fake.docx"),
                    "evidence_path": str(copied_evidence),
                    "source_overview": str(local_overview),
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    script = PROJECT_ROOT / "scripts" / "refresh_large_scene_metadata.py"
    env = dict(os.environ)
    env["SAR_VLM_URL"] = "http://vlm.local/v1"
    env["SAR_VLM_MODEL"] = "vlm-model"

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url

        def describe(self, _image: str) -> str:
            return "图像显示锚地水域内目标分布较为分散"

    old_argv = sys.argv
    with patch.dict(os.environ, env, clear=False), \
         patch("modules.report.vlm_describer.VLMDescriber", _FakeVLM):
        try:
            sys.argv = ["refresh_large_scene_metadata.py", "--manifest", str(manifest_path)]
            refresh_metadata_script.main()
        finally:
            sys.argv = old_argv

    updated = json.loads(copied_evidence.read_text(encoding="utf-8"))
    assert updated["scene"]["scene_description"] == "图像显示锚地水域内目标分布较为分散"


def test_refresh_large_scene_metadata_require_vlm_fails_on_empty_description(tmp_path: Path) -> None:
    root = tmp_path / "experiments" / "output" / "large_scene_original_format" / "ship2_large_tiff"
    root.mkdir(parents=True)

    source_evidence = PROJECT_ROOT / "output" / "large_scene_original_format" / "ship2_large_tiff" / "ship2_large_tiff_original_format_evidence.json"
    copied_evidence = root / source_evidence.name
    shutil.copy2(source_evidence, copied_evidence)

    source_overview = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    local_overview = root / source_overview.name
    shutil.copy2(source_overview, local_overview)

    manifest_path = tmp_path / "experiments" / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "case_name": "ship2_large_tiff",
                    "docx_path": str(root / "fake.docx"),
                    "evidence_path": str(copied_evidence),
                    "source_overview": str(local_overview),
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["SAR_VLM_URL"] = "http://vlm.local/v1"
    env["SAR_VLM_MODEL"] = "vlm-model"
    env["SAR_REQUIRE_VLM"] = "1"

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url

        def describe(self, _image: str) -> str:
            return ""

    old_argv = sys.argv
    with patch.dict(os.environ, env, clear=False), \
         patch("modules.report.vlm_describer.VLMDescriber", _FakeVLM):
        try:
            sys.argv = ["refresh_large_scene_metadata.py", "--manifest", str(manifest_path)]
            with pytest.raises(RuntimeError, match="VLM scene description is required"):
                refresh_metadata_script.main()
        finally:
            sys.argv = old_argv
