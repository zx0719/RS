from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.report.docx_assembler import DocxAssembler


def test_patch_existing_docx_updates_body_and_footer(tmp_path: Path) -> None:
    source_docx = PROJECT_ROOT / "output" / "large_scene_original_format" / "ship2_large_tiff" / "sar-20260428-61A68F_draft.docx"
    copied = tmp_path / source_docx.name
    shutil.copy2(source_docx, copied)

    evidence_path = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["report"]["body"] = "据GF-3卫星2025年5月14日对某某军港实施侦察，共发现舰船2艘。"
    evidence["trace"]["pipeline_run_id"] = "pipe-test-docx"
    evidence["trace"]["acceptance_run_id"] = "accept-test-docx"
    evidence["input"]["metadata"]["satellite"] = "GF-3"
    evidence["input"]["metadata"]["sensor"] = "SAR"
    evidence["input"]["metadata"]["acquisition_time"] = "2025-05-14T10:22:31Z"

    DocxAssembler.patch_existing_docx(str(copied), evidence, draft_mode=True)

    with ZipFile(copied, "r") as zf:
        document_xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        footer_xml = "\n".join(
            zf.read(name).decode("utf-8", errors="ignore")
            for name in zf.namelist()
            if name.startswith("word/footer") and name.endswith(".xml")
        )

    assert "共发现舰船2艘" in document_xml
    assert "自动生成草稿，待人工审核" in footer_xml
    assert "流水线：pipe-test-docx" in footer_xml
    assert "验收批次：accept-test-docx" in footer_xml


def test_patch_existing_docx_rejects_missing_body(tmp_path: Path) -> None:
    source_docx = PROJECT_ROOT / "output" / "large_scene_original_format" / "ship2_large_tiff" / "sar-20260428-61A68F_draft.docx"
    copied = tmp_path / source_docx.name
    shutil.copy2(source_docx, copied)

    evidence = {"report": {}, "input": {"metadata": {}}, "trace": {}}

    with pytest.raises(ValueError, match="report.body is required"):
        DocxAssembler.patch_existing_docx(str(copied), evidence, draft_mode=True)


def test_assemble_rejects_missing_body(tmp_path: Path) -> None:
    try:
        import docx  # noqa: F401
    except ImportError:
        pytest.skip("python-docx is not installed")

    evidence = {
        "package_id": "missing-body",
        "status": "REPORT_DRAFTED",
        "input": {
            "metadata": {"acquisition_time": "2026-05-25T00:00:00+00:00"},
            "mission": {"region_name": "测试区域"},
        },
        "report": {},
        "objects": [],
        "attachments": {},
        "trace": {},
    }

    with pytest.raises(ValueError, match="report.body is required"):
        DocxAssembler().assemble(evidence, output_dir=str(tmp_path))
