from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_validate_vlm_outputs_script_produces_report() -> None:
    script = PROJECT_ROOT / "scripts" / "validate_vlm_outputs.py"
    manifest = PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    output = PROJECT_ROOT / "output" / "large_scene_original_format" / "vlm_validation_report.json"

    subprocess.run(
        [sys.executable, str(script), "--manifest", str(manifest), "--output", str(output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"]
    assert all("scene_description_present" in row for row in report["cases"])
    assert "summary" in report


def _write_vlm_manifest(
    tmp_path: Path,
    scene_description: str | None,
    report_body: str | None = None,
    trace: dict | None = None,
) -> Path:
    evidence = {
        "scene": {},
        "report": {"body": report_body if report_body is not None else scene_description or ""},
    }
    if scene_description is not None:
        evidence["scene"]["scene_description"] = scene_description
    if trace is not None:
        evidence["scene"]["scene_description_trace"] = trace

    evidence_path = tmp_path / "case_a_evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "case_name": "case_a",
                    "evidence_path": str(evidence_path),
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_validate_vlm_outputs_can_require_vlm(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_vlm_outputs.py"
    manifest = _write_vlm_manifest(tmp_path, None)
    output = tmp_path / "vlm_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--require-vlm",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["total_cases"] == 1
    assert report["summary"]["scene_description_cases"] == 0
    assert report["summary"]["body_mentions_scene_description_cases"] == 0
    assert report["summary"]["require_vlm"] is True
    assert report["summary"]["failures"]


def test_validate_vlm_outputs_passes_when_required_description_exists(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_vlm_outputs.py"
    manifest = _write_vlm_manifest(tmp_path, "港口区域目标分布较集中。")
    output = tmp_path / "vlm_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--require-vlm",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["scene_description_cases"] == 1
    assert report["summary"]["body_mentions_scene_description_cases"] == 1
    assert report["summary"]["failures"] == []


def test_validate_vlm_outputs_requires_trace_when_requested(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_vlm_outputs.py"
    manifest = _write_vlm_manifest(tmp_path, "港口区域目标分布较集中。")
    output = tmp_path / "vlm_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--require-vlm",
            "--require-vlm-trace",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["vlm_trace_cases"] == 0
    assert any(failure["type"] == "missing_vlm_trace" for failure in report["summary"]["failures"])


def test_validate_vlm_outputs_accepts_required_trace(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_vlm_outputs.py"
    manifest = _write_vlm_manifest(
        tmp_path,
        "港口区域目标分布较集中。",
        trace={
            "source": "vlm_v1",
            "mode": "api",
            "model_name": "qwen3-vl-4b",
            "base_url": "http://127.0.0.1:8101/v1",
            "status": "used",
            "description_present": True,
        },
    )
    output = tmp_path / "vlm_report.json"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--require-vlm",
            "--require-vlm-trace",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["vlm_trace_cases"] == 1
    assert report["summary"]["failures"] == []


def test_validate_vlm_outputs_requires_body_to_use_description(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_vlm_outputs.py"
    manifest = _write_vlm_manifest(
        tmp_path,
        "港口区域目标分布较集中。",
        report_body="据GF-3卫星侦察，发现舰船2艘。",
    )
    output = tmp_path / "vlm_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--require-vlm",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["scene_description_cases"] == 1
    assert report["summary"]["body_mentions_scene_description_cases"] == 0
    assert any(
        failure["type"] == "vlm_scene_description_not_used_in_body"
        for failure in report["summary"]["failures"]
    )
