from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_docx_validation_and_release_manifest_scripts() -> None:
    validate_script = PROJECT_ROOT / "scripts" / "validate_docx_outputs.py"
    release_script = PROJECT_ROOT / "scripts" / "build_large_scene_release_manifest.py"
    manifest_path = PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    validation_output = PROJECT_ROOT / "output" / "large_scene_original_format" / "docx_validation_report.json"
    release_output = PROJECT_ROOT / "output" / "large_scene_original_format" / "release_manifest.json"

    subprocess.run(
        [sys.executable, str(validate_script), "--manifest", str(manifest_path), "--output", str(validation_output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    validation = json.loads(validation_output.read_text(encoding="utf-8"))
    assert validation["cases"]
    assert all("title_ok" in row for row in validation["cases"])

    subprocess.run(
        [sys.executable, str(release_script), "--input-manifest", str(manifest_path), "--output", str(release_output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    release = json.loads(release_output.read_text(encoding="utf-8"))
    assert release["cases"]
    assert all("case_name" in row and "report_source" in row for row in release["cases"])
    assert "summary" in release
    assert "strict_release_ready_cases" in release["summary"]
    assert all("model_participation_ok" in row for row in release["cases"])
    assert all("fresh_model_output_ok" in row for row in release["cases"])
    assert all("local_gpu_requirement_ok" in row for row in release["cases"])
    assert all("vlm_trace_ok" in row for row in release["cases"])


def test_release_readiness_rejects_current_non_strict_outputs(tmp_path: Path) -> None:
    """A package lacking GPU/VLM traces must be rejected under the strict gate.

    Historically this test pointed at the committed large-scene outputs, which
    were non-strict. Those outputs have since been regenerated WITH full
    GPU+VLM traces, so the gate correctly accepts them now. To keep testing the
    rejection path itself, we build a self-contained non-strict fixture here
    (no GPU flags, no VLM scene-description trace) and assert it is rejected.
    """
    script = PROJECT_ROOT / "scripts" / "validate_release_readiness.py"
    existing_manifest = json.loads(
        (PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    docx_path = existing_manifest[0]["docx_path"]
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "release_readiness.json"

    evidence_path.write_text(
        json.dumps(
            {
                # No scene_description_trace -> VLM requirement fails.
                "scene": {},
                "report_context": {
                    "generation_trace": {
                        "selected_source": "local_llm_v1",
                        "selected_backend": {
                            "role": "local_llm",
                            "source": "local_llm_v1",
                            "mode": "local",
                            "model_path": "/models/qwen3-4b",
                            # No require_gpu / cuda_available -> GPU requirement fails.
                        },
                        "available_backends": {},
                        "attempts": [{"role": "local_llm", "status": "used"}],
                    }
                },
                "report": {
                    "body": "据GF-3卫星侦察，发现舰船2艘。",
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "local_llm_v1"}
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "case_name": "non-strict-case",
                    "docx_path": docx_path,
                    "evidence_path": str(evidence_path),
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--require-model-participation",
            "--require-gpu",
            "--require-vlm",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["ready_cases"] < report["summary"]["total_cases"]
    assert report["summary"]["require_gpu"] is True
    assert report["summary"]["require_vlm"] is True
    assert any(row["gpu_requirement_ok"] is False for row in report["cases"])
    assert any(row["vlm_scene_description_present"] is False for row in report["cases"])


def test_release_readiness_accepts_gpu_vlm_model_trace(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_release_readiness.py"
    existing_manifest = json.loads(
        (PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_docx_path = Path(existing_manifest[0]["docx_path"])
    docx_path = tmp_path / source_docx_path.name
    docx_path.write_bytes(source_docx_path.read_bytes())
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "release_readiness.json"
    scene_description = "图像显示港区内舰船目标集中分布"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "acceptance_run_id": "accept-ready",
                    "generation_trace": {
                        "selected_source": "local_llm_v1",
                        "selected_backend": {
                            "role": "local_llm",
                            "source": "local_llm_v1",
                            "mode": "local",
                            "model_path": "/models/qwen3-4b",
                            "require_gpu": True,
                            "cuda_available": True,
                        },
                        "available_backends": {},
                        "attempts": [{"role": "local_llm", "status": "used"}],
                    }
                },
                "scene": {
                    "scene_description": scene_description,
                    "scene_description_trace": {
                        "source": "vlm_v1",
                        "mode": "api",
                        "model_name": "qwen3-vl-4b",
                        "base_url": "http://127.0.0.1:8101/v1",
                        "status": "used",
                        "description_present": True,
                    },
                },
                "report": {
                    "body": f"据GF-3卫星侦察，发现舰船2艘。{scene_description}，建议持续关注。",
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "local_llm_v1"}
                    ],
                },
                "trace": {"acceptance_run_id": "accept-ready"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    from modules.report.docx_assembler import DocxAssembler

    DocxAssembler.patch_existing_docx(str(docx_path), json.loads(evidence_path.read_text(encoding="utf-8")), draft_mode=True)
    manifest_path.write_text(
        json.dumps(
            [
                    {
                        "case_name": "ready-case",
                        "docx_path": str(docx_path),
                        "evidence_path": str(evidence_path),
                        "acceptance_run_id": "accept-ready",
                    }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--require-model-participation",
            "--require-gpu",
            "--require-local-gpu",
            "--require-vlm",
            "--require-vlm-trace",
            "--require-acceptance-run-id",
            "accept-ready",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["ready_cases"] == 1
    assert report["summary"]["require_acceptance_run_id"] == "accept-ready"
    assert report["cases"][0]["ready"] is True


def test_release_readiness_rejects_api_backend_for_local_gpu_gate(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_release_readiness.py"
    existing_manifest = json.loads(
        (PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    docx_path = existing_manifest[0]["docx_path"]
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "release_readiness.json"

    evidence_path.write_text(
        json.dumps(
            {
                "scene": {},
                "report_context": {
                    "generation_trace": {
                        "selected_source": "llm_v1",
                        "selected_backend": {
                            "role": "llm",
                            "source": "llm_v1",
                            "mode": "api",
                            "model_name": "api-model",
                            "base_url": "http://127.0.0.1:8100/v1",
                        },
                        "available_backends": {},
                        "attempts": [{"role": "llm", "status": "used"}],
                    }
                },
                "report": {
                    "body": "据GF-3卫星侦察，发现舰船2艘。",
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "llm_v1"}
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "case_name": "api-case",
                    "docx_path": docx_path,
                    "evidence_path": str(evidence_path),
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--require-model-participation",
            "--require-gpu",
            "--require-local-gpu",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"][0]["gpu_requirement_ok"] is True
    assert report["cases"][0]["local_gpu_requirement_ok"] is False
    assert report["summary"]["ready_cases"] == 0


def test_release_readiness_rejects_local_gpu_verified_without_cuda_fields(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_release_readiness.py"
    existing_manifest = json.loads(
        (PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    docx_path = existing_manifest[0]["docx_path"]
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "release_readiness.json"

    evidence_path.write_text(
        json.dumps(
            {
                "scene": {},
                "report_context": {
                    "generation_trace": {
                        "selected_source": "local_llm_v1",
                        "selected_backend": {
                            "role": "local_llm",
                            "source": "local_llm_v1",
                            "mode": "local",
                            "model_path": "/models/qwen3-4b",
                            "require_gpu": False,
                            "cuda_available": False,
                            "local_gpu_verified": True,
                        },
                        "available_backends": {},
                        "attempts": [{"role": "local_llm", "status": "used"}],
                    }
                },
                "report": {
                    "body": "据GF-3卫星侦察，发现舰船2艘。",
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "local_llm_v1"}
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            [{"case_name": "forged-local-gpu", "docx_path": docx_path, "evidence_path": str(evidence_path)}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--require-model-participation",
            "--require-local-gpu",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"][0]["model_participation_ok"] is True
    assert report["cases"][0]["local_gpu_requirement_ok"] is False
    assert report["summary"]["ready_cases"] == 0


def test_release_readiness_rejects_missing_vlm_trace(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_release_readiness.py"
    existing_manifest = json.loads(
        (PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    docx_path = existing_manifest[0]["docx_path"]
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "release_readiness.json"
    scene_description = "图像显示港区内舰船目标集中分布"

    evidence_path.write_text(
        json.dumps(
            {
                "scene": {"scene_description": scene_description},
                "report_context": {
                    "generation_trace": {
                        "selected_source": "local_llm_v1",
                        "selected_backend": {
                            "role": "local_llm",
                            "source": "local_llm_v1",
                            "mode": "local",
                            "model_path": "/models/qwen3-4b",
                            "require_gpu": True,
                            "cuda_available": True,
                            "local_gpu_verified": True,
                        },
                        "available_backends": {},
                        "attempts": [{"role": "local_llm", "status": "used"}],
                    }
                },
                "report": {
                    "body": f"据GF-3卫星侦察，发现舰船2艘。{scene_description}，建议持续关注。",
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "local_llm_v1"}
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            [{"case_name": "missing-vlm-trace", "docx_path": docx_path, "evidence_path": str(evidence_path)}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--require-model-participation",
            "--require-gpu",
            "--require-local-gpu",
            "--require-vlm",
            "--require-vlm-trace",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"][0]["vlm_trace_ok"] is False


def test_release_readiness_rejects_source_without_selected_backend(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_release_readiness.py"
    existing_manifest = json.loads(
        (PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    docx_path = existing_manifest[0]["docx_path"]
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "release_readiness.json"

    evidence_path.write_text(
        json.dumps(
            {
                "scene": {},
                "report_context": {
                    "generation_trace": {
                        "selected_source": "small_llm_v1",
                        "available_backends": {
                            "small_llm": {
                                "mode": "api",
                                "model_name": "small-api",
                                "base_url": None,
                            }
                        },
                        "attempts": [{"role": "small_llm", "status": "used"}],
                    }
                },
                "report": {
                    "body": "据GF-3卫星侦察，发现舰船2艘。",
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "small_llm_v1"}
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            [{"case_name": "fake-llm-source", "docx_path": docx_path, "evidence_path": str(evidence_path)}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--require-model-participation",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"][0]["model_participation_ok"] is False
    assert "selected_backend is missing" in report["cases"][0]["model_participation_failures"]
    assert report["cases"][0]["fresh_model_output_ok"] is False


def test_release_readiness_fresh_gate_rejects_missing_selected_backend(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_release_readiness.py"
    existing_manifest = json.loads(
        (PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    docx_path = existing_manifest[0]["docx_path"]
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "release_readiness.json"

    evidence_path.write_text(
        json.dumps(
            {
                "scene": {},
                "report_context": {
                    "generation_trace": {
                        "selected_source": "small_llm_v1",
                        "available_backends": {
                            "small_llm": {
                                "mode": "api",
                                "model_name": "small-api",
                                "base_url": None,
                            }
                        },
                        "attempts": [{"role": "small_llm", "status": "used"}],
                    }
                },
                "report": {
                    "body": "据GF-3卫星侦察，发现舰船2艘。",
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "small_llm_v1"}
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            [{"case_name": "missing-backend", "docx_path": docx_path, "evidence_path": str(evidence_path)}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--require-fresh-model-output",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"][0]["fresh_model_output_ok"] is False
    assert report["summary"]["fresh_model_output_cases"] == 0


def test_release_readiness_rejects_mismatched_acceptance_run_id(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_release_readiness.py"
    existing_manifest = json.loads(
        (PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    docx_path = existing_manifest[0]["docx_path"]
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "release_readiness.json"
    scene_description = "图像显示港区内舰船目标集中分布"

    evidence_path.write_text(
        json.dumps(
            {
                "trace": {"acceptance_run_id": "accept-old"},
                "report_context": {
                    "acceptance_run_id": "accept-old",
                    "generation_trace": {
                        "selected_source": "local_llm_v1",
                        "selected_backend": {
                            "role": "local_llm",
                            "source": "local_llm_v1",
                            "mode": "local",
                            "model_path": "/models/qwen3-4b",
                            "require_gpu": True,
                            "cuda_available": True,
                            "local_gpu_verified": True,
                        },
                        "available_backends": {},
                        "attempts": [{"role": "local_llm", "status": "used"}],
                    },
                },
                "scene": {
                    "scene_description": scene_description,
                    "scene_description_trace": {
                        "source": "vlm_v1",
                        "mode": "api",
                        "model_name": "qwen3-vl-4b",
                        "base_url": "http://127.0.0.1:8101/v1",
                        "status": "used",
                        "description_present": True,
                    },
                },
                "report": {
                    "body": f"据GF-3卫星侦察，发现舰船2艘。{scene_description}，建议持续关注。",
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "local_llm_v1"}
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "case_name": "stale-run",
                    "docx_path": docx_path,
                    "evidence_path": str(evidence_path),
                    "acceptance_run_id": "accept-old",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--require-model-participation",
            "--require-gpu",
            "--require-local-gpu",
            "--require-vlm",
            "--require-vlm-trace",
            "--require-acceptance-run-id",
            "accept-new",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"][0]["acceptance_run_id_ok"] is False
    assert report["summary"]["ready_cases"] == 0


def test_release_readiness_rejects_acceptance_run_id_missing_from_docx(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_release_readiness.py"
    existing_manifest = json.loads(
        (PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    docx_path = existing_manifest[0]["docx_path"]
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "release_readiness.json"
    scene_description = "图像显示港区内舰船目标集中分布"

    evidence_path.write_text(
        json.dumps(
            {
                "trace": {"acceptance_run_id": "accept-docx-missing"},
                "report_context": {
                    "acceptance_run_id": "accept-docx-missing",
                    "generation_trace": {
                        "selected_source": "local_llm_v1",
                        "selected_backend": {
                            "role": "local_llm",
                            "source": "local_llm_v1",
                            "mode": "local",
                            "model_path": "/models/qwen3-4b",
                            "require_gpu": True,
                            "cuda_available": True,
                            "local_gpu_verified": True,
                        },
                        "available_backends": {},
                        "attempts": [{"role": "local_llm", "status": "used"}],
                    },
                },
                "scene": {
                    "scene_description": scene_description,
                    "scene_description_trace": {
                        "source": "vlm_v1",
                        "mode": "api",
                        "model_name": "qwen3-vl-4b",
                        "base_url": "http://127.0.0.1:8101/v1",
                        "status": "used",
                        "description_present": True,
                    },
                },
                "report": {
                    "body": f"据GF-3卫星侦察，发现舰船2艘。{scene_description}，建议持续关注。",
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "local_llm_v1"}
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "case_name": "docx-missing-run-id",
                    "docx_path": docx_path,
                    "evidence_path": str(evidence_path),
                    "acceptance_run_id": "accept-docx-missing",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--require-model-participation",
            "--require-gpu",
            "--require-local-gpu",
            "--require-vlm",
            "--require-vlm-trace",
            "--require-acceptance-run-id",
            "accept-docx-missing",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"][0]["acceptance_run_id_ok"] is True
    assert report["cases"][0]["docx_acceptance_run_id_ok"] is False


def test_release_readiness_rejects_cache_when_fresh_required(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_release_readiness.py"
    existing_manifest = json.loads(
        (PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    docx_path = existing_manifest[0]["docx_path"]
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "release_readiness.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_trace": {
                        "selected_source": "small_llm_cache_v1",
                        "selected_backend": {
                            "role": "small_llm",
                            "source": "small_llm_cache_v1",
                            "selection_status": "cache_hit",
                            "mode": "local",
                            "model_path": "/models/qwen",
                            "require_gpu": True,
                            "cuda_available": True,
                            "local_gpu_verified": True,
                        },
                        "available_backends": {},
                        "attempts": [{"role": "small_llm", "status": "cache_hit"}],
                    }
                },
                "report": {
                    "body": "据GF-3卫星侦察，发现舰船2艘。",
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "small_llm_cache_v1"}
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            [{"case_name": "cache-hit", "docx_path": docx_path, "evidence_path": str(evidence_path)}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--require-model-participation",
            "--require-gpu",
            "--require-local-gpu",
            "--require-fresh-model-output",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"][0]["model_participation_ok"] is True
    assert report["cases"][0]["fresh_model_output_ok"] is False
    assert report["summary"]["fresh_model_output_cases"] == 0
