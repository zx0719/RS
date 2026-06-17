from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_validate_collaborative_outputs_script_emits_batch_report() -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    manifest = PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    output = PROJECT_ROOT / "output" / "large_scene_original_format" / "collaborative_validation_report.json"

    subprocess.run(
        [sys.executable, str(script), "--manifest", str(manifest), "--output", str(output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"]
    assert all("generation_route" in row for row in report["cases"])
    assert all("generation_reason" in row for row in report["cases"])
    assert all("attempts" in row for row in report["cases"])
    assert "summary" in report


def test_validate_collaborative_outputs_can_require_model_participation(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_route": "small_llm",
                    "generation_trace": {
                        "selected_source": "template_v1",
                        "attempts": [{"role": "template", "status": "used"}],
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "template_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "template-case", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-model-participation",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["model_participation_cases"] == 0
    assert report["summary"]["failures"]


def test_validate_collaborative_outputs_can_require_gpu_for_local_model(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_route": "small_llm",
                    "generation_trace": {
                        "selected_source": "small_llm_v1",
                        "available_backends": {
                            "small_llm": {
                                "mode": "local",
                                "model_path": "/models/small",
                                "require_gpu": True,
                                "cuda_available": False,
                            }
                        },
                        "selected_backend": {
                            "role": "small_llm",
                            "source": "small_llm_v1",
                            "mode": "local",
                            "model_path": "/models/small",
                            "require_gpu": True,
                            "cuda_available": False,
                        },
                        "attempts": [{"role": "small_llm", "status": "used"}],
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "small_llm_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "cpu-local-case", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-model-participation",
            "--require-gpu",
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["cases"][0]["has_model_participation"] is True
    assert report["cases"][0]["gpu_requirement_ok"] is False
    assert report["summary"]["failures"]


def test_validate_collaborative_outputs_accepts_gpu_verified_selected_backend(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_route": "small_llm",
                    "generation_trace": {
                        "selected_source": "small_llm_v1",
                        "available_backends": {
                            "small_llm": {
                                "mode": "local",
                                "model_path": "/models/small",
                                "require_gpu": True,
                                "cuda_available": False,
                            }
                        },
                        "selected_backend": {
                            "role": "small_llm",
                            "source": "small_llm_v1",
                            "mode": "local",
                            "model_path": "/models/small",
                            "require_gpu": True,
                            "cuda_available": True,
                        },
                        "attempts": [{"role": "small_llm", "status": "used"}],
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "small_llm_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "gpu-local-case", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-model-participation",
            "--require-gpu",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["cases"][0]["gpu_requirement_ok"] is True
    assert report["summary"]["gpu_requirement_cases"] == 1
    assert report["summary"]["failures"] == []


def test_validate_collaborative_outputs_rejects_api_backend_without_url_for_gpu_gate(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_route": "small_llm",
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
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "small_llm_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "api-without-url", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-model-participation",
            "--require-gpu",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["cases"][0]["gpu_requirement_ok"] is False


def test_validate_collaborative_outputs_accepts_api_backend_with_url_for_gpu_gate(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_route": "small_llm",
                    "generation_trace": {
                        "selected_source": "small_llm_v1",
                        "selected_backend": {
                            "role": "small_llm",
                            "source": "small_llm_v1",
                            "mode": "api",
                            "model_name": "small-api",
                            "base_url": "http://127.0.0.1:8100/v1",
                        },
                        "available_backends": {},
                        "attempts": [{"role": "small_llm", "status": "used"}],
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "small_llm_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "api-with-url", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-model-participation",
            "--require-gpu",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["cases"][0]["gpu_requirement_ok"] is True
    assert report["cases"][0]["has_model_participation"] is True


def test_validate_collaborative_outputs_rejects_api_backend_for_local_gpu_gate(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_route": "small_llm",
                    "generation_trace": {
                        "selected_source": "small_llm_v1",
                        "selected_backend": {
                            "role": "small_llm",
                            "source": "small_llm_v1",
                            "mode": "api",
                            "model_name": "small-api",
                            "base_url": "http://127.0.0.1:8100/v1",
                        },
                        "available_backends": {},
                        "attempts": [{"role": "small_llm", "status": "used"}],
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "small_llm_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "api-with-url", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-model-participation",
            "--require-gpu",
            "--require-local-gpu",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["cases"][0]["gpu_requirement_ok"] is True
    assert report["cases"][0]["local_gpu_requirement_ok"] is False
    assert report["summary"]["local_gpu_requirement_cases"] == 0


def test_validate_collaborative_outputs_accepts_direct_local_llm_gpu_trace(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_route": "direct_llm",
                    "generation_trace": {
                        "selected_source": "local_llm_v1",
                        "selected_backend": {
                            "role": "local_llm",
                            "source": "local_llm_v1",
                            "mode": "local",
                            "model_path": "/models/qwen",
                            "require_gpu": True,
                            "cuda_available": True,
                        },
                        "available_backends": {},
                        "attempts": [{"role": "local_llm", "status": "used"}],
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "local_llm_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "direct-local", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-model-participation",
            "--require-gpu",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["model_participation_cases"] == 1
    assert report["cases"][0]["gpu_requirement_ok"] is True
    assert report["cases"][0]["local_gpu_requirement_ok"] is True


def test_validate_collaborative_outputs_rejects_local_gpu_verified_without_cuda_fields(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_route": "direct_llm",
                    "generation_trace": {
                        "selected_source": "local_llm_v1",
                        "selected_backend": {
                            "role": "local_llm",
                            "source": "local_llm_v1",
                            "mode": "local",
                            "model_path": "/models/qwen",
                            "require_gpu": False,
                            "cuda_available": False,
                            "local_gpu_verified": True,
                        },
                        "available_backends": {},
                        "attempts": [{"role": "local_llm", "status": "used"}],
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "local_llm_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "forged-local-gpu", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-model-participation",
            "--require-local-gpu",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["cases"][0]["has_model_participation"] is True
    assert report["cases"][0]["local_gpu_requirement_ok"] is False
    assert report["summary"]["local_gpu_requirement_cases"] == 0


def test_validate_collaborative_outputs_accepts_direct_api_llm_trace(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_route": "direct_llm",
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
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "llm_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "direct-api", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-model-participation",
            "--require-gpu",
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["model_participation_cases"] == 1
    assert report["cases"][0]["gpu_requirement_ok"] is True


def test_validate_collaborative_outputs_rejects_source_without_selected_backend(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_route": "small_llm",
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
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "small_llm_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "fake-llm-source", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-model-participation",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["model_participation_cases"] == 0
    assert report["cases"][0]["has_model_participation"] is False
    assert "selected_backend is missing" in report["cases"][0]["model_participation_failures"]
    assert report["cases"][0]["fresh_model_output_ok"] is False


def test_validate_collaborative_outputs_fresh_gate_rejects_missing_selected_backend(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

    evidence_path.write_text(
        json.dumps(
            {
                "report_context": {
                    "generation_route": "small_llm",
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
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "small_llm_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "missing-backend", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-fresh-model-output",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["fresh_model_output_cases"] == 0
    assert report["cases"][0]["fresh_model_output_ok"] is False


def test_validate_collaborative_outputs_rejects_mismatched_acceptance_run_id(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

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
                            "model_path": "/models/qwen",
                            "require_gpu": True,
                            "cuda_available": True,
                        },
                        "available_backends": {},
                        "attempts": [{"role": "local_llm", "status": "used"}],
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "local_llm_v1"}
                    ]
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
            str(output_path),
            "--require-model-participation",
            "--require-acceptance-run-id",
            "accept-new",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["model_participation_cases"] == 1
    assert report["summary"]["require_acceptance_run_id"] == "accept-new"
    assert report["cases"][0]["manifest_acceptance_run_id"] == "accept-old"
    assert report["cases"][0]["evidence_acceptance_run_id"] == "accept-old"


def test_validate_collaborative_outputs_rejects_cache_when_fresh_required(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "validate_collaborative_outputs.py"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "validation.json"

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
                        },
                        "available_backends": {},
                        "attempts": [{"role": "small_llm", "status": "cache_hit"}],
                    },
                },
                "report": {
                    "body_sections": [
                        {"section_name": "summary", "content": "x", "source": "small_llm_cache_v1"}
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"case_name": "cache-hit", "evidence_path": str(evidence_path)}]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--require-model-participation",
            "--require-fresh-model-output",
        ],
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["model_participation_cases"] == 1
    assert report["summary"]["fresh_model_output_cases"] == 0
    assert report["cases"][0]["fresh_model_output_ok"] is False
