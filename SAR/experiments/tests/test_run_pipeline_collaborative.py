from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_pipeline
import modules.evidence as evidence_module
import modules.report.collaborative as collaborative_module
import modules.report.pipeline as pipeline_module
import modules.report.docx_assembler as docx_assembler_module
import modules.eval as eval_module


def test_run_pipeline_uses_collaborative_llm_path(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.tif"
    image_path.write_text("dummy", encoding="utf-8")
    output_dir = tmp_path / "out"

    created: dict[str, object] = {}

    class _FakeDetector:
        def detect(self, _image: str):
            return [], None

    class _FakeBuilder:
        def build(self, _image: str, mission: dict, _objects: list):
            return {
                "schema_version": "1.0.0",
                "package_id": "sar-test-collab",
                "task_type": "intel_brief",
                "status": "READY_FOR_NLG",
                "created_at": "2026-05-25T00:00:00+00:00",
                "updated_at": "2026-05-25T00:00:00+00:00",
                "trace": {"pipeline_run_id": "pipe-test"},
                "input": {
                    "input_id": "input",
                    "image": {"uri": f"file://{_image}", "file_name": "scene.tif", "format": "GeoTIFF", "sha256": "x"},
                    "metadata": {"satellite": "GF-3", "sensor": "SAR", "acquisition_time": "2026-05-25T00:00:00+00:00"},
                    "mission": mission,
                },
                "scene": {"scene_type_cn": "港口"},
                "objects": [],
                "statistics": {
                    "totals": {"all_objects": 0, "ships": 0, "aircraft": 0},
                    "by_class": [],
                    "by_super_class": [],
                    "spatial_summary": {"distribution": "无目标", "cluster_count": 0, "nearest_neighbor_mean_m": 0.0},
                    "confidence_summary": {"mean_confidence": None, "low_confidence_count": 0, "review_required_count": 0},
                },
                "attachments": {},
                "report": {},
                "quality": {},
                "errors": [],
            }

    class _FakeCollaborative:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def generate(self, pkg: dict) -> dict:
            pkg = dict(pkg)
            report = dict(pkg.get("report", {}))
            report["body"] = "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。"
            report["body_sections"] = [{"section_name": "summary", "content": report["body"], "source": "small_llm_v1"}]
            report["tables"] = {"component_table": [], "equipment_table": []}
            pkg["report"] = report
            pkg["status"] = "REPORT_DRAFTED"
            return pkg

        def describe_backends(self):
            return {
                "small_llm": {"model_name": created["small_model_name"], "base_url": created["small_base_url"]},
                "large_llm": {"model_name": created["large_model_name"], "base_url": created["large_base_url"]},
            }

    class _FakeAssembler:
        def assemble(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True, include_image=True):
            path = Path(output_dir) / "fake.docx"
            path.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(path)

    class _FakeQualityGate:
        def evaluate(self, package: dict) -> dict:
            package["quality"] = {"review_gate": {"needs_human_review": False}}
            return package

    argv = [
        "--image", str(image_path),
        "--region", "测试区域",
        "--llm-url", "http://small.local/v1",
        "--llm-model-name", "small-model",
        "--large-llm-url", "http://large.local/v1",
        "--large-llm-model-name", "large-model",
        "--use-collaborative-llm",
        "--output-dir", str(output_dir),
    ]

    fake_detector_module = types.ModuleType("modules.detector.mock_detector")
    fake_detector_module.MockDetector = lambda: _FakeDetector()

    called = {"factory": False}

    def _fake_from_collaborative_models(**kwargs):
        called["factory"] = True
        created.update(kwargs)

        class _FakePipeline:
            def __init__(self):
                self._generator = _FakeCollaborative(**kwargs)

            def run(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True):
                pkg = self._generator.generate(evidence_package)
                pkg["report"]["docx"] = {"uri": f"file://{Path(output_dir) / 'fake.docx'}"}
                pkg["status"] = "DOCX_RENDERED"
                return pkg

        return _FakePipeline()

    with patch.dict(sys.modules, {"modules.detector.mock_detector": fake_detector_module}), \
         patch.object(evidence_module, "EvidenceBuilder", return_value=_FakeBuilder()), \
         patch.object(collaborative_module, "CollaborativeReportGenerator", _FakeCollaborative), \
         patch.object(pipeline_module.ReportPipeline, "from_collaborative_models", side_effect=_fake_from_collaborative_models), \
         patch.object(docx_assembler_module, "DocxAssembler", return_value=_FakeAssembler()), \
         patch.object(eval_module, "QualityGate", return_value=_FakeQualityGate()):
        rc = run_pipeline.main(argv)

    assert rc == 0
    assert called["factory"] is True
    assert created["small_base_url"] == "http://small.local/v1"
    assert created["large_base_url"] == "http://large.local/v1"
    assert created["small_model_name"] == "small-model"
    assert created["large_model_name"] == "large-model"
    assert (output_dir / "sar-test-collab_evidence.json").exists()


def test_run_pipeline_can_read_collaborative_urls_from_env_file(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.tif"
    image_path.write_text("dummy", encoding="utf-8")
    output_dir = tmp_path / "out"
    env_file = tmp_path / ".env.collab"
    env_file.write_text(
        "SAR_SMALL_LLM_URL=http://small.from.env/v1\n"
        "SAR_SMALL_LLM_MODEL=small-env-model\n"
        "SAR_LARGE_LLM_URL=http://large.from.env/v1\n"
        "SAR_LARGE_LLM_MODEL=large-env-model\n",
        encoding="utf-8",
    )

    created: dict[str, object] = {}

    class _FakeDetector:
        def detect(self, _image: str):
            return [], None

    class _FakeBuilder:
        def build(self, _image: str, mission: dict, _objects: list):
            return {
                "schema_version": "1.0.0",
                "package_id": "sar-test-envfile",
                "task_type": "intel_brief",
                "status": "READY_FOR_NLG",
                "created_at": "2026-05-25T00:00:00+00:00",
                "updated_at": "2026-05-25T00:00:00+00:00",
                "trace": {"pipeline_run_id": "pipe-test"},
                "input": {
                    "input_id": "input",
                    "image": {"uri": f"file://{_image}", "file_name": "scene.tif", "format": "GeoTIFF", "sha256": "x"},
                    "metadata": {"satellite": "GF-3", "sensor": "SAR", "acquisition_time": "2026-05-25T00:00:00+00:00"},
                    "mission": mission,
                },
                "scene": {"scene_type_cn": "港口"},
                "objects": [],
                "statistics": {
                    "totals": {"all_objects": 0, "ships": 0, "aircraft": 0},
                    "by_class": [],
                    "by_super_class": [],
                    "spatial_summary": {"distribution": "无目标", "cluster_count": 0, "nearest_neighbor_mean_m": 0.0},
                    "confidence_summary": {"mean_confidence": None, "low_confidence_count": 0, "review_required_count": 0},
                },
                "attachments": {},
                "report": {},
                "quality": {},
                "errors": [],
            }

    class _FakeAssembler:
        def assemble(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True, include_image=True):
            path = Path(output_dir) / "fake.docx"
            path.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(path)

    class _FakeQualityGate:
        def evaluate(self, package: dict) -> dict:
            package["quality"] = {"review_gate": {"needs_human_review": False}}
            return package

    def _fake_from_collaborative_models(**kwargs):
        created.update(kwargs)

        class _FakePipeline:
            def run(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True):
                evidence_package["report"] = {
                    "body": "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。",
                    "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                }
                evidence_package["status"] = "DOCX_RENDERED"
                evidence_package["report"]["docx"] = {"uri": f"file://{Path(output_dir) / 'fake.docx'}"}
                return evidence_package

        return _FakePipeline()

    fake_detector_module = types.ModuleType("modules.detector.mock_detector")
    fake_detector_module.MockDetector = lambda: _FakeDetector()

    argv = [
        "--image", str(image_path),
        "--region", "测试区域",
        "--use-collaborative-llm",
        "--env-file", str(env_file),
        "--output-dir", str(output_dir),
    ]

    with patch.dict(sys.modules, {"modules.detector.mock_detector": fake_detector_module}), \
         patch.object(evidence_module, "EvidenceBuilder", return_value=_FakeBuilder()), \
         patch.object(pipeline_module.ReportPipeline, "from_collaborative_models", side_effect=_fake_from_collaborative_models), \
         patch.object(docx_assembler_module, "DocxAssembler", return_value=_FakeAssembler()), \
         patch.object(eval_module, "QualityGate", return_value=_FakeQualityGate()):
        rc = run_pipeline.main(argv)

    assert rc == 0
    assert created["small_base_url"] == "http://small.from.env/v1"
    assert created["large_base_url"] == "http://large.from.env/v1"


def test_run_pipeline_can_use_collaborative_local_model_paths_from_env_file(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.tif"
    image_path.write_text("dummy", encoding="utf-8")
    output_dir = tmp_path / "out"
    env_file = tmp_path / ".env.collab.local"
    small_model = tmp_path / "small-model"
    large_model = tmp_path / "large-model"
    small_model.mkdir()
    large_model.mkdir()
    env_file.write_text(
        f"SAR_SMALL_LLM_PATH={small_model}\n"
        f"SAR_LARGE_LLM_PATH={large_model}\n"
        "SAR_LLM_DEVICE=cpu\n",
        encoding="utf-8",
    )

    created: dict[str, object] = {}

    class _FakeDetector:
        def detect(self, _image: str):
            return [], None

    class _FakeBuilder:
        def build(self, _image: str, mission: dict, _objects: list):
            return {
                "schema_version": "1.0.0",
                "package_id": "sar-test-local-envfile",
                "task_type": "intel_brief",
                "status": "READY_FOR_NLG",
                "created_at": "2026-05-25T00:00:00+00:00",
                "updated_at": "2026-05-25T00:00:00+00:00",
                "trace": {"pipeline_run_id": "pipe-test"},
                "input": {
                    "input_id": "input",
                    "image": {"uri": f"file://{_image}", "file_name": "scene.tif", "format": "GeoTIFF", "sha256": "x"},
                    "metadata": {"satellite": "GF-3", "sensor": "SAR", "acquisition_time": "2026-05-25T00:00:00+00:00"},
                    "mission": mission,
                },
                "scene": {"scene_type_cn": "港口"},
                "objects": [],
                "statistics": {
                    "totals": {"all_objects": 0, "ships": 0, "aircraft": 0},
                    "by_class": [],
                    "by_super_class": [],
                    "spatial_summary": {"distribution": "无目标", "cluster_count": 0, "nearest_neighbor_mean_m": 0.0},
                    "confidence_summary": {"mean_confidence": None, "low_confidence_count": 0, "review_required_count": 0},
                },
                "attachments": {},
                "report": {},
                "quality": {},
                "errors": [],
            }

    class _FakeAssembler:
        def assemble(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True, include_image=True):
            path = Path(output_dir) / "fake.docx"
            path.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(path)

    class _FakeQualityGate:
        def evaluate(self, package: dict) -> dict:
            package["quality"] = {"review_gate": {"needs_human_review": False}}
            return package

    def _fake_from_collaborative_models(**kwargs):
        created.update(kwargs)

        class _FakePipeline:
            def run(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True):
                evidence_package["report"] = {
                    "body": "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。",
                    "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                }
                evidence_package["status"] = "DOCX_RENDERED"
                evidence_package["report"]["docx"] = {"uri": f"file://{Path(output_dir) / 'fake.docx'}"}
                return evidence_package

        return _FakePipeline()

    fake_detector_module = types.ModuleType("modules.detector.mock_detector")
    fake_detector_module.MockDetector = lambda: _FakeDetector()

    argv = [
        "--image", str(image_path),
        "--region", "测试区域",
        "--use-collaborative-llm",
        "--env-file", str(env_file),
        "--output-dir", str(output_dir),
    ]

    with patch.dict(sys.modules, {"modules.detector.mock_detector": fake_detector_module}), \
         patch.object(evidence_module, "EvidenceBuilder", return_value=_FakeBuilder()), \
         patch.object(pipeline_module.ReportPipeline, "from_collaborative_models", side_effect=_fake_from_collaborative_models), \
         patch.object(docx_assembler_module, "DocxAssembler", return_value=_FakeAssembler()), \
         patch.object(eval_module, "QualityGate", return_value=_FakeQualityGate()):
        rc = run_pipeline.main(argv)

    assert rc == 0
    assert str(created["small_model_path"]) == str(small_model)
    assert str(created["large_model_path"]) == str(large_model)
    assert created["local_device"] == "cpu"


def test_run_pipeline_propagates_strict_collaborative_flags_from_env_file(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.tif"
    image_path.write_text("dummy", encoding="utf-8")
    output_dir = tmp_path / "out"
    env_file = tmp_path / ".env.collab.strict"
    small_model = tmp_path / "small-model"
    small_model.mkdir()
    env_file.write_text(
        f"SAR_SMALL_LLM_PATH={small_model}\n"
        "SAR_REQUIRE_GPU=1\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=0\n",
        encoding="utf-8",
    )

    created: dict[str, object] = {}

    class _FakeDetector:
        def detect(self, _image: str):
            return [], None

    class _FakeBuilder:
        def build(self, _image: str, mission: dict, _objects: list):
            return {
                "schema_version": "1.0.0",
                "package_id": "sar-test-strict-envfile",
                "task_type": "intel_brief",
                "status": "READY_FOR_NLG",
                "created_at": "2026-05-25T00:00:00+00:00",
                "updated_at": "2026-05-25T00:00:00+00:00",
                "trace": {"pipeline_run_id": "pipe-test"},
                "input": {
                    "input_id": "input",
                    "image": {"uri": f"file://{_image}", "file_name": "scene.tif", "format": "GeoTIFF", "sha256": "x"},
                    "metadata": {"satellite": "GF-3", "sensor": "SAR", "acquisition_time": "2026-05-25T00:00:00+00:00"},
                    "mission": mission,
                },
                "scene": {"scene_type_cn": "港口"},
                "objects": [],
                "statistics": {
                    "totals": {"all_objects": 0, "ships": 0, "aircraft": 0},
                    "by_class": [],
                    "by_super_class": [],
                    "spatial_summary": {"distribution": "无目标", "cluster_count": 0, "nearest_neighbor_mean_m": 0.0},
                    "confidence_summary": {"mean_confidence": None, "low_confidence_count": 0, "review_required_count": 0},
                },
                "attachments": {},
                "report": {},
                "quality": {},
                "errors": [],
            }

    def _fake_from_collaborative_models(**kwargs):
        created.update(kwargs)

        class _FakePipeline:
            def run(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True):
                evidence_package["report"] = {
                    "body": "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。",
                    "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                    "docx": {"uri": f"file://{Path(output_dir) / 'fake.docx'}"},
                }
                evidence_package["status"] = "DOCX_RENDERED"
                return evidence_package

        return _FakePipeline()

    class _FakeQualityGate:
        def evaluate(self, package: dict) -> dict:
            return package

    fake_detector_module = types.ModuleType("modules.detector.mock_detector")
    fake_detector_module.MockDetector = lambda: _FakeDetector()

    argv = [
        "--image", str(image_path),
        "--region", "测试区域",
        "--env-file", str(env_file),
        "--output-dir", str(output_dir),
    ]

    with patch.dict(sys.modules, {"modules.detector.mock_detector": fake_detector_module}), \
         patch.object(evidence_module, "EvidenceBuilder", return_value=_FakeBuilder()), \
         patch.object(pipeline_module.ReportPipeline, "from_collaborative_models", side_effect=_fake_from_collaborative_models), \
         patch.object(eval_module, "QualityGate", return_value=_FakeQualityGate()):
        rc = run_pipeline.main(argv)

    assert rc == 0
    assert created["require_gpu"] is True
    assert created["allow_template_fallback"] is False
    assert str(created["small_model_path"]) == str(small_model)


def test_run_pipeline_passes_strict_flags_to_direct_local_model(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.tif"
    image_path.write_text("dummy", encoding="utf-8")
    output_dir = tmp_path / "out"
    local_model = tmp_path / "direct-local-model"
    local_model.mkdir()
    env_file = tmp_path / ".env.strict"
    env_file.write_text(
        "SAR_LLM_DEVICE=cpu\n"
        "SAR_LLM_MAX_NEW_TOKENS=1024\n"
        "SAR_REQUIRE_GPU=0\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=1\n",
        encoding="utf-8",
    )

    created: dict[str, object] = {}

    class _FakeDetector:
        def detect(self, _image: str):
            return [], None

    class _FakeBuilder:
        def build(self, _image: str, mission: dict, _objects: list):
            return {
                "schema_version": "1.0.0",
                "package_id": "sar-test-direct-local",
                "task_type": "intel_brief",
                "status": "READY_FOR_NLG",
                "created_at": "2026-05-25T00:00:00+00:00",
                "updated_at": "2026-05-25T00:00:00+00:00",
                "trace": {"pipeline_run_id": "pipe-test"},
                "input": {
                    "input_id": "input",
                    "image": {"uri": f"file://{_image}", "file_name": "scene.tif", "format": "GeoTIFF", "sha256": "x"},
                    "metadata": {"satellite": "GF-3", "sensor": "SAR", "acquisition_time": "2026-05-25T00:00:00+00:00"},
                    "mission": mission,
                },
                "scene": {"scene_type_cn": "港口"},
                "objects": [],
                "statistics": {
                    "totals": {"all_objects": 0, "ships": 0, "aircraft": 0},
                    "by_class": [],
                    "by_super_class": [],
                    "spatial_summary": {"distribution": "无目标", "cluster_count": 0, "nearest_neighbor_mean_m": 0.0},
                    "confidence_summary": {"mean_confidence": None, "low_confidence_count": 0, "review_required_count": 0},
                },
                "attachments": {},
                "report": {},
                "quality": {},
                "errors": [],
            }

    def _fake_from_local_model(model_path: str, **kwargs):
        created["model_path"] = model_path
        created.update(kwargs)

        class _FakePipeline:
            def run(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True):
                evidence_package["report"] = {
                    "body": "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。",
                    "body_sections": [{"section_name": "summary", "content": "x", "source": "local_llm_v1"}],
                    "docx": {"uri": f"file://{Path(output_dir) / 'fake.docx'}"},
                }
                evidence_package["status"] = "DOCX_RENDERED"
                return evidence_package

        return _FakePipeline()

    class _FakeQualityGate:
        def evaluate(self, package: dict) -> dict:
            return package

    fake_detector_module = types.ModuleType("modules.detector.mock_detector")
    fake_detector_module.MockDetector = lambda: _FakeDetector()

    argv = [
        "--image", str(image_path),
        "--region", "测试区域",
        "--llm-path", str(local_model),
        "--env-file", str(env_file),
        "--llm-device", "cuda:0",
        "--llm-max-new-tokens", "256",
        "--require-gpu",
        "--no-template-fallback",
        "--output-dir", str(output_dir),
    ]

    with patch.dict(sys.modules, {"modules.detector.mock_detector": fake_detector_module}), \
         patch.object(evidence_module, "EvidenceBuilder", return_value=_FakeBuilder()), \
         patch.object(pipeline_module.ReportPipeline, "from_local_model", side_effect=_fake_from_local_model), \
         patch.object(eval_module, "QualityGate", return_value=_FakeQualityGate()):
        rc = run_pipeline.main(argv)

    assert rc == 0
    assert created["model_path"] == str(local_model)
    assert created["device"] == "cuda:0"
    assert created["max_new_tokens"] == 256
    assert created["require_gpu"] is True
    assert created["allow_template_fallback"] is False


def test_run_pipeline_fails_when_report_generation_fails(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.tif"
    image_path.write_text("dummy", encoding="utf-8")
    output_dir = tmp_path / "out"
    env_file = tmp_path / ".env.strict"
    env_file.write_text("SAR_ALLOW_TEMPLATE_FALLBACK=0\n", encoding="utf-8")

    class _FakeDetector:
        def detect(self, _image: str):
            return [], None

    class _FakeBuilder:
        def build(self, _image: str, mission: dict, _objects: list):
            return {
                "schema_version": "1.0.0",
                "package_id": "sar-report-failed",
                "task_type": "intel_brief",
                "status": "READY_FOR_NLG",
                "created_at": "2026-05-25T00:00:00+00:00",
                "updated_at": "2026-05-25T00:00:00+00:00",
                "trace": {"pipeline_run_id": "pipe-test"},
                "input": {
                    "input_id": "input",
                    "image": {"uri": f"file://{_image}", "file_name": "scene.tif", "format": "GeoTIFF", "sha256": "x"},
                    "metadata": {"satellite": "GF-3", "sensor": "SAR", "acquisition_time": "2026-05-25T00:00:00+00:00"},
                    "mission": mission,
                },
                "scene": {"scene_type_cn": "港口"},
                "objects": [],
                "statistics": {
                    "totals": {"all_objects": 0, "ships": 0, "aircraft": 0},
                    "by_class": [],
                    "by_super_class": [],
                    "spatial_summary": {"distribution": "无目标", "cluster_count": 0, "nearest_neighbor_mean_m": 0.0},
                    "confidence_summary": {"mean_confidence": None, "low_confidence_count": 0, "review_required_count": 0},
                },
                "attachments": {},
                "report": {},
                "quality": {},
                "errors": [],
            }

    class _FailingPipeline:
        def run(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True):
            raise RuntimeError("no valid LLM output")

    fake_detector_module = types.ModuleType("modules.detector.mock_detector")
    fake_detector_module.MockDetector = lambda: _FakeDetector()

    argv = [
        "--image", str(image_path),
        "--region", "测试区域",
        "--env-file", str(env_file),
        "--output-dir", str(output_dir),
    ]

    with patch.dict(sys.modules, {"modules.detector.mock_detector": fake_detector_module}), \
         patch.object(evidence_module, "EvidenceBuilder", return_value=_FakeBuilder()), \
         patch.object(pipeline_module, "ReportPipeline", return_value=_FailingPipeline()):
        rc = run_pipeline.main(argv)

    assert rc == 1
    assert not (output_dir / "sar-report-failed_evidence.json").exists()
