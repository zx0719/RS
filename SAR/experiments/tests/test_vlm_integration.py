from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_large_scene_tests.py"
SPEC = importlib.util.spec_from_file_location("run_large_scene_tests_vlm", SCRIPT_PATH)
large_scene_script = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = large_scene_script
SPEC.loader.exec_module(large_scene_script)

import run_pipeline


def test_run_pipeline_injects_scene_description_from_vlm(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.tif"
    image_path.write_text("dummy", encoding="utf-8")
    output_dir = tmp_path / "out"

    class _FakeDetector:
        def detect(self, _image: str):
            return [], None

    class _FakeBuilder:
        def build(self, _image: str, mission: dict, _objects: list):
            return {
                "schema_version": "1.0.0",
                "package_id": "sar-test-vlm",
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

    class _FakePipeline:
        def run(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True):
            evidence_package["report"] = {
                "body": "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "template_v1"}],
                "docx": {"uri": f"file://{Path(output_dir) / 'fake.docx'}"},
            }
            evidence_package["status"] = "DOCX_RENDERED"
            return evidence_package

    class _FakeQualityGate:
        def evaluate(self, package: dict) -> dict:
            package["quality"] = {"review_gate": {"needs_human_review": False}}
            return package

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url

        def describe(self, _image: str) -> str:
            return "图像显示港口水域较为空旷，场景特征明显"

    fake_detector_module = type(sys)("modules.detector.mock_detector")
    fake_detector_module.MockDetector = lambda: _FakeDetector()

    argv = [
        "--image", str(image_path),
        "--region", "测试区域",
        "--output-dir", str(output_dir),
        "--vlm-url", "http://vlm.local/v1",
    ]

    with patch.dict(sys.modules, {"modules.detector.mock_detector": fake_detector_module}), \
         patch("modules.evidence.EvidenceBuilder", return_value=_FakeBuilder()), \
         patch("modules.report.pipeline.ReportPipeline", return_value=_FakePipeline()), \
         patch("modules.report.pipeline.ReportPipeline.from_local_model", return_value=_FakePipeline()), \
         patch("modules.eval.QualityGate", return_value=_FakeQualityGate()), \
         patch("modules.report.vlm_describer.VLMDescriber", _FakeVLM):
        rc = run_pipeline.main(argv)

    assert rc == 0
    evidence = json.loads((output_dir / "sar-test-vlm_evidence.json").read_text(encoding="utf-8"))
    assert evidence["scene"]["scene_description"] == "图像显示港口水域较为空旷，场景特征明显"


def test_run_pipeline_can_read_vlm_url_from_env_file(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.tif"
    image_path.write_text("dummy", encoding="utf-8")
    output_dir = tmp_path / "out"
    env_file = tmp_path / ".env.vlm"
    env_file.write_text("SAR_VLM_URL=http://vlm.from.env/v1\nSAR_VLM_MODEL=vlm-env-model\n", encoding="utf-8")

    class _FakeDetector:
        def detect(self, _image: str):
            return [], None

    class _FakeBuilder:
        def build(self, _image: str, mission: dict, _objects: list):
            return {
                "schema_version": "1.0.0",
                "package_id": "sar-test-vlm-env",
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

    class _FakePipeline:
        def run(self, evidence_package: dict, output_dir: str, template_path=None, draft_mode=True):
            evidence_package["report"] = {
                "body": "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "template_v1"}],
                "docx": {"uri": f"file://{Path(output_dir) / 'fake.docx'}"},
            }
            evidence_package["status"] = "DOCX_RENDERED"
            return evidence_package

    class _FakeQualityGate:
        def evaluate(self, package: dict) -> dict:
            package["quality"] = {"review_gate": {"needs_human_review": False}}
            return package

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url
            self.model_name = model_name

        def describe(self, _image: str) -> str:
            return "图像显示港口水域较为空旷，场景特征明显"

    fake_detector_module = type(sys)("modules.detector.mock_detector")
    fake_detector_module.MockDetector = lambda: _FakeDetector()

    argv = [
        "--image", str(image_path),
        "--region", "测试区域",
        "--output-dir", str(output_dir),
        "--env-file", str(env_file),
    ]

    with patch.dict(sys.modules, {"modules.detector.mock_detector": fake_detector_module}), \
         patch("modules.evidence.EvidenceBuilder", return_value=_FakeBuilder()), \
         patch("modules.report.pipeline.ReportPipeline", return_value=_FakePipeline()), \
         patch("modules.report.pipeline.ReportPipeline.from_local_model", return_value=_FakePipeline()), \
         patch("modules.eval.QualityGate", return_value=_FakeQualityGate()), \
         patch("modules.report.vlm_describer.VLMDescriber", _FakeVLM):
        rc = run_pipeline.main(argv)

    assert rc == 0
    evidence = json.loads((output_dir / "sar-test-vlm-env_evidence.json").read_text(encoding="utf-8"))
    assert evidence["scene"]["scene_description"] == "图像显示港口水域较为空旷，场景特征明显"


def test_run_pipeline_require_vlm_fails_on_empty_description(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.tif"
    image_path.write_text("dummy", encoding="utf-8")
    output_dir = tmp_path / "out"
    env_file = tmp_path / ".env.vlm.strict"
    env_file.write_text(
        "SAR_VLM_URL=http://vlm.from.env/v1\n"
        "SAR_REQUIRE_VLM=1\n",
        encoding="utf-8",
    )

    class _FakeDetector:
        def detect(self, _image: str):
            return [], None

    class _FakeBuilder:
        def build(self, _image: str, mission: dict, _objects: list):
            return {
                "schema_version": "1.0.0",
                "package_id": "sar-test-vlm-required",
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

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url

        def describe(self, _image: str) -> str:
            return ""

    fake_detector_module = type(sys)("modules.detector.mock_detector")
    fake_detector_module.MockDetector = lambda: _FakeDetector()

    argv = [
        "--image", str(image_path),
        "--region", "测试区域",
        "--output-dir", str(output_dir),
        "--env-file", str(env_file),
    ]

    with patch.dict(sys.modules, {"modules.detector.mock_detector": fake_detector_module}), \
         patch("modules.evidence.EvidenceBuilder", return_value=_FakeBuilder()), \
         patch("modules.report.vlm_describer.VLMDescriber", _FakeVLM):
        rc = run_pipeline.main(argv)

    assert rc == 1


def test_run_pipeline_require_vlm_cli_fails_when_vlm_url_missing(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.tif"
    image_path.write_text("dummy", encoding="utf-8")
    output_dir = tmp_path / "out"
    env_file = tmp_path / ".env.vlm.loose"
    env_file.write_text("SAR_REQUIRE_VLM=0\n", encoding="utf-8")

    class _FakeDetector:
        def detect(self, _image: str):
            return [], None

    class _FakeBuilder:
        def build(self, _image: str, mission: dict, _objects: list):
            return {
                "schema_version": "1.0.0",
                "package_id": "sar-test-vlm-cli-required",
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

    fake_detector_module = type(sys)("modules.detector.mock_detector")
    fake_detector_module.MockDetector = lambda: _FakeDetector()

    argv = [
        "--image", str(image_path),
        "--region", "测试区域",
        "--output-dir", str(output_dir),
        "--env-file", str(env_file),
        "--require-vlm",
    ]

    with patch.dict(sys.modules, {"modules.detector.mock_detector": fake_detector_module}), \
         patch("modules.evidence.EvidenceBuilder", return_value=_FakeBuilder()):
        rc = run_pipeline.main(argv)

    assert rc == 1
    assert not (output_dir / "sar-test-vlm-cli-required_evidence.json").exists()


def test_large_scene_build_evidence_can_inject_scene_description(tmp_path: Path) -> None:
    manifest = large_scene_script.load_manifest(
        PROJECT_ROOT / "tests" / "fixtures" / "large_scene_manifest.json"
    )
    case = large_scene_script.select_cases(manifest, suite="smoke", names=["ship2_large_tiff"])[0]
    package = json.loads(
        (PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json").read_text(encoding="utf-8")
    )
    objects = package["objects"]
    overview_path = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    run_stats = package["trace"]["large_scene_test"]["run_stats"]

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url

        def describe(self, _image: str) -> str:
            return "图像显示锚地水域内目标分布较为分散"

    with patch("modules.report.vlm_describer.VLMDescriber", _FakeVLM):
        rebuilt = large_scene_script.build_evidence(case, objects, overview_path, run_stats, vlm_url="http://vlm.local/v1")

    assert rebuilt["scene"]["scene_description"] == "图像显示锚地水域内目标分布较为分散"


def test_large_scene_build_evidence_require_vlm_fails_on_empty_description(tmp_path: Path) -> None:
    manifest = large_scene_script.load_manifest(
        PROJECT_ROOT / "tests" / "fixtures" / "large_scene_manifest.json"
    )
    case = large_scene_script.select_cases(manifest, suite="smoke", names=["ship2_large_tiff"])[0]
    package = json.loads(
        (PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_evidence.json").read_text(encoding="utf-8")
    )
    objects = package["objects"]
    overview_path = PROJECT_ROOT / "output" / "large_scene" / "ship2_large_tiff" / "ship2_large_tiff_overview.jpg"
    run_stats = package["trace"]["large_scene_test"]["run_stats"]
    env_file = tmp_path / ".env.vlm.strict"
    env_file.write_text(
        "SAR_VLM_URL=http://vlm.local/v1\n"
        "SAR_REQUIRE_VLM=1\n",
        encoding="utf-8",
    )

    class _FakeVLM:
        def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None):
            self.base_url = base_url

        def describe(self, _image: str) -> str:
            return ""

    with patch("modules.report.vlm_describer.VLMDescriber", _FakeVLM):
        try:
            large_scene_script.build_evidence(case, objects, overview_path, run_stats, env_file=str(env_file))
        except RuntimeError as exc:
            assert "VLM scene description is required" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
