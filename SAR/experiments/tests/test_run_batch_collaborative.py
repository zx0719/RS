from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_batch_test


def test_run_one_accepts_collaborative_generator_result(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.jpg"
    image_path.write_text("dummy", encoding="utf-8")

    class _FakeDetector:
        def detect(self, _image: str, save_vis=True, vis_dir=None):
            return [], None

    class _FakeAssembler:
        def assemble(self, evidence_package: dict, output_dir: str, draft_mode=True, include_image=True):
            out = Path(output_dir) / "fake.docx"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(evidence_package["report"]["body"], encoding="utf-8")
            return str(out)

    class _FakeGenerator:
        def generate(self, package: dict) -> dict:
            package = dict(package)
            package["report"] = {
                "body": "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            package["status"] = "REPORT_DRAFTED"
            return package

    fake_class_map_module = types.ModuleType("modules.detector.class_map")
    fake_class_map_module.CLASS_MAP = {}

    with patch.dict(sys.modules, {"modules.detector.class_map": fake_class_map_module}), \
         patch("PIL.Image.open") as fake_open:
        fake_open.return_value.__enter__.return_value.size = (256, 256)
        result = run_batch_test.run_one(
            str(image_path),
            "测试区域",
            _FakeDetector(),
            _FakeAssembler(),
            _FakeGenerator(),
        )

    assert result["docx_path"].endswith("fake.docx")
    assert result["timings"]["nlg"] >= 0.0
