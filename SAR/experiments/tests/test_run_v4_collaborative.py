from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_v4_test_report as report_cli


def test_generate_report_package_supports_collaborative_mode() -> None:
    package = {
        "status": "READY_FOR_NLG",
        "input": {"mission": {"region_name": "测试区域"}},
        "report": {},
        "objects": [],
        "statistics": {
            "totals": {"all_objects": 0, "ships": 0, "aircraft": 0},
            "by_class": [],
            "spatial_summary": {"distribution": "无目标", "cluster_count": 0, "nearest_neighbor_mean_m": 0.0},
            "confidence_summary": {"mean_confidence": None, "low_confidence_count": 0, "review_required_count": 0},
        },
    }

    created: dict[str, object] = {}

    class _FakeCollaborative:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def generate(self, pkg: dict) -> dict:
            pkg = dict(pkg)
            pkg["report"] = {
                "body": "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            pkg["status"] = "REPORT_DRAFTED"
            return pkg

    with patch("modules.report.collaborative.CollaborativeReportGenerator", _FakeCollaborative):
        updated = report_cli.generate_report_package(
            package,
            llm_url="http://small.local/v1",
            large_llm_url="http://large.local/v1",
            use_collaborative_llm=True,
            large_llm_model="large-model",
        )

    assert updated["report"]["body_sections"][0]["source"] == "small_llm_v1"
    assert created["small_base_url"] == "http://small.local/v1"
    assert created["large_base_url"] == "http://large.local/v1"


def test_generate_report_package_reads_strict_collaborative_flags_from_env_file(tmp_path: Path) -> None:
    package = {
        "status": "READY_FOR_NLG",
        "input": {"mission": {"region_name": "测试区域"}},
        "report": {},
        "objects": [],
        "statistics": {
            "totals": {"all_objects": 0, "ships": 0, "aircraft": 0},
            "by_class": [],
            "spatial_summary": {"distribution": "无目标", "cluster_count": 0, "nearest_neighbor_mean_m": 0.0},
            "confidence_summary": {"mean_confidence": None, "low_confidence_count": 0, "review_required_count": 0},
        },
    }
    env_file = tmp_path / ".env.collab"
    model_dir = tmp_path / "qwen3-4b"
    model_dir.mkdir()
    env_file.write_text(
        f"SAR_SMALL_LLM_PATH={model_dir}\n"
        "SAR_REQUIRE_GPU=1\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=0\n",
        encoding="utf-8",
    )

    created: dict[str, object] = {}

    class _FakeCollaborative:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def generate(self, pkg: dict) -> dict:
            pkg = dict(pkg)
            pkg["report"] = {
                "body": "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            pkg["status"] = "REPORT_DRAFTED"
            return pkg

    with patch("modules.report.collaborative.CollaborativeReportGenerator", _FakeCollaborative):
        updated = report_cli.generate_report_package(package, env_file=str(env_file))

    assert updated["report"]["body_sections"][0]["source"] == "small_llm_v1"
    assert str(created["small_model_path"]) == str(model_dir)
    assert created["require_gpu"] is True
    assert created["allow_template_fallback"] is False


def test_generate_report_package_cli_strict_overrides_env_file(tmp_path: Path) -> None:
    package = {
        "status": "READY_FOR_NLG",
        "input": {"mission": {"region_name": "测试区域"}},
        "report": {},
        "objects": [],
        "statistics": {
            "totals": {"all_objects": 0, "ships": 0, "aircraft": 0},
            "by_class": [],
            "spatial_summary": {"distribution": "无目标", "cluster_count": 0, "nearest_neighbor_mean_m": 0.0},
            "confidence_summary": {"mean_confidence": None, "low_confidence_count": 0, "review_required_count": 0},
        },
    }
    env_file = tmp_path / ".env.collab"
    model_dir = tmp_path / "qwen3-4b"
    model_dir.mkdir()
    env_file.write_text(
        f"SAR_SMALL_LLM_PATH={model_dir}\n"
        "SAR_LLM_DEVICE=cpu\n"
        "SAR_LLM_MAX_NEW_TOKENS=1024\n"
        "SAR_REQUIRE_GPU=0\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=1\n",
        encoding="utf-8",
    )

    created: dict[str, object] = {}

    class _FakeCollaborative:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def generate(self, pkg: dict) -> dict:
            pkg = dict(pkg)
            pkg["report"] = {
                "body": "据GF-3卫星2026年5月25日对测试区域实施侦察，未发现目标。",
                "body_sections": [{"section_name": "summary", "content": "x", "source": "small_llm_v1"}],
                "tables": {"component_table": [], "equipment_table": []},
            }
            pkg["status"] = "REPORT_DRAFTED"
            return pkg

    with patch("modules.report.collaborative.CollaborativeReportGenerator", _FakeCollaborative):
        updated = report_cli.generate_report_package(
            package,
            env_file=str(env_file),
            local_device="cuda:0",
            local_max_new_tokens=256,
            require_gpu=True,
            allow_template_fallback=False,
        )

    assert updated["report"]["body_sections"][0]["source"] == "small_llm_v1"
    assert str(created["small_model_path"]) == str(model_dir)
    assert created["local_device"] == "cuda:0"
    assert created["local_max_new_tokens"] == 256
    assert created["require_gpu"] is True
    assert created["allow_template_fallback"] is False
