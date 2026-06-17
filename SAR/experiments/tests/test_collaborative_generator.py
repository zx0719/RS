from __future__ import annotations

import sys
import types
from pathlib import Path
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.report.collaborative import CollaborativeReportGenerator
from modules.report.generator import LocalModelGenerator, ReportGenerator
from tests.test_e2e_smoke import _fresh_mock_pkg


class _BackendStub:
    def __init__(self, name: str, body: str | None = None, error: Exception | None = None) -> None:
        self.model_name = name
        self.base_url = f"http://stub/{name}"
        self._body = body
        self._error = error

    def _call_llm(self, _evidence: dict) -> str:
        if self._error is not None:
            raise self._error
        assert self._body is not None
        return self._body

    def _post_validate(self, body: str, _statistics: dict) -> tuple[bool, str]:
        return (False, "forced invalid") if "无效" in body else (True, "")

    def _template_fallback(self, evidence: dict) -> str:
        return f"模板回退：{evidence['input']['mission']['region_name']}"

    def call_llm_with_prompt(self, _system_prompt: str, _user_prompt: str) -> str:
        return self._call_llm({})


def _large_scene_pkg() -> dict:
    pkg = _fresh_mock_pkg()
    pkg["attachments"]["overview_image"] = {
        "uri": "file:///tmp/overview.jpg",
        "role": "large_scene_overview",
    }
    pkg["trace"]["large_scene_test"] = {
        "case_name": "stage2-case",
        "relative_path": "dummy/path.tif",
        "scene_category": "ship",
        "usage": "test collaborative generator",
        "run_stats": {
            "tile_count": 12,
            "raw_object_count": 2,
            "deduped_object_count": 2,
        },
    }
    return pkg


def test_collaborative_generator_uses_small_model_and_caches(tmp_path: Path) -> None:
    gen = CollaborativeReportGenerator(cache_dir=str(tmp_path))
    gen._small_backend = _BackendStub(
        "small-model",
        body="据GF-3卫星2025年5月14日对某某军港（港口）实施侦察，共发现舰船2艘，其中驱逐舰1艘、护卫舰1艘，目标集中分布于港池北侧码头区域。",
    )
    gen._large_backend = _BackendStub("large-model", body="无效输出")

    result = gen.generate(_large_scene_pkg())
    assert result["report"]["body_sections"][0]["source"] == "small_llm_v1"
    trace = result["report_context"]["generation_trace"]
    assert trace["route"] == "small_llm"
    assert trace["attempts"][0]["role"] == "small_llm"
    assert trace["available_backends"]["small_llm"]["model_name"] == "small-model"
    assert trace["selected_backend"]["role"] == "small_llm"
    assert trace["selected_backend"]["source"] == "small_llm_v1"

    cached = gen.generate(_large_scene_pkg())
    assert cached["report"]["body_sections"][0]["source"] == "small_llm_cache_v1"
    cached_trace = cached["report_context"]["generation_trace"]
    assert cached_trace["selected_backend"]["selection_status"] == "cache_hit"


def test_collaborative_generator_escalates_to_large_model(tmp_path: Path) -> None:
    pkg = _large_scene_pkg()
    pkg["statistics"]["totals"]["all_objects"] = 150
    pkg["statistics"]["totals"]["ships"] = 120
    pkg["statistics"]["totals"]["aircraft"] = 20
    pkg["statistics"]["by_class"] = [
        {"code": "ship", "name_cn": "ship", "count": 120},
        {"code": "aircraft", "name_cn": "aircraft", "count": 20},
        {"code": "tank", "name_cn": "tank", "count": 10},
    ]
    gen = CollaborativeReportGenerator(cache_dir=str(tmp_path))
    gen._small_backend = _BackendStub("small-model", error=RuntimeError("small failed"))
    gen._large_backend = _BackendStub(
        "large-model",
        body="据GF-3卫星2025年5月14日对某某军港（港口）实施侦察，共发现舰船120艘、飞机20架、地面目标10个，目标分布于多个区域。",
    )

    result = gen.generate(pkg)
    assert result["report"]["body_sections"][0]["source"] == "large_llm_v1"
    attempts = result["report_context"]["generation_trace"]["attempts"]
    assert attempts[0]["role"] == "large_llm"
    assert attempts[0]["status"] == "used"
    assert result["report_context"]["generation_trace"]["available_backends"]["large_llm"]["model_name"] == "large-model"


def test_collaborative_generator_falls_back_to_template(tmp_path: Path) -> None:
    gen = CollaborativeReportGenerator(cache_dir=str(tmp_path))
    gen._small_backend = _BackendStub("small-model", error=RuntimeError("small failed"))
    gen._large_backend = _BackendStub("large-model", error=RuntimeError("large failed"))

    result = gen.generate(_large_scene_pkg())
    assert result["report"]["body_sections"][0]["source"] == "template_v1"
    assert "模板回退" in result["report"]["body"]


def test_collaborative_generator_can_disable_template_fallback(tmp_path: Path) -> None:
    gen = CollaborativeReportGenerator(
        cache_dir=str(tmp_path),
        allow_template_fallback=False,
    )
    gen._small_backend = _BackendStub("small-model", error=RuntimeError("small failed"))
    gen._large_backend = _BackendStub("large-model", error=RuntimeError("large failed"))

    with pytest.raises(RuntimeError, match="template fallback is disabled"):
        gen.generate(_large_scene_pkg())


def test_collaborative_generator_propagates_template_fallback_flag(tmp_path: Path) -> None:
    small_model = tmp_path / "small-model"
    large_model = tmp_path / "large-model"
    small_model.mkdir()
    large_model.mkdir()

    gen = CollaborativeReportGenerator(
        small_model_path=str(small_model),
        large_model_path=str(large_model),
        cache_dir=str(tmp_path / "cache"),
        allow_template_fallback=False,
    )

    assert gen.small_backend.allow_template_fallback is False
    assert gen.large_backend is not None
    assert gen.large_backend.allow_template_fallback is False


def test_post_validate_requires_domain_units_for_key_counts(tmp_path: Path) -> None:
    pkg = _large_scene_pkg()
    stats = {
        "totals": {"all_objects": 30, "ships": 18, "aircraft": 6},
        "by_class": [
            {"code": "ship", "name_cn": "ship", "count": 18},
            {"code": "aircraft", "name_cn": "aircraft", "count": 6},
            {"code": "tank", "name_cn": "tank", "count": 6},
        ],
    }
    gen = CollaborativeReportGenerator(cache_dir=str(tmp_path))

    bad_body = "据GF-3卫星侦察，共发现30个目标，其中舰船18个、飞机6个、坦克/装甲车6个。"
    ok, reason = gen.small_backend._post_validate(bad_body, stats)
    assert ok is False
    assert "舰船数量 18" in reason

    good_body = "据GF-3卫星侦察，共发现30个目标，其中舰船18艘、飞机6架、坦克/装甲车6辆。"
    ok, reason = gen.small_backend._post_validate(good_body, stats)
    assert ok is True
    assert reason == ""


def test_collaborative_generator_uses_large_model_to_refine_small_draft(tmp_path: Path) -> None:
    pkg = _large_scene_pkg()
    pkg["statistics"]["totals"]["all_objects"] = 40
    pkg["statistics"]["totals"]["ships"] = 20
    pkg["statistics"]["totals"]["aircraft"] = 10
    pkg["statistics"]["by_class"] = [
        {"code": "ship", "name_cn": "ship", "count": 20},
        {"code": "aircraft", "name_cn": "aircraft", "count": 10},
        {"code": "tank", "name_cn": "tank", "count": 10},
    ]
    pkg["report_context"] = {"generation_route": "large_refine"}

    gen = CollaborativeReportGenerator(cache_dir=str(tmp_path))
    gen._small_backend = _BackendStub(
        "small-model",
        body="据GF-3卫星2025年5月14日对某某军港实施侦察，共发现舰船20艘、飞机10架、地面目标10个，目标分布于多个区域。",
    )
    gen._large_backend = _BackendStub(
        "large-model",
        body="据GF-3卫星2025年5月14日对某某军港（港口）实施侦察，共发现舰船20艘、飞机10架、地面目标10个。各目标分布于多个区域，建议持续关注。",
    )

    result = gen.generate(pkg)
    assert result["report"]["body_sections"][0]["source"] == "large_llm_refine_v1"
    attempts = result["report_context"]["generation_trace"]["attempts"]
    assert any(item["role"] == "large_refine" and item["status"] == "used" for item in attempts)


def test_collaborative_generator_health_status_reports_disabled_backends(tmp_path: Path) -> None:
    gen = CollaborativeReportGenerator(cache_dir=str(tmp_path))
    status = gen.health_status()
    assert status["small_llm"]["healthy"] is False
    assert status["small_llm"]["reason"] == "disabled"
    assert status["large_llm"] is None


def test_collaborative_generator_health_status_reports_local_backends(tmp_path: Path) -> None:
    small_model = tmp_path / "small-model"
    large_model = tmp_path / "large-model"
    small_model.mkdir()
    large_model.mkdir()
    gen = CollaborativeReportGenerator(
        small_model_path=str(small_model),
        large_model_path=str(large_model),
        cache_dir=str(tmp_path / "cache"),
    )
    status = gen.health_status()
    assert status["small_llm"]["healthy"] is True
    assert status["small_llm"]["reason"] == "local_model_exists"
    assert status["large_llm"]["healthy"] is True


def test_collaborative_generator_health_status_can_require_gpu(tmp_path: Path) -> None:
    small_model = tmp_path / "small-model"
    small_model.mkdir()
    gen = CollaborativeReportGenerator(
        small_model_path=str(small_model),
        cache_dir=str(tmp_path / "cache"),
        require_gpu=True,
    )
    status = gen.health_status()
    assert status["small_llm"]["require_gpu"] is True
    assert "cuda_available" in status["small_llm"]
    if status["small_llm"]["cuda_available"] is False:
        assert status["small_llm"]["healthy"] is False
        assert status["small_llm"]["reason"] == "cuda_not_visible"


def test_collaborative_backend_metadata_marks_local_gpu_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "small-model"
    model_dir.mkdir()
    gen = CollaborativeReportGenerator(
        small_model_path=str(model_dir),
        cache_dir=str(tmp_path / "cache"),
        require_gpu=True,
    )
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backends = gen.describe_backends()

    assert backends["small_llm"]["mode"] == "local"
    assert backends["small_llm"]["local_gpu_verified"] is True


def test_template_fallback_can_include_scene_description(tmp_path: Path) -> None:
    pkg = _large_scene_pkg()
    pkg["scene"]["scene_description"] = "图像显示港口水域内目标较为密集"
    gen = CollaborativeReportGenerator(cache_dir=str(tmp_path))
    real_small = gen.small_backend
    real_small.base_url = "http://stub/small"
    gen._small_backend = real_small
    gen._small_backend._call_llm = lambda _pkg: (_ for _ in ()).throw(RuntimeError("small failed"))  # type: ignore[method-assign]
    gen._large_backend = _BackendStub("large-model", error=RuntimeError("large failed"))

    result = gen.generate(pkg)
    assert "图像显示港口水域内目标较为密集" in result["report"]["body"]


def test_report_generator_can_disable_template_fallback() -> None:
    pkg = _large_scene_pkg()
    gen = ReportGenerator(
        model_name="small-model",
        base_url="http://stub/small",
        allow_template_fallback=False,
    )
    gen._call_llm = lambda _pkg: (_ for _ in ()).throw(RuntimeError("api failed"))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="template fallback is disabled"):
        gen.generate(pkg)


def test_report_generator_records_direct_llm_backend_trace() -> None:
    pkg = _large_scene_pkg()
    gen = ReportGenerator(model_name="small-model", base_url="http://stub/small")
    gen._call_llm = lambda _pkg: (  # type: ignore[method-assign]
        "据GF-3卫星2025年5月14日对某某军港实施侦察，共发现舰船2艘，其中驱逐舰1艘、护卫舰1艘，目标集中分布于港池北侧码头区域。"
    )

    result = gen.generate(pkg)

    assert result["report"]["body_sections"][0]["source"] == "llm_v1"
    trace = result["report_context"]["generation_trace"]
    assert trace["selected_source"] == "llm_v1"
    assert trace["selected_backend"]["role"] == "llm"
    assert trace["selected_backend"]["mode"] == "api"
    assert trace["selected_backend"]["base_url"] == "http://stub/small"
    assert trace["available_backends"]["llm"]["model_name"] == "small-model"


def test_local_model_generator_can_disable_template_fallback() -> None:
    pkg = _large_scene_pkg()
    gen = LocalModelGenerator(
        model_path="/tmp/nonexistent-model",
        allow_template_fallback=False,
    )
    gen._call_llm = lambda _pkg: (_ for _ in ()).throw(RuntimeError("local failed"))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="template fallback is disabled"):
        gen.generate(pkg)


def test_local_model_generator_records_direct_local_backend_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _large_scene_pkg()
    gen = LocalModelGenerator(
        model_path="/tmp/nonexistent-model",
        require_gpu=True,
    )
    gen._call_llm = lambda _pkg: (  # type: ignore[method-assign]
        "据GF-3卫星2025年5月14日对某某军港实施侦察，共发现舰船2艘，其中驱逐舰1艘、护卫舰1艘，目标集中分布于港池北侧码头区域。"
    )
    monkeypatch.setattr(
        gen,
        "_backend_trace_metadata",
        lambda: {
            "mode": "local",
            "model_name": gen.model_name,
            "base_url": None,
            "model_path": gen.model_path,
            "model_path_exists": False,
            "require_gpu": True,
            "cuda_available": True,
            "local_gpu_verified": True,
            "allow_template_fallback": True,
        },
    )

    result = gen.generate(pkg)

    assert result["report"]["body_sections"][0]["source"] == "local_llm_v1"
    trace = result["report_context"]["generation_trace"]
    assert trace["selected_source"] == "local_llm_v1"
    assert trace["selected_backend"]["role"] == "local_llm"
    assert trace["selected_backend"]["require_gpu"] is True
    assert trace["selected_backend"]["cuda_available"] is True
    assert trace["selected_backend"]["local_gpu_verified"] is True
    assert trace["available_backends"]["local_llm"]["mode"] == "local"
