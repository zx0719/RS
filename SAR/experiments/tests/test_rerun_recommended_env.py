from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_rerun_with_recommended_env_script_produces_second_report(tmp_path: Path) -> None:
    sweep_script = PROJECT_ROOT / "scripts" / "sweep_collaborative_thresholds.py"
    recommend_script = PROJECT_ROOT / "scripts" / "recommend_collaborative_thresholds.py"
    write_env_script = PROJECT_ROOT / "scripts" / "write_recommended_env.py"
    rerun_script = PROJECT_ROOT / "scripts" / "rerun_with_recommended_env.py"

    sweep_output = PROJECT_ROOT / "output" / "large_scene_original_format" / "collaborative_threshold_sweep.json"
    recommendation_output = PROJECT_ROOT / "output" / "large_scene_original_format" / "collaborative_threshold_recommendation.json"
    base_env = tmp_path / ".env.base"
    recommended_env = tmp_path / ".env.collaborative.recommended"
    rerun_output = tmp_path / "collaborative_acceptance_rerun_report.json"
    base_env.write_text(
        "\n".join(
            [
                "SAR_REQUIRE_GPU=0",
                "SAR_REQUIRE_LOCAL_GPU=0",
                "SAR_ALLOW_TEMPLATE_FALLBACK=1",
                "SAR_REQUIRE_VLM=0",
                "SAR_ROUTE_LARGE_REFINE_CLASS_THRESHOLD=2",
                "SAR_ROUTE_LARGE_REFINE_OBJECT_THRESHOLD=24",
                "SAR_ROUTE_LARGE_REFINE_REVIEW_THRESHOLD=4",
                "SAR_ROUTE_LARGE_MODEL_CLUSTER_THRESHOLD=12",
                "SAR_ROUTE_LARGE_MODEL_CLASS_THRESHOLD=3",
                "SAR_ROUTE_LARGE_MODEL_OBJECT_THRESHOLD=120",
                "SAR_ROUTE_LARGE_MODEL_REVIEW_THRESHOLD=10",
                "SAR_ROUTE_LARGE_MODEL_LOW_CONF_RATIO=0.28",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(sweep_script), "--output", str(sweep_output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    subprocess.run(
        [sys.executable, str(recommend_script), "--input", str(sweep_output), "--output", str(recommendation_output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    subprocess.run(
        [
            sys.executable,
            str(write_env_script),
            "--recommendation",
            str(recommendation_output),
            "--base-env",
            str(base_env),
            "--output",
            str(recommended_env),
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    subprocess.run(
        [sys.executable, str(rerun_script), "--env-file", str(recommended_env), "--output", str(rerun_output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    report = json.loads(rerun_output.read_text(encoding="utf-8"))
    assert report["steps"]
    assert any(step["step"] == "recommended_env" for step in report["steps"])
