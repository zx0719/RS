from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_recommend_collaborative_thresholds_script_produces_recommendation() -> None:
    sweep_script = PROJECT_ROOT / "scripts" / "sweep_collaborative_thresholds.py"
    sweep_output = PROJECT_ROOT / "output" / "large_scene_original_format" / "collaborative_threshold_sweep.json"
    subprocess.run(
        [sys.executable, str(sweep_script), "--refine-object-thresholds", "24", "--refine-review-thresholds", "4", "--large-object-thresholds", "120", "--large-low-conf-ratios", "0.28", "--output", str(sweep_output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    recommend_script = PROJECT_ROOT / "scripts" / "recommend_collaborative_thresholds.py"
    output = PROJECT_ROOT / "output" / "large_scene_original_format" / "collaborative_threshold_recommendation.json"
    subprocess.run(
        [sys.executable, str(recommend_script), "--input", str(sweep_output), "--output", str(output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["best"] is not None
    assert "thresholds" in data["best"]
