from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_threshold_sweep_script_produces_results() -> None:
    script = PROJECT_ROOT / "scripts" / "sweep_collaborative_thresholds.py"
    manifest = PROJECT_ROOT / "output" / "large_scene_original_format" / "original_format_docx_manifest.json"
    output = PROJECT_ROOT / "output" / "large_scene_original_format" / "collaborative_threshold_sweep.json"

    subprocess.run(
        [sys.executable, str(script), "--manifest", str(manifest), "--refine-object-thresholds", "24", "--refine-review-thresholds", "4", "--large-object-thresholds", "120", "--large-low-conf-ratios", "0.28", "--output", str(output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["results"]
    assert "thresholds" in data["results"][0]
    assert "routes" in data["results"][0]
