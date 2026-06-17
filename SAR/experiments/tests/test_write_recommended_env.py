from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_write_recommended_env_script_writes_thresholds(tmp_path: Path) -> None:
    recommendation = tmp_path / "recommendation.json"
    recommendation.write_text(
        json.dumps(
            {
                "best": {
                    "thresholds": {
                        "large_refine_class_threshold": 3,
                        "large_refine_object_threshold": 30,
                        "large_refine_review_threshold": 5,
                        "large_model_cluster_threshold": 15,
                        "large_model_class_threshold": 4,
                        "large_model_object_threshold": 150,
                        "large_model_review_threshold": 12,
                        "large_model_low_conf_ratio": 0.31,
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output = tmp_path / ".env.recommended"
    script = PROJECT_ROOT / "scripts" / "write_recommended_env.py"
    base_env = PROJECT_ROOT / ".env.collaborative.example"

    subprocess.run(
        [sys.executable, str(script), "--recommendation", str(recommendation), "--base-env", str(base_env), "--output", str(output)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    text = output.read_text(encoding="utf-8")
    assert "SAR_ROUTE_LARGE_REFINE_OBJECT_THRESHOLD=30" in text
    assert "SAR_ROUTE_LARGE_MODEL_LOW_CONF_RATIO=0.31" in text
