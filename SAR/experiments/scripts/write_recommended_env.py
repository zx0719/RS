#!/usr/bin/env python3
"""
Write recommended collaborative thresholds into a new env file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write recommended collaborative env file.")
    parser.add_argument(
        "--recommendation",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/collaborative_threshold_recommendation.json"),
    )
    parser.add_argument(
        "--base-env",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/.env.collaborative.example"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/.env.collaborative.recommended"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recommendation = json.loads(args.recommendation.read_text(encoding="utf-8"))
    best = recommendation.get("best") or {}
    thresholds = best.get("thresholds", {})

    base_lines = args.base_env.read_text(encoding="utf-8").splitlines()
    rendered: list[str] = []
    for line in base_lines:
        if line.startswith("SAR_ROUTE_") and "=" in line:
            key = line.split("=", 1)[0]
            if key == "SAR_ROUTE_LARGE_REFINE_CLASS_THRESHOLD":
                line = f"{key}={thresholds.get('large_refine_class_threshold', line.split('=',1)[1])}"
            elif key == "SAR_ROUTE_LARGE_REFINE_OBJECT_THRESHOLD":
                line = f"{key}={thresholds.get('large_refine_object_threshold', line.split('=',1)[1])}"
            elif key == "SAR_ROUTE_LARGE_REFINE_REVIEW_THRESHOLD":
                line = f"{key}={thresholds.get('large_refine_review_threshold', line.split('=',1)[1])}"
            elif key == "SAR_ROUTE_LARGE_MODEL_CLUSTER_THRESHOLD":
                line = f"{key}={thresholds.get('large_model_cluster_threshold', line.split('=',1)[1])}"
            elif key == "SAR_ROUTE_LARGE_MODEL_CLASS_THRESHOLD":
                line = f"{key}={thresholds.get('large_model_class_threshold', line.split('=',1)[1])}"
            elif key == "SAR_ROUTE_LARGE_MODEL_OBJECT_THRESHOLD":
                line = f"{key}={thresholds.get('large_model_object_threshold', line.split('=',1)[1])}"
            elif key == "SAR_ROUTE_LARGE_MODEL_REVIEW_THRESHOLD":
                line = f"{key}={thresholds.get('large_model_review_threshold', line.split('=',1)[1])}"
            elif key == "SAR_ROUTE_LARGE_MODEL_LOW_CONF_RATIO":
                line = f"{key}={thresholds.get('large_model_low_conf_ratio', line.split('=',1)[1])}"
        rendered.append(line)

    args.output.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
