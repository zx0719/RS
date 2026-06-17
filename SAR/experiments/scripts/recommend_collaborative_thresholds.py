#!/usr/bin/env python3
"""
Recommend a threshold set from collaborative threshold sweep results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend a threshold set from sweep results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/collaborative_threshold_sweep.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/collaborative_threshold_recommendation.json"),
    )
    return parser.parse_args()


def _score(routes: dict[str, str]) -> float:
    score = 0.0
    for route in routes.values():
        if route == "large_refine":
            score += 2.0
        elif route == "small_llm":
            score += 1.0
        elif route == "large_llm":
            score += 0.5
        elif route == "template":
            score -= 2.0
    return score


def main() -> None:
    args = parse_args()
    sweep = json.loads(args.input.read_text(encoding="utf-8"))
    ranked = sorted(
        (
            {
                "thresholds": item["thresholds"],
                "routes": item["routes"],
                "score": _score(item["routes"]),
            }
            for item in sweep["results"]
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    recommendation = {
        "best": ranked[0] if ranked else None,
        "top3": ranked[:3],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
