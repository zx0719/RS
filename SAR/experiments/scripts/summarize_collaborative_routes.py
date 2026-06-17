#!/usr/bin/env python3
"""
Summarize collaborative route/source statistics from validation outputs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize collaborative route/source stats.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/collaborative_validation_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/collaborative_route_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    route_counter = Counter()
    source_counter = Counter()
    model_participation = 0
    reason_counter = Counter()

    for row in report["cases"]:
        route_counter[row.get("generation_route") or "unknown"] += 1
        source_counter[row.get("report_source") or "unknown"] += 1
        reason_counter[row.get("generation_reason") or "unknown"] += 1
        if row.get("has_model_participation"):
            model_participation += 1

    summary = {
        "total_cases": len(report["cases"]),
        "routes": dict(route_counter),
        "reasons": dict(reason_counter),
        "sources": dict(source_counter),
        "model_participation_cases": model_participation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
