#!/usr/bin/env python3
"""
Sweep collaborative-routing thresholds against existing large-scene evidence.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.report.collab_config import RouteThresholds
from modules.report.large_scene import build_evidence_digest, choose_generation_route


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep collaborative route thresholds.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/original_format_docx_manifest.json"),
    )
    parser.add_argument("--refine-object-thresholds", default="20,24,30")
    parser.add_argument("--refine-review-thresholds", default="2,4,6")
    parser.add_argument("--large-object-thresholds", default="80,120,160")
    parser.add_argument("--large-low-conf-ratios", default="0.2,0.28,0.35")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/collaborative_threshold_sweep.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    evidences = []
    for item in items:
        evidence = json.loads(Path(item["evidence_path"]).read_text(encoding="utf-8"))
        evidences.append((item["case_name"], evidence, build_evidence_digest(evidence)))

    results: list[dict[str, object]] = []
    for refine_object, refine_review, large_object, low_conf_ratio in itertools.product(
        parse_csv_ints(args.refine_object_thresholds),
        parse_csv_ints(args.refine_review_thresholds),
        parse_csv_ints(args.large_object_thresholds),
        parse_csv_floats(args.large_low_conf_ratios),
    ):
        thresholds = RouteThresholds(
            large_refine_object_threshold=refine_object,
            large_refine_review_threshold=refine_review,
            large_model_object_threshold=large_object,
            large_model_low_conf_ratio=low_conf_ratio,
        )
        routes = {}
        for case_name, evidence, digest in evidences:
            routes[case_name] = choose_generation_route(evidence, digest=digest, thresholds=thresholds)
        results.append(
            {
                "thresholds": thresholds.__dict__,
                "routes": routes,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
