#!/usr/bin/env python3
"""
Profile large-scene prompt payload size and routing decisions.

Usage:
    python scripts/profile_large_scene_prompts.py \
        --summary experiments/output/large_scene/full_run_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile large-scene prompt inputs.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene/full_run_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene/prompt_profile.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    import sys

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from modules.report.large_scene import build_evidence_digest, choose_generation_route, profile_prompt_input

    rows: list[dict[str, object]] = []
    for case in summary["cases"]:
        evidence_path = Path(case["evidence_path"])
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        digest = build_evidence_digest(evidence)
        route = choose_generation_route(evidence, digest)
        profile = profile_prompt_input(evidence, digest, route)
        rows.append(
            {
                "case_name": case["case_name"],
                "route": route,
                "prompt_chars": profile.prompt_chars,
                "prompt_lines": profile.prompt_lines,
                "digest_bytes": profile.digest_bytes,
                "digest_id": digest["digest_id"],
                "object_count": digest["global_summary"]["object_count"],
                "class_count": len(digest["class_digest"]),
                "cluster_count": digest["cluster_summary"]["cluster_count"],
            }
        )

    result = {"summary_path": str(args.summary), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
