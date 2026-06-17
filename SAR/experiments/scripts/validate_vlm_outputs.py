#!/usr/bin/env python3
"""
Validate whether VLM scene descriptions were written into evidence/report outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _body_mentions_scene_description(body: str, scene_desc: str | None) -> bool:
    if not scene_desc:
        return False
    normalized_body = "".join(str(body).split())
    normalized_desc = "".join(str(scene_desc).split())
    if not normalized_body or not normalized_desc:
        return False
    candidates = {
        normalized_desc,
        normalized_desc[:8],
        normalized_desc[:12],
    }
    if len(normalized_desc) > 12:
        candidates.add(normalized_desc[-8:])
    return any(fragment and fragment in normalized_body for fragment in candidates)


def _vlm_trace_ok(trace: object) -> bool:
    if not isinstance(trace, dict):
        return False
    return (
        trace.get("source") == "vlm_v1"
        and trace.get("mode") == "api"
        and bool(trace.get("base_url"))
        and bool(trace.get("model_name"))
        and trace.get("status") == "used"
        and trace.get("description_present") is True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate VLM-enriched evidence outputs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/original_format_docx_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/vlm_validation_report.json"),
    )
    parser.add_argument(
        "--require-vlm",
        action="store_true",
        help="Fail if any case lacks a VLM scene description.",
    )
    parser.add_argument(
        "--require-vlm-trace",
        action="store_true",
        help="Fail if VLM scene descriptions lack auditable VLM service trace metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for item in items:
        evidence = json.loads(Path(item["evidence_path"]).read_text(encoding="utf-8"))
        scene_desc = evidence.get("scene", {}).get("scene_description")
        scene_desc_trace = evidence.get("scene", {}).get("scene_description_trace")
        body = evidence.get("report", {}).get("body", "")
        body_mentions_scene_description = _body_mentions_scene_description(body, scene_desc)
        vlm_trace_ok = _vlm_trace_ok(scene_desc_trace)
        rows.append(
            {
                "case_name": item.get("case_name"),
                "scene_description_present": bool(scene_desc),
                "scene_description": scene_desc,
                "scene_description_trace": scene_desc_trace,
                "vlm_trace_ok": vlm_trace_ok,
                "body_mentions_scene_description": body_mentions_scene_description,
            }
        )

    total_cases = len(rows)
    scene_description_cases = sum(1 for row in rows if row["scene_description_present"])
    body_mentions_cases = sum(1 for row in rows if row["body_mentions_scene_description"])
    vlm_trace_cases = sum(1 for row in rows if row["vlm_trace_ok"])
    failures = []
    if args.require_vlm and scene_description_cases != total_cases:
        missing = [row["case_name"] for row in rows if not row["scene_description_present"]]
        failures.append(
            {
                "type": "missing_vlm_scene_description",
                "message": "VLM scene description is required but missing for one or more cases.",
                "cases": missing,
            }
        )
    if args.require_vlm and body_mentions_cases != total_cases:
        missing = [
            row["case_name"]
            for row in rows
            if not row["body_mentions_scene_description"]
        ]
        failures.append(
            {
                "type": "vlm_scene_description_not_used_in_body",
                "message": "VLM scene description is required but not reflected in report body.",
                "cases": missing,
            }
        )
    if args.require_vlm_trace and vlm_trace_cases != total_cases:
        missing = [
            row["case_name"]
            for row in rows
            if not row["vlm_trace_ok"]
        ]
        failures.append(
            {
                "type": "missing_vlm_trace",
                "message": "VLM trace metadata is required but missing or incomplete.",
                "cases": missing,
            }
        )

    report = {
        "cases": rows,
        "summary": {
            "total_cases": total_cases,
            "scene_description_cases": scene_description_cases,
            "body_mentions_scene_description_cases": body_mentions_cases,
            "vlm_trace_cases": vlm_trace_cases,
            "require_vlm": args.require_vlm,
            "require_vlm_trace": args.require_vlm_trace,
            "failures": failures,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
