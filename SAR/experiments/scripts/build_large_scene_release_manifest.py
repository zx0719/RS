#!/usr/bin/env python3
"""
Build a consolidated release manifest for large-scene outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_release_readiness import (
    _body_mentions_scene_description,
    _evidence_acceptance_run_id,
    _fresh_model_output_ok,
    _gpu_requirement_ok,
    _local_gpu_requirement_ok,
    _model_participation_failures,
    _vlm_trace_ok,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consolidated large-scene release manifest.")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/original_format_docx_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/release_manifest.json"),
    )
    parser.add_argument(
        "--require-acceptance-run-id",
        default=None,
        metavar="RUN_ID",
        help="Annotate readiness against this acceptance run id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for item in items:
        evidence_path = Path(item["evidence_path"])
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        report_context = evidence.get("report_context", {})
        generation_trace = report_context.get("generation_trace", {})
        report = evidence.get("report", {})
        body = str(report.get("body") or "")
        body_sections = report.get("body_sections", [])
        report_source = body_sections[0].get("source") if body_sections else None
        scene = evidence.get("scene", {})
        scene_description = scene.get("scene_description")
        scene_description_trace = scene.get("scene_description_trace")
        selected_backend = generation_trace.get("selected_backend")
        manifest_acceptance_run_id = item.get("acceptance_run_id")
        evidence_acceptance_run_id = _evidence_acceptance_run_id(evidence)
        model_participation_failures = _model_participation_failures(report_source, generation_trace)
        model_participation_ok = not model_participation_failures
        fresh_model_output_ok = _fresh_model_output_ok(report_source, generation_trace)
        gpu_requirement_ok = _gpu_requirement_ok(
            report_source,
            generation_trace.get("available_backends"),
            selected_backend,
        )
        local_gpu_requirement_ok = _local_gpu_requirement_ok(
            report_source,
            generation_trace.get("available_backends"),
            selected_backend,
        )
        vlm_scene_description_present = bool(scene_description)
        vlm_scene_description_used = _body_mentions_scene_description(body, scene_description)
        vlm_trace_ok = _vlm_trace_ok(scene_description_trace)
        acceptance_run_id_ok = (
            manifest_acceptance_run_id == evidence_acceptance_run_id
            and bool(manifest_acceptance_run_id)
            and (
                args.require_acceptance_run_id is None
                or manifest_acceptance_run_id == args.require_acceptance_run_id
            )
        )
        strict_release_ready = (
            model_participation_ok
            and fresh_model_output_ok is True
            and gpu_requirement_ok is True
            and local_gpu_requirement_ok is True
            and vlm_scene_description_present
            and vlm_scene_description_used
            and vlm_trace_ok
            and acceptance_run_id_ok
        )
        rows.append(
            {
                "case_name": item.get("case_name"),
                "docx_path": item.get("docx_path"),
                "evidence_path": item.get("evidence_path"),
                "manifest_acceptance_run_id": manifest_acceptance_run_id,
                "evidence_acceptance_run_id": evidence_acceptance_run_id,
                "acceptance_run_id_ok": acceptance_run_id_ok,
                "report_source": report_source,
                "generation_route": report_context.get("generation_route", item.get("generation_route")),
                "digest_id": generation_trace.get("digest_id"),
                "selected_source": generation_trace.get("selected_source"),
                "selected_backend": selected_backend,
                "available_backends": generation_trace.get("available_backends"),
                "model_participation_ok": model_participation_ok,
                "model_participation_failures": model_participation_failures,
                "fresh_model_output_ok": fresh_model_output_ok,
                "gpu_requirement_ok": gpu_requirement_ok,
                "local_gpu_requirement_ok": local_gpu_requirement_ok,
                "vlm_scene_description_present": vlm_scene_description_present,
                "vlm_scene_description_used": vlm_scene_description_used,
                "vlm_trace_ok": vlm_trace_ok,
                "strict_release_ready": strict_release_ready,
                "quality_warnings": evidence.get("quality_warnings", []),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "cases": rows,
        "summary": {
            "total_cases": len(rows),
            "strict_release_ready_cases": sum(1 for row in rows if row["strict_release_ready"]),
            "model_participation_cases": sum(1 for row in rows if row["model_participation_ok"]),
            "fresh_model_output_cases": sum(1 for row in rows if row["fresh_model_output_ok"] is True),
            "local_gpu_cases": sum(1 for row in rows if row["local_gpu_requirement_ok"] is True),
            "vlm_trace_cases": sum(1 for row in rows if row["vlm_trace_ok"]),
            "acceptance_run_id_cases": sum(1 for row in rows if row["acceptance_run_id_ok"]),
            "require_acceptance_run_id": args.require_acceptance_run_id,
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
