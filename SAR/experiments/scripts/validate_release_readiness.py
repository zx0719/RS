#!/usr/bin/env python3
"""
Validate whether large-scene report outputs are ready for release.

This is the final gate: it checks the evidence, report body, generation trace,
VLM scene description usage, and DOCX package directly from the manifest.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile


MODEL_SOURCES = {
    "llm_v1",
    "local_llm_v1",
    "small_llm_v1",
    "small_llm_cache_v1",
    "large_llm_v1",
    "large_llm_cache_v1",
    "large_llm_refine_v1",
    "large_llm_refine_cache_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate release readiness for SAR reports.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/original_format_docx_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/release_readiness_report.json"),
    )
    parser.add_argument(
        "--require-model-participation",
        action="store_true",
        help="Require every report body to come from an LLM backend, not template fallback.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="For local model outputs, require trace evidence that CUDA was visible and required.",
    )
    parser.add_argument(
        "--require-local-gpu",
        action="store_true",
        help="Require every model output to come from a local backend with require_gpu=true and cuda_available=true.",
    )
    parser.add_argument(
        "--require-vlm",
        action="store_true",
        help="Require VLM scene description to be present and reflected in the report body.",
    )
    parser.add_argument(
        "--require-vlm-trace",
        action="store_true",
        help="Require VLM scene descriptions to include auditable VLM service trace metadata.",
    )
    parser.add_argument(
        "--require-acceptance-run-id",
        default=None,
        metavar="RUN_ID",
        help="Require every manifest item and evidence trace to match this acceptance run id.",
    )
    parser.add_argument(
        "--require-fresh-model-output",
        action="store_true",
        help="Reject cache-backed LLM sources; every model output must be freshly generated in this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    for item in items:
        evidence_path = Path(item["evidence_path"])
        docx_path = Path(item["docx_path"])
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        report = evidence.get("report", {})
        body = str(report.get("body") or "")
        body_sections = report.get("body_sections", [])
        source = body_sections[0].get("source") if body_sections else None
        report_context = evidence.get("report_context", {})
        generation_trace = report_context.get("generation_trace", {})
        evidence_acceptance_run_id = _evidence_acceptance_run_id(evidence)
        manifest_acceptance_run_id = item.get("acceptance_run_id")
        scene_description = evidence.get("scene", {}).get("scene_description")
        scene_description_trace = evidence.get("scene", {}).get("scene_description_trace")

        failures: list[str] = []
        docx_ok, docx_failures, docx_text = _validate_docx(docx_path)
        failures.extend(docx_failures)

        if not body.strip() or "正文待生成" in body:
            failures.append("report body is empty or still a placeholder")

        model_participation_failures = _model_participation_failures(source, generation_trace)
        fresh_model_output_ok = _fresh_model_output_ok(source, generation_trace)
        model_participation_ok = not model_participation_failures
        if args.require_model_participation and not model_participation_ok:
            failures.extend(model_participation_failures)
        if args.require_fresh_model_output and source in MODEL_SOURCES and fresh_model_output_ok is not True:
            failures.append("fresh non-cache model output is required")

        selected_source = generation_trace.get("selected_source")
        if not args.require_model_participation and source in MODEL_SOURCES and selected_source != source:
            failures.append(
                f"generation trace selected_source {selected_source!r} does not match report source {source!r}"
            )

        gpu_requirement_ok = _gpu_requirement_ok(
            source,
            generation_trace.get("available_backends"),
            generation_trace.get("selected_backend"),
        )
        if args.require_gpu and model_participation_ok and gpu_requirement_ok is not True:
            failures.append("GPU-backed local generation is not proven by generation_trace")
        local_gpu_requirement_ok = _local_gpu_requirement_ok(
            source,
            generation_trace.get("available_backends"),
            generation_trace.get("selected_backend"),
        )
        if args.require_local_gpu and model_participation_ok and local_gpu_requirement_ok is not True:
            failures.append("local GPU model backend is required but not proven by generation_trace")

        vlm_present = bool(scene_description)
        vlm_used = _body_mentions_scene_description(body, scene_description)
        if args.require_vlm and not vlm_present:
            failures.append("VLM scene_description is missing")
        if args.require_vlm and not vlm_used:
            failures.append("VLM scene_description is not reflected in report body")
        vlm_trace_ok = _vlm_trace_ok(scene_description_trace)
        if args.require_vlm_trace and not vlm_trace_ok:
            failures.append("VLM scene_description trace is missing or incomplete")
        acceptance_run_id_ok = (
            args.require_acceptance_run_id is None
            or (
                manifest_acceptance_run_id == args.require_acceptance_run_id
                and evidence_acceptance_run_id == args.require_acceptance_run_id
            )
        )
        if not acceptance_run_id_ok:
            failures.append(f"acceptance_run_id does not match required run id {args.require_acceptance_run_id!r}")
        docx_acceptance_run_id_ok = (
            args.require_acceptance_run_id is None
            or args.require_acceptance_run_id in docx_text
        )
        if not docx_acceptance_run_id_ok:
            failures.append(f"DOCX does not contain required acceptance_run_id {args.require_acceptance_run_id!r}")

        rows.append(
            {
                "case_name": item.get("case_name"),
                "evidence_path": str(evidence_path),
                "docx_path": str(docx_path),
                "report_source": source,
                "selected_source": selected_source,
                "selected_backend": generation_trace.get("selected_backend"),
                "manifest_acceptance_run_id": manifest_acceptance_run_id,
                "evidence_acceptance_run_id": evidence_acceptance_run_id,
                "acceptance_run_id_ok": acceptance_run_id_ok,
                "docx_acceptance_run_id_ok": docx_acceptance_run_id_ok,
                "docx_ok": docx_ok,
                "body_ok": bool(body.strip()) and "正文待生成" not in body,
                "model_participation_ok": model_participation_ok,
                "model_participation_failures": model_participation_failures,
                "fresh_model_output_ok": fresh_model_output_ok,
                "gpu_requirement_ok": gpu_requirement_ok,
                "local_gpu_requirement_ok": local_gpu_requirement_ok,
                "vlm_scene_description_present": vlm_present,
                "vlm_scene_description_used": vlm_used,
                "vlm_trace_ok": vlm_trace_ok,
                "ready": not failures,
                "failures": failures,
            }
        )

    failed_cases = [row["case_name"] for row in rows if not row["ready"]]
    report = {
        "cases": rows,
        "summary": {
            "total_cases": len(rows),
            "ready_cases": sum(1 for row in rows if row["ready"]),
            "require_model_participation": args.require_model_participation,
            "require_gpu": args.require_gpu,
            "require_local_gpu": args.require_local_gpu,
            "require_vlm": args.require_vlm,
            "require_vlm_trace": args.require_vlm_trace,
            "require_acceptance_run_id": args.require_acceptance_run_id,
            "require_fresh_model_output": args.require_fresh_model_output,
            "fresh_model_output_cases": sum(1 for row in rows if row.get("fresh_model_output_ok") is True),
            "failures": failed_cases,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    if failed_cases:
        raise SystemExit(1)


def _validate_docx(docx_path: Path) -> tuple[bool, list[str], str]:
    failures: list[str] = []
    if not docx_path.exists():
        return False, [f"DOCX does not exist: {docx_path}"], ""
    try:
        with ZipFile(docx_path, "r") as zf:
            names = set(zf.namelist())
            document_xml = (
                zf.read("word/document.xml").decode("utf-8", errors="ignore")
                if "word/document.xml" in names
                else ""
            )
            footer_xml = "\n".join(
                zf.read(name).decode("utf-8", errors="ignore")
                for name in names
                if name.startswith("word/footer") and name.endswith(".xml")
            )
    except BadZipFile:
        return False, [f"DOCX is not a valid zip package: {docx_path}"], ""

    required_texts = ("航天通报", "附件1", "附件2", "附件3", "目标分布图", "组成分布统计表", "装备分布统计表")
    missing = [text for text in required_texts if text not in document_xml]
    if missing:
        failures.append("DOCX missing required text: " + ", ".join(missing))
    if "自动生成草稿，待人工审核" not in footer_xml and "自动生成草稿，待人工审核" not in document_xml:
        failures.append("DOCX missing draft footer")
    if not re.search(r"\d{4}年\d{1,2}月\d{1,2}日", document_xml):
        failures.append("DOCX missing report date")
    return not failures, failures, document_xml + "\n" + footer_xml


def _gpu_requirement_ok(
    source: object,
    available_backends: object,
    selected_backend: object,
) -> bool | None:
    role = _source_role(source)
    if role is None:
        return None
    if isinstance(selected_backend, dict):
        if selected_backend.get("mode") != "local":
            return bool(selected_backend.get("base_url"))
        return (
            selected_backend.get("role") in {role, "large_refine"}
            and selected_backend.get("require_gpu") is True
            and selected_backend.get("cuda_available") is True
        )
    if not isinstance(available_backends, dict):
        return False
    backend = available_backends.get(role)
    if not isinstance(backend, dict):
        return False
    if backend.get("mode") != "local":
        return bool(backend.get("base_url"))
    return backend.get("require_gpu") is True and backend.get("cuda_available") is True


def _local_gpu_requirement_ok(
    source: object,
    available_backends: object,
    selected_backend: object,
) -> bool | None:
    role = _source_role(source)
    if role is None:
        return None
    if isinstance(selected_backend, dict):
        return _backend_is_local_gpu(selected_backend, role)
    if not isinstance(available_backends, dict):
        return False
    backend = available_backends.get(role)
    if not isinstance(backend, dict):
        return False
    return _backend_is_local_gpu(backend, role)


def _backend_is_local_gpu(backend: dict, role: str) -> bool:
    complete_gpu_fields = (
        backend.get("mode") == "local"
        and backend.get("role", role) in {role, "large_refine"}
        and backend.get("require_gpu") is True
        and backend.get("cuda_available") is True
    )
    if backend.get("local_gpu_verified") is None:
        return complete_gpu_fields
    return complete_gpu_fields and backend.get("local_gpu_verified") is True


def _model_participation_failures(source: object, generation_trace: dict) -> list[str]:
    if source not in MODEL_SOURCES:
        return [f"report source is not an LLM source: {source}"]
    if not isinstance(generation_trace, dict):
        return ["generation_trace is missing"]
    selected_source = generation_trace.get("selected_source")
    if selected_source != source:
        return [f"selected_source {selected_source!r} does not match report source {source!r}"]
    selected_backend = generation_trace.get("selected_backend")
    if not isinstance(selected_backend, dict):
        return ["selected_backend is missing"]
    if selected_backend.get("source") != source:
        return [f"selected_backend source {selected_backend.get('source')!r} does not match report source {source!r}"]
    role = _source_role(source)
    if role is None:
        return [f"cannot map source to backend role: {source}"]
    if selected_backend.get("role") not in {role, "large_refine"}:
        return [f"selected_backend role {selected_backend.get('role')!r} does not match source role {role!r}"]
    if not _backend_has_real_model_endpoint(selected_backend):
        return ["selected_backend does not prove a real model endpoint or local model path"]
    attempts = generation_trace.get("attempts")
    if isinstance(attempts, list) and attempts:
        if not _has_used_attempt(attempts, selected_backend.get("role")):
            return ["attempts do not contain a used/cache_hit attempt for the selected backend"]
    return []


def _evidence_acceptance_run_id(evidence: dict) -> object:
    trace = evidence.get("trace", {})
    if isinstance(trace, dict) and trace.get("acceptance_run_id"):
        return trace.get("acceptance_run_id")
    report_context = evidence.get("report_context", {})
    if isinstance(report_context, dict):
        return report_context.get("acceptance_run_id")
    return None


def _fresh_model_output_ok(source: object, generation_trace: dict) -> bool | None:
    if source not in MODEL_SOURCES:
        return None
    if not isinstance(generation_trace, dict):
        return False
    if _model_participation_failures(source, generation_trace):
        return False
    if isinstance(source, str) and source.endswith("_cache_v1"):
        return False
    selected_backend = generation_trace.get("selected_backend")
    if not isinstance(selected_backend, dict):
        return False
    if selected_backend.get("selection_status") == "cache_hit":
        return False
    role = selected_backend.get("role")
    attempts = generation_trace.get("attempts")
    if isinstance(attempts, list) and attempts:
        return _has_used_attempt(attempts, role, allowed_statuses={"used"})
    return False


def _backend_has_real_model_endpoint(backend: dict) -> bool:
    mode = backend.get("mode")
    if mode == "local":
        return bool(backend.get("model_path"))
    if mode == "api":
        return bool(backend.get("base_url"))
    return False


def _has_used_attempt(
    attempts: list,
    role: object,
    *,
    allowed_statuses: set[str] | None = None,
) -> bool:
    statuses = allowed_statuses or {"used", "cache_hit"}
    valid_roles = {"large_refine"} if role == "large_refine" else {role}
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        if attempt.get("role") in valid_roles and attempt.get("status") in statuses:
            return True
    return False


def _source_role(source: object) -> str | None:
    if source == "llm_v1":
        return "llm"
    if source == "local_llm_v1":
        return "local_llm"
    if source in {"small_llm_v1", "small_llm_cache_v1"}:
        return "small_llm"
    if source in {
        "large_llm_v1",
        "large_llm_cache_v1",
        "large_llm_refine_v1",
        "large_llm_refine_cache_v1",
    }:
        return "large_llm"
    return None


def _body_mentions_scene_description(body: str, scene_desc: object) -> bool:
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


if __name__ == "__main__":
    main()
