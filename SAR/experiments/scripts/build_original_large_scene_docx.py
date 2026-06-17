from __future__ import annotations

import json
import os
import sys
import argparse
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.report.docx_assembler import DocxAssembler
from modules.report.collab_config import collaborative_generator_kwargs, resolve_collaborative_config, resolve_vlm_config
from modules.report.collaborative import CollaborativeReportGenerator
from modules.report.large_scene import attach_large_scene_metadata
from modules.report.vlm_describer import VLMDescriber
from modules.report.vlm_trace import attach_vlm_scene_description
from scripts.validate_release_readiness import (
    _body_mentions_scene_description,
    _evidence_acceptance_run_id,
    _fresh_model_output_ok,
    _gpu_requirement_ok,
    _local_gpu_requirement_ok,
    _model_participation_failures,
    _vlm_trace_ok,
)


SOURCE_ROOT = Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene")
OUTPUT_ROOT = Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format")
_GEOGRAPHIC_CRS_VALUES = {"EPSG:4326", "WGS84", "WGS 84", "OGC:CRS84", "CRS84"}
TEMPLATE = "/home/zhuxiang/RS/SAR/需求/成品.docx"
CACHE_ROOT = OUTPUT_ROOT / ".report_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build original-format large-scene docx reports.")
    parser.add_argument(
        "env_file_positional",
        nargs="?",
        type=Path,
        help="Optional legacy positional .env file.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file for collaborative LLM/VLM settings.",
    )
    parser.add_argument(
        "--acceptance-run-id",
        default=None,
        help="Optional strict acceptance run id to stamp into evidence and manifest outputs.",
    )
    parser.add_argument(
        "--require-model-participation",
        action="store_true",
        help="Fail before writing DOCX/evidence unless report text proves a real model backend.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail before writing DOCX/evidence unless generation trace proves a GPU-capable backend.",
    )
    parser.add_argument(
        "--require-local-gpu",
        action="store_true",
        help="Fail before writing DOCX/evidence unless generation trace proves a local GPU backend.",
    )
    parser.add_argument(
        "--require-vlm",
        action="store_true",
        help="Fail before writing DOCX/evidence unless VLM scene description is present and used.",
    )
    parser.add_argument(
        "--require-fresh-model-output",
        action="store_true",
        help="Fail before writing DOCX/evidence unless report text was freshly generated, not cache-backed.",
    )
    parser.add_argument(
        "--strict-release",
        action="store_true",
        help="Shortcut for release-grade local GPU + fresh LLM + VLM + acceptance-run checks.",
    )
    return parser.parse_args()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_geographic_crs(package: dict) -> bool:
    metadata = package.get("input", {}).get("metadata", {})
    crs = metadata.get("crs")
    if not crs:
        return False
    return str(crs).strip().upper() in _GEOGRAPHIC_CRS_VALUES


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _is_valid_lon_lat(lon: object, lat: object) -> bool:
    lon_val = _as_float(lon)
    lat_val = _as_float(lat)
    return (
        lon_val is not None
        and lat_val is not None
        and -180.0 <= lon_val <= 180.0
        and -90.0 <= lat_val <= 90.0
    )


def _sanitize_report_table_coordinates(package: dict) -> None:
    report = package.setdefault("report", {})
    tables = report.setdefault("tables", {})
    geographic_crs = _is_geographic_crs(package)
    cleared_rows = 0

    for table_name in ("component_table", "equipment_table"):
        rows = tables.get(table_name, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if geographic_crs and _is_valid_lon_lat(row.get("lon"), row.get("lat")):
                continue
            if row.get("lon") is not None or row.get("lat") is not None:
                cleared_rows += 1
            row["lon"] = None
            row["lat"] = None

    if cleared_rows:
        package.setdefault("quality_warnings", []).append(
            {
                "code": "NON_WGS84_COORDINATES_SUPPRESSED",
                "message": (
                    "Table lon/lat fields were suppressed because the source "
                    "coordinates are not confirmed WGS84 longitude/latitude."
                ),
                "cleared_rows": cleared_rows,
            }
        )


def _ensure_vlm_description_in_report(package: dict) -> None:
    scene = package.get("scene", {})
    if not isinstance(scene, dict):
        return
    scene_description = scene.get("scene_description")
    if not scene_description:
        return

    report = package.setdefault("report", {})
    body = str(report.get("body") or "").strip()
    if not body or _body_mentions_scene_description(body, scene_description):
        return

    description = str(scene_description).strip().rstrip("。")
    if not description:
        return
    suffix = f"VLM场景补充显示，{description}。"
    if body.endswith("。"):
        body = f"{body}{suffix}"
    else:
        body = f"{body}。{suffix}"
    report["body"] = body

    body_sections = report.get("body_sections")
    if isinstance(body_sections, list) and body_sections:
        section = body_sections[0]
        if isinstance(section, dict):
            section["content"] = body


def _prepare_docx_image(overview_path: Path, out_dir: Path, case_name: str) -> Path:
    docx_image_path = out_dir / f"{case_name}_overview_docx.png"
    with Image.open(overview_path) as image:
        image = image.convert("RGB")
        image.thumbnail((1600, 1600))
        image.save(docx_image_path, format="PNG", optimize=True)
    return docx_image_path


def _commit_docx_image(prepared_path: Path, out_dir: Path, case_name: str) -> Path:
    final_path = out_dir / f"{case_name}_overview_docx.png"
    if prepared_path == final_path:
        return final_path
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prepared_path, final_path)
    return final_path


def _path_in_case_dir(path: str, *, case_name: str, staging_root: Path, final_root: Path) -> str:
    path_obj = Path(path)
    try:
        relative = path_obj.relative_to(staging_root / case_name)
    except ValueError:
        return path
    return str(final_root / case_name / relative)


def _rewrite_package_paths(package: dict, *, case_name: str, staging_root: Path, final_root: Path) -> None:
    attachments = package.get("attachments", {})
    if isinstance(attachments, dict):
        for attachment in attachments.values():
            if not isinstance(attachment, dict):
                continue
            uri = attachment.get("uri")
            if isinstance(uri, str) and uri.startswith("file://"):
                local_path = uri[len("file://") :]
                final_path = _path_in_case_dir(
                    local_path,
                    case_name=case_name,
                    staging_root=staging_root,
                    final_root=final_root,
                )
                attachment["uri"] = f"file://{final_path}"

    report_docx = package.get("report", {}).get("docx")
    if isinstance(report_docx, dict):
        uri = report_docx.get("uri")
        if isinstance(uri, str) and uri.startswith("file://"):
            local_path = uri[len("file://") :]
            final_path = _path_in_case_dir(
                local_path,
                case_name=case_name,
                staging_root=staging_root,
                final_root=final_root,
            )
            report_docx["uri"] = f"file://{final_path}"


def _commit_strict_outputs(
    staged_items: list[dict[str, object]],
    *,
    staging_root: Path,
    final_root: Path,
) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for item in staged_items:
        case_name = str(item["case_name"])
        stage_case_dir = staging_root / case_name
        final_case_dir = final_root / case_name
        final_case_dir.mkdir(parents=True, exist_ok=True)
        for staged_file in stage_case_dir.iterdir():
            if staged_file.is_file():
                shutil.copy2(staged_file, final_case_dir / staged_file.name)

        staged_docx_path = str(item["docx_path"])
        staged_evidence_path = str(item["evidence_path"])
        final_docx_path = _path_in_case_dir(
            staged_docx_path,
            case_name=case_name,
            staging_root=staging_root,
            final_root=final_root,
        )
        final_evidence_path = _path_in_case_dir(
            staged_evidence_path,
            case_name=case_name,
            staging_root=staging_root,
            final_root=final_root,
        )
        package = json.loads(Path(final_evidence_path).read_text(encoding="utf-8"))
        _rewrite_package_paths(
            package,
            case_name=case_name,
            staging_root=staging_root,
            final_root=final_root,
        )
        package.setdefault("report", {}).setdefault("docx", {})["uri"] = f"file://{final_docx_path}"
        Path(final_evidence_path).write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")

        outputs.append(
            {
                "case_name": case_name,
                "docx_path": final_docx_path,
                "evidence_path": final_evidence_path,
                "source_overview": str(item["source_overview"]),
                "report_source": str(item["report_source"]),
                "generation_route": str(item["generation_route"]),
                "acceptance_run_id": item.get("acceptance_run_id"),  # type: ignore[dict-item]
            }
        )
    return outputs


def _strict_package_failures(
    package: dict,
    *,
    case_name: str,
    acceptance_run_id: str | None,
    require_model_participation: bool,
    require_gpu: bool,
    require_local_gpu: bool,
    require_vlm: bool,
    require_fresh_model_output: bool,
) -> list[str]:
    failures: list[str] = []
    report = package.get("report", {})
    body = str(report.get("body") or "")
    if not body.strip() or "正文待生成" in body:
        failures.append("report body is empty or still a placeholder")

    body_sections = report.get("body_sections", [])
    source = body_sections[0].get("source") if body_sections else None
    report_context = package.get("report_context", {})
    generation_trace = report_context.get("generation_trace", {})
    selected_backend = generation_trace.get("selected_backend") if isinstance(generation_trace, dict) else None

    if require_model_participation:
        failures.extend(_model_participation_failures(source, generation_trace))
    if require_fresh_model_output and _fresh_model_output_ok(source, generation_trace) is not True:
        failures.append("fresh non-cache model output is required")
    if require_gpu:
        gpu_ok = _gpu_requirement_ok(
            source,
            generation_trace.get("available_backends") if isinstance(generation_trace, dict) else None,
            selected_backend,
        )
        if gpu_ok is not True:
            failures.append("GPU-backed generation is not proven by generation_trace")
    if require_local_gpu:
        local_gpu_ok = _local_gpu_requirement_ok(
            source,
            generation_trace.get("available_backends") if isinstance(generation_trace, dict) else None,
            selected_backend,
        )
        if local_gpu_ok is not True:
            failures.append("local GPU model backend is required but not proven by generation_trace")

    scene = package.get("scene", {})
    scene_description = scene.get("scene_description") if isinstance(scene, dict) else None
    scene_description_trace = scene.get("scene_description_trace") if isinstance(scene, dict) else None
    if require_vlm:
        if not scene_description:
            failures.append("VLM scene_description is missing")
        if not _body_mentions_scene_description(body, scene_description):
            failures.append("VLM scene_description is not reflected in report body")
        if not _vlm_trace_ok(scene_description_trace):
            failures.append("VLM scene_description trace is missing or incomplete")

    if acceptance_run_id and _evidence_acceptance_run_id(package) != acceptance_run_id:
        failures.append(f"acceptance_run_id does not match {acceptance_run_id!r}")
    return [f"{case_name}: {failure}" for failure in failures]


def main() -> None:
    args = parse_args()
    env_file = args.env_file or args.env_file_positional
    acceptance_run_id = args.acceptance_run_id or os.getenv("SAR_ACCEPTANCE_RUN_ID")
    summary = json.loads((SOURCE_ROOT / "full_run_summary.json").read_text(encoding="utf-8"))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, object]] = []
    assembler = DocxAssembler()
    config = resolve_collaborative_config(cache_dir=str(CACHE_ROOT), env_file=env_file)
    generator = CollaborativeReportGenerator(**collaborative_generator_kwargs(config))
    vlm_config = resolve_vlm_config(env_file=env_file)
    require_model_participation = (
        args.require_model_participation
        or args.strict_release
        or not config.allow_template_fallback
    )
    require_gpu = args.require_gpu or args.strict_release or config.require_gpu
    require_local_gpu = args.require_local_gpu or args.strict_release or _env_bool("SAR_REQUIRE_LOCAL_GPU")
    require_vlm = args.require_vlm or args.strict_release or vlm_config.require_vlm
    require_fresh_model_output = (
        args.require_fresh_model_output
        or args.strict_release
        or (bool(acceptance_run_id) and require_model_participation)
    )
    if args.strict_release and not acceptance_run_id:
        raise RuntimeError("--strict-release requires --acceptance-run-id or SAR_ACCEPTANCE_RUN_ID.")
    defer_output_artifacts = args.strict_release
    describer = None
    if vlm_config.vlm_base_url:
        describer = VLMDescriber(
            base_url=vlm_config.vlm_base_url,
            model_name=vlm_config.vlm_model_name or "qwen3-vl-4b",
            api_key=vlm_config.api_key or "EMPTY",
            timeout=vlm_config.timeout,
        )
    elif require_vlm:
        raise RuntimeError("SAR_REQUIRE_VLM=1 but SAR_VLM_URL is empty.")

    with tempfile.TemporaryDirectory(prefix="sar_strict_docx_") as tmp_dir:
        temp_root = Path(tmp_dir)
        staging_root = temp_root / "staging" if defer_output_artifacts else OUTPUT_ROOT
        for case in summary["cases"]:
            case_name = case["case_name"]
            case_dir = SOURCE_ROOT / case_name
            evidence_path = case_dir / f"{case_name}_evidence.json"
            overview_path = case_dir / f"{case_name}_overview.jpg"
            final_out_dir = OUTPUT_ROOT / case_name
            out_dir = staging_root / case_name if defer_output_artifacts else final_out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            image_work_dir = temp_root / case_name if defer_output_artifacts else out_dir
            image_work_dir.mkdir(parents=True, exist_ok=True)

            package = json.loads(evidence_path.read_text(encoding="utf-8"))
            package = attach_large_scene_metadata(package)
            if acceptance_run_id:
                package.setdefault("trace", {})["acceptance_run_id"] = acceptance_run_id
                package.setdefault("report_context", {})["acceptance_run_id"] = acceptance_run_id
            package["status"] = "READY_FOR_NLG"
            _sanitize_report_table_coordinates(package)

            attachments = package.setdefault("attachments", {})
            prepared_docx_image_path: Path | None = None
            if overview_path.exists():
                prepared_docx_image_path = _prepare_docx_image(overview_path, image_work_dir, case_name)
                attachments["annotated_image"] = {
                    "uri": f"file://{prepared_docx_image_path}",
                    "role": "large_scene_overview_as_target_distribution",
                }
                if describer is not None:
                    scene_desc = describer.describe(str(overview_path))
                    if scene_desc:
                        attach_vlm_scene_description(
                            package,
                            description=scene_desc,
                            describer=describer,
                            image_path=str(overview_path),
                        )
                    elif require_vlm:
                        raise RuntimeError(f"VLM scene description is required but empty for {case_name}.")
            elif require_vlm:
                raise RuntimeError(f"VLM scene description is required but overview image is missing for {case_name}.")

            report = package.setdefault("report", {})
            report.setdefault("title", "航天通报")
            package = generator.generate(package)
            if acceptance_run_id:
                package.setdefault("trace", {})["acceptance_run_id"] = acceptance_run_id
                package.setdefault("report_context", {})["acceptance_run_id"] = acceptance_run_id
            generated_body = package.get("report", {}).get("body")
            if not isinstance(generated_body, str) or not generated_body.strip():
                raise RuntimeError(f"Report body generation failed for {case_name}.")
            _ensure_vlm_description_in_report(package)
            strict_failures = _strict_package_failures(
                package,
                case_name=case_name,
                acceptance_run_id=acceptance_run_id,
                require_model_participation=require_model_participation,
                require_gpu=require_gpu,
                require_local_gpu=require_local_gpu,
                require_vlm=require_vlm,
                require_fresh_model_output=require_fresh_model_output,
            )
            if strict_failures:
                raise RuntimeError("Strict report validation failed before DOCX write: " + "; ".join(strict_failures))

            if prepared_docx_image_path is not None:
                final_docx_image_path = _commit_docx_image(prepared_docx_image_path, out_dir, case_name)
                attachments["annotated_image"]["uri"] = f"file://{final_docx_image_path}"

            existing_docx = next(iter(sorted(out_dir.glob("*_draft.docx"))), None)
            try:
                docx_path = assembler.assemble(
                    package,
                    output_dir=str(out_dir),
                    template_path=TEMPLATE,
                    draft_mode=True,
                    include_image=True,
                )
            except ImportError:
                if existing_docx is None:
                    raise
                docx_path = DocxAssembler.patch_existing_docx(
                    str(existing_docx),
                    package,
                    draft_mode=True,
                )

            package.setdefault("report", {})["docx"] = {
                "uri": f"file://{docx_path}",
                "template_version": DocxAssembler.TEMPLATE_VERSION,
            }
            package["status"] = "DOCX_RENDERED"
            package["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

            updated_evidence_path = out_dir / f"{case_name}_original_format_evidence.json"
            updated_evidence_path.write_text(
                json.dumps(package, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            outputs.append(
                {
                    "case_name": case_name,
                    "docx_path": docx_path,
                    "evidence_path": str(updated_evidence_path),
                    "source_overview": str(overview_path),
                    "report_source": package.get("report", {}).get("body_sections", [{}])[0].get("source", "unknown"),
                    "generation_route": package.get("report_context", {}).get("generation_route", "unknown"),
                    "acceptance_run_id": acceptance_run_id,
                }
            )

        if defer_output_artifacts:
            outputs = _commit_strict_outputs(
                outputs,
                staging_root=staging_root,
                final_root=OUTPUT_ROOT,
            )

    manifest_path = OUTPUT_ROOT / "original_format_docx_manifest.json"
    manifest_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in outputs:
        print(item["docx_path"])


if __name__ == "__main__":
    main()
