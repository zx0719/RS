"""
run_pipeline.py — SAR Intelligence Reporting Pipeline Entry Point

Standalone CLI script that runs the full SAR intel pipeline:
  M2 Detection → M4 Evidence Building → M5+M6 Report Generation → M7 Quality Gate

Usage
-----
    python run_pipeline.py --image <path> --region <name> [options]

Examples
--------
    # Minimal (MockDetector, template fallback)
    python run_pipeline.py --image test.tif --region 某某军港

    # With real model weights and local LLM
    python run_pipeline.py --image scene.tif --region 某某军港 \\
        --model runs/best.pt --llm-path /mnt/data/zhuxiang/Qwen/Qwen3-4B

    # With API-based LLM and final (no watermark) output
    python run_pipeline.py --image scene.tif --region 某某军港 \\
        --llm-url http://localhost:8000/v1 --final
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from modules.report.collab_config import collaborative_generator_kwargs, resolve_collaborative_config, resolve_vlm_config

# ---------------------------------------------------------------------------
# Logging setup — configured before any module imports so handlers are ready
# ---------------------------------------------------------------------------

def _configure_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="SAR Intelligence Reporting Pipeline — end-to-end CLI runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--image",
        required=True,
        metavar="PATH",
        help="SAR image path (GeoTIFF or JPEG)",
    )
    parser.add_argument(
        "--region",
        required=True,
        metavar="NAME",
        help='Region name for the report, e.g. "某某军港"',
    )

    # Optional — scene
    parser.add_argument(
        "--region-type",
        default="harbor",
        choices=["harbor", "airport", "anchorage", "airbase", "unknown"],
        metavar="TYPE",
        help="Scene type: harbor|airport|anchorage|airbase|unknown (default: harbor)",
    )

    # Optional — detector (M1)
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Path to YOLOv8 .pt weights (if omitted, uses MockDetector)",
    )

    # Optional — Branch B: Gate + M2 classifiers + M3 segmentation
    parser.add_argument(
        "--gate-model",
        default=None,
        metavar="PATH",
        help="Path to Gate ResNet-18 scene classifier weights (.pt)",
    )
    parser.add_argument(
        "--ship-cls-model",
        default=None,
        metavar="PATH",
        help="Path to M2a ship fine-grained classifier weights (.pt)",
    )
    parser.add_argument(
        "--aircraft-cls-model",
        default=None,
        metavar="PATH",
        help="Path to M2b aircraft fine-grained classifier weights (.pt)",
    )

    # Optional — LLM
    parser.add_argument(
        "--llm-path",
        default=None,
        metavar="PATH",
        help="Local LLM model path. Prefer SAR_SMALL_LLM_PATH in --env-file for collaborative generation.",
    )
    parser.add_argument(
        "--small-llm-path",
        default=None,
        metavar="PATH",
        help="Optional local small-model path for collaborative generation",
    )
    parser.add_argument(
        "--large-llm-path",
        default=None,
        metavar="PATH",
        help="Optional local large-model path for collaborative generation",
    )
    parser.add_argument(
        "--llm-url",
        default=None,
        metavar="URL",
        help="OpenAI-compatible API base URL (alternative to --llm-path)",
    )
    parser.add_argument(
        "--llm-model-name",
        default="Qwen2.5-7B-Instruct",
        metavar="NAME",
        help="Model name passed to the API when --llm-url is used (default: Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--large-llm-url",
        default=None,
        metavar="URL",
        help="Optional large-model OpenAI-compatible API base URL for collaborative generation",
    )
    parser.add_argument(
        "--large-llm-model-name",
        default="Qwen2.5-72B-Instruct",
        metavar="NAME",
        help="Large-model name used together with --large-llm-url",
    )
    parser.add_argument(
        "--use-collaborative-llm",
        action="store_true",
        help="Use CollaborativeReportGenerator for large-scene-aware routing and escalation",
    )
    parser.add_argument(
        "--vlm-url",
        default=None,
        metavar="URL",
        help="Optional VLM OpenAI-compatible API base URL for scene-level description",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        metavar="NAME",
        help="Optional VLM model name (overrides env-file if provided)",
    )
    parser.add_argument(
        "--require-vlm",
        action="store_true",
        default=None,
        help="Require a successful VLM scene description; fail if VLM is missing or empty",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        metavar="PATH",
        help="Optional .env-style file for collaborative LLM settings",
    )
    parser.add_argument(
        "--llm-device",
        default=None,
        metavar="DEVICE",
        help="Optional local device for collaborative local models, e.g. auto/cpu/cuda:0",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Optional max_new_tokens for local LLM generation",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        default=None,
        help="Require CUDA-visible GPU for local LLM inference; fail instead of using CPU",
    )
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument(
        "--allow-template-fallback",
        dest="allow_template_fallback",
        action="store_true",
        default=None,
        help="Allow rule-template report fallback when LLM generation is unavailable",
    )
    fallback_group.add_argument(
        "--no-template-fallback",
        dest="allow_template_fallback",
        action="store_false",
        default=None,
        help="Disable rule-template report fallback; require a real LLM backend",
    )

    # Optional — output
    parser.add_argument(
        "--output-dir",
        default="./output",
        metavar="DIR",
        help="Output directory (default: ./output)",
    )

    # Draft / final mode (mutually exclusive, draft is the default)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--draft",
        action="store_true",
        default=False,
        help="Output draft mode with review watermark (default behaviour)",
    )
    mode_group.add_argument(
        "--final",
        action="store_true",
        default=False,
        help="Output final mode (no watermark)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        metavar="LEVEL",
        help="Logging level: DEBUG|INFO|WARNING (default: INFO)",
    )

    return parser


# ---------------------------------------------------------------------------
# Helper: determine draft_mode from parsed args
# ---------------------------------------------------------------------------

def _resolve_draft_mode(args: argparse.Namespace) -> bool:
    """Return True (draft) unless --final was explicitly passed."""
    if args.final:
        return False
    return True  # --draft or neither → draft mode


# ---------------------------------------------------------------------------
# Helper: summarise detected object counts
# ---------------------------------------------------------------------------

def _summarise_objects(objects: list[dict[str, Any]]) -> str:
    """Return a compact string like '3 (ships: 2, aircraft: 1)'."""
    total = len(objects)
    ships = sum(1 for o in objects if o.get("class", {}).get("super_class") == "ship")
    aircraft = sum(1 for o in objects if o.get("class", {}).get("super_class") == "aircraft")
    parts: list[str] = []
    if ships:
        parts.append(f"ships: {ships}")
    if aircraft:
        parts.append(f"aircraft: {aircraft}")
    detail = ", ".join(parts) if parts else "unknown types"
    return f"{total} ({detail})"


# ---------------------------------------------------------------------------
# Helper: derive LLM source tag from the pipeline instance
# ---------------------------------------------------------------------------

def _llm_source_tag(pipeline: Any) -> str:
    from modules.report.generator import LocalModelGenerator  # type: ignore[import]
    from modules.report.collaborative import CollaborativeReportGenerator  # type: ignore[import]
    gen = getattr(pipeline, "_generator", None)
    if gen is None:
        return "unknown"
    if isinstance(gen, CollaborativeReportGenerator):
        backends = gen.describe_backends()
        small = backends["small_llm"]
        large = backends["large_llm"]
        return (
            f"Collaborative(small={small['model_name']}@{small['base_url'] or 'disabled'}, "
            f"large={large['model_name']}@{large['base_url'] if large else 'disabled'})"
        )
    if isinstance(gen, LocalModelGenerator):
        return "LocalModelGenerator"
    base_url = getattr(gen, "base_url", None)
    if base_url:
        return f"API({base_url})"
    return "template_fallback"


# ---------------------------------------------------------------------------
# Helper: extract DOCX path from package
# ---------------------------------------------------------------------------

def _docx_uri(package: dict[str, Any]) -> str:
    try:
        uri: str = package["report"]["docx"]["uri"]
        # Strip file:// prefix for display
        if uri.startswith("file://"):
            return uri[7:]
        return uri
    except (KeyError, TypeError):
        return "<not available>"


# ---------------------------------------------------------------------------
# Helper: quality gate result string
# ---------------------------------------------------------------------------

def _quality_result(package: dict[str, Any]) -> str:
    quality = package.get("quality", {})
    review_gate = quality.get("review_gate", {})
    needs_review: bool = review_gate.get("needs_human_review", False)
    return "REVIEW REQUIRED" if needs_review else "PASSED"


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    package: dict[str, Any],
    region: str,
    region_type: str,
    objects: list[dict[str, Any]],
    llm_source: str,
    evidence_path: Path,
) -> None:
    sep = "═" * 43
    docx_path = _docx_uri(package)
    quality_str = _quality_result(package)
    pkg_id = package.get("package_id", "<unknown>")
    obj_summary = _summarise_objects(objects)

    print(sep)
    print("SAR Intelligence Report Generated")
    print(sep)
    print(f"Package ID   : {pkg_id}")
    print(f"Region       : {region} ({region_type})")
    print(f"Objects      : {obj_summary}")
    print(f"LLM Source   : {llm_source}")
    print(f"Output DOCX  : {docx_path}")
    print(f"Evidence JSON: {evidence_path}")
    print(f"Quality Gate : {quality_str}")
    print(sep)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logger = _configure_logging(args.log_level)
    # Resolve draft mode
    draft_mode = _resolve_draft_mode(args)
    mode_label = "draft" if draft_mode else "final"
    logger.info("Pipeline starting | image=%s | region=%s | mode=%s",
                args.image, args.region, mode_label)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Output directory: %s", output_dir)
    except OSError as exc:
        logger.error("Cannot create output directory %s: %s", output_dir, exc)
        return 1

    # ── Step 0: Detector selection ────────────────────────────────────────────
    logger.info("Step 0: Initialising detector")
    try:
        if args.model and Path(args.model).exists():
            from modules.detector import DetectorTool  # type: ignore[import]
            # Use SSDD class map for models trained on SSDD (single "ship" class)
            class_map = None
            if "ssdd" in Path(args.model).parts or "ssdd" in args.model.lower():
                from modules.detector.class_map import SSDD_CLASS_MAP  # type: ignore[import]
                class_map = SSDD_CLASS_MAP
            detector = DetectorTool(args.model, class_map=class_map)
            logger.info("Using DetectorTool with weights: %s", args.model)
        else:
            from modules.detector.mock_detector import MockDetector  # type: ignore[import]
            detector = MockDetector()
            if args.model:
                logger.warning(
                    "Model weights not found at %s — using MockDetector", args.model
                )
            else:
                logger.warning("No model weights provided — using MockDetector")
    except Exception as exc:
        logger.error("Fatal: detector initialisation failed: %s", exc)
        return 1

    # ── Step 1: Detect ────────────────────────────────────────────────────────
    logger.info("Step 1: Running detection on %s", args.image)
    objects: list[dict[str, Any]] = []
    try:
        objects, _vis = detector.detect(args.image)
        logger.info("Detected %d objects", len(objects))
    except Exception as exc:
        logger.error("Detection failed: %s", exc)
        logger.warning("Continuing with empty objects list")
        objects = []

    # ── Step 1.5a: Gate scene classification (Branch B) ──────────────────────
    gate_result: dict[str, Any] | None = None
    if args.gate_model and Path(args.gate_model).exists():
        logger.info("Step 1.5a: Running Gate scene classifier")
        try:
            from modules.classifier import SceneGate  # type: ignore[import]
            gate = SceneGate(args.gate_model)
            gate_result = gate.predict(args.image)
            logger.info(
                "Gate result: %s (conf=%.3f, trigger=%s)",
                gate_result["scene_class"],
                gate_result["confidence"],
                gate_result["gate_decision"],
            )
            # Override region_type with gate result
            if gate_result["gate_decision"]:
                args.region_type = gate_result["scene_class"]
        except Exception as exc:
            logger.warning("Gate classification failed: %s — continuing without gate", exc)
            gate_result = None
    elif args.gate_model:
        logger.warning("Gate model not found at %s — skipping", args.gate_model)

    # ── Step 1.5b: Fine-grained classification (M2a/M2b) ─────────────────────
    if objects:
        # Ship fine-grained classification
        ship_objs = [o for o in objects if o.get("class", {}).get("super_class") == "ship"]
        if ship_objs and args.ship_cls_model and Path(args.ship_cls_model).exists():
            logger.info("Step 1.5b: M2a ship fine-grained classification (%d ships)", len(ship_objs))
            try:
                from modules.classifier import ShipClassifier  # type: ignore[import]
                ship_clf = ShipClassifier(args.ship_cls_model)
                ship_clf.classify_batch(args.image, objects)
            except Exception as exc:
                logger.warning("Ship classification failed: %s", exc)

        # Aircraft fine-grained classification
        aircraft_objs = [o for o in objects if o.get("class", {}).get("super_class") == "aircraft"]
        if aircraft_objs and args.aircraft_cls_model and Path(args.aircraft_cls_model).exists():
            logger.info(
                "Step 1.5b: M2b aircraft fine-grained classification (%d aircraft)",
                len(aircraft_objs),
            )
            try:
                from modules.classifier import AircraftClassifier  # type: ignore[import]
                aircraft_clf = AircraftClassifier(args.aircraft_cls_model)
                aircraft_clf.classify_batch(args.image, objects)
            except Exception as exc:
                logger.warning("Aircraft classification failed: %s", exc)

    # ── Step 1.5c: M3 FastSAM segmentation (conditional on Gate) ────────────
    segmentation: dict[str, Any] | None = None
    if gate_result and gate_result.get("gate_decision"):
        logger.info(
            "Step 1.5c: M3 FastSAM segmentation (prompt=%s)",
            gate_result["scene_class"],
        )
        try:
            from modules.segmentation import FastSAMSegmenter  # type: ignore[import]
            fastsam = FastSAMSegmenter()
            segmentation = fastsam.segment(
                args.image,
                prompt=gate_result["scene_class"],
            )
            if segmentation.get("success"):
                logger.info(
                    "FastSAM: area=%.4f km² (mask_px=%d)",
                    segmentation.get("mask_area_km2", 0),
                    segmentation.get("mask_area_px", 0),
                )
            else:
                logger.info("FastSAM: no valid segmentation found")
        except Exception as exc:
            logger.warning("FastSAM segmentation failed: %s", exc)
            segmentation = None

    # ── Step 2: Build Evidence Package ────────────────────────────────────────
    logger.info("Step 2: Building Evidence Package")
    package: dict[str, Any] = {}
    try:
        from modules.evidence import EvidenceBuilder  # type: ignore[import]
        mission: dict[str, Any] = {
            "region_name": args.region,
            "region_type": args.region_type,
            "priority": "HIGH",
            "user_prompt": None,
        }
        builder = EvidenceBuilder()
        package = builder.build(
            args.image, mission, objects,
            segmentation=segmentation,
            gate_result=gate_result,
        )
        logger.info(
            "Evidence Package built: %s | status=%s",
            package.get("package_id"), package.get("status"),
        )
        vlm_config = resolve_vlm_config(
            env_file=args.env_file,
            vlm_url=args.vlm_url,
            vlm_model=args.vlm_model,
            require_vlm=args.require_vlm,
        )
        if vlm_config.vlm_base_url:
            from modules.report.vlm_describer import VLMDescriber  # type: ignore[import]
            from modules.report.vlm_trace import attach_vlm_scene_description  # type: ignore[import]
            describer = VLMDescriber(
                base_url=vlm_config.vlm_base_url,
                model_name=vlm_config.vlm_model_name or "qwen3-vl-4b",
                api_key=vlm_config.api_key or "EMPTY",
                timeout=vlm_config.timeout,
            )
            desc = describer.describe(args.image)
            if desc:
                attach_vlm_scene_description(
                    package,
                    description=desc,
                    describer=describer,
                    image_path=args.image,
                )
                logger.info("VLM 场景描述已写入 evidence")
            elif vlm_config.require_vlm:
                raise RuntimeError("SAR_REQUIRE_VLM=1 but VLM returned an empty scene description.")
        elif vlm_config.require_vlm:
            raise RuntimeError("SAR_REQUIRE_VLM=1 but SAR_VLM_URL/--vlm-url is empty.")
    except Exception as exc:
        logger.error("Fatal: Evidence Package build failed: %s", exc)
        return 1

    # ── Step 3: LLM selection + Report generation ─────────────────────────────
    logger.info("Step 3: Initialising ReportPipeline")
    pipeline = None
    try:
        from modules.report.pipeline import ReportPipeline  # type: ignore[import]
        collab_config = resolve_collaborative_config(
            cache_dir=str(output_dir / ".report_cache"),
            env_file=args.env_file,
            small_url=args.llm_url,
            small_model=args.llm_model_name if args.llm_url else None,
            small_model_path=args.small_llm_path,
            large_url=args.large_llm_url,
            large_model=args.large_llm_model_name if args.large_llm_url else None,
            large_model_path=args.large_llm_path,
            local_device=args.llm_device,
            local_max_new_tokens=args.llm_max_new_tokens,
            require_gpu=args.require_gpu,
            allow_template_fallback=args.allow_template_fallback,
        )
        effective_llm_url = collab_config.small_base_url
        effective_llm_model = collab_config.small_model_name
        effective_large_llm_url = collab_config.large_base_url
        effective_large_llm_model = collab_config.large_model_name
        effective_small_local_path = collab_config.small_model_path
        effective_large_local_path = collab_config.large_model_path
        if effective_llm_url or effective_small_local_path:
            if (
                args.use_collaborative_llm
                or effective_large_llm_url
                or effective_large_local_path
                or effective_small_local_path
                or collab_config.require_gpu
                or not collab_config.allow_template_fallback
            ):
                pipeline = ReportPipeline.from_collaborative_models(
                    **collaborative_generator_kwargs(collab_config),
                )
                logger.info(
                    "Using collaborative LLMs: small=%s (%s|%s), large=%s (%s|%s)",
                    collab_config.small_model_name,
                    collab_config.small_base_url,
                    collab_config.small_model_path,
                    collab_config.large_model_name,
                    collab_config.large_base_url,
                    collab_config.large_model_path,
                )
            else:
                pipeline = ReportPipeline(
                    base_url=effective_llm_url,
                    model_name=effective_llm_model,
                    allow_template_fallback=collab_config.allow_template_fallback,
                )
                logger.info("Using API-based LLM: %s", effective_llm_url)
        elif args.llm_path and Path(args.llm_path).exists():
            pipeline = ReportPipeline.from_local_model(
                args.llm_path,
                device=args.llm_device or collab_config.local_device,
                max_new_tokens=collab_config.local_max_new_tokens,
                require_gpu=collab_config.require_gpu,
                allow_template_fallback=collab_config.allow_template_fallback,
            )
            logger.info("Using local LLM: %s", args.llm_path)
        else:
            if not collab_config.allow_template_fallback:
                logger.error("LLM is required but no model backend is configured.")
                return 1
            pipeline = ReportPipeline()
            logger.warning(
                "LLM not configured — using template fallback",
            )
    except Exception as exc:
        logger.error("Fatal: ReportPipeline initialisation failed: %s", exc)
        return 1

    llm_source = _llm_source_tag(pipeline)

    logger.info("Step 3b: Generating report (draft_mode=%s)", draft_mode)
    try:
        package = pipeline.run(
            package,
            output_dir=str(output_dir),
            draft_mode=draft_mode,
        )
        logger.info(
            "Report generated | status=%s | DOCX=%s",
            package.get("status"), _docx_uri(package),
        )
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        return 1

    # ── Step 4: Quality check ─────────────────────────────────────────────────
    logger.info("Step 4: Running Quality Gate")
    try:
        from modules.eval import QualityGate  # type: ignore[import]
        package = QualityGate().evaluate(package)
        qr = _quality_result(package)
        logger.info("Quality Gate result: %s", qr)
    except Exception as exc:
        logger.error("Quality Gate evaluation failed: %s", exc)
        logger.warning("Quality block not populated")

    # ── Step 5: Save evidence JSON ────────────────────────────────────────────
    pkg_id = package.get("package_id", "sar-unknown")
    evidence_path = output_dir / f"{pkg_id}_evidence.json"
    logger.info("Step 5: Saving evidence JSON to %s", evidence_path)
    try:
        with open(evidence_path, "w", encoding="utf-8") as fh:
            json.dump(package, fh, ensure_ascii=False, indent=2)
        logger.info("Evidence JSON saved: %s", evidence_path)
    except OSError as exc:
        logger.error("Failed to save evidence JSON: %s", exc)
        # Non-fatal — continue to summary

    # ── Step 6: Print summary ─────────────────────────────────────────────────
    _print_summary(
        package=package,
        region=args.region,
        region_type=args.region_type,
        objects=objects,
        llm_source=llm_source,
        evidence_path=evidence_path,
    )

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
