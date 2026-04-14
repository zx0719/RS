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

    # Optional — detector
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Path to YOLOv8 .pt weights (if omitted, uses MockDetector)",
    )

    # Optional — LLM
    parser.add_argument(
        "--llm-path",
        default="/mnt/data/zhuxiang/Qwen/Qwen3-4B",
        metavar="PATH",
        help="Local LLM model path (default: /mnt/data/zhuxiang/Qwen/Qwen3-4B)",
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
    gen = getattr(pipeline, "_generator", None)
    if gen is None:
        return "unknown"
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
            detector = DetectorTool(args.model)
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
        objects = detector.detect(args.image)
        logger.info("Detected %d objects", len(objects))
    except Exception as exc:
        logger.error("Detection failed: %s", exc)
        logger.warning("Continuing with empty objects list")
        objects = []

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
        package = builder.build(args.image, mission, objects)
        logger.info(
            "Evidence Package built: %s | status=%s",
            package.get("package_id"), package.get("status"),
        )
    except Exception as exc:
        logger.error("Fatal: Evidence Package build failed: %s", exc)
        return 1

    # ── Step 3: LLM selection + Report generation ─────────────────────────────
    logger.info("Step 3: Initialising ReportPipeline")
    pipeline = None
    try:
        from modules.report.pipeline import ReportPipeline  # type: ignore[import]
        if args.llm_url:
            pipeline = ReportPipeline(
                base_url=args.llm_url,
                model_name=args.llm_model_name,
            )
            logger.info("Using API-based LLM: %s", args.llm_url)
        elif Path(args.llm_path).exists():
            pipeline = ReportPipeline.from_local_model(args.llm_path)
            logger.info("Using local LLM: %s", args.llm_path)
        else:
            pipeline = ReportPipeline()
            logger.warning(
                "LLM not available (path not found: %s) — using template fallback",
                args.llm_path,
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
        logger.warning("Continuing with un-rendered package")

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
