#!/usr/bin/env python3
"""
Refresh large-scene evidence metadata for collaborative-LLM participation.

This updates report_context / generation_trace fields on existing evidence
artifacts without regenerating the docx payload itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.report.collab_config import (
    cache_dir_from_manifest,
    collaborative_generator_kwargs,
    resolve_collaborative_config,
    resolve_vlm_config,
)
from modules.report.collaborative import CollaborativeReportGenerator
from modules.report.large_scene import attach_large_scene_metadata


DEFAULT_INPUT_MANIFEST = Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/original_format_docx_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh large-scene evidence metadata.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_INPUT_MANIFEST,
        help="Path to original_format_docx_manifest.json",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="可选 .env 配置文件，用于加载协同模型参数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_collaborative_config(
        cache_dir=cache_dir_from_manifest(args.manifest),
        env_file=args.env_file,
    )
    generator = CollaborativeReportGenerator(**collaborative_generator_kwargs(config))
    vlm_config = resolve_vlm_config(env_file=args.env_file)
    describer = None
    if vlm_config.vlm_base_url:
        from modules.report.vlm_describer import VLMDescriber
        from modules.report.vlm_trace import attach_vlm_scene_description
        describer = VLMDescriber(
            base_url=vlm_config.vlm_base_url,
            model_name=vlm_config.vlm_model_name or "qwen3-vl-4b",
            api_key=vlm_config.api_key or "EMPTY",
            timeout=vlm_config.timeout,
        )
    elif vlm_config.require_vlm:
        raise RuntimeError("SAR_REQUIRE_VLM=1 but SAR_VLM_URL is empty.")

    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    for item in items:
        evidence_path = Path(item["evidence_path"])
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence = attach_large_scene_metadata(evidence)
        if describer is not None:
            scene_desc = _resolve_scene_description_source(item, evidence_path)
            if scene_desc is not None:
                desc = describer.describe(str(scene_desc))
                if desc:
                    attach_vlm_scene_description(
                        evidence,
                        description=desc,
                        describer=describer,
                        image_path=str(scene_desc),
                    )
                elif vlm_config.require_vlm:
                    raise RuntimeError(
                        f"VLM scene description is required but empty for {item.get('case_name')}."
                    )
            elif vlm_config.require_vlm:
                raise RuntimeError(
                    f"VLM scene description is required but overview image is missing for {item.get('case_name')}."
                )
        report_context = evidence.setdefault("report_context", {})
        generation_trace = report_context.setdefault("generation_trace", {})
        generation_trace["available_backends"] = generator.describe_backends()
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        print(evidence_path)


def _resolve_scene_description_source(item: dict, evidence_path: Path) -> Path | None:
    source_overview = item.get("source_overview")
    if source_overview:
        overview_path = Path(source_overview)
        if overview_path.exists():
            return overview_path

    case_name = item.get("case_name")
    if case_name:
        fallback = evidence_path.parent / f"{case_name}_overview_docx.png"
        if fallback.exists():
            return fallback
    return None


if __name__ == "__main__":
    main()
