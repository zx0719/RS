#!/usr/bin/env python3
"""
Build NLG training data for Qwen3-4B fine-tuning.

Parses SAR caption datasets and generates:
  - Synthetic Evidence JSON packages (input to LLM)
  - Standard Chinese military report text (target output)

Output format: JSONL with each line being:
{
  "messages": [
    {"role": "system", "content": "<system_prompt>"},
    {"role": "user", "content": "<evidence_json_prompt>"},
    {"role": "assistant", "content": "<report_body>"}
  ]
}

Usage:
    python data/build_nlg_training_data.py \\
        --fsar-cap /mnt/data/mm_data/SAR/FSAR-Cap/FSAR-Captrain.json \\
        --sarlang /mnt/data/mm_data/SAR/SARLANG-1M/Text/Caption/train/Caption_train.json \\
        --output data/nlg_training/train.jsonl \\
        [--max-samples 5000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow imports from the experiments root
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.report.prompt_templates import build_system_prompt, build_user_prompt
from modules.report.generator import ReportGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "1.0.0"
_TASK_TYPE = "intel_brief"

_SHIP_CODES = frozenset({
    "carrier", "destroyer", "frigate", "replenishment",
    "amphibious", "other_vessel",
})
_AIRCRAFT_CODES = frozenset({
    "fighter", "bomber", "transport", "aew", "helicopter", "other_aircraft",
})

_CODE_TO_CN: dict[str, str] = {
    "carrier": "航空母舰",
    "destroyer": "驱逐舰",
    "frigate": "护卫舰",
    "replenishment": "补给舰",
    "amphibious": "两栖舰",
    "other_vessel": "其他舰船",
    "fighter": "战斗机",
    "bomber": "轰炸机",
    "transport": "运输机",
    "aew": "预警机",
    "helicopter": "直升机",
    "other_aircraft": "其他飞机",
}

# Number word → integer
NUMBER_WORDS: dict[str, int] = {
    "one": 1, "a": 1, "an": 1, "single": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
    "several": 3, "multiple": 4, "numerous": 6, "many": 5,
}

# Caption → class code mapping  (regex pattern → code)
CAPTION_SHIP_PATTERNS: list[tuple[str, str]] = [
    (r"\baircraft\s+carrier", "carrier"),
    (r"\bcarrier\b", "carrier"),
    (r"\bwarship", "destroyer"),
    (r"\bdestroyer", "destroyer"),
    (r"\bfrigate", "frigate"),
    (r"\breplenishment\s+ship", "replenishment"),
    (r"\btanker", "replenishment"),
    (r"\bamphibious", "amphibious"),
    (r"\blanding\s+ship", "amphibious"),
    (r"\bcargo\s+ship", "other_vessel"),
    (r"\bcontainer\s+ship", "other_vessel"),
    (r"\bbulk\s+carrier", "other_vessel"),
    (r"\bvessel", "other_vessel"),
    (r"\bship\b", "other_vessel"),
    (r"\bboat\b", "other_vessel"),
]

CAPTION_AIRCRAFT_PATTERNS: list[tuple[str, str]] = [
    (r"\bfighter\b", "fighter"),
    (r"\bbomber\b", "bomber"),
    (r"\btransport\s+aircraft", "transport"),
    (r"\bcargo\s+aircraft", "transport"),
    (r"\baircraft\b", "fighter"),   # generic fallback
    (r"\bplane\b", "fighter"),
    (r"\baew\b", "aew"),
    (r"\bearlywarning", "aew"),
    (r"\bhelicopter\b", "helicopter"),
]

# Location hints from caption → Chinese spatial description
LOCATION_PATTERNS: list[tuple[str, str]] = [
    (r"\bharbor\b", "港口区域"),
    (r"\bport\b", "港口区域"),
    (r"\bpier\b", "码头"),
    (r"\bdock", "船坞"),
    (r"\banchorage\b", "锚地"),
    (r"\bcoast", "沿岸区域"),
    (r"\bairport\b", "机场"),
    (r"\bairbase\b", "空军基地"),
    (r"\bairfield\b", "机场"),
    (r"\btop[\s\-]right", "图像右上方"),
    (r"\btop[\s\-]left", "图像左上方"),
    (r"\bbottom[\s\-]right", "图像右下方"),
    (r"\bbottom[\s\-]left", "图像左下方"),
    (r"\bcenter", "图像中央"),
    (r"\bopen\s+water", "开阔水域"),
    (r"\bopen\s+sea", "开阔海域"),
]

# Satellite name inference from image filename prefix
_SATELLITE_FROM_PREFIX: dict[str, str] = {
    "GF3": "高分三号",
    "GF-3": "高分三号",
    "S1": "哨兵一号",
    "SENTINEL": "哨兵一号",
    "RS2": "雷达卫星-2",
    "PALSAR": "ALOS-PALSAR",
    "TSX": "TerraSAR-X",
    "COSMO": "COSMO-SkyMed",
}

# Region types for scene
_REGION_TYPE_FROM_LOCATION: dict[str, str] = {
    "港口区域": "harbor",
    "码头": "harbor",
    "船坞": "shipyard",
    "锚地": "anchorage",
    "沿岸区域": "coastal_area",
    "机场": "airport",
    "空军基地": "airbase",
    "开阔水域": "coastal_area",
    "开阔海域": "coastal_area",
    "图像右上方": "unknown",
    "图像左上方": "unknown",
    "图像右下方": "unknown",
    "图像左下方": "unknown",
    "图像中央": "unknown",
}

_SCENE_TYPE_CN: dict[str, str] = {
    "harbor": "港口",
    "shipyard": "船坞",
    "anchorage": "锚地",
    "coastal_area": "沿岸区域",
    "airport": "机场",
    "airbase": "空军基地",
    "unknown": "未知",
}

_DISTRIBUTION_TEMPLATES: list[str] = [
    "集中分布于同一区域",
    "分散于{location}",
    "目标相对集中",
    "目标分布于多个区域",
]


# ---------------------------------------------------------------------------
# CaptionToEvidenceConverter
# ---------------------------------------------------------------------------

class CaptionToEvidenceConverter:
    """Convert English SAR captions into synthetic Evidence JSON packages.

    Main methods
    ------------
    parse_caption(caption_text) -> dict
        Extract structured info (counts, types, locations) from caption.
    build_synthetic_evidence(parsed, image_path) -> dict
        Construct a minimal valid Evidence JSON from parsed info.
    generate_chinese_report(evidence) -> str
        Generate a Chinese military report using the template fallback.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._fallback_gen = ReportGenerator(base_url=None)

    # ------------------------------------------------------------------
    # Step 1: parse caption
    # ------------------------------------------------------------------

    def parse_caption(self, caption_text: str) -> dict[str, Any]:
        """Extract structured info from an English SAR caption.

        Parameters
        ----------
        caption_text:
            Raw English caption string, e.g.
            "A SAR image shows five large cargo ships in a harbor area."

        Returns
        -------
        dict with keys:
            objects (list[dict]): each has "code" and "count"
            location_hint (str): Chinese spatial description, may be empty
            spatial_summary (str): qualitative distribution string
        """
        text_lower = caption_text.lower()
        objects: list[dict[str, Any]] = []

        # ── Ship detection ──────────────────────────────────────────────────
        for pattern, code in CAPTION_SHIP_PATTERNS:
            for m in re.finditer(pattern, text_lower):
                count = self._extract_count_before(text_lower, m.start())
                if count > 0:
                    objects.append({"code": code, "count": count})
                    break  # one match per pattern is sufficient

        # ── Aircraft detection ──────────────────────────────────────────────
        for pattern, code in CAPTION_AIRCRAFT_PATTERNS:
            for m in re.finditer(pattern, text_lower):
                count = self._extract_count_before(text_lower, m.start())
                if count > 0:
                    objects.append({"code": code, "count": count})
                    break

        # ── Merge duplicates (sum counts for same code) ─────────────────────
        merged: dict[str, int] = {}
        for obj in objects:
            merged[obj["code"]] = merged.get(obj["code"], 0) + obj["count"]
        objects = [{"code": code, "count": cnt} for code, cnt in merged.items()]

        # ── Location hint ───────────────────────────────────────────────────
        location_hint = ""
        for pattern, desc in LOCATION_PATTERNS:
            if re.search(pattern, text_lower):
                location_hint = desc
                break

        # ── Spatial summary ─────────────────────────────────────────────────
        total = sum(o["count"] for o in objects)
        if total == 0:
            spatial_summary = ""
        elif total == 1:
            spatial_summary = "单目标" + (f"，位于{location_hint}" if location_hint else "")
        elif location_hint:
            spatial_summary = f"分散于{location_hint}"
        else:
            spatial_summary = "目标集中分布于同一区域"

        return {
            "objects": objects,
            "location_hint": location_hint,
            "spatial_summary": spatial_summary,
        }

    # ------------------------------------------------------------------
    # Step 2: build synthetic Evidence JSON
    # ------------------------------------------------------------------

    def build_synthetic_evidence(
        self,
        parsed: dict[str, Any],
        image_path: str,
    ) -> dict[str, Any]:
        """Construct a minimal but valid Evidence JSON from parsed caption.

        Parameters
        ----------
        parsed:
            Output of parse_caption().
        image_path:
            Path to the source image (used to infer satellite name and region).

        Returns
        -------
        Full Evidence Package dict with status "READY_FOR_NLG".
        """
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        filename = os.path.basename(image_path)
        stem = os.path.splitext(filename)[0]

        # ── Satellite inference ─────────────────────────────────────────────
        satellite = "SAR卫星"
        for prefix, sat_name in _SATELLITE_FROM_PREFIX.items():
            if stem.upper().startswith(prefix.upper()):
                satellite = sat_name
                break

        # ── Region / scene inference ────────────────────────────────────────
        location_hint = parsed.get("location_hint", "")
        region_name = "某SAR监测区域"
        region_type = "unknown"
        if location_hint:
            region_type = _REGION_TYPE_FROM_LOCATION.get(location_hint, "unknown")
            if region_type not in ("unknown", ""):
                region_name = f"某{_SCENE_TYPE_CN.get(region_type, '区域')}"

        scene_type_cn = _SCENE_TYPE_CN.get(region_type, "未知")

        # ── Objects list ────────────────────────────────────────────────────
        objects_list: list[dict[str, Any]] = []
        obj_index = 1
        for obj in parsed.get("objects", []):
            code = obj["code"]
            count = obj["count"]
            super_class = "ship" if code in _SHIP_CODES else "aircraft"
            for _ in range(count):
                # Place objects in a small synthetic pixel grid
                cx = self._rng.uniform(100, 900)
                cy = self._rng.uniform(100, 900)
                objects_list.append({
                    "object_id": f"obj-{obj_index:06d}",
                    "class": {
                        "code": code,
                        "name_cn": _CODE_TO_CN.get(code, code),
                        "super_class": super_class,
                    },
                    "score": {
                        "confidence": round(self._rng.uniform(0.65, 0.97), 3),
                    },
                    "geometry": {
                        "pixel": {
                            "center_x": round(cx, 1),
                            "center_y": round(cy, 1),
                        },
                        "geo": {
                            "center_lon": None,
                            "center_lat": None,
                        },
                    },
                })
                obj_index += 1

        # ── Statistics ──────────────────────────────────────────────────────
        ship_count = sum(
            1 for o in objects_list
            if o["class"]["code"] in _SHIP_CODES
        )
        aircraft_count = sum(
            1 for o in objects_list
            if o["class"]["code"] in _AIRCRAFT_CODES
        )
        all_objects = len(objects_list)

        # by_class
        class_counter: dict[str, int] = {}
        for o in objects_list:
            code = o["class"]["code"]
            class_counter[code] = class_counter.get(code, 0) + 1

        by_class = [
            {
                "code": code,
                "name_cn": _CODE_TO_CN.get(code, code),
                "count": cnt,
            }
            for code, cnt in sorted(
                class_counter.items(), key=lambda kv: -kv[1]
            )
        ]

        by_super_class = [
            {"code": "ship", "count": ship_count},
            {"code": "aircraft", "count": aircraft_count},
        ]

        spatial_summary_text = parsed.get("spatial_summary", "")
        if not spatial_summary_text:
            spatial_summary_text = (
                "目标集中分布于同一区域" if all_objects > 1 else "单目标"
            )

        statistics = {
            "totals": {
                "all_objects": all_objects,
                "ships": ship_count,
                "aircraft": aircraft_count,
            },
            "by_class": by_class,
            "by_super_class": by_super_class,
            "spatial_summary": {
                "distribution": spatial_summary_text,
                "cluster_count": 1,
                "nearest_neighbor_mean_m": 0.0,
            },
            "confidence_summary": {
                "mean_confidence": None,
                "low_confidence_count": 0,
                "review_required_count": 0,
            },
        }

        # ── Fake acquisition time (random recent date) ──────────────────────
        days_ago = self._rng.randint(0, 730)
        fake_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Simple offset: use stem hash to pick a deterministic date
        stem_hash = int(hashlib.md5(stem.encode()).hexdigest(), 16)
        offset_days = stem_hash % 730
        from datetime import timedelta
        acq_dt = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=offset_days)
        acq_iso = acq_dt.isoformat()

        # ── Fake image sha256 ───────────────────────────────────────────────
        fake_sha256 = hashlib.sha256(stem.encode()).hexdigest()

        # ── Package ID ──────────────────────────────────────────────────────
        pkg_id = f"sar-{acq_dt.strftime('%Y%m%d')}-{stem_hash % 0xFFFFFF:06X}"

        package: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "package_id": pkg_id,
            "task_type": _TASK_TYPE,
            "status": "READY_FOR_NLG",
            "created_at": now_iso,
            "updated_at": now_iso,
            "trace": {
                "request_id": f"req-{stem_hash % 0xFFFF:04x}",
                "pipeline_run_id": f"pipe-{stem_hash % 0xFFFF:04x}",
                "operator": "nlg_data_builder",
                "source_system": "fsar-cap-synthetic",
            },
            "input": {
                "input_id": f"inp-{stem_hash % 0xFFFFFF:06x}",
                "image": {
                    "uri": image_path,
                    "file_name": filename,
                    "format": Path(filename).suffix.lstrip(".").upper() or "PNG",
                    "sha256": fake_sha256,
                },
                "metadata": {
                    "satellite": satellite,
                    "sensor": "SAR",
                    "acquisition_time": acq_iso,
                },
                "mission": {
                    "region_name": region_name,
                    "region_type": region_type,
                    "priority": "NORMAL",
                },
            },
            "scene": {
                "scene_id": f"scene-{stem_hash % 0xFFFF:04x}",
                "scene_type": region_type,
                "scene_type_cn": scene_type_cn,
                "scene_confidence": None,
                "geo_bounds": None,
                "image_center": None,
                "scale": {"gsd_m": None, "pixel_area_m2": None},
                "environment": {
                    "is_near_coast": region_type in ("harbor", "anchorage", "coastal_area"),
                    "is_airport": region_type in ("airport", "airbase"),
                    "is_harbor": region_type == "harbor",
                },
            },
            "objects": objects_list,
            "statistics": statistics,
            "attachments": {},
            "report": {},
            "quality": {},
            "errors": [],
        }

        return package

    # ------------------------------------------------------------------
    # Step 3: generate Chinese report
    # ------------------------------------------------------------------

    def generate_chinese_report(self, evidence: dict[str, Any]) -> str:
        """Generate a Chinese military-style report using the template fallback.

        Parameters
        ----------
        evidence:
            Evidence Package dict with status "READY_FOR_NLG".

        Returns
        -------
        Chinese report body string (100-300 characters).
        """
        return self._fallback_gen._template_fallback(evidence)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_count_before(self, text: str, match_start: int) -> int:
        """Extract a numeric or word count in the window before *match_start*.

        Looks back up to 30 characters for a digit or number word.
        Returns 0 if nothing is found (signals: skip this match).
        """
        window_start = max(0, match_start - 30)
        window = text[window_start:match_start]

        # Try digit first (e.g. "3 cargo ships")
        digit_match = re.search(r"(\d+)\s*$", window)
        if digit_match:
            val = int(digit_match.group(1))
            return val if 1 <= val <= 50 else 0

        # Try number word (e.g. "five cargo ships")
        tokens = window.split()
        for token in reversed(tokens):
            clean = token.strip(".,;:")
            if clean in NUMBER_WORDS:
                return NUMBER_WORDS[clean]

        return 0


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def load_caption_dataset(path: str) -> list[dict[str, Any]]:
    """Load a FSAR-Cap or SARLANG-1M caption JSONL/JSON file."""
    p = Path(path)
    if not p.exists():
        logger.warning("Dataset file not found: %s", path)
        return []

    with open(p, "r", encoding="utf-8") as fh:
        raw = fh.read().strip()

    # Support JSON array or JSONL
    if raw.startswith("["):
        return json.loads(raw)

    items = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def process_dataset(
    items: list[dict[str, Any]],
    converter: CaptionToEvidenceConverter,
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert a list of caption items to training triples.

    Returns
    -------
    training_items : list[dict]
        List of {"messages": [...]} dicts ready for JSONL output.
    class_dist : dict[str, int]
        Counts per class code observed across all processed items.
    """
    system_prompt = build_system_prompt()
    training_items: list[dict[str, Any]] = []
    class_dist: dict[str, int] = {}
    skipped = 0

    for item in items:
        if max_samples is not None and len(training_items) >= max_samples:
            break

        try:
            messages = item.get("messages", [])
            if len(messages) < 2:
                skipped += 1
                continue
            caption = messages[1].get("content", "")
            images = item.get("images", [])
            image_path = images[0] if images else "unknown.png"
        except (KeyError, IndexError, TypeError):
            skipped += 1
            continue

        parsed = converter.parse_caption(caption)
        if not parsed.get("objects"):
            skipped += 1
            continue

        evidence = converter.build_synthetic_evidence(parsed, image_path)
        chinese_report = converter.generate_chinese_report(evidence)

        if not chinese_report or len(chinese_report) < 30:
            skipped += 1
            continue

        user_prompt = build_user_prompt(evidence)
        training_item: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": chinese_report},
            ]
        }
        training_items.append(training_item)

        # Track class distribution
        for obj in parsed["objects"]:
            code = obj["code"]
            class_dist[code] = class_dist.get(code, 0) + obj["count"]

    logger.info(
        "Processed %d items → %d training triples (%d skipped)",
        len(items), len(training_items), skipped,
    )
    return training_items, class_dist


def save_jsonl(items: list[dict[str, Any]], path: str) -> None:
    """Write items as JSONL (one JSON object per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(items), path)


def split_train_val(
    items: list[dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split items into train and validation sets."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_size:], shuffled[:val_size]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build NLG training data for Qwen3-4B fine-tuning."
    )
    parser.add_argument(
        "--fsar-cap",
        default="/mnt/data/mm_data/SAR/FSAR-Cap/FSAR-Captrain.json",
        help="Path to FSAR-Cap JSON file.",
    )
    parser.add_argument(
        "--sarlang",
        default="/mnt/data/mm_data/SAR/SARLANG-1M/Text/Caption/train/Caption_train.json",
        help="Path to SARLANG-1M Caption JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/nlg_training/train.jsonl",
        help="Output path for training JSONL.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total number of training samples to generate.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    converter = CaptionToEvidenceConverter(seed=args.seed)
    all_items: list[dict[str, Any]] = []
    all_class_dist: dict[str, int] = {}

    # Load and process FSAR-Cap
    fsar_data = load_caption_dataset(args.fsar_cap)
    if fsar_data:
        logger.info("FSAR-Cap: %d raw items", len(fsar_data))
        fsar_limit = args.max_samples  # will be trimmed globally later
        fsar_items, fsar_dist = process_dataset(fsar_data, converter, max_samples=fsar_limit)
        all_items.extend(fsar_items)
        for k, v in fsar_dist.items():
            all_class_dist[k] = all_class_dist.get(k, 0) + v

    # Load and process SARLANG-1M
    sarlang_data = load_caption_dataset(args.sarlang)
    if sarlang_data:
        logger.info("SARLANG-1M: %d raw items", len(sarlang_data))
        remaining = (args.max_samples - len(all_items)) if args.max_samples else None
        sarlang_items, sarlang_dist = process_dataset(
            sarlang_data, converter, max_samples=remaining
        )
        all_items.extend(sarlang_items)
        for k, v in sarlang_dist.items():
            all_class_dist[k] = all_class_dist.get(k, 0) + v

    if not all_items:
        logger.error(
            "No training items generated. Check that data files exist and "
            "contain recognisable ship/aircraft captions."
        )
        sys.exit(1)

    # Apply global max_samples limit
    if args.max_samples and len(all_items) > args.max_samples:
        rng = random.Random(args.seed)
        rng.shuffle(all_items)
        all_items = all_items[: args.max_samples]

    # Split train / val
    train_items, val_items = split_train_val(all_items, val_ratio=args.val_ratio, seed=args.seed)

    # Output paths
    out_dir = Path(args.output).parent
    train_path = str(out_dir / "train.jsonl")
    val_path = str(out_dir / "val.jsonl")
    stats_path = str(out_dir / "stats.json")

    save_jsonl(train_items, train_path)
    save_jsonl(val_items, val_path)

    # Compute avg report length
    report_lengths = [
        len(item["messages"][2]["content"])
        for item in all_items
        if len(item["messages"]) >= 3
    ]
    avg_len = round(sum(report_lengths) / len(report_lengths), 1) if report_lengths else 0.0

    stats: dict[str, Any] = {
        "total_samples": len(all_items),
        "train": len(train_items),
        "val": len(val_items),
        "avg_report_length": avg_len,
        "class_distribution": all_class_dist,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "sources": {
            "fsar_cap": args.fsar_cap,
            "sarlang": args.sarlang,
        },
    }

    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    logger.info("Stats saved to %s", stats_path)

    print(f"\n=== NLG Training Data Summary ===")
    print(f"  Total samples : {len(all_items)}")
    print(f"  Train         : {len(train_items)}")
    print(f"  Val           : {len(val_items)}")
    print(f"  Avg report len: {avg_len:.0f} chars")
    print(f"  Class dist    : {all_class_dist}")
    print(f"  Output dir    : {out_dir}")


if __name__ == "__main__":
    main()
