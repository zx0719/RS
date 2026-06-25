#!/usr/bin/env python3
"""Diagnose SAR training data for gate/ship/aircraft classifiers.

This script inspects:
- detection image/label/metadata alignment
- label class_id distribution and label format (HBB/OBB)
- metadata coarse_class/fine_class distribution
- simulated gate label assignment
- simulated ship/aircraft training label assignment
- common failure patterns that make training collapse

Usage:
  python scripts/diagnose_training_data.py --data-root /path/to/data_root
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.class_labels import _CODE_TO_CN

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEBUG_MAX = 20
HARBOR_CLASS_ID = 0
AIRPORT_CLASS_ID = 22

SHIP_CLASS_ID_TO_CODE = {
    1: "military_auxiliary",
    2: "combat_ship",
    3: "liquid_cargo",
    4: "harbor_service",
    5: "bulk_carrier",
    6: "other_vessel",
    7: "survey_vessel",
    10: "amphibious",
    11: "container_ship",
    12: "engineering_vessel",
    13: "fishing_vessel",
    14: "tug_boat",
    15: "passenger_ship",
    16: "ro_ro_ship",
    19: "sailing_vessel",
    20: "research_vessel",
}

AIRCRAFT_CLASS_ID_TO_CODE = {
    8: "combat_aircraft",
    9: "transport_aircraft",
    17: "combat_support_aircraft",
    18: "helicopter",
    21: "other_aircraft",
}

SHIP_COARSE = {
    "舰船", "军用辅助舰船", "作战舰船", "液货船", "港务船", "散货船",
    "舰船_其他", "调查船", "两栖舰船", "集装箱船", "工程船", "渔船",
    "拖船", "客船", "滚装船", "帆船", "研究船",
    "ship", "vessel", "destroyer", "frigate", "carrier",
    "replenishment", "amphibious", "cargo", "tanker", "fishing",
}

AIRCRAFT_COARSE = {
    "飞机", "作战飞机", "运输机", "作战支援飞机", "直升机", "飞机_其他",
    "aircraft", "fighter", "bomber", "transport", "helicopter",
    "aew", "awacs",
}

SCENE_LABELS = {0: "harbor", 1: "airport", 2: "none"}
CN_TO_CODE = {v: k for k, v in _CODE_TO_CN.items()}


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _canonicalize_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in CN_TO_CODE:
        return CN_TO_CODE[text]
    lowered = text.lower().replace(" ", "_")
    return lowered


def _print_header(title: str) -> None:
    print("")
    print("=" * 88)
    print(title)
    print("=" * 88)


def _print_counter(counter: Counter, title: str, limit: int = 20) -> None:
    print(title)
    if not counter:
        print("  <empty>")
        return
    for key, value in counter.most_common(limit):
        print(f"  {key}: {value}")


def _iter_images(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if not image_dir.is_dir():
        return paths
    for ext in IMAGE_EXTS:
        paths.extend(sorted(image_dir.glob(f"*{ext}")))
    return sorted(set(paths))


def _parse_label_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            row: dict[str, Any] = {
                "line_no": line_no,
                "num_cols": len(parts),
                "parts": parts,
                "class_id": _safe_int(parts[0]) if parts else None,
            }
            rows.append(row)
    return rows


def _load_metadata(meta_path: Path) -> tuple[dict[str, dict[str, Any]], Counter, Counter]:
    by_name: dict[str, dict[str, Any]] = {}
    dup_names = Counter()
    instance_count_hist = Counter()
    if not meta_path.exists():
        return by_name, dup_names, instance_count_hist

    with open(meta_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_name = Path(str(rec.get("image_path", ""))).name
            if image_name in by_name:
                dup_names[image_name] += 1
            rec["_line_no"] = line_no
            by_name[image_name] = rec
            instances = rec.get("instances", [])
            instance_count_hist[len(instances)] += 1
    return by_name, dup_names, instance_count_hist


def _scan_polygon_class_ids(poly_path: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    with open(poly_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    items = data if isinstance(data, list) else []
    raw_class_ids = []
    for item in items:
        if isinstance(item, dict):
            raw_class_ids.append(item.get("class_id", item.get("class__id")))
    return items, raw_class_ids


def _determine_gate_label(poly_items: list[dict[str, Any]]) -> tuple[int, str]:
    class_ids = set()
    for item in poly_items:
        if not isinstance(item, dict):
            continue
        class_id = _safe_int(item.get("class_id", item.get("class__id")))
        if class_id is not None:
            class_ids.add(class_id)
    if AIRPORT_CLASS_ID in class_ids:
        return 1, "class_id"
    if HARBOR_CLASS_ID in class_ids:
        return 0, "class_id"

    airport_keywords = {"机场", "airport", "airbase", "airfield", "跑道", "停机坪",
                        "滑行道", "机库", "机棚", "塔台", "弹药库", "端保险道", "联络道",
                        "飞机掩蔽库", "runway", "taxiway", "apron", "hangar"}
    harbor_keywords = {"港口", "harbor", "dock", "pier", "berth", "军港", "码头"}
    for item in poly_items:
        if not isinstance(item, dict):
            continue
        coarse = str(item.get("coarse_class", ""))
        fine = str(item.get("fine_class", ""))
        tokens = set((coarse + " " + fine).lower().split())
        if any(kw in tokens for kw in airport_keywords):
            return 1, "keyword"
        if any(kw in tokens for kw in harbor_keywords):
            return 0, "keyword"
    return 2, "none"


def analyze_detection(data_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    image_dir = data_root / "detection" / "train" / "images"
    label_dir = data_root / "detection" / "train" / "labels"
    meta_path = data_root / "detection" / "metadata" / "samples.jsonl"

    images = _iter_images(image_dir)
    label_files = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
    metadata_by_name, dup_names, instance_count_hist = _load_metadata(meta_path)

    image_names = {p.name for p in images}
    label_stems = {p.stem for p in label_files}
    image_stems = {p.stem for p in images}
    metadata_names = set(metadata_by_name.keys())

    col_hist = Counter()
    class_id_hist = Counter()
    per_file_row_count = {}
    obb_examples = []
    mismatched_instance_count = []

    for label_path in label_files:
        rows = _parse_label_file(label_path)
        per_file_row_count[label_path.stem] = len(rows)
        for row in rows:
            col_hist[row["num_cols"]] += 1
            class_id_hist[row["class_id"]] += 1
            if row["num_cols"] == 9 and len(obb_examples) < DEBUG_MAX:
                obb_examples.append(
                    f"{label_path.name}:{row['line_no']} parts={row['parts'][:9]}"
                )
        image_name = f"{label_path.stem}.jpg"
        meta = metadata_by_name.get(image_name) or metadata_by_name.get(f"{label_path.stem}.png") \
            or metadata_by_name.get(f"{label_path.stem}.bmp")
        if meta is not None:
            instances = meta.get("instances", [])
            if len(instances) != len(rows):
                mismatched_instance_count.append(
                    (label_path.name, len(rows), len(instances), meta.get("_line_no"))
                )

    summary = {
        "image_count": len(images),
        "label_count": len(label_files),
        "metadata_count": len(metadata_by_name),
        "images_without_labels": sorted(image_stems - label_stems),
        "labels_without_images": sorted(label_stems - image_stems),
        "images_without_metadata": sorted(image_names - metadata_names),
        "metadata_without_images": sorted(metadata_names - image_names),
        "label_col_hist": col_hist,
        "label_class_id_hist": class_id_hist,
        "metadata_dup_names": dup_names,
        "metadata_instance_count_hist": instance_count_hist,
        "obb_examples": obb_examples,
        "mismatched_instance_count": mismatched_instance_count[:DEBUG_MAX],
        "metadata_by_name": metadata_by_name,
    }
    return summary, metadata_by_name


def analyze_metadata_semantics(metadata_by_name: dict[str, dict[str, Any]]) -> dict[str, Counter]:
    coarse_hist = Counter()
    fine_hist = Counter()
    source_tif_hist = Counter()

    for rec in metadata_by_name.values():
        source_tif = str(rec.get("source_tif", "")).strip()
        if source_tif:
            source_tif_hist[Path(source_tif).suffix.lower() or "<no_ext>"] += 1
        for inst in rec.get("instances", []):
            coarse_hist[str(inst.get("coarse_class", "") or "<empty>")] += 1
            fine_hist[str(inst.get("fine_class", "") or "<empty>")] += 1
    return {
        "coarse_hist": coarse_hist,
        "fine_hist": fine_hist,
        "source_tif_hist": source_tif_hist,
    }


def analyze_gate(data_root: Path, metadata_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    image_dir = data_root / "detection" / "train" / "images"
    poly_dir = data_root / "segmentation" / "train" / "polygons"
    images = _iter_images(image_dir)

    label_hist = Counter()
    source_hist = Counter()
    missing_poly = 0
    debug_lines = []
    polygon_class_id_hist = Counter()

    for img_path in images:
        rec = metadata_by_name.get(img_path.name)
        source_tif = str(rec.get("source_tif", "")).strip() if rec else ""
        poly_path = poly_dir / f"{Path(source_tif).stem}.json" if source_tif else None
        if poly_path is None or not poly_path.exists():
            label_hist["none"] += 1
            source_hist["missing_polygon"] += 1
            missing_poly += 1
            if len(debug_lines) < DEBUG_MAX:
                debug_lines.append(
                    f"img={img_path.name} source_tif={source_tif or '-'} poly=- label=none reason=missing_polygon"
                )
            continue

        items, raw_class_ids = _scan_polygon_class_ids(poly_path)
        for cid in raw_class_ids:
            polygon_class_id_hist[str(cid)] += 1
        label_id, label_source = _determine_gate_label(items)
        label = SCENE_LABELS[label_id]
        label_hist[label] += 1
        source_hist[label_source] += 1
        if len(debug_lines) < DEBUG_MAX:
            debug_lines.append(
                f"img={img_path.name} source_tif={source_tif or '-'} poly={poly_path.name} "
                f"label={label} source={label_source} class_ids={raw_class_ids[:8]}"
            )

    return {
        "label_hist": label_hist,
        "source_hist": source_hist,
        "missing_poly": missing_poly,
        "debug_lines": debug_lines,
        "polygon_class_id_hist": polygon_class_id_hist,
    }


def _simulate_classifier(
    data_root: Path,
    metadata_by_name: dict[str, dict[str, Any]],
    target_name: str,
    class_id_to_code: dict[int, str],
    coarse_accept: set[str],
    fallback_other: str,
) -> dict[str, Any]:
    label_dir = data_root / "detection" / "train" / "labels"
    label_files = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []

    target_hist = Counter()
    source_hist = Counter()
    skipped_hist = Counter()
    canonical_but_skipped = Counter()
    debug_lines = []

    valid_codes = set(class_id_to_code.values())

    for label_path in label_files:
        rows = _parse_label_file(label_path)
        candidate_names = [f"{label_path.stem}.jpg", f"{label_path.stem}.png", f"{label_path.stem}.bmp"]
        rec = None
        for name in candidate_names:
            rec = metadata_by_name.get(name)
            if rec is not None:
                break
        instances = rec.get("instances", []) if rec else []

        for idx, row in enumerate(rows):
            cls_id = row["class_id"]
            class_code = class_id_to_code.get(cls_id) if cls_id is not None else None
            fine_class = None
            coarse_class = ""
            if idx < len(instances):
                inst = instances[idx]
                coarse_class = str(inst.get("coarse_class", "") or "")
                fine_class = inst.get("fine_class")

            is_target = class_code is not None
            reason = "class_id" if class_code is not None else ""
            if not is_target:
                if coarse_class in coarse_accept:
                    is_target = True
                    reason = "coarse"
                elif fine_class and str(fine_class) in coarse_accept:
                    is_target = True
                    reason = "fine_membership"

            fine_canonical = _canonicalize_name(fine_class)
            coarse_canonical = _canonicalize_name(coarse_class)
            if not is_target:
                skipped_hist["not_target"] += 1
                if fine_canonical in valid_codes:
                    canonical_but_skipped[fine_canonical] += 1
                continue

            target_code = class_code
            if target_code is None and fine_canonical in valid_codes:
                target_code = fine_canonical
                reason = "fine_canonical"
            if target_code is None and coarse_canonical in valid_codes:
                target_code = coarse_canonical
                reason = "coarse_canonical"
            if target_code is None:
                target_code = fallback_other
                if reason in ("coarse", "fine_membership"):
                    reason = f"{reason}->default_other"
                else:
                    reason = "default_other"

            target_hist[target_code] += 1
            source_hist[reason] += 1
            if len(debug_lines) < DEBUG_MAX:
                debug_lines.append(
                    f"img={label_path.stem} idx={idx} cls_id={cls_id} cols={row['num_cols']} "
                    f"class_code={class_code or '-'} coarse={coarse_class or '-'} "
                    f"fine={fine_class or '-'} target={target_code} source={reason}"
                )

    return {
        "target_name": target_name,
        "target_hist": target_hist,
        "source_hist": source_hist,
        "skipped_hist": skipped_hist,
        "canonical_but_skipped": canonical_but_skipped,
        "debug_lines": debug_lines,
    }


def print_warnings(
    detection_summary: dict[str, Any],
    gate_summary: dict[str, Any],
    ship_summary: dict[str, Any],
    aircraft_summary: dict[str, Any],
) -> None:
    _print_header("Warnings")

    class_ids = {cid for cid in detection_summary["label_class_id_hist"] if cid is not None}
    if class_ids and class_ids.issubset({0, 1}):
        print("[WARN] labels/*.txt 第一列只出现 0/1，像是 2 类粗检测标签，不是 23 类细分类标签。")

    if detection_summary["label_col_hist"].get(9, 0) > 0:
        print("[WARN] 发现 9 列标签（OBB）。当前 ship/aircraft 训练脚本仍按前 5 列当 HBB 读，裁图可能严重错误。")

    if detection_summary["mismatched_instance_count"]:
        print("[WARN] label 行数和 metadata instances 数量存在不一致，idx fallback 对齐不可靠。")

    if gate_summary["label_hist"].get("harbor", 0) == 0 and gate_summary["label_hist"].get("airport", 0) == 0:
        print("[WARN] Gate 场景标签全是 none。请检查 source_tif 和 segmentation/train/polygons 的对齐。")

    if ship_summary["canonical_but_skipped"]:
        print("[WARN] 有 ship fine_class 看起来已经是 canonical subtype，但仍被 current logic 跳过。")

    if aircraft_summary["canonical_but_skipped"]:
        print("[WARN] 有 aircraft fine_class 看起来已经是 canonical subtype，但仍被 current logic 跳过。")

    if not any([
        class_ids and class_ids.issubset({0, 1}),
        detection_summary["label_col_hist"].get(9, 0) > 0,
        detection_summary["mismatched_instance_count"],
        gate_summary["label_hist"].get("harbor", 0) == 0 and gate_summary["label_hist"].get("airport", 0) == 0,
        ship_summary["canonical_but_skipped"],
        aircraft_summary["canonical_but_skipped"],
    ]):
        print("No high-confidence structural warning was triggered.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SAR training dataset")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    detection_summary, metadata_by_name = analyze_detection(data_root)
    metadata_semantics = analyze_metadata_semantics(metadata_by_name)
    gate_summary = analyze_gate(data_root, metadata_by_name)
    ship_summary = _simulate_classifier(
        data_root, metadata_by_name, "ship", SHIP_CLASS_ID_TO_CODE, SHIP_COARSE, "other_vessel"
    )
    aircraft_summary = _simulate_classifier(
        data_root, metadata_by_name, "aircraft", AIRCRAFT_CLASS_ID_TO_CODE, AIRCRAFT_COARSE, "other_aircraft"
    )

    _print_header("Paths")
    print(f"data_root: {data_root}")
    print(f"detection images: {data_root / 'detection/train/images'}")
    print(f"detection labels: {data_root / 'detection/train/labels'}")
    print(f"metadata: {data_root / 'detection/metadata/samples.jsonl'}")
    print(f"segmentation polygons: {data_root / 'segmentation/train/polygons'}")

    _print_header("Detection Summary")
    print(f"images={detection_summary['image_count']} labels={detection_summary['label_count']} "
          f"metadata_records={detection_summary['metadata_count']}")
    print(f"images_without_labels={len(detection_summary['images_without_labels'])}")
    print(f"labels_without_images={len(detection_summary['labels_without_images'])}")
    print(f"images_without_metadata={len(detection_summary['images_without_metadata'])}")
    print(f"metadata_without_images={len(detection_summary['metadata_without_images'])}")
    print(f"duplicate_metadata_basenames={sum(detection_summary['metadata_dup_names'].values())}")
    print(f"label/metadata instance mismatches={len(detection_summary['mismatched_instance_count'])}")
    _print_counter(detection_summary["label_col_hist"], "label column count histogram:")
    _print_counter(detection_summary["label_class_id_hist"], "label class_id histogram:", limit=30)
    if detection_summary["obb_examples"]:
        print("sample 9-column label rows:")
        for line in detection_summary["obb_examples"][:DEBUG_MAX]:
            print(f"  {line}")
    if detection_summary["mismatched_instance_count"]:
        print("sample label/instance mismatches:")
        for name, row_cnt, inst_cnt, line_no in detection_summary["mismatched_instance_count"]:
            print(f"  {name}: label_rows={row_cnt} metadata_instances={inst_cnt} metadata_line={line_no}")

    _print_header("Metadata Semantics")
    _print_counter(metadata_semantics["source_tif_hist"], "source_tif suffix histogram:")
    _print_counter(metadata_semantics["coarse_hist"], "top coarse_class values:", limit=30)
    _print_counter(metadata_semantics["fine_hist"], "top fine_class values:", limit=30)
    _print_counter(detection_summary["metadata_instance_count_hist"], "instances-per-image histogram:", limit=20)

    _print_header("Gate Simulation")
    _print_counter(gate_summary["label_hist"], "gate label histogram:")
    _print_counter(gate_summary["source_hist"], "gate label source histogram:")
    _print_counter(gate_summary["polygon_class_id_hist"], "polygon raw class_id/class__id histogram:", limit=30)
    print("sample gate mappings:")
    for line in gate_summary["debug_lines"]:
        print(f"  {line}")

    _print_header("Ship Simulation")
    _print_counter(ship_summary["target_hist"], "ship target histogram:", limit=30)
    _print_counter(ship_summary["source_hist"], "ship label source histogram:", limit=30)
    _print_counter(ship_summary["skipped_hist"], "ship skipped histogram:", limit=10)
    _print_counter(ship_summary["canonical_but_skipped"], "ship canonical fine_class but skipped:", limit=20)
    print("sample ship mappings:")
    for line in ship_summary["debug_lines"]:
        print(f"  {line}")

    _print_header("Aircraft Simulation")
    _print_counter(aircraft_summary["target_hist"], "aircraft target histogram:", limit=20)
    _print_counter(aircraft_summary["source_hist"], "aircraft label source histogram:", limit=20)
    _print_counter(aircraft_summary["skipped_hist"], "aircraft skipped histogram:", limit=10)
    _print_counter(aircraft_summary["canonical_but_skipped"], "aircraft canonical fine_class but skipped:", limit=20)
    print("sample aircraft mappings:")
    for line in aircraft_summary["debug_lines"]:
        print(f"  {line}")

    print_warnings(detection_summary, gate_summary, ship_summary, aircraft_summary)


if __name__ == "__main__":
    main()
