#!/usr/bin/env python3
"""
Validate generated docx outputs using zip/XML inspection.

This validator does not require python-docx.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated docx outputs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/original_format_docx_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/zhuxiang/RS/SAR/experiments/output/large_scene_original_format/docx_validation_report.json"),
    )
    return parser.parse_args()


def _xml_text(docx_path: Path, member: str) -> str:
    with ZipFile(docx_path, "r") as zf:
        if member not in zf.namelist():
            return ""
        return zf.read(member).decode("utf-8", errors="ignore")


def _footer_text(docx_path: Path) -> str:
    with ZipFile(docx_path, "r") as zf:
        return "\n".join(
            zf.read(name).decode("utf-8", errors="ignore")
            for name in zf.namelist()
            if name.startswith("word/footer") and name.endswith(".xml")
        )


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    for item in manifest:
        docx_path = Path(item["docx_path"])
        document_xml = _xml_text(docx_path, "word/document.xml")
        footer_xml = _footer_text(docx_path)
        title_ok = "航天通报" in document_xml
        attachment_ok = all(
            text in document_xml
            for text in ("附件1", "附件2", "附件3", "目标分布图", "组成分布统计表", "装备分布统计表")
        )
        footer_ok = "自动生成草稿，待人工审核" in footer_xml or "自动生成草稿，待人工审核" in document_xml
        date_ok = bool(re.search(r"\d{4}年\d{1,2}月\d{1,2}日", document_xml))
        rows.append(
            {
                "case_name": item.get("case_name"),
                "docx_path": str(docx_path),
                "title_ok": title_ok,
                "attachment_ok": attachment_ok,
                "footer_ok": footer_ok,
                "date_ok": date_ok,
                "all_ok": title_ok and attachment_ok and footer_ok and date_ok,
            }
        )

    failures = [str(row["case_name"]) for row in rows if not row["all_ok"]]
    report = {
        "cases": rows,
        "summary": {
            "total_cases": len(rows),
            "ok_cases": sum(1 for row in rows if row["all_ok"]),
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
