#!/usr/bin/env python3
"""
Rerun collaborative acceptance using the generated recommended env file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "large_scene_original_format"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun acceptance with recommended collaborative env.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "original_format_docx_manifest.json",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env.collaborative.recommended",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "collaborative_acceptance_rerun_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_collaborative_acceptance.py"),
        "--manifest",
        str(args.manifest),
        "--env-file",
        str(args.env_file),
        "--output",
        str(args.output),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    print(args.output)


if __name__ == "__main__":
    main()
