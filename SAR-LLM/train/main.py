from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trainer.stage1_trainer import run_stage1  # noqa: E402
from src.trainer.stage2_trainer import run_stage2  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SAR-LLM training entrypoint.")
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2"],
        required=True,
        help="Select which training stage to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "stage1":
        run_stage1()
        return
    run_stage2()


if __name__ == "__main__":
    main()
