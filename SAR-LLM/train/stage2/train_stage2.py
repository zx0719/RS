from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trainer.stage2_trainer import (  # noqa: E402
    Stage2Trainer,
    apply_lora,
    build_datasets,
    evaluate_loss,
    forward_vqa,
    main,
    run_stage2,
)

__all__ = [
    "Stage2Trainer",
    "apply_lora",
    "build_datasets",
    "evaluate_loss",
    "forward_vqa",
    "main",
    "run_stage2",
]


if __name__ == "__main__":
    main()
