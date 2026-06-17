from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trainer.stage1_trainer import (  # noqa: E402
    Stage1Trainer,
    build_pt_datasets_and_collate,
    evaluate_loss,
    forward_pt,
    main,
    run_stage1,
)
from src.model.qwen3_sar_model import (  # noqa: E402
    SarQwenVLForCausalLM,
    infer_qwen3_vl_text_hidden_size,
)
from src.model.sarclip_module import TokenLinearProjector  # noqa: E402

__all__ = [
    "Stage1Trainer",
    "SarQwenVLForCausalLM",
    "TokenLinearProjector",
    "build_pt_datasets_and_collate",
    "evaluate_loss",
    "forward_pt",
    "infer_qwen3_vl_text_hidden_size",
    "main",
    "run_stage1",
]


if __name__ == "__main__":
    main()
