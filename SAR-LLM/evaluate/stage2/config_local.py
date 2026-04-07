from __future__ import annotations
import sys
from pathlib import Path

# 复用 stage2 训练目录的 config
STAGE2_TRAIN_DIR = Path(__file__).resolve().parent.parent.parent / "train" / "stage2"
if str(STAGE2_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_TRAIN_DIR))

STAGE1_DIR = Path(__file__).resolve().parent.parent.parent / "train" / "stage1"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from config_local import PATHS, TRAIN  # noqa: F401
