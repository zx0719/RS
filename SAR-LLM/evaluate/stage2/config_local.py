from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


STAGE2_CONFIG = Path(__file__).resolve().parents[2] / "train" / "stage2" / "config_local.py"
spec = importlib.util.spec_from_file_location("sar_llm_train_stage2_config_local", STAGE2_CONFIG)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载 stage2 config: {STAGE2_CONFIG}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

PATHS = module.PATHS
TRAIN = module.TRAIN
