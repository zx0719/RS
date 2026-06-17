"""Stage configuration entry points."""

from src.configs.stage1 import PATHS as STAGE1_PATHS
from src.configs.stage1 import TRAIN as STAGE1_TRAIN
from src.configs.stage2 import PATHS as STAGE2_PATHS
from src.configs.stage2 import TRAIN as STAGE2_TRAIN

__all__ = [
    "STAGE1_PATHS",
    "STAGE1_TRAIN",
    "STAGE2_PATHS",
    "STAGE2_TRAIN",
]
