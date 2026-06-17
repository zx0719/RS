#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python - <<'PY'
import os

from src.configs.stage2 import PATHS, TRAIN
from src.trainer.stage2_trainer import run_stage2


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


smoke_test = env_flag("SMOKE_TEST", False)

TRAIN.device = "cuda:0"
TRAIN.main_device = "cuda:0"
TRAIN.use_multi_gpu = False
TRAIN.qwen_device_map = None
TRAIN.is_smoke_test = smoke_test

TRAIN.batch_size = int(os.environ.get("BATCH_SIZE", "1" if smoke_test else "1"))
TRAIN.max_steps = int(os.environ.get("MAX_STEPS", "5" if smoke_test else str(TRAIN.max_steps)))
TRAIN.max_length = int(os.environ.get("MAX_LENGTH", "512" if smoke_test else str(TRAIN.max_length)))
TRAIN.num_workers = int(os.environ.get("NUM_WORKERS", "0" if smoke_test else str(TRAIN.num_workers)))
TRAIN.gradient_checkpointing = env_flag("GRADIENT_CHECKPOINTING", True if smoke_test else True)
TRAIN.use_lora = env_flag("USE_LORA", TRAIN.use_lora)
TRAIN.fp16 = env_flag("FP16", TRAIN.fp16)
TRAIN.use_swanlab = env_flag("USE_SWANLAB", True if smoke_test else False)

default_exp_name = TRAIN.swanlab_experiment_name
if smoke_test and not default_exp_name.endswith("_smoke"):
    default_exp_name = f"{default_exp_name}_smoke"
TRAIN.swanlab_experiment_name = os.environ.get("SWANLAB_EXPERIMENT_NAME", default_exp_name)
TRAIN.swanlab_mode = os.environ.get("SWANLAB_MODE", "offline" if smoke_test else "")
if TRAIN.swanlab_mode == "":
    TRAIN.swanlab_mode = None
os.environ.pop("SWANLAB_MODE", None)
TRAIN.swanlab_group = os.environ.get("SWANLAB_GROUP", "stage2-single-gpu" if smoke_test else "")
if TRAIN.swanlab_group == "":
    TRAIN.swanlab_group = None
tag_parts = [x.strip() for x in os.environ.get("SWANLAB_TAGS", "").split(",") if x.strip()]
if smoke_test:
    tag_parts.extend(["smoke_test", "single_gpu"])
TRAIN.swanlab_tags = sorted(set(tag_parts))
TRAIN.swanlab_description = os.environ.get(
    "SWANLAB_DESCRIPTION",
    "Stage2 single-GPU smoke test" if smoke_test else "",
)
if TRAIN.swanlab_description == "":
    TRAIN.swanlab_description = None
TRAIN.swanlab_logdir = os.environ.get("SWANLAB_LOGDIR", "")
if TRAIN.swanlab_logdir == "":
    TRAIN.swanlab_logdir = None

if smoke_test:
    TRAIN.debug_print_every = 1
    TRAIN.save_full_checkpoint = False
    PATHS.sarvqa_max_samples = int(os.environ.get("SARVQA_MAX_SAMPLES", "16"))
    PATHS.sartext_max_samples = int(os.environ.get("SARTEXT_MAX_SAMPLES", "16"))
    PATHS.sarlang_vqa_max_samples = int(os.environ.get("SARLANG_VQA_MAX_SAMPLES", "16"))

if "LR" in os.environ:
    TRAIN.lr = float(os.environ["LR"])
if "LORA_LR" in os.environ:
    TRAIN.lora_lr = float(os.environ["LORA_LR"])
if "SAVE_DIR" in os.environ:
    TRAIN.save_dir = os.environ["SAVE_DIR"]
if "SAVE_NAME" in os.environ:
    TRAIN.save_name = os.environ["SAVE_NAME"]
if "RESUME_CKPT" in os.environ:
    TRAIN.resume_ckpt = os.environ["RESUME_CKPT"]
if "STAGE1_PROJECTOR_CKPT" in os.environ:
    PATHS.stage1_projector_ckpt = os.environ["STAGE1_PROJECTOR_CKPT"]

if "MIXED_WEIGHT_SARVQA" in os.environ:
    TRAIN.mixed_weight_sarvqa = float(os.environ["MIXED_WEIGHT_SARVQA"])
if "MIXED_WEIGHT_SARTEXT" in os.environ:
    TRAIN.mixed_weight_sartext = float(os.environ["MIXED_WEIGHT_SARTEXT"])
if "MIXED_WEIGHT_SARLANG_VQA" in os.environ:
    TRAIN.mixed_weight_sarlang_vqa = float(os.environ["MIXED_WEIGHT_SARLANG_VQA"])

print("[INFO] Launch stage2 on a single visible GPU")
print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
print(f"[INFO] smoke_test={smoke_test}")
print(f"[INFO] batch_size={TRAIN.batch_size}, max_length={TRAIN.max_length}, max_steps={TRAIN.max_steps}")
print(f"[INFO] gradient_checkpointing={TRAIN.gradient_checkpointing}, use_lora={TRAIN.use_lora}, fp16={TRAIN.fp16}")
print(f"[INFO] use_swanlab={TRAIN.use_swanlab}, swanlab_mode={TRAIN.swanlab_mode}")
print(f"[INFO] stage1_projector_ckpt={PATHS.stage1_projector_ckpt}")
print(f"[INFO] save_dir={TRAIN.save_dir}")
if smoke_test:
    print(
        "[INFO] dataset caps="
        f"sarvqa:{PATHS.sarvqa_max_samples}, "
        f"sartext:{PATHS.sartext_max_samples}, "
        f"sarlang_vqa:{PATHS.sarlang_vqa_max_samples}"
    )

run_stage2()
PY
