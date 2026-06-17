#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/zhuxiang/.conda/envs/qwen3-4B/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${VLLM_TENSOR_PARALLEL_SIZE:-1}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION:-0.85}}"
MODEL_LEN="${MODEL_LEN:-${VLLM_MAX_MODEL_LEN:-6144}}"
DTYPE="${DTYPE:-${VLLM_DTYPE:-bfloat16}}"
ENFORCE_EAGER="${ENFORCE_EAGER:-${VLLM_ENFORCE_EAGER:-0}}"

unset VLLM_TENSOR_PARALLEL_SIZE
unset VLLM_GPU_MEMORY_UTILIZATION
unset VLLM_MAX_MODEL_LEN
unset VLLM_DTYPE
unset VLLM_ENFORCE_EAGER

ARGS=(
  --stage stage2
  --batch_size "${BATCH_SIZE:-64}"
  --max_gen_samples "${MAX_GEN_SAMPLES:--1}"
  --max_new_tokens "${MAX_NEW_TOKENS:-200}"
  --progress_every "${PROGRESS_EVERY:-20}"
  --vllm_tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --vllm_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --vllm_dtype "${DTYPE}"
)

if [[ -n "${MODEL_PATH:-}" ]]; then
  ARGS+=(--model_path "${MODEL_PATH}")
fi
if [[ -n "${DATASETS:-}" ]]; then
  ARGS+=(--datasets "${DATASETS}")
fi
if [[ -n "${MODEL_LEN}" ]]; then
  ARGS+=(--vllm_max_model_len "${MODEL_LEN}")
fi
if [[ -n "${MIN_PIXELS:-}" ]]; then
  ARGS+=(--min_pixels "${MIN_PIXELS}")
fi
if [[ -n "${MAX_PIXELS:-262144}" ]]; then
  ARGS+=(--max_pixels "${MAX_PIXELS:-262144}")
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ARGS+=(--vllm_enforce_eager)
fi

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Running direct Qwen3-VL stage2 evaluation"
"${PYTHON}" -u evaluate/qwen3_vl_direct/eval_vllm.py "${ARGS[@]}" "$@"
