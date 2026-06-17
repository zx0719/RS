#!/usr/bin/env bash
# start_vllm_vlm.sh — 启动 Qwen3-VL-4B-Instruct VLM vLLM 服务（端口 8101）
#
# 用法：
#   bash scripts/start_vllm_vlm.sh [GPU_ID]
#   GPU_ID 默认 1

set -euo pipefail

GPU_ID="${1:-1}"
MODEL_PATH="/mnt/data/zhuxiang/Qwen/Qwen3-VL-4B-Instruct"
PORT=8101
MAX_MODEL_LEN=8192

echo "[VLM] 启动 Qwen3-VL-4B-Instruct vLLM 服务"
echo "  模型: ${MODEL_PATH}"
echo "  GPU:  ${GPU_ID}"
echo "  端口: ${PORT}"

PYTHON="/home/zhuxiang/.conda/envs/qwen3-4B/bin/python"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "qwen3-vl-4b" \
    --port "${PORT}" \
    --host "0.0.0.0" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.60 \
    --max-num-seqs 8 \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code
