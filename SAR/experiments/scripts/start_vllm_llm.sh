#!/usr/bin/env bash
# start_vllm_llm.sh — 启动 Qwen3-4B LLM vLLM 服务（端口 8100）
#
# 用法：
#   bash scripts/start_vllm_llm.sh [GPU_ID]
#   GPU_ID 默认 1（避免与 YOLOv8 推理的 GPU 0 冲突）

set -euo pipefail

GPU_ID="${1:-1}"
MODEL_PATH="/mnt/data/zhuxiang/Qwen/Qwen3-4B"
PORT=8100
MAX_MODEL_LEN=8192   # 情报通报场景不需要超长上下文

echo "[LLM] 启动 Qwen3-4B vLLM 服务"
echo "  模型: ${MODEL_PATH}"
echo "  GPU:  ${GPU_ID}"
echo "  端口: ${PORT}"
echo "  最大上下文: ${MAX_MODEL_LEN}"

PYTHON="/home/zhuxiang/.conda/envs/qwen3-4B/bin/python"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "qwen3-4b" \
    --port "${PORT}" \
    --host "0.0.0.0" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.60 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --trust-remote-code
