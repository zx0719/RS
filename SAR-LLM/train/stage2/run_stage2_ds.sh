#!/bin/bash
# Stage 2 训练启动脚本
# 双卡 DDP: bash train/stage2/run_stage2_ds.sh          (默认)
# 单卡:     bash train/stage2/run_stage2_ds.sh --num_gpus 1

set -e
cd "$(dirname "$0")/../.."

NUM_GPUS=2
for arg in "$@"; do
    case $arg in
        --num_gpus) shift; NUM_GPUS=$1; shift ;;
        --num_gpus=*) NUM_GPUS="${arg#*=}" ;;
    esac
done

TORCHRUN=/home/zhuxiang/.conda/envs/qwen3-4B/bin/torchrun
PYTHON=/home/zhuxiang/.conda/envs/qwen3-4B/bin/python

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1   # GPU0/GPU1 PCIe P2P 有问题，走 SHM/socket

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "[INFO] 双卡 DDP + bf16 + flash-attn 启动 (num_gpus=$NUM_GPUS)"
    CUDA_VISIBLE_DEVICES=0,1 \
    USE_DDP=1 \
    $TORCHRUN \
        --nproc_per_node $NUM_GPUS \
        --master_port 29501 \
        train/stage2/train_stage2.py \
        2>&1 | tee /tmp/stage2_ds.log
else
    echo "[INFO] 单卡 bf16 + flash-attn 启动 (GPU0)"
    CUDA_VISIBLE_DEVICES=0 \
    $PYTHON -u train/stage2/train_stage2.py \
        2>&1 | tee /tmp/stage2_train.log
fi
