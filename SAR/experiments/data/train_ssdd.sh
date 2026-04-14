#!/bin/bash
# Train YOLOv8m-OBB on SSDD
#
# Usage:
#   bash data/train_ssdd.sh [--epochs N] [--device DEVICE] [--batch N]
#
# Defaults:
#   --epochs 100
#   --device 0
#   --batch  16
#
# Example (multi-GPU):
#   bash data/train_ssdd.sh --device 0,1 --batch 32

set -euo pipefail

EPOCHS=100
DEVICE=0
BATCH=16

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)  EPOCHS="$2";  shift 2 ;;
        --device)  DEVICE="$2";  shift 2 ;;
        --batch)   BATCH="$2";   shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

echo "Training YOLOv8m-OBB on SSDD"
echo "  epochs : ${EPOCHS}"
echo "  device : ${DEVICE}"
echo "  batch  : ${BATCH}"

python -m ultralytics train \
    model=yolov8m-obb.pt \
    data=/home/zhuxiang/RS/SAR/experiments/data/ssdd_yolo_obb/dataset.yaml \
    epochs="${EPOCHS}" \
    imgsz=640 \
    batch="${BATCH}" \
    device="${DEVICE}" \
    project=/home/zhuxiang/RS/SAR/experiments/runs/detect \
    name=ssdd_obb \
    patience=20 \
    save=True \
    val=True
