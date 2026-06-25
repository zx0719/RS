#!/usr/bin/env bash
# Rebuild a fully offline-installable SAR training pack.
#
# Usage:
#   bash scripts/build_offline_training_pack.sh
#
# This script runs on a networked Linux x86_64 machine and downloads:
#   - Python 3.11 wheels for the training/inference dependencies
#   - FastSAM-s.pt model weights
# Then it rebuilds offline_training_pack.tar.gz.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACK_DIR="$ROOT_DIR/offline_training_pack"
WHEELS_DIR="$PACK_DIR/wheels"
MODELS_DIR="$PACK_DIR/assets/models"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_DOWNLOAD_BIN=("$PYTHON_BIN" -m pip download)
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
FASTSAM_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt"

mkdir -p "$WHEELS_DIR" "$MODELS_DIR"
rm -f "$WHEELS_DIR"/*.whl

PYVER=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PYVER" != "3.11" ]; then
  echo "[ERROR] build_offline_training_pack.sh must run with Python 3.11, got $PYVER"
  echo "        Example:"
  echo "        PYTHON_BIN=/home/zhuxiang/.conda/envs/sar-intel/bin/python bash scripts/build_offline_training_pack.sh"
  exit 2
fi

echo "[1/4] Downloading PyTorch CUDA 13.0 wheels..."
"${PIP_DOWNLOAD_BIN[@]}" \
  --only-binary=:all: \
  --dest "$WHEELS_DIR" \
  --index-url "$PYTORCH_INDEX_URL" \
  torch==2.11.0+cu130 \
  torchvision==0.26.0+cu130 \
  numpy==2.4.4 \
  pillow==11.3.0

echo "[2/4] Downloading remaining runtime wheels..."
"${PIP_DOWNLOAD_BIN[@]}" \
  --only-binary=:all: \
  --dest "$WHEELS_DIR" \
  opencv-python==4.13.0.92 \
  matplotlib==3.11.0 \
  pyyaml==6.0.3 \
  requests==2.34.2 \
  scipy==1.17.1 \
  psutil==7.2.2 \
  polars==1.41.2

echo "[3/4] Downloading ultralytics wheels without re-resolving torch..."
"${PIP_DOWNLOAD_BIN[@]}" \
  --only-binary=:all: \
  --no-deps \
  --dest "$WHEELS_DIR" \
  ultralytics==8.4.37 \
  ultralytics-thop==2.0.20

echo "[4/4] Downloading bundled FastSAM checkpoint..."
curl -4 -L --fail --output "$MODELS_DIR/FastSAM-s.pt" "$FASTSAM_URL"

echo "Rebuilding offline_training_pack.tar.gz ..."
tar -czf "$ROOT_DIR/offline_training_pack.tar.gz" -C "$ROOT_DIR" offline_training_pack

echo ""
echo "Build complete:"
echo "  tarball: $ROOT_DIR/offline_training_pack.tar.gz"
echo "  wheels:  $(find "$WHEELS_DIR" -maxdepth 1 -type f | wc -l) files"
echo "  model:   $MODELS_DIR/FastSAM-s.pt"
