#!/bin/bash
# install.sh — Offline installation for SAR classifier training
# Run on the air-gapped machine before training.
#
# Usage:
#   1. Copy the offline_training_pack/ to the air-gapped machine
#   2. cd offline_training_pack/
#   3. bash install.sh
#
# Prerequisites:
#   - Linux x86_64
#   - Python 3.11.x available as python3
#   - NVIDIA driver/GPU compatible with CUDA 13.0

set -euo pipefail

echo "=== SAR Classifier Offline Training Pack ==="
echo ""

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYVER=$("$PYTHON_BIN" --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
ARCH=$(uname -m)
OS_NAME=$(uname -s)

echo "Python version: $PYVER"
echo "Platform: $OS_NAME $ARCH"

if [ "$OS_NAME" != "Linux" ] || [ "$ARCH" != "x86_64" ]; then
    echo "[ERROR] This pack only supports Linux x86_64."
    exit 2
fi

if [ "$PYVER" != "3.11" ]; then
    echo "[ERROR] This pack bundles cp311 wheels and requires Python 3.11.x."
    exit 3
fi

if [ ! -d "wheels/" ] || [ -z "$(ls -A wheels/ 2>/dev/null)" ]; then
    echo "[ERROR] Local wheels/ directory is missing or empty."
    echo "        This pack is intended for fully offline installation."
    exit 4
fi

if [ ! -f "assets/models/FastSAM-s.pt" ]; then
    echo "[ERROR] Missing bundled FastSAM model: assets/models/FastSAM-s.pt"
    exit 5
fi

echo "Installing bootstrap tooling from local wheels..."
"$PYTHON_BIN" -m ensurepip --upgrade

echo "Installing runtime dependencies from local wheels..."
"$PYTHON_BIN" -m pip install --no-index --find-links ./wheels/ -r code/requirements_offline.txt

# Verify installation
"$PYTHON_BIN" -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | CUDA build: {torch.version.cuda}')"
"$PYTHON_BIN" -c "import torchvision; print(f'TorchVision {torchvision.__version__}')"
"$PYTHON_BIN" -c "import cv2; print(f'OpenCV {cv2.__version__}')"
"$PYTHON_BIN" -c "import numpy; print(f'NumPy {numpy.__version__}')"
"$PYTHON_BIN" -c "import ultralytics; print(f'Ultralytics {ultralytics.__version__}')"
"$PYTHON_BIN" -c "from pathlib import Path; p = Path('assets/models/FastSAM-s.pt'); print(f'FastSAM model: {p} ({p.stat().st_size / (1024 * 1024):.1f} MB)')"

echo ""
echo "=== Installation complete ==="
echo "Next steps:"
echo "  1. Place your private dataset images"
echo "  2. python code/modules/classifier/train_gate.py --help"
echo "  3. python code/modules/classifier/train_ship_cls.py --help"
echo "  4. python code/modules/classifier/train_aircraft_cls.py --help"
echo "  5. Optional FastSAM test: python -c \"from modules.segmentation.fastsam_segmenter import FastSAMSegmenter\""
