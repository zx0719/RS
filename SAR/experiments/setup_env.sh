#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Legacy conda bootstrap for the SAR Intelligence Pipeline
#
# Usage:
#   bash setup_env.sh
#
# What this script does:
#   1. Creates (or reuses) the conda env "sar-intel" with Python 3.10
#   2. Installs PyTorch 2.1.x with CUDA 12.1 support from pytorch.org
#   3. Installs all remaining dependencies from requirements.txt
#
# Requirements:
#   - conda (Miniconda or Anaconda) must be on PATH
#   - Internet access (or a local conda/pip mirror configured)
#
# After setup, activate the environment with:
#   conda activate sar-intel
# =============================================================================

set -euo pipefail

echo "[WARN] setup_env.sh is the legacy conda installer."
echo "[WARN] Prefer uv profiles instead:"
echo "       bash setup_uv.sh base|dev|geo|det|api|llm|all"
echo "       see ENVIRONMENT.md"
echo ""

ENV_NAME="sar-intel"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Step 1: Create or reuse the conda environment
# ---------------------------------------------------------------------------
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[INFO] Conda environment '${ENV_NAME}' already exists — skipping creation."
else
    echo "[INFO] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION} ..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    echo "[INFO] Environment created."
fi

# Resolve the path to the env's Python/pip so we don't need to activate
CONDA_BASE="$(conda info --base)"
ENV_PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
ENV_PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

# ---------------------------------------------------------------------------
# Step 2: Install PyTorch with CUDA 12.1 (from pytorch.org index)
# ---------------------------------------------------------------------------
echo "[INFO] Installing PyTorch 2.1 (cu121) ..."
"${ENV_PIP}" install \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# Step 3: Install project requirements (torch already satisfied above)
# ---------------------------------------------------------------------------
echo "[INFO] Installing project requirements from requirements.txt ..."
"${ENV_PIP}" install -r "${SCRIPT_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# Step 4: Install the project itself in editable mode (for imports)
# ---------------------------------------------------------------------------
echo "[INFO] Installing sar-intel-pipeline in editable mode ..."
"${ENV_PIP}" install -e "${SCRIPT_DIR}"

echo ""
echo "============================================================"
echo " Setup complete."
echo " Activate the environment with:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo " Verify GPU availability:"
echo "   python -c \"import torch; print(torch.cuda.is_available())\""
echo ""
echo " Qwen3-4B model path: /mnt/data/zhuxiang/Qwen/Qwen3-4B"
echo "============================================================"
