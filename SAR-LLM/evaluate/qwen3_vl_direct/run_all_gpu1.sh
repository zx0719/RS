#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[INFO] Running stage1 then stage2 on GPU1"
"${SCRIPT_DIR}/run_stage1_gpu1.sh" "$@"
"${SCRIPT_DIR}/run_stage2_gpu1.sh" "$@"
