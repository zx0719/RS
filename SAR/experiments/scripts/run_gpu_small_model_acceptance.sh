#!/usr/bin/env bash
# Run strict GPU-only acceptance for local Qwen3-4B collaborative reports.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/home/zhuxiang/.conda/envs/qwen3-4B/bin/python}"
MODEL_PATH="${SAR_SMALL_LLM_PATH:-/mnt/data/zhuxiang/Qwen/Qwen3-4B}"
DEVICE="${SAR_LLM_DEVICE:-cuda:0}"
VLM_URL="${SAR_VLM_URL:-}"
VLM_MODEL="${SAR_VLM_MODEL:-qwen3-vl-4b}"
REQUIRE_VLM="${SAR_REQUIRE_VLM:-1}"
MANIFEST="${1:-output/large_scene_original_format/original_format_docx_manifest.json}"
OUTPUT="${2:-output/large_scene_original_format/collaborative_acceptance_gpu_report.json}"
ACCEPTANCE_RUN_ID="${SAR_ACCEPTANCE_RUN_ID:-accept-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export SAR_ACCEPTANCE_RUN_ID="${ACCEPTANCE_RUN_ID}"

case "${REQUIRE_VLM,,}" in
  1|true|yes|on) REQUIRE_VLM=1 ;;
  *) REQUIRE_VLM=0 ;;
esac

echo "[gpu-acceptance] python=${PYTHON}"
echo "[gpu-acceptance] model=${MODEL_PATH}"
echo "[gpu-acceptance] device=${DEVICE}"
echo "[gpu-acceptance] require_vlm=${REQUIRE_VLM}"
echo "[gpu-acceptance] acceptance_run_id=${ACCEPTANCE_RUN_ID}"
if [[ -n "${VLM_URL}" ]]; then
  echo "[gpu-acceptance] vlm=${VLM_URL} (${VLM_MODEL})"
fi
if [[ "${REQUIRE_VLM}" == "1" && -z "${VLM_URL}" ]]; then
  echo "[gpu-acceptance] ERROR: strict release acceptance requires SAR_VLM_URL. Start VLM or set SAR_REQUIRE_VLM=0 only for non-release debugging." >&2
fi

PREFLIGHT_ARGS=()
if [[ -n "${VLM_URL}" ]]; then
  PREFLIGHT_ARGS+=(--vlm-url "${VLM_URL}" --vlm-model "${VLM_MODEL}")
fi
if [[ "${REQUIRE_VLM}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--require-vlm)
fi

"${PYTHON}" scripts/preflight_gpu_acceptance.py \
  --env-file .env.collaborative.example \
  --manifest "${MANIFEST}" \
  --model-path "${MODEL_PATH}" \
  --python "${PYTHON}" \
  --device "${DEVICE}" \
  --require-gpu \
  --require-local-gpu \
  "${PREFLIGHT_ARGS[@]}" \
  --output output/large_scene_original_format/gpu_acceptance_preflight.json

if ! nvidia-smi >/dev/null; then
  echo "[gpu-acceptance] ERROR: nvidia-smi failed; GPU is not visible in this shell." >&2
  exit 9
fi
"${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to this Python process.")
print("cuda_available=True")
print("device_count=", torch.cuda.device_count())
print("device0=", torch.cuda.get_device_name(0))
PY

# Avoid accepting cache produced by a CPU-only dry run.
rm -rf output/large_scene_original_format/.report_cache

VLM_ARGS=()
if [[ -n "${VLM_URL}" ]]; then
  VLM_ARGS+=(--vlm-url "${VLM_URL}" --vlm-model "${VLM_MODEL}")
fi
if [[ "${REQUIRE_VLM}" == "1" ]]; then
  VLM_ARGS+=(--require-vlm)
fi

"${PYTHON}" scripts/run_collaborative_acceptance.py \
  --manifest "${MANIFEST}" \
  --env-file .env.collaborative.example \
  --small-model-path "${MODEL_PATH}" \
  --small-model qwen3-4b \
  --local-device "${DEVICE}" \
  --max-new-tokens "${SAR_LLM_MAX_NEW_TOKENS:-256}" \
  --rebuild-reports \
  --require-model-participation \
  --require-gpu \
  --require-local-gpu \
  "${VLM_ARGS[@]}" \
  --skip-refresh \
  --output "${OUTPUT}"

"${PYTHON}" scripts/validate_collaborative_outputs.py \
  --manifest output/large_scene_original_format/original_format_docx_manifest.json \
  --output output/large_scene_original_format/collaborative_validation_gpu_strict.json \
  --require-model-participation \
  --require-gpu \
  --require-local-gpu \
  --require-acceptance-run-id "${ACCEPTANCE_RUN_ID}" \
  --require-fresh-model-output

READINESS_ARGS=(
  --manifest output/large_scene_original_format/original_format_docx_manifest.json
  --output output/large_scene_original_format/release_readiness_gpu_strict.json
  --require-model-participation
  --require-gpu
  --require-local-gpu
  --require-acceptance-run-id
  "${ACCEPTANCE_RUN_ID}"
  --require-fresh-model-output
)
if [[ "${REQUIRE_VLM}" == "1" ]]; then
  "${PYTHON}" scripts/validate_vlm_outputs.py \
    --manifest output/large_scene_original_format/original_format_docx_manifest.json \
    --output output/large_scene_original_format/vlm_validation_gpu_strict.json \
    --require-vlm \
    --require-vlm-trace
  READINESS_ARGS+=(--require-vlm --require-vlm-trace)
else
  echo "[gpu-acceptance] WARNING: SAR_REQUIRE_VLM=0; this is not a full LLM+VLM release gate." >&2
fi

"${PYTHON}" scripts/validate_release_readiness.py "${READINESS_ARGS[@]}"

echo "[gpu-acceptance] PASS: ${OUTPUT}"
