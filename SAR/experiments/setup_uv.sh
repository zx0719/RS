#!/usr/bin/env bash
# Bootstrap SAR experiments with uv.
#
# Usage:
#   bash setup_uv.sh            # lightweight runtime + tests
#   bash setup_uv.sh base       # lightweight runtime only
#   bash setup_uv.sh geo        # add GeoTIFF/rasterio support
#   bash setup_uv.sh det        # add YOLO/Ultralytics support
#   bash setup_uv.sh api        # add OpenAI-compatible API client
#   bash setup_uv.sh llm        # add local Transformers/PyTorch support
#   bash setup_uv.sh all        # install every optional dependency

set -euo pipefail

PROFILE="${1:-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv not found. Install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

cd "$SCRIPT_DIR"

case "$PROFILE" in
    base)
        uv sync
        ;;
    dev|test)
        uv sync --extra dev
        ;;
    geo)
        uv sync --extra dev --extra geo
        ;;
    det)
        uv sync --extra dev --extra det
        ;;
    api)
        uv sync --extra dev --extra api
        ;;
    llm)
        uv sync --extra dev --extra llm
        ;;
    all)
        uv sync --all-extras
        ;;
    *)
        echo "[ERROR] Unknown profile: $PROFILE"
        echo "Valid profiles: base, dev, test, geo, det, api, llm, all"
        exit 2
        ;;
esac

echo ""
echo "[OK] uv environment is ready at: $SCRIPT_DIR/.venv"
echo "Run tests with:"
echo "  uv run pytest tests/ -q"
