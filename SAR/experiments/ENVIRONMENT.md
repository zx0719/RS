# SAR Environment Guide

This directory is now managed by `uv` for day-to-day work. The old conda bootstrap remains available for compatibility, but the recommended path is to install only the capability you need.

## Why

The previous `sar-intel` conda environment is large and mixed several unrelated profiles:

- Core report generation and mock tests.
- GeoTIFF metadata parsing.
- YOLO/Ultralytics detection.
- Local Qwen/Transformers inference.
- API inference and notebooks.

Keeping all of these in one environment pulled in large packages such as PyTorch/CUDA, Ultralytics, Transformers, rasterio, notebooks, and unrelated packages. Use the `uv` profiles below to avoid duplicating heavy dependencies unless a task requires them.

## Profiles

| Profile | Command | Purpose |
| --- | --- | --- |
| base | `bash setup_uv.sh base` | Lightweight runtime: mock detector, evidence builder, fallback report, Word assembly. |
| dev | `bash setup_uv.sh` | Base + pytest/ipykernel. Default for code edits and unit tests. |
| geo | `bash setup_uv.sh geo` | Adds `rasterio` for GeoTIFF metadata/georeferencing. |
| det | `bash setup_uv.sh det` | Adds `ultralytics`; this will also pull PyTorch through Ultralytics. |
| api | `bash setup_uv.sh api` | Adds OpenAI-compatible API client. |
| llm | `bash setup_uv.sh llm` | Adds local Transformers/PyTorch inference dependencies. |
| all | `bash setup_uv.sh all` | Installs every optional dependency. Use only when necessary. |

## Common Commands

```bash
cd /home/zhuxiang/RS/SAR/experiments

# Create/update the lightweight dev environment
bash setup_uv.sh

# Run tests
uv run pytest tests/ -q

# Run the mock/template pipeline
uv run python run_pipeline.py --image /path/to/image.tif --region 某某军港

# Check installed profiles and CUDA visibility
uv run python scripts/check_uv_env.py
```

## Notes

- `.venv/` is ignored by git.
- `uv.lock` should be committed when dependency changes are intentional.
- Avoid installing `all` by default. It duplicates GPU stacks and makes the environment large.
- Keep the existing conda env until GPU detection and local Qwen workflows have been validated under `uv`.

## Current Full Profile Baseline

The full profile has been locked to stable major versions instead of latest unbounded releases:

- `numpy<2`
- `torch>=2.4,<2.5`
- `torchvision>=0.19,<0.20`
- `transformers>=4.40,<5`
- `ultralytics>=8,<9`
- `rasterio>=1.3,<2`
- `openai>=1,<3`

This avoids unexpected jumps such as `torch 2.11/cu13` or `transformers 5.x` when running a full sync.
