# Direct Qwen3-VL vLLM Evaluation

This directory is an extension only. It does not modify the existing stage1 or stage2 evaluation scripts.

It directly evaluates `/mnt/data/zhuxiang/Qwen/Qwen3-VL-4B-Instruct` on the same stage1 caption and stage2 VQA test JSONs, using raw image files and vLLM multimodal inference.

## Run on GPU1

```bash
bash evaluate/qwen3_vl_direct/run_stage1_gpu1.sh
bash evaluate/qwen3_vl_direct/run_stage2_gpu1.sh
```

Run both:

```bash
bash evaluate/qwen3_vl_direct/run_all_gpu1.sh
```

Common overrides:

```bash
MAX_GEN_SAMPLES=-1 BATCH_SIZE=8 bash evaluate/qwen3_vl_direct/run_stage1_gpu1.sh
DATASETS=sarvqa MAX_GEN_SAMPLES=100 bash evaluate/qwen3_vl_direct/run_stage2_gpu1.sh
MODEL_LEN=6144 GPU_MEMORY_UTILIZATION=0.8 bash evaluate/qwen3_vl_direct/run_stage1_gpu1.sh
```

GPU binding is only in the shell scripts through `CUDA_VISIBLE_DEVICES=1`; the Python code does not hard-code device IDs.

## Tuned GPU1 Defaults

The GPU1 scripts default to full evaluation and use the fastest small-batch setting measured on the local A800 80GB GPU:

- `BATCH_SIZE=64`
- `MAX_GEN_SAMPLES=-1`
- `MODEL_LEN=6144`
- `MAX_PIXELS=262144`
- `GPU_MEMORY_UTILIZATION=0.85`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`

`MODEL_LEN=6144` is used because SARText contains images whose Qwen3-VL multimodal prompt length can exceed 4096 tokens. Batch-level failures are retried one sample at a time, and only unrecoverable single samples are skipped.

Short benchmark results with Qwen3-VL-4B-Instruct:

- Stage1 `sarlang`, 128 samples, batch 64: about `20.2 samples/s`
- Stage2 `sarvqa`, 128 samples, batch 64: about `6.2 samples/s`

Expected full runtime on GPU1:

- Stage1 default datasets: `sarlang` 6190 + `sartext` 13621 = 19811 samples, roughly 30-40 minutes plus startup/metric overhead under full-run I/O.
- Stage2 default usable dataset: `sarvqa` 2488 samples, roughly 7-10 minutes plus startup/metric overhead. `sarlang_vqa` test JSON is currently empty and is skipped.
- End-to-end `run_all_gpu1.sh`: roughly 40-55 minutes under similar GPU load.

Monitor a running full job:

```bash
tail -f evaluate/qwen3_vl_direct/logs/direct_qwen3vl_stage1_<timestamp>.log
tail -f evaluate/qwen3_vl_direct/logs/direct_qwen3vl_stage2_<timestamp>.log
```

## Outputs

Results are saved under:

- `evaluate/qwen3_vl_direct/outputs/direct_qwen3vl_<stage>_<timestamp>/`
- `evaluate/qwen3_vl_direct/metrics/direct_qwen3vl_<stage>_<timestamp>/`
- `evaluate/qwen3_vl_direct/logs/direct_qwen3vl_<stage>_<timestamp>.log`
