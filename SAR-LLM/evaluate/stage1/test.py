"""
test.py
=======
Stage1 统一评测入口。

- 默认只跑生成评测，输出每条样本结果与 BLEU-1/2/3/4、ROUGE-L
- 如需保留旧的含 loss 评测流程，可在同一入口上加 `--with_loss` 或 `--loss_only`
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(os.environ.get("SARCLIP_PROJECT_DIR", "/home/qianwentao/SARClip/SARCLIP+QwenVL-pt"))
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config_local import TRAIN                                                      # noqa: E402
from test_pt import (                                                               # noqa: E402
    _build_inference_collate,
    _log_generation_metrics,
    _log_sample_records,
    _write_json,
    _write_jsonl,
    build_test_datasets,
    evaluate_generation,
    evaluate_generation_vllm,
    main as legacy_full_eval_main,
    load_model,
    load_vllm_caption_backend,
    release_cuda_resources,
    resolve_generation_backend,
)


def parse_args():
    parser = argparse.ArgumentParser(description="test.py：Stage1 统一评测入口（默认仅生成）")
    parser.add_argument("--projector", type=str, required=True,
                        help="projector 权重路径（.pt 纯权重 或 完整 checkpoint）")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备，如 cuda:0 / cuda:1 / cpu")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="推理 batch size")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="生成时最大新 token 数")
    parser.add_argument("--max_gen_samples", type=int, default=500,
                        help="每个数据集最大样本数；-1 表示全量")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集名，e.g. sarlang,sartext；默认跑 config 里启用的全部")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader num_workers（默认 0，避免 CUDA fork 死锁）")
    parser.add_argument("--progress_every", type=int, default=20,
                        help="每隔多少个 batch 打印一次生成吞吐与耗时")
    parser.add_argument("--gen_backend", type=str, default="auto",
                        choices=["auto", "hf", "vllm"],
                        help="生成评测后端：auto 优先 vllm，缺失时回退 hf")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1,
                        help="vLLM tensor parallel size")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--vllm_max_model_len", type=int, default=None,
                        help="vLLM max_model_len；默认使用模型配置")
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16",
                        help="vLLM dtype，如 auto / float16 / bfloat16")
    parser.add_argument("--vllm_enforce_eager", action="store_true",
                        help="为 vLLM 打开 enforce_eager，便于兼容性排查")
    parser.add_argument("--no_print_samples", action="store_true",
                        help="不在终端/日志中打印每一条样本结果")
    parser.add_argument("--print_prompt_for_each_sample", action="store_true",
                        help="打印每条样本的完整 prompt（默认只打印 ref/hyp）")
    parser.add_argument("--with_loss", action="store_true",
                        help="走兼容实现，追加 Test Loss 评测")
    parser.add_argument("--loss_only", action="store_true",
                        help="走兼容实现，只计算 Test Loss")
    return parser.parse_args()


def main():
    raw_args = sys.argv[1:]
    if "--loss_only" in raw_args or "--with_loss" in raw_args:
        print("[INFO] `test.py` 已统一入口；当前请求包含 loss，转到兼容实现 `test_pt.py`。")
        argv_backup = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]] + [arg for arg in raw_args if arg != "--with_loss"]
            legacy_full_eval_main()
        finally:
            sys.argv = argv_backup
        return

    args = parse_args()
    max_gen = None if args.max_gen_samples == -1 else args.max_gen_samples
    wanted = [s.strip() for s in args.datasets.split(",")] if args.datasets else None
    gen_backend = resolve_generation_backend(args.gen_backend)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"test_pt_gen_{ts}"
    stage1_dir = Path(__file__).parent
    log_dir = stage1_dir / "logs"
    output_root = stage1_dir / "outputs"
    metrics_root = stage1_dir / "metrics"
    log_dir.mkdir(exist_ok=True)
    output_root.mkdir(exist_ok=True)
    metrics_root.mkdir(exist_ok=True)
    log_path = log_dir / f"test_pt_gen_{ts}.log"
    output_dir = output_root / run_name
    metrics_dir = metrics_root / run_name
    output_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    log(f"[INFO] log       → {log_path}")
    log(f"[INFO] projector → {args.projector}")
    log(f"[INFO] device    → {args.device}")
    log(f"[INFO] datasets  → {args.datasets or '(config defaults)'}")
    log(f"[INFO] max_gen_samples = {args.max_gen_samples}  max_new_tokens = {args.max_new_tokens}")
    log(f"[INFO] progress_every = {args.progress_every}")
    log(f"[INFO] generation backend = {gen_backend}")
    log(f"[INFO] outputs dir = {output_dir}")
    log(f"[INFO] metrics dir = {metrics_dir}")
    if gen_backend == "vllm":
        log("[INFO] vLLM 选卡建议通过 CUDA_VISIBLE_DEVICES 控制；--device 主要约束 projector/HF 路径")

    test_datasets = build_test_datasets(wanted)
    log(f"[INFO] datasets to evaluate: {list(test_datasets)}")

    all_results: Dict[str, Dict] = {}
    model = None

    try:
        if gen_backend == "hf":
            model = load_model(args.projector, device=args.device)
            collate_gen = _build_inference_collate(
                tokenizer=model.tokenizer,
                image_token=model.image_token,
                num_image_tokens=TRAIN.num_image_tokens,
                max_length=TRAIN.max_length,
            )

            for ds_name, ds in test_datasets.items():
                log(f"\n{'='*60}")
                log(f"  数据集: {ds_name}   test size = {len(ds)}")
                log(f"{'='*60}")
                log(f"  (max_gen_samples={max_gen}, max_new_tokens={args.max_new_tokens})")

                t0 = time.time()
                gen_loader = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=collate_gen,
                    pin_memory=torch.cuda.is_available(),
                )
                gen_metrics = evaluate_generation(
                    model=model,
                    data_loader=gen_loader,
                    dataset_name=ds_name,
                    max_new_tokens=args.max_new_tokens,
                    max_samples=max_gen,
                    progress_every=args.progress_every,
                    progress_callback=log,
                )
                all_results[ds_name] = gen_metrics
                _log_generation_metrics(log, gen_metrics, time.time() - t0)
                if not args.no_print_samples:
                    _log_sample_records(
                        log,
                        gen_metrics.get("sample_records", []),
                        print_prompt=args.print_prompt_for_each_sample,
                    )
                _write_jsonl(output_dir / f"{ds_name}.jsonl", gen_metrics.get("sample_records", []))
                _write_json(
                    metrics_dir / f"{ds_name}.json",
                    {
                        "dataset": ds_name,
                        "metrics": {
                            "bleu1": gen_metrics["bleu1"],
                            "bleu2": gen_metrics["bleu2"],
                            "bleu3": gen_metrics["bleu3"],
                            "bleu4": gen_metrics["bleu4"],
                            "rougeL": gen_metrics["rougeL"],
                            "num_samples": gen_metrics["num_samples"],
                            "samples_per_sec": gen_metrics["samples_per_sec"],
                            "tokens_per_sec": gen_metrics["tokens_per_sec"],
                            "avg_output_tokens": gen_metrics["avg_output_tokens"],
                            "elapsed_sec": gen_metrics["elapsed_sec"],
                        },
                    },
                )
        else:
            prompt_builder, projector, generator = load_vllm_caption_backend(args)
            try:
                collate_gen = _build_inference_collate(
                    tokenizer=prompt_builder.tokenizer,
                    image_token=prompt_builder.image_token,
                    num_image_tokens=TRAIN.num_image_tokens,
                    max_length=TRAIN.max_length,
                )

                for ds_name, ds in test_datasets.items():
                    log(f"\n{'='*60}")
                    log(f"  数据集: {ds_name}   test size = {len(ds)}")
                    log(f"{'='*60}")
                    log(f"  (max_gen_samples={max_gen}, max_new_tokens={args.max_new_tokens})")

                    t0 = time.time()
                    gen_loader = DataLoader(
                        ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=collate_gen,
                        pin_memory=torch.cuda.is_available(),
                    )
                    gen_metrics = evaluate_generation_vllm(
                        prompt_builder=prompt_builder,
                        projector=projector,
                        generator=generator,
                        data_loader=gen_loader,
                        dataset_name=ds_name,
                        max_new_tokens=args.max_new_tokens,
                        max_samples=max_gen,
                        progress_every=args.progress_every,
                        progress_callback=log,
                    )
                    all_results[ds_name] = gen_metrics
                    _log_generation_metrics(log, gen_metrics, time.time() - t0)
                    if not args.no_print_samples:
                        _log_sample_records(
                            log,
                            gen_metrics.get("sample_records", []),
                            print_prompt=args.print_prompt_for_each_sample,
                        )
                    _write_jsonl(output_dir / f"{ds_name}.jsonl", gen_metrics.get("sample_records", []))
                    _write_json(
                        metrics_dir / f"{ds_name}.json",
                        {
                            "dataset": ds_name,
                            "metrics": {
                                "bleu1": gen_metrics["bleu1"],
                                "bleu2": gen_metrics["bleu2"],
                                "bleu3": gen_metrics["bleu3"],
                                "bleu4": gen_metrics["bleu4"],
                                "rougeL": gen_metrics["rougeL"],
                                "num_samples": gen_metrics["num_samples"],
                                "samples_per_sec": gen_metrics["samples_per_sec"],
                                "tokens_per_sec": gen_metrics["tokens_per_sec"],
                                "avg_output_tokens": gen_metrics["avg_output_tokens"],
                                "elapsed_sec": gen_metrics["elapsed_sec"],
                            },
                        },
                    )
            finally:
                release_cuda_resources(prompt_builder, projector, generator)
    finally:
        if model is not None:
            release_cuda_resources(model)

    log(f"\n{'='*60}")
    log("  汇总结果")
    log(f"{'='*60}")
    header = (
        f"{'数据集':<12}  {'BLEU-1':>8}  {'BLEU-2':>8}  "
        f"{'BLEU-3':>8}  {'BLEU-4':>8}  {'ROUGE-L':>8}  {'样本数':>8}"
    )
    log(header)
    log("-" * len(header))
    for ds_name, r in all_results.items():
        log(
            f"{ds_name:<12}  "
            f"{r['bleu1']:>8.4f}  "
            f"{r['bleu2']:>8.4f}  "
            f"{r['bleu3']:>8.4f}  "
            f"{r['bleu4']:>8.4f}  "
            f"{r['rougeL']:>8.4f}  "
            f"{r['num_samples']:>8}"
        )

    _write_json(
        metrics_dir / "summary.json",
        {
            "projector": args.projector,
            "device": args.device,
            "generation_backend": gen_backend,
            "datasets": {
                ds_name: {
                    "bleu1": r["bleu1"],
                    "bleu2": r["bleu2"],
                    "bleu3": r["bleu3"],
                    "bleu4": r["bleu4"],
                    "rougeL": r["rougeL"],
                    "num_samples": r["num_samples"],
                    "samples_per_sec": r["samples_per_sec"],
                    "tokens_per_sec": r["tokens_per_sec"],
                    "avg_output_tokens": r["avg_output_tokens"],
                    "elapsed_sec": r["elapsed_sec"],
                }
                for ds_name, r in all_results.items()
            },
        },
    )

    log(f"\n[INFO] 完整日志已保存至: {log_path}")
    log(f"[INFO] 样本输出已保存至: {output_dir}")
    log(f"[INFO] 指标汇总已保存至: {metrics_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
