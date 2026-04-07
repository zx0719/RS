"""
test_pt_gen_only.py
===================
只跑生成评测（BLEU-4 / ROUGE-L），跳过 Test Loss 计算，速度更快。
全部逻辑复用 test_pt.py，不重复造轮子。

用法：
  # 跑 config 里启用的全部数据集，各取 500 条
  python test_pt_gen_only.py --projector /path/to/projector.pt

  # 只跑指定数据集
  python test_pt_gen_only.py --projector /path/to/projector.pt --datasets sarlang,sartext

  # 全量生成
  python test_pt_gen_only.py --projector /path/to/projector.pt --max_gen_samples -1
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(os.environ.get("SARCLIP_PROJECT_DIR", "/home/qianwentao/SARClip/SARCLIP+QwenVL-pt"))
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config_local import PATHS, TRAIN                                               # noqa: E402
from test_pt import (                                                                # noqa: E402
    load_model,
    build_test_datasets,
    _build_inference_collate,
    evaluate_generation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="test_pt_gen_only.py：只跑生成评测 BLEU-4 / ROUGE-L")
    parser.add_argument("--projector", type=str, required=True,
                        help="projector 权重路径（.pt 纯权重 或 完整 checkpoint）")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="推理 batch size")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="生成时最大新 token 数")
    parser.add_argument("--max_gen_samples", type=int, default=500,
                        help="每个数据集最大样本数；-1 表示全量")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集名，e.g. sarlang,sartext；默认跑 config 里启用的全部")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader num_workers")
    return parser.parse_args()


def main():
    args    = parse_args()
    max_gen = None if args.max_gen_samples == -1 else args.max_gen_samples
    wanted  = [s.strip() for s in args.datasets.split(",")] if args.datasets else None

    # ── 日志文件 ──────────────────────────────────────────
    log_dir  = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"test_pt_gen_{ts}.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    log(f"[INFO] log       → {log_path}")
    log(f"[INFO] projector → {args.projector}")
    log(f"[INFO] datasets  → {args.datasets or '(config defaults)'}")
    log(f"[INFO] max_gen_samples = {args.max_gen_samples}  max_new_tokens = {args.max_new_tokens}")

    # ── 加载模型 ──────────────────────────────────────────
    model = load_model(args.projector)

    # ── 数据集 ────────────────────────────────────────────
    test_datasets = build_test_datasets(wanted)
    log(f"[INFO] datasets to evaluate: {list(test_datasets)}")

    collate_gen = _build_inference_collate(
        tokenizer        = model.tokenizer,
        image_token      = model.image_token,
        num_image_tokens = TRAIN.num_image_tokens,
        max_length       = TRAIN.max_length,
    )

    # ── 逐数据集评测 ──────────────────────────────────────
    all_results: Dict[str, Dict] = {}

    for ds_name, ds in test_datasets.items():
        log(f"\n{'='*60}")
        log(f"  数据集: {ds_name}   test size = {len(ds)}")
        log(f"{'='*60}")
        log(f"  (max_gen_samples={max_gen}, max_new_tokens={args.max_new_tokens})")

        t0 = time.time()
        gen_loader = DataLoader(
            ds,
            batch_size   = args.batch_size,
            shuffle      = False,
            num_workers  = args.num_workers,
            collate_fn   = collate_gen,
            pin_memory   = torch.cuda.is_available(),
        )
        gen_metrics = evaluate_generation(
            model          = model,
            data_loader    = gen_loader,
            max_new_tokens = args.max_new_tokens,
            max_samples    = max_gen,
        )
        all_results[ds_name] = gen_metrics

        log(f"  BLEU-4     = {gen_metrics['bleu4']:.4f}")
        log(f"  ROUGE-L    = {gen_metrics['rougeL']:.4f}")
        log(f"  评测样本数 = {gen_metrics['num_samples']}  ({time.time()-t0:.0f}s)")
        if gen_metrics.get("prompt_examples"):
            log("\n  首条真实 prompt:")
            log(gen_metrics["prompt_examples"][0])

        log(f"\n  样本示例（前 3 条）:")
        for j, (ref, hyp) in enumerate(gen_metrics.get("examples", [])):
            log(f"  [{j+1}] Ref: {ref}")
            log(f"       Hyp: {hyp}")
            log()

    # ── 汇总 ─────────────────────────────────────────────
    log(f"\n{'='*60}")
    log("  汇总结果")
    log(f"{'='*60}")
    header = f"{'数据集':<12}  {'BLEU-4':>8}  {'ROUGE-L':>8}  {'样本数':>8}"
    log(header)
    log("-" * len(header))
    for ds_name, r in all_results.items():
        log(f"{ds_name:<12}  {r['bleu4']:>8.4f}  {r['rougeL']:>8.4f}  {r['num_samples']:>8}")

    log(f"\n[INFO] 完整日志已保存至: {log_path}")
    log_file.close()


if __name__ == "__main__":
    main()
