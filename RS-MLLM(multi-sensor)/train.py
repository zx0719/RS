import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.dataset.ben_stage1_feature_dataset import (
    BENStage1FeatureDataset,
    ben_stage1_collate_fn,
)
from src.loss.stage1_losses import Stage1Loss
from src.model.qwen3_feature_vlm import Qwen3FeatureVLM
from src.trainer.stage1_trainer import Stage1Trainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qwen_path",
        type=str,
        default="/mnt/data/zhuxiang/Qwen/Qwen3-VL-4B-Instruct",
    )
    parser.add_argument(
        "--ms_mapping_csv",
        type=str,
        default="/mnt/data/mm_data/ben-mm/encoder_outputs/feature_mapping.csv",
    )
    parser.add_argument(
        "--sar_mapping_csv",
        type=str,
        default="/mnt/data/qianwentao/benmm_s1_CHW/feature_mapping.csv",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="/mnt/data/mm_data/ben-mm/metadata.parquet",
    )

    parser.add_argument("--image_token", type=str, default="<rs_patch>")
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="Describe the land-cover content of this remote sensing patch.",
    )

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)

    parser.add_argument("--num_ms_queries", type=int, default=32)
    parser.add_argument("--num_sar_tokens", type=int, default=4)
    parser.add_argument("--bridge_dim", type=int, default=512)
    parser.add_argument("--num_resampler_layers", type=int, default=2)
    parser.add_argument("--num_fusion_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_align", type=float, default=0.05)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./outputs/stage1_run1")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()
    set_seed(args.seed + rank)

    os.makedirs(args.save_dir, exist_ok=True)

    if is_main_process():
        print(f"[init] distributed={distributed} rank={rank} world_size={world_size} local_rank={local_rank}")

    train_dataset = BENStage1FeatureDataset(
        ms_mapping_csv=args.ms_mapping_csv,
        sar_mapping_csv=args.sar_mapping_csv,
        metadata_path=args.metadata_path,
        split="train",
        image_token=args.image_token,
        prompt_template=args.prompt_template,
        max_samples=args.max_train_samples,
    )
    val_dataset = BENStage1FeatureDataset(
        ms_mapping_csv=args.ms_mapping_csv,
        sar_mapping_csv=args.sar_mapping_csv,
        metadata_path=args.metadata_path,
        split="val",
        image_token=args.image_token,
        prompt_template=args.prompt_template,
        max_samples=args.max_val_samples,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=ben_stage1_collate_fn,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=ben_stage1_collate_fn,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    model = Qwen3FeatureVLM(
        qwen_name_or_path=args.qwen_path,
        image_token=args.image_token,
        bridge_dim=args.bridge_dim,
        num_ms_queries=args.num_ms_queries,
        num_sar_tokens=args.num_sar_tokens,
        num_resampler_layers=args.num_resampler_layers,
        num_fusion_layers=args.num_fusion_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_llm=True,
    )

    criterion = Stage1Loss(lambda_align=args.lambda_align)

    trainer = Stage1Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        save_dir=args.save_dir,
        bf16=args.bf16,
        fp16=args.fp16,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        distributed=distributed,
        local_rank=local_rank,
        train_sampler=train_sampler,
    )

    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
