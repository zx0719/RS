from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.configs.stage2 import PATHS, TRAIN
from src.dataset.stage2_vqa_dataset import (
    MultiPtVqaDataset,
    PtVqaDataset,
    build_vqa_collate_fn,
)
from src.model.qwen3_sar_model import (
    SarQwenVLForCausalLM,
    infer_qwen3_vl_text_hidden_size,
)
from src.model.sarclip_module import BridgeGuidedProjector, TokenLinearProjector


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(cfg_device: str) -> torch.device:
    if cfg_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] config device=cuda，但当前环境没有 CUDA，已回退到 CPU")
        return torch.device("cpu")
    return torch.device(cfg_device)


def init_swanlab():
    if not getattr(TRAIN, "use_swanlab", False):
        return None
    try:
        import swanlab
    except ImportError:
        print("[WARN] swanlab 未安装，已跳过。pip install swanlab")
        return None
    tags = list(getattr(TRAIN, "swanlab_tags", []) or [])
    if getattr(TRAIN, "is_smoke_test", False) and "smoke_test" not in tags:
        tags.append("smoke_test")
    if not TRAIN.use_multi_gpu and "single_gpu" not in tags:
        tags.append("single_gpu")
    if TRAIN.use_lora and "lora" not in tags:
        tags.append("lora")

    try:
        return swanlab.init(
            project=TRAIN.swanlab_project,
            experiment_name=TRAIN.swanlab_experiment_name,
            description=getattr(TRAIN, "swanlab_description", None),
            group=getattr(TRAIN, "swanlab_group", None),
            tags=tags or None,
            logdir=getattr(TRAIN, "swanlab_logdir", None),
            mode=getattr(TRAIN, "swanlab_mode", None),
            config={
                "stage": TRAIN.stage,
                "batch_size": TRAIN.batch_size,
                "lr": TRAIN.lr,
                "lora_lr": TRAIN.lora_lr,
                "max_steps": TRAIN.max_steps,
                "max_length": TRAIN.max_length,
                "num_image_tokens": TRAIN.num_image_tokens,
                "fp16": TRAIN.fp16,
                "use_lora": TRAIN.use_lora,
                "lora_r": TRAIN.lora_r,
                "gradient_checkpointing": TRAIN.gradient_checkpointing,
                "use_multi_gpu": TRAIN.use_multi_gpu,
                "smoke_test": bool(getattr(TRAIN, "is_smoke_test", False)),
                "save_dir": TRAIN.save_dir,
                "stage1_projector_ckpt": PATHS.stage1_projector_ckpt,
                "mixed_weight_sarvqa": TRAIN.mixed_weight_sarvqa,
                "mixed_weight_sartext": TRAIN.mixed_weight_sartext,
                "mixed_weight_sarlang_vqa": TRAIN.mixed_weight_sarlang_vqa,
                "dataset_caps": {
                    "sarvqa": getattr(PATHS, "sarvqa_max_samples", None),
                    "sartext": getattr(PATHS, "sartext_max_samples", None),
                    "sarlang_vqa": getattr(PATHS, "sarlang_vqa_max_samples", None),
                },
            },
        )
    except Exception as e:
        print(f"[WARN] swanlab 初始化失败，已跳过：{e}")
        return None


def load_stage1_projector(model: SarQwenVLForCausalLM) -> None:
    ckpt_path = Path(PATHS.stage1_projector_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"stage1 projector 不存在: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "projector" in state:
        state = state["projector"]
    missing, unexpected = model.projector.load_state_dict(state, strict=False)
    print(f"[INFO] loaded stage1 projector from: {ckpt_path}")
    if missing or unexpected:
        print(f"[WARN] projector 权重加载不干净: missing={missing}, unexpected={unexpected}")


def apply_lora(model: SarQwenVLForCausalLM) -> None:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError("需要安装 peft: pip install peft") from e

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        base_model_name_or_path=PATHS.qwen_path,
        r=TRAIN.lora_r,
        lora_alpha=TRAIN.lora_alpha,
        lora_dropout=TRAIN.lora_dropout,
        target_modules=TRAIN.lora_target_modules,
        bias="none",
    )

    for p in model.vl_model.parameters():
        p.requires_grad = False

    if hasattr(model.llm_causal, "name_or_path"):
        model.llm_causal.name_or_path = PATHS.qwen_path
    model.llm_causal = get_peft_model(model.llm_causal, lora_config)
    if hasattr(model.llm_causal, "peft_config"):
        for cfg in model.llm_causal.peft_config.values():
            cfg.base_model_name_or_path = PATHS.qwen_path
    model.llm_causal.print_trainable_parameters()

    if hasattr(model.llm_causal, "get_input_embeddings"):
        model.llm_backbone = model.llm_causal
    elif hasattr(model.llm_causal, "model"):
        model.llm_backbone = model.llm_causal.model

    print("[INFO] LoRA applied to LLM")


def save_checkpoint(model, optim, scaler, step: int, loss_value: float, save_dir: Path) -> None:
    import os as _os
    import shutil
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 先写到本地 /tmp，避免 RAID 脏页限速（balance_dirty_pages）导致崩溃
    local_dir = Path("/tmp/stage2_ckpt_local")
    local_dir.mkdir(parents=True, exist_ok=True)

    fname = f"checkpoint_step_{step:06d}.pt"
    local_tmp   = local_dir / (fname + ".tmp")
    local_final = local_dir / fname
    final_path  = save_dir / fname

    lora_state = None
    if TRAIN.use_lora:
        try:
            from peft import get_peft_model_state_dict
            lora_state = get_peft_model_state_dict(model.llm_causal)
        except Exception as e:
            print(f"[WARN] 无法保存 LoRA 权重: {e}")

    ckpt = {
        "step": step,
        "loss": float(loss_value),
        "projector": model.projector.state_dict(),
        "lora": lora_state,
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(ckpt, str(local_tmp))
    _os.replace(str(local_tmp), str(local_final))   # 本地原子 rename
    shutil.copy2(str(local_final), str(final_path)) # 再 copy 到 RAID
    print(f"[OK] Saved checkpoint to: {final_path}")
    cleanup_old_checkpoints(save_dir, keep_last=TRAIN.keep_last_n_checkpoints)
    cleanup_old_checkpoints(local_dir, keep_last=TRAIN.keep_last_n_checkpoints)


def cleanup_old_checkpoints(save_dir: Path, keep_last: int = 5) -> None:
    ckpts = sorted(save_dir.glob("checkpoint_step_*.pt"))
    for p in ckpts[:-keep_last]:
        try:
            p.unlink()
            print(f"[INFO] removed old checkpoint: {p}")
        except Exception as e:
            print(f"[WARN] failed to remove {p}: {e}")


def maybe_resume(model, optim, scaler) -> int:
    if not TRAIN.resume_ckpt:
        return 0
    ckpt_path = Path(TRAIN.resume_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"resume_ckpt 不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.projector.load_state_dict(ckpt["projector"], strict=False)
    if ckpt.get("lora") and TRAIN.use_lora:
        try:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model.llm_causal, ckpt["lora"])
        except Exception as e:
            print(f"[WARN] 无法恢复 LoRA 权重: {e}")
    if "optimizer" in ckpt:
        optim.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_step = int(ckpt.get("step", 0))
    print(f"[INFO] resumed from step={start_step}")
    return start_step


def forward_vqa(
    model: SarQwenVLForCausalLM,
    sar_feats: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Optional[torch.Tensor]:
    main_device = model.lm_input_device  # 动态读取，DS 接管后是 GPU

    patch_tokens = sar_feats.to(main_device)
    if not torch.isfinite(patch_tokens).all():
        raise RuntimeError("sar_feats 含 NaN/Inf")

    with torch.amp.autocast("cuda", enabled=False):
        sar_token_embeds = model.projector(patch_tokens.float())

    model._check_projector_dim(sar_token_embeds)

    target_scale = float(model._emb_scale)
    cur_scale = sar_token_embeds.abs().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    sar_token_embeds = sar_token_embeds / (cur_scale + 1e-6) * target_scale
    sar_token_embeds = torch.clamp(
        sar_token_embeds,
        min=-3.0 * target_scale,
        max=3.0 * target_scale,
    )
    sar_token_embeds = sar_token_embeds.to(model.lm_emb_dtype)

    lm_dev = model.lm_input_device
    input_ids = input_ids.to(lm_dev)
    attention_mask = attention_mask.to(lm_dev)
    labels = labels.to(lm_dev)
    sar_token_embeds = sar_token_embeds.to(lm_dev)

    inputs_embeds = model._inject_sar_embeds(input_ids, sar_token_embeds)
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    outputs = model.llm_backbone(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    logits = model.lm_head(outputs.last_hidden_state)
    if not torch.isfinite(logits).all():
        raise RuntimeError("logits 含 NaN/Inf")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if shift_labels.ne(-100).sum() == 0:
        return None

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def build_datasets(model: SarQwenVLForCausalLM):
    collate_fn = build_vqa_collate_fn(
        tokenizer=model.tokenizer,
        image_token=model.image_token,
        num_image_tokens=TRAIN.num_image_tokens,
        max_length=TRAIN.max_length,
    )

    datasets = {}
    if TRAIN.mixed_weight_sarvqa > 0:
        datasets["sarvqa"] = PtVqaDataset(
            root=PATHS.sarvqa_root,
            pt_json=PATHS.sarvqa_pt_train_json,
            max_samples=PATHS.sarvqa_max_samples,
        )
    if TRAIN.mixed_weight_sartext > 0:
        datasets["sartext"] = PtVqaDataset(
            root=PATHS.sartext_root,
            pt_json=PATHS.sartext_pt_train_json,
            max_samples=PATHS.sartext_max_samples,
        )
    if TRAIN.mixed_weight_sarlang_vqa > 0:
        datasets["sarlang_vqa"] = PtVqaDataset(
            root=PATHS.sarlang_vqa_root,
            pt_json=PATHS.sarlang_vqa_pt_train_json,
            max_samples=PATHS.sarlang_vqa_max_samples,
        )
    if not datasets:
        raise ValueError("没有启用任何数据集，请至少打开一个数据集")

    train_ds = MultiPtVqaDataset(datasets=datasets, seed=TRAIN.mixed_seed)
    return train_ds, collate_fn


@torch.no_grad()
def evaluate_loss(model, data_loader) -> Optional[float]:
    model.eval()
    total_loss, total_count = 0.0, 0
    for batch in data_loader:
        if batch is None:
            continue
        try:
            loss = forward_vqa(
                model=model,
                sar_feats=batch["sar_feats"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        except RuntimeError as e:
            print(f"[WARN] eval forward error: {e}, skipping")
            continue
        if loss is not None:
            bs = batch["input_ids"].size(0)
            total_loss += float(loss.item()) * bs
            total_count += bs
    model.train()
    return total_loss / total_count if total_count > 0 else None


class Stage2Trainer:
    def __init__(self) -> None:
        set_seed(42)

        self.use_deepspeed = False  # 改用 DDP，DS ZeRO-2 不适合大量冻结参数的场景
        self.use_ddp = (os.environ.get("USE_DDP", "0") == "1") or \
                       (os.environ.get("USE_DEEPSPEED", "0") == "1")  # 复用启动脚本

        if self.use_ddp:
            import torch.distributed as dist
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.cuda.set_device(self.local_rank)
            self.main_device = torch.device(f"cuda:{self.local_rank}")
            self.is_main = (self.local_rank == 0)
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # PCIe P2P 不稳定，走 SHM
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
        else:
            self.local_rank = 0
            self.world_size = 1
            self.is_main = True
            self.main_device = pick_device(TRAIN.main_device)

        self.train_start_time = 0.0

        print(f"[INFO] main_device={self.main_device}  ddp={self.use_ddp}  world_size={self.world_size}")
        self.model = self._build_model()
        self.train_loader = self._build_train_loader()
        self.optimizer, self.param_groups = self._build_optimizer()

        if self.use_ddp:
            import torch.distributed as dist
            # 不用 DDP 包装（DDP 构造时会 broadcast 全部 4B 参数，极慢）
            # 改为在 backward 后手动 all_reduce 可训练参数的梯度
            # 各 rank 模型权重初始状态相同（同一个 checkpoint），无需 broadcast
            dist.barrier()
            print(f"[INFO] DDP (manual grad all_reduce) ready, world_size={self.world_size}")
            self.swanlab_run = init_swanlab() if self.is_main else None
        else:
            self.swanlab_run = init_swanlab()

        self.model_engine = None

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        # 单卡用 bf16 autocast；DeepSpeed 自己管精度
        self.use_amp = (TRAIN.fp16 or getattr(TRAIN, "bf16", False)) and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if getattr(TRAIN, "bf16", False) else torch.float16
        self.scaler = torch.amp.GradScaler("cuda") if (TRAIN.fp16 and not getattr(TRAIN, "bf16", False)) else None
        if self.use_amp:
            print(f"[INFO] AMP autocast enabled ({self.amp_dtype})")

        self.step = maybe_resume(self.model, self.optimizer, self.scaler)

    def _build_model(self) -> SarQwenVLForCausalLM:
        llm_hidden = infer_qwen3_vl_text_hidden_size(PATHS.qwen_path)
        print(f"[INFO] qwen3-vl text hidden size = {llm_hidden}")

        # 与 Stage 1 保持一致：有 bridge_ckpt 就用 BridgeGuidedProjector
        bridge_ckpt = getattr(TRAIN, "bridge_ckpt", None)
        if bridge_ckpt and Path(bridge_ckpt).exists():
            dim_hidden = int(getattr(TRAIN, "bridge_dim_hidden", 1024))
            projector = BridgeGuidedProjector(
                dim_sar=768, dim_qwen=llm_hidden,
                dim_hidden=dim_hidden,
                bridge_ckpt_path=str(bridge_ckpt),
            ).to(self.main_device)
            print(f"[INFO] Stage2 using BridgeGuidedProjector")
        else:
            projector = TokenLinearProjector(in_dim=768, llm_hidden_size=llm_hidden).to(self.main_device)
        qwen_device_map = None  # DeepSpeed 自己管设备分配，不用 device_map
        # bf16 模式下直接用 bfloat16 加载，节省显存
        load_dtype = torch.bfloat16 if getattr(TRAIN, "bf16", False) else (torch.float16 if TRAIN.fp16 else torch.float32)
        model = SarQwenVLForCausalLM(
            qwen_path=PATHS.qwen_path,
            projector=projector,
            device=str(self.main_device),
            torch_dtype=load_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=qwen_device_map,
            gradient_checkpointing=TRAIN.gradient_checkpointing,
        )
        load_stage1_projector(model)
        if TRAIN.use_lora:
            apply_lora(model)
        return model

    def _build_train_loader(self) -> DataLoader:
        train_ds, collate_fn = build_datasets(self.model)
        print(f"[INFO] stage = {TRAIN.stage}")
        print(f"[INFO] train dataset size = {len(train_ds)}")
        return DataLoader(
            train_ds,
            batch_size=TRAIN.batch_size,
            shuffle=TRAIN.shuffle,
            num_workers=TRAIN.num_workers,
            pin_memory=False,   # 关掉 pin_memory，避免 /dev/shm 耗尽
            persistent_workers=(TRAIN.num_workers > 0 and TRAIN.persistent_workers),
            prefetch_factor=TRAIN.prefetch_factor if TRAIN.num_workers > 0 else None,
            collate_fn=collate_fn,
            drop_last=TRAIN.drop_last,
        )

    def _build_optimizer(self):
        param_groups = [
            {"params": self.model.projector.parameters(), "lr": TRAIN.lr},
        ]
        if TRAIN.use_lora:
            lora_params = [p for p in self.model.llm_causal.parameters() if p.requires_grad]
            if lora_params:
                param_groups.append({"params": lora_params, "lr": TRAIN.lora_lr})
        optimizer = torch.optim.AdamW(param_groups, lr=TRAIN.lr, weight_decay=0.01)
        return optimizer, param_groups

    def _allreduce_grads(self, params) -> None:
        """手动 all_reduce 可训练参数的梯度，替代 DDP 的自动梯度同步。"""
        import torch.distributed as dist
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return
        # 打包成一个 flat tensor 做一次 all_reduce，减少通信次数
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)
        offset = 0
        for g in grads:
            numel = g.numel()
            g.copy_(flat[offset:offset + numel].reshape(g.shape))
            offset += numel

    def _lr_lambda(self, step: int) -> float:
        if step < TRAIN.warmup_steps:
            return float(step + 1) / float(TRAIN.warmup_steps)
        # cosine decay to lr_min=1e-6
        progress = (step - TRAIN.warmup_steps) / max(1, TRAIN.max_steps - TRAIN.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_min_ratio = 1e-6 / TRAIN.lr
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    def _log_step(self, avg_loss: float) -> None:
        if self.step % TRAIN.debug_print_every != 0:
            return
        elapsed = time.time() - self.train_start_time
        lr_now = self.optimizer.param_groups[0]["lr"]
        print(
            f"[step {self.step:6d}/{TRAIN.max_steps}] "
            f"loss={avg_loss:.4f}  lr={lr_now:.2e}  elapsed={elapsed:.1f}s"
        )
        if self.swanlab_run is not None:
            self.swanlab_run.log(
                {"train/loss": avg_loss, "train/step": self.step, "train/lr": lr_now},
                step=self.step + 1,
            )

    def save_final_artifacts(self) -> dict[str, str]:
        save_dir = Path(TRAIN.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        projector_path = save_dir / "projector_stage2.pt"
        torch.save(self.model.projector.state_dict(), projector_path)
        artifacts = {"projector": str(projector_path)}

        if TRAIN.use_lora:
            try:
                adapter_dir = save_dir / "lora_adapter"
                self.model.llm_causal.save_pretrained(str(adapter_dir))
                print(f"[OK] Saved LoRA adapter to: {adapter_dir}")
                artifacts["lora_adapter"] = str(adapter_dir)
            except Exception as e:
                print(f"[WARN] 无法保存 LoRA adapter: {e}")

        tokenizer_dir = save_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.model.tokenizer.save_pretrained(str(tokenizer_dir))
        artifacts["tokenizer"] = str(tokenizer_dir)
        print(f"[OK] Saved projector to: {projector_path}")
        print(f"[OK] Saved tokenizer to: {tokenizer_dir}")
        return artifacts

    def train(self) -> None:
        self.model.train()
        if self.is_main:
            proj_trainable = sum(p.numel() for p in self.model.projector.parameters() if p.requires_grad)
            lora_trainable = sum(p.numel() for p in self.model.llm_causal.parameters() if p.requires_grad) if TRAIN.use_lora else 0
            print(f"[INFO] trainable projector params = {proj_trainable:,}")
            print(f"[INFO] trainable LoRA params      = {lora_trainable:,}")

        self.train_start_time = time.time()
        loss_accum = 0.0
        accum_steps = 0

        while self.step < TRAIN.max_steps:
            for batch in self.train_loader:
                if batch is None:
                    continue

                self.optimizer.zero_grad(set_to_none=True)

                try:
                    if self.use_amp:
                        with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                            loss = forward_vqa(
                                model=self.model,
                                sar_feats=batch["sar_feats"],
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"],
                            )
                    else:
                        loss = forward_vqa(
                            model=self.model,
                            sar_feats=batch["sar_feats"],
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                except RuntimeError as e:
                    if self.is_main:
                        print(f"[WARN] step {self.step} forward error: {e}, skipping")
                    self.step += 1
                    continue

                if loss is None:
                    self.step += 1
                    continue

                params = [p for group in self.param_groups for p in group["params"] if p.requires_grad]
                if self.scaler is not None:
                    scale_before = self.scaler.get_scale()
                    self.scaler.scale(loss).backward()
                    if self.use_ddp:
                        self._allreduce_grads(params)
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(params, TRAIN.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    optimizer_stepped = self.scaler.get_scale() >= scale_before
                else:
                    loss.backward()
                    if self.use_ddp:
                        self._allreduce_grads(params)
                    torch.nn.utils.clip_grad_norm_(params, TRAIN.grad_clip_norm)
                    self.optimizer.step()
                    optimizer_stepped = True

                if optimizer_stepped:
                    self.scheduler.step()

                loss_accum += float(loss.item())
                accum_steps += 1
                if self.is_main and self.step % TRAIN.debug_print_every == 0:
                    avg_loss = loss_accum / max(accum_steps, 1)
                    self._log_step(avg_loss)
                    loss_accum = 0.0
                    accum_steps = 0

                if self.is_main and TRAIN.save_full_checkpoint and self.step > 0 and self.step % TRAIN.save_every_steps == 0:
                    save_checkpoint(
                        model=self.model,
                        optim=self.optimizer,
                        scaler=self.scaler,
                        step=self.step,
                        loss_value=float(loss.item()),
                        save_dir=Path(TRAIN.save_dir),
                    )

                self.step += 1
                if self.step >= TRAIN.max_steps:
                    break

        if self.is_main:
            elapsed = time.time() - self.train_start_time
            artifacts = self.save_final_artifacts()
            if self.swanlab_run is not None:
                final_log = {
                    "train/final_step": self.step,
                    "train/elapsed_sec": elapsed,
                    "artifacts/projector_saved": 1,
                    "artifacts/tokenizer_saved": 1,
                    "artifacts/lora_saved": int("lora_adapter" in artifacts),
                }
                try:
                    import swanlab
                    final_log["artifacts/paths"] = swanlab.Text(
                        "\n".join(f"{name}: {path}" for name, path in artifacts.items()),
                        caption="Saved artifacts",
                    )
                except Exception:
                    pass
                self.swanlab_run.log(final_log, step=max(self.step + 1, 1))
            if self.swanlab_run is not None:
                try:
                    self.swanlab_run.finish()
                except Exception:
                    pass


def run_stage2() -> None:
    trainer = Stage2Trainer()
    trainer.train()


def main() -> None:
    run_stage2()


if __name__ == "__main__":
    main()
