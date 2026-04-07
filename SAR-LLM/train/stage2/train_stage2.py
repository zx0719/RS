"""
train_stage2.py
===============
Stage2：在 Stage1 projector 基础上，用 VQA 数据微调 projector + LLM（LoRA）。

前提：
  1. Stage1 已训好 projector，路径填入 config_local.PATHS.stage1_projector_ckpt
  2. VQA 数据的 .pt 特征已提取，pt_json 路径已填写
  3. 如需 LoRA，pip install peft

用法：
  python train_stage2.py
"""

from __future__ import annotations

import os
import sys
import time
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

STAGE1_DIR = Path(__file__).resolve().parent.parent / "stage1"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config_local import PATHS, TRAIN                          # noqa: E402
from sarclip_module import TokenLinearProjector                # noqa: E402
from qwen3_sar_model import SarQwenVLForCausalLM, infer_qwen3_vl_text_hidden_size  # noqa: E402
from vqa_dataset_pt import (                                   # noqa: E402
    PtVqaDataset,
    MultiPtVqaDataset,
    build_vqa_collate_fn,
)

# ─────────────────────────────────────────────
# 1. 工具函数
# ─────────────────────────────────────────────

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
    return swanlab.init(
        project=TRAIN.swanlab_project,
        experiment_name=TRAIN.swanlab_experiment_name,
        config={
            "stage":            TRAIN.stage,
            "batch_size":       TRAIN.batch_size,
            "lr":               TRAIN.lr,
            "lora_lr":          TRAIN.lora_lr,
            "max_steps":        TRAIN.max_steps,
            "max_length":       TRAIN.max_length,
            "num_image_tokens": TRAIN.num_image_tokens,
            "fp16":             TRAIN.fp16,
            "use_lora":         TRAIN.use_lora,
            "lora_r":           TRAIN.lora_r,
        },
    )


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
    """对 LLM 应用 LoRA，使其可训练。"""
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise ImportError("需要安装 peft: pip install peft")

    lora_config = LoraConfig(
        task_type        = TaskType.CAUSAL_LM,
        r                = TRAIN.lora_r,
        lora_alpha       = TRAIN.lora_alpha,
        lora_dropout     = TRAIN.lora_dropout,
        target_modules   = TRAIN.lora_target_modules,
        bias             = "none",
    )

    # 先解冻 LLM，再应用 LoRA
    for p in model.vl_model.parameters():
        p.requires_grad = False

    model.llm_causal = get_peft_model(model.llm_causal, lora_config)
    model.llm_causal.print_trainable_parameters()

    # 更新 backbone 引用（LoRA 包装后结构不变）
    if hasattr(model.llm_causal, "get_input_embeddings"):
        model.llm_backbone = model.llm_causal
    elif hasattr(model.llm_causal, "model"):
        model.llm_backbone = model.llm_causal.model

    print("[INFO] LoRA applied to LLM")


def save_checkpoint(model, optim, scaler, step, loss_value, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"checkpoint_step_{step:06d}.pt"

    # 保存 projector 权重
    projector_state = model.projector.state_dict()

    # 保存 LoRA 权重（如果有）
    lora_state = None
    if TRAIN.use_lora:
        try:
            from peft import get_peft_model_state_dict
            lora_state = get_peft_model_state_dict(model.llm_causal)
        except Exception as e:
            print(f"[WARN] 无法保存 LoRA 权重: {e}")

    ckpt = {
        "step":       step,
        "loss":       float(loss_value),
        "projector":  projector_state,
        "lora":       lora_state,
        "optimizer":  optim.state_dict(),
        "scaler":     scaler.state_dict() if scaler is not None else None,
    }
    torch.save(ckpt, save_path)
    print(f"[OK] Saved checkpoint to: {save_path}")
    _cleanup_old_checkpoints(save_dir, keep_last=TRAIN.keep_last_n_checkpoints)


def _cleanup_old_checkpoints(save_dir: Path, keep_last: int = 5) -> None:
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


# ─────────────────────────────────────────────
# 2. Forward
# ─────────────────────────────────────────────

def forward_vqa(
    model:          SarQwenVLForCausalLM,
    sar_feats:      torch.Tensor,   # (B, 195, 768)
    input_ids:      torch.Tensor,   # (B, L)
    attention_mask: torch.Tensor,   # (B, L)
    labels:         torch.Tensor,   # (B, L)
) -> Optional[torch.Tensor]:
    """
    Stage2 forward：projector + LoRA LLM 均参与梯度计算。
    """
    main_device = model.main_device

    patch_tokens = sar_feats.to(main_device)
    if not torch.isfinite(patch_tokens).all():
        raise RuntimeError("sar_feats 含 NaN/Inf")

    with torch.amp.autocast("cuda", enabled=False):
        sar_token_embeds = model.projector(patch_tokens.float())

    model._check_projector_dim(sar_token_embeds)

    target_scale = float(model._emb_scale)
    cur_scale    = sar_token_embeds.abs().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    sar_token_embeds = sar_token_embeds / (cur_scale + 1e-6) * target_scale
    sar_token_embeds = torch.clamp(sar_token_embeds,
                                   min=-3.0 * target_scale, max=3.0 * target_scale)
    sar_token_embeds = sar_token_embeds.to(model.lm_emb_dtype)

    lm_dev = model.lm_input_device
    input_ids      = input_ids.to(lm_dev)
    attention_mask = attention_mask.to(lm_dev)
    labels         = labels.to(lm_dev)
    sar_token_embeds = sar_token_embeds.to(lm_dev)

    inputs_embeds = model._inject_sar_embeds(input_ids, sar_token_embeds)
    position_ids  = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    # Stage2：LLM 参与梯度（LoRA 参数可训练）
    outputs = model.llm_backbone(
        inputs_embeds  = inputs_embeds,
        attention_mask = attention_mask,
        position_ids   = position_ids,
        use_cache      = False,
        return_dict    = True,
    )
    hidden_states = outputs.last_hidden_state
    logits        = model.lm_head(hidden_states)

    if not torch.isfinite(logits).all():
        raise RuntimeError("logits 含 NaN/Inf")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if shift_labels.ne(-100).sum() == 0:
        return None

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


# ─────────────────────────────────────────────
# 3. 数据集构建
# ─────────────────────────────────────────────

def build_datasets(model: SarQwenVLForCausalLM):
    collate_fn = build_vqa_collate_fn(
        tokenizer        = model.tokenizer,
        image_token      = model.image_token,
        num_image_tokens = TRAIN.num_image_tokens,
        max_length       = TRAIN.max_length,
    )

    datasets = {}

    if TRAIN.mixed_weight_sarvqa > 0:
        datasets["sarvqa"] = PtVqaDataset(
            root        = PATHS.sarvqa_root,
            pt_json     = PATHS.sarvqa_pt_train_json,
            max_samples = PATHS.sarvqa_max_samples,
        )

    if TRAIN.mixed_weight_sartext > 0:
        datasets["sartext"] = PtVqaDataset(
            root        = PATHS.sartext_root,
            pt_json     = PATHS.sartext_pt_train_json,
            max_samples = PATHS.sartext_max_samples,
        )

    if TRAIN.mixed_weight_sarlang_vqa > 0:
        datasets["sarlang_vqa"] = PtVqaDataset(
            root        = PATHS.sarlang_vqa_root,
            pt_json     = PATHS.sarlang_vqa_pt_train_json,
            max_samples = PATHS.sarlang_vqa_max_samples,
        )

    if not datasets:
        raise ValueError("没有启用任何数据集，请至少打开一个数据集")

    train_ds = MultiPtVqaDataset(datasets=datasets, seed=TRAIN.mixed_seed)
    return train_ds, collate_fn


# ─────────────────────────────────────────────
# 4. 验证 loss
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_loss(model, data_loader) -> Optional[float]:
    model.eval()
    total_loss, total_count = 0.0, 0
    for batch in data_loader:
        if batch is None:
            continue
        try:
            loss = forward_vqa(
                model          = model,
                sar_feats      = batch["sar_feats"],
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                labels         = batch["labels"],
            )
        except RuntimeError as e:
            print(f"[WARN] eval forward error: {e}, skipping")
            continue
        if loss is not None:
            bs = batch["input_ids"].size(0)
            total_loss  += float(loss.item()) * bs
            total_count += bs
    model.train()
    return total_loss / total_count if total_count > 0 else None


# ─────────────────────────────────────────────
# 5. main
# ─────────────────────────────────────────────

def main():
    set_seed(42)
    swanlab_run = init_swanlab()

    main_device = pick_device(TRAIN.main_device)
    print(f"[INFO] main_device = {main_device}")

    # 1) 构建模型
    llm_hidden = infer_qwen3_vl_text_hidden_size(PATHS.qwen_path)
    print(f"[INFO] qwen3-vl text hidden size = {llm_hidden}")

    projector = TokenLinearProjector(in_dim=768, llm_hidden_size=llm_hidden).to(main_device)

    qwen_device_map = TRAIN.qwen_device_map if TRAIN.use_multi_gpu else None
    model = SarQwenVLForCausalLM(
        qwen_path              = PATHS.qwen_path,
        projector              = projector,
        device                 = str(main_device),
        torch_dtype            = torch.float16 if TRAIN.fp16 else torch.float32,
        trust_remote_code      = True,
        low_cpu_mem_usage      = True,
        device_map             = qwen_device_map,
        gradient_checkpointing = TRAIN.gradient_checkpointing,
    )

    # 2) 加载 stage1 projector
    load_stage1_projector(model)

    # 3) 应用 LoRA（解冻 LLM 的 LoRA 参数）
    if TRAIN.use_lora:
        apply_lora(model)

    # 4) 数据集
    train_ds, collate_fn = build_datasets(model)
    print(f"[INFO] stage = {TRAIN.stage}")
    print(f"[INFO] train dataset size = {len(train_ds)}")

    dl = DataLoader(
        train_ds,
        batch_size         = TRAIN.batch_size,
        shuffle            = TRAIN.shuffle,
        num_workers        = TRAIN.num_workers,
        pin_memory         = torch.cuda.is_available(),
        persistent_workers = (TRAIN.num_workers > 0 and TRAIN.persistent_workers),
        prefetch_factor    = TRAIN.prefetch_factor if TRAIN.num_workers > 0 else None,
        collate_fn         = collate_fn,
        drop_last          = TRAIN.drop_last,
    )

    # 5) 优化器：projector + LoRA 参数
    param_groups = [
        {"params": model.projector.parameters(), "lr": TRAIN.lr},
    ]
    if TRAIN.use_lora:
        lora_params = [p for p in model.llm_causal.parameters() if p.requires_grad]
        if lora_params:
            param_groups.append({"params": lora_params, "lr": TRAIN.lora_lr})

    optim = torch.optim.AdamW(param_groups, lr=TRAIN.lr, weight_decay=0.01)

    def get_lr(step: int) -> float:
        if step < TRAIN.warmup_steps:
            return float(step + 1) / float(TRAIN.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=get_lr)

    # 6) AMP GradScaler
    use_amp = TRAIN.fp16 and torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("[INFO] AMP GradScaler enabled (fp16 training)")

    # 7) 断点续训
    start_step = maybe_resume(model, optim, scaler)

    # 8) 训练
    model.train()
    proj_trainable = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
    lora_trainable = sum(p.numel() for p in model.llm_causal.parameters() if p.requires_grad) if TRAIN.use_lora else 0
    print(f"[INFO] trainable projector params = {proj_trainable:,}")
    print(f"[INFO] trainable LoRA params      = {lora_trainable:,}")

    step       = start_step
    epoch      = 0
    loss_accum = 0.0
    t0         = time.time()

    while step < TRAIN.max_steps:
        epoch += 1
        for batch in dl:
            if batch is None:
                continue

            optim.zero_grad()

            try:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = forward_vqa(
                            model          = model,
                            sar_feats      = batch["sar_feats"],
                            input_ids      = batch["input_ids"],
                            attention_mask = batch["attention_mask"],
                            labels         = batch["labels"],
                        )
                else:
                    loss = forward_vqa(
                        model          = model,
                        sar_feats      = batch["sar_feats"],
                        input_ids      = batch["input_ids"],
                        attention_mask = batch["attention_mask"],
                        labels         = batch["labels"],
                    )
            except RuntimeError as e:
                print(f"[WARN] step {step} forward error: {e}, skipping")
                step += 1
                continue

            if loss is None:
                step += 1
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"] if p.requires_grad],
                    TRAIN.grad_clip_norm,
                )
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"] if p.requires_grad],
                    TRAIN.grad_clip_norm,
                )
                optim.step()

            scheduler.step()

            loss_accum += float(loss.item())

            if step % TRAIN.debug_print_every == 0:
                elapsed = time.time() - t0
                avg_loss = loss_accum / TRAIN.debug_print_every
                loss_accum = 0.0
                lr_now = optim.param_groups[0]["lr"]
                print(
                    f"[step {step:6d}/{TRAIN.max_steps}] "
                    f"loss={avg_loss:.4f}  lr={lr_now:.2e}  "
                    f"elapsed={elapsed:.1f}s"
                )
                if swanlab_run is not None:
                    try:
                        import swanlab
                        swanlab.log({"train/loss": avg_loss, "train/step": step, "train/lr": lr_now})
                    except Exception:
                        pass

            if TRAIN.save_full_checkpoint and step > 0 and step % TRAIN.save_every_steps == 0:
                save_checkpoint(
                    model=model, optim=optim, scaler=scaler,
                    step=step, loss_value=loss.item(),
                    save_dir=Path(TRAIN.save_dir),
                )

            step += 1
            if step >= TRAIN.max_steps:
                break

            if torch.cuda.is_available() and step % 100 == 0:
                torch.cuda.empty_cache()

    # 9) 最终保存
    save_dir = Path(TRAIN.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存 projector
    torch.save(model.projector.state_dict(), save_dir / "projector_stage2.pt")

    # 保存 LoRA adapter
    if TRAIN.use_lora:
        try:
            model.llm_causal.save_pretrained(str(save_dir / "lora_adapter"))
            print(f"[OK] Saved LoRA adapter to: {save_dir / 'lora_adapter'}")
        except Exception as e:
            print(f"[WARN] 无法保存 LoRA adapter: {e}")

    # 保存 tokenizer
    tokenizer_dir = save_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model.tokenizer.save_pretrained(str(tokenizer_dir))

    print(f"[OK] Saved projector to: {save_dir / 'projector_stage2.pt'}")
    print(f"[OK] Saved tokenizer to: {tokenizer_dir}")

    if swanlab_run is not None:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
