"""
train_pt.py
===========
直接加载预提取的 SARCLIP .pt 特征进行训练，跳过图像读取和 encoder 推理。

前提：已用 get_pt_*.py 脚本预提取完所有 .pt 文件，
      且 config_local.py 里各数据集的 *_pt_root / *_pt_train_json 已填写正确。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.data import DataLoader

PROJECT_DIR = Path("/home/qianwentao/SARClip/SARCLIP+QwenVL-pt")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config_local import PATHS, TRAIN                          # noqa: E402
from sarclip_module import TokenLinearProjector                # noqa: E402
from qwen3_sar_model import SarQwenVLForCausalLM, infer_qwen3_vl_text_hidden_size  # noqa: E402
from mixed_dataset_pt import (                                 # noqa: E402
    PtCaptionDataset,
    MultiPtCaptionDataset,
    build_pt_collate_fn,
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
            "max_steps":        TRAIN.max_steps,
            "max_length":       TRAIN.max_length,
            "num_image_tokens": TRAIN.num_image_tokens,
            "fp16":             TRAIN.fp16,
        },
    )


def maybe_load_projector(model: SarQwenVLForCausalLM) -> None:
    if not TRAIN.projector_ckpt:
        return
    ckpt = Path(TRAIN.projector_ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"projector_ckpt 不存在: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "projector" in state:
        state = state["projector"]
    missing, unexpected = model.projector.load_state_dict(state, strict=False)
    print(f"[INFO] loaded projector from: {ckpt}")
    if missing or unexpected:
        raise RuntimeError(f"projector 权重加载不干净: missing={missing}, unexpected={unexpected}")


def save_training_checkpoint(model, optim, scaler, step, loss_value, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"checkpoint_step_{step:06d}.pt"
    ckpt = {
        "step":          step,
        "loss":          float(loss_value),
        "projector":     model.projector.state_dict(),
        "optimizer":     optim.state_dict(),
        "scaler":        scaler.state_dict() if scaler is not None else None,
        "train_config":  vars(TRAIN),
        "path_config":   vars(PATHS),
        "image_token":   model.image_token,
        "tokenizer_len": len(model.tokenizer),
    }
    torch.save(ckpt, save_path)
    print(f"[OK] Saved checkpoint to: {save_path}")
    cleanup_old_checkpoints(save_dir, keep_last=TRAIN.keep_last_n_checkpoints)


def cleanup_old_checkpoints(save_dir: Path, keep_last: int = 5) -> None:
    ckpts = sorted(save_dir.glob("checkpoint_step_*.pt"))
    for p in ckpts[:-keep_last]:
        try:
            p.unlink()
            print(f"[INFO] removed old checkpoint: {p}")
        except Exception as e:
            print(f"[WARN] failed to remove {p}: {e}")


def maybe_resume_training(model, optim, scaler) -> int:
    if not TRAIN.resume_ckpt:
        return 0
    ckpt_path = Path(TRAIN.resume_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"resume_ckpt 不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "projector" not in ckpt:
        raise ValueError(f"不是完整 checkpoint，缺少 projector: {ckpt_path}")
    model.projector.load_state_dict(ckpt["projector"], strict=False)
    if "optimizer" in ckpt:
        optim.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_step = int(ckpt.get("step", 0))
    print(f"[INFO] resumed from step={start_step}")
    return start_step


def forward_pt(
    model:          SarQwenVLForCausalLM,
    sar_feats:      torch.Tensor,          # (B, 195, 768)
    input_ids:      torch.Tensor,          # (B, L)
    attention_mask: torch.Tensor,          # (B, L)
    labels:         torch.Tensor,          # (B, L)
    batch_meta:     dict | None = None,    # 新增：用于打印 pt_path / source / image_id
):
    """
    sar_feats: 预提取的 SARCLIP patch token 特征 (B, N, C)。
    仅 projector 参与计算图，Qwen3-VL 全程 no_grad。
    """
    main_device = model.main_device
    debug_nan = bool(getattr(TRAIN, "debug_nan", False))
    feat_warn_th = float(getattr(TRAIN, "debug_feat_absmax_threshold", 1e4))
    embed_warn_th = float(getattr(TRAIN, "debug_embed_absmax_threshold", 1e4))

    def _meta_str() -> str:
        if not batch_meta:
            return ""
        keys = ["sources", "image_ids", "pt_paths"]
        parts = []
        for k in keys:
            v = batch_meta.get(k, None)
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                show = list(v[:2])
                if len(v) > 2:
                    parts.append(f"{k}={show}...(n={len(v)})")
                else:
                    parts.append(f"{k}={show}")
            else:
                parts.append(f"{k}={v}")
        return " | " + " | ".join(parts) if parts else ""

    def _tensor_stat_str(name: str, x: torch.Tensor) -> str:
        xf = x.detach().float()
        finite = torch.isfinite(xf)
        finite_ratio = float(finite.float().mean().item())
        if finite.any():
            valid = xf[finite]
            abs_mean = float(valid.abs().mean().item())
            abs_max  = float(valid.abs().max().item())
            min_v    = float(valid.min().item())
            max_v    = float(valid.max().item())
        else:
            abs_mean = float("nan")
            abs_max  = float("nan")
            min_v    = float("nan")
            max_v    = float("nan")
        return (
            f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
            f"finite_ratio={finite_ratio:.6f}, abs_mean={abs_mean:.6e}, "
            f"abs_max={abs_max:.6e}, min={min_v:.6e}, max={max_v:.6e}"
        )

    patch_tokens = sar_feats.to(main_device)   # (B, 195, 768)

    if not torch.isfinite(patch_tokens).all():
        raise RuntimeError(
            "sar_feats 含 NaN/Inf\n"
            + _tensor_stat_str("patch_tokens", patch_tokens)
            + _meta_str()
        )

    patch_abs_max = float(patch_tokens.detach().abs().max().item())
    if debug_nan and patch_abs_max > feat_warn_th:
        print("[WARN] patch_tokens abs_max too large:",
              f"{patch_abs_max:.6e}",
              _meta_str())

    # projector 参数前检查
    if debug_nan:
        for name, p in model.projector.named_parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(
                    f"projector 参数在 forward 前已损坏: {name}\n"
                    + _tensor_stat_str(f"param[{name}]", p)
                    + _meta_str()
                )

    # 强制在 fp32 下运行 projector，防止 fp16 autocast 下 matmul 溢出
    with torch.amp.autocast("cuda", enabled=False):
        sar_token_embeds_fp32 = model.projector(patch_tokens.float())   # fp32

    model._check_projector_dim(sar_token_embeds_fp32)

    if not torch.isfinite(sar_token_embeds_fp32).all():
        msg = [
            "projector 原始输出出现 NaN/Inf",
            _tensor_stat_str("patch_tokens", patch_tokens),
            _tensor_stat_str("sar_token_embeds_fp32_raw", sar_token_embeds_fp32),
        ]
        if debug_nan:
            for name, p in model.projector.named_parameters():
                msg.append(_tensor_stat_str(f"param[{name}]", p))
        raise RuntimeError("\n".join(msg) + _meta_str())

    # 尺度对齐（保持 fp32）
    target_scale = float(model._emb_scale)
    cur_scale = sar_token_embeds_fp32.abs().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)

    if not torch.isfinite(cur_scale).all():
        raise RuntimeError(
            "cur_scale 出现 NaN/Inf\n"
            + _tensor_stat_str("sar_token_embeds_fp32_raw", sar_token_embeds_fp32)
            + "\n"
            + _tensor_stat_str("cur_scale", cur_scale)
            + _meta_str()
        )

    sar_token_embeds_fp32 = sar_token_embeds_fp32 / (cur_scale + 1e-6) * target_scale

    if not torch.isfinite(sar_token_embeds_fp32).all():
        raise RuntimeError(
            "sar_token_embeds 在 scale 后出现 NaN/Inf\n"
            + _tensor_stat_str("cur_scale", cur_scale)
            + "\n"
            + _tensor_stat_str("sar_token_embeds_fp32_scaled", sar_token_embeds_fp32)
            + _meta_str()
        )

    sar_token_embeds_fp32 = torch.clamp(
        sar_token_embeds_fp32,
        min=-3.0 * target_scale,
        max= 3.0 * target_scale,
    )

    if not torch.isfinite(sar_token_embeds_fp32).all():
        raise RuntimeError(
            "sar_token_embeds 在 clamp 后出现 NaN/Inf\n"
            + _tensor_stat_str("sar_token_embeds_fp32_clamped", sar_token_embeds_fp32)
            + _meta_str()
        )

    embed_abs_max = float(sar_token_embeds_fp32.detach().abs().max().item())
    if debug_nan and embed_abs_max > embed_warn_th:
        print("[WARN] sar_token_embeds_fp32 abs_max too large:",
              f"{embed_abs_max:.6e}",
              _meta_str())

    # 转换为 Qwen embedding 的 dtype（通常 fp16）
    sar_token_embeds = sar_token_embeds_fp32.to(model.lm_emb_dtype)

    if not torch.isfinite(sar_token_embeds).all():
        raise RuntimeError(
            "sar_token_embeds 在 cast 到 lm_emb_dtype 后出现 NaN/Inf\n"
            + _tensor_stat_str("sar_token_embeds_fp32_clamped", sar_token_embeds_fp32)
            + "\n"
            + _tensor_stat_str("sar_token_embeds_casted", sar_token_embeds)
            + _meta_str()
        )

    lm_dev = model.lm_input_device
    input_ids = input_ids.to(lm_dev)
    attention_mask = attention_mask.to(lm_dev)
    labels = labels.to(lm_dev)
    sar_token_embeds = sar_token_embeds.to(lm_dev)

    inputs_embeds = model._inject_sar_embeds(input_ids, sar_token_embeds)

    if not torch.isfinite(inputs_embeds).all():
        raise RuntimeError(
            "inputs_embeds 出现 NaN/Inf（来自 _inject_sar_embeds 或文本 embedding）\n"
            + _tensor_stat_str("sar_token_embeds", sar_token_embeds)
            + "\n"
            + _meta_str()
        )

    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    outputs = model.llm_backbone(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state

    if not torch.isfinite(hidden_states).all():
        raise RuntimeError(
            "hidden_states 出现 NaN/Inf\n"
            + _tensor_stat_str("hidden_states", hidden_states)
            + _meta_str()
        )

    logits = model.lm_head(hidden_states)

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "logits 出现 NaN/Inf\n"
            + _tensor_stat_str("logits", logits)
            + _meta_str()
        )

    loss = None
    if labels is not None:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        if shift_labels.ne(-100).sum() > 0:
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

    # return type("PtOutput", (), {"loss": loss, "logits": logits})()
    return loss

# ─────────────────────────────────────────────
# 3. 数据集构建
# ─────────────────────────────────────────────

def build_pt_datasets_and_collate(model: SarQwenVLForCausalLM):
    """
    选择启用的数据集后，直接拼接成一个大训练集。
    随机打乱交给 DataLoader(shuffle=True)。
    """
    if TRAIN.stage.strip().lower() != "stage1_mixed_caption":
        raise ValueError(
            f"train_pt.py 仅支持 stage=stage1_mixed_caption，当前: {TRAIN.stage}"
        )

    collate_fn = build_pt_collate_fn(
        tokenizer        = model.tokenizer,
        image_token      = model.image_token,
        num_image_tokens = TRAIN.num_image_tokens,
        max_length       = TRAIN.max_length,
    )

    def _make(root, pt_json, max_s):
        return PtCaptionDataset(root=root, pt_json=pt_json, max_samples=max_s)

    datasets = {}

    # 保留你原来的“>0 表示启用，=0 表示不启用”习惯
    if TRAIN.mixed_weight_sarlang > 0:
        datasets["sarlang"] = _make(
            PATHS.sarlang_root,
            PATHS.sarlang_pt_train_json,
            PATHS.sarlang_max_samples,
        )

    if TRAIN.mixed_weight_sarcap > 0:
        datasets["sarcap"] = _make(
            PATHS.sarcap_root,
            PATHS.sarcap_pt_train_json,
            PATHS.sarcap_max_samples,
        )

    if TRAIN.mixed_weight_sartext > 0:
        datasets["sartext"] = _make(
            PATHS.sartext_root,
            PATHS.sartext_pt_train_json,
            PATHS.sartext_max_samples,
        )

    if TRAIN.mixed_weight_fsarcap > 0:
        datasets["fsarcap"] = _make(
            PATHS.fsarcap_root,
            PATHS.fsarcap_pt_train_json,
            PATHS.fsarcap_max_samples,
        )

    if not datasets:
        raise ValueError("没有启用任何数据集，请至少打开一个数据集")

    train_ds = MultiPtCaptionDataset(
        datasets=datasets,
        seed=TRAIN.mixed_seed,
    )
    return train_ds, collate_fn

# def build_pt_datasets_and_collate(model: SarQwenVLForCausalLM):
#     """
#     统一使用 stage1_mixed_caption，通过 mixed_weight_* 控制各数据集比例。
#     将某个数据集权重设为 0 即可排除该数据集，无需改 stage。
#     """
#     if TRAIN.stage.strip().lower() != "stage1_mixed_caption":
#         raise ValueError(
#             f"train_pt.py 仅支持 stage=stage1_mixed_caption，当前: {TRAIN.stage}\n"
#             "通过 mixed_weight_sarlang / sarcap / sartext / fsarcap 控制各数据集比例。"
#         )

#     collate_fn = build_pt_collate_fn(
#         tokenizer        = model.tokenizer,
#         image_token      = model.image_token,
#         num_image_tokens = TRAIN.num_image_tokens,
#         max_length       = TRAIN.max_length,
#     )

#     def _make(root, pt_json, max_s):
#         return PtCaptionDataset(root=root, pt_json=pt_json, max_samples=max_s)

#     datasets = {}
#     weights  = {}

#     if TRAIN.mixed_weight_sarlang > 0:
#         datasets["sarlang"] = _make(PATHS.sarlang_root, PATHS.sarlang_pt_train_json, PATHS.sarlang_max_samples)
#         weights["sarlang"]  = TRAIN.mixed_weight_sarlang

#     if TRAIN.mixed_weight_sarcap > 0:
#         datasets["sarcap"]  = _make(PATHS.sarcap_root,  PATHS.sarcap_pt_train_json,  PATHS.sarcap_max_samples)
#         weights["sarcap"]   = TRAIN.mixed_weight_sarcap

#     if TRAIN.mixed_weight_sartext > 0:
#         datasets["sartext"] = _make(PATHS.sartext_root, PATHS.sartext_pt_train_json, PATHS.sartext_max_samples)
#         weights["sartext"]  = TRAIN.mixed_weight_sartext

#     if TRAIN.mixed_weight_fsarcap > 0:
#         datasets["fsarcap"] = _make(PATHS.fsarcap_root, PATHS.fsarcap_pt_train_json, PATHS.fsarcap_max_samples)
#         weights["fsarcap"]  = TRAIN.mixed_weight_fsarcap

#     if not datasets:
#         raise ValueError("所有数据集权重均为 0，请至少将一个 mixed_weight_* 设为正数")

#     train_ds = MultiPtCaptionDataset(
#         datasets     = datasets,
#         weights      = weights,
#         epoch_length = TRAIN.mixed_epoch_length,
#         seed         = TRAIN.mixed_seed,
#     )
#     return train_ds, collate_fn


# ─────────────────────────────────────────────
# 4. 验证 loss
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_loss(model, data_loader) -> Optional[float]:
    if data_loader is None:
        return None
    model.eval()
    total_loss, total_count = 0.0, 0
    for batch in data_loader:
        if batch is None:
            continue
        out = forward_pt(
            model          = model,
            sar_feats      = batch["sar_feats"],
            input_ids      = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            labels         = batch["labels"],
            batch_meta={
                "sources": batch.get("sources"),
                "image_ids": batch.get("image_ids"),
                "pt_paths": batch.get("pt_paths"),
            },
        )
        if out.loss is not None:
            bs = batch["input_ids"].size(0)
            total_loss  += float(out.loss.item()) * bs
            total_count += bs
    model.train()
    return total_loss / total_count if total_count > 0 else None


# ─────────────────────────────────────────────
# 5. main
# ─────────────────────────────────────────────

def main():
    set_seed(9982443)
    device = pick_device(TRAIN.device)

    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  gpu[{i}] = {torch.cuda.get_device_name(i)}")
    print("picked device =", device)
    print("[INFO] train_pt.py 模式：跳过 SARCLIP encoder，直接加载预提取 .pt 特征")

    swanlab_run = init_swanlab()
    main_device = TRAIN.main_device if str(device) != "cpu" else "cpu"

    # 1) 读 Qwen hidden size
    llm_hidden = infer_qwen3_vl_text_hidden_size(PATHS.qwen_path)
    print(f"[INFO] qwen3-vl text hidden size = {llm_hidden}")

    # 2) projector（ViT-B/16 patch token dim=768，与 .pt 文件一致）
    SARCLIP_TOKEN_DIM = 768
    projector = TokenLinearProjector(
        in_dim          = SARCLIP_TOKEN_DIM,
        llm_hidden_size = llm_hidden,
    ).to(main_device)

    if TRAIN.num_image_tokens != 195:
        raise ValueError(
            f"TRAIN.num_image_tokens={TRAIN.num_image_tokens}，"
            f"但预提取特征固定为 195 tokens，请在 config 中设置 num_image_tokens=195"
        )

    # 3) Qwen3-VL
    #    双卡推荐：projector 在 cuda:0，Qwen 整体放 cuda:1（减少 GPU 间通信开销）
    #    若 GPU 显存不足可改回 "balanced"
    qwen_device_map = "cpu" if str(device) == "cpu" else (
        TRAIN.qwen_device_map if TRAIN.use_multi_gpu else None
    )
    model = SarQwenVLForCausalLM(
        qwen_path              = PATHS.qwen_path,
        projector              = projector,
        device                 = main_device,
        torch_dtype            = torch.float16 if TRAIN.fp16 else torch.float32,
        trust_remote_code      = True,
        low_cpu_mem_usage      = True,
        device_map             = qwen_device_map,
        gradient_checkpointing = TRAIN.gradient_checkpointing,
    )
    print(f"[INFO] model.llm_hidden_size = {model.llm_hidden_size}")
    print(f"[INFO] projector device = {main_device}, lm_input_device = {model.lm_input_device}")

    maybe_load_projector(model)

    # 4) 数据集
    train_ds, collate_fn = build_pt_datasets_and_collate(model)
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

    # 5) 优化器：只训 projector
    optim = torch.optim.AdamW(model.projector.parameters(), lr=TRAIN.lr)
    warmup_steps = 500

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return TRAIN.lr * float(step + 1) / float(warmup_steps)
        return TRAIN.lr

    # 6) AMP GradScaler（fp16 训练时防止梯度下溢）
    use_amp = TRAIN.fp16 and torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("[INFO] AMP GradScaler enabled (fp16 training)")

    # 7) 训练
    model.train()
    proj_trainable = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
    vl_trainable   = sum(p.numel() for p in model.vl_model.parameters()  if p.requires_grad)
    print(f"[INFO] trainable projector params = {proj_trainable}")
    print(f"[INFO] trainable vl params = {vl_trainable} (期望 0)")

    step = maybe_resume_training(model, optim, scaler)
    train_start_time = time.time()

    while step < TRAIN.max_steps:
        for batch in dl:
            if batch is None:
                print("[WARN] skip empty batch")
                continue

            optim.zero_grad(set_to_none=True)

            cur_lr = get_lr(step)
            for pg in optim.param_groups:
                pg["lr"] = cur_lr

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = forward_pt(
                    model          = model,
                    sar_feats      = batch["sar_feats"],
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    labels         = batch["labels"],
                )
            # loss = out.loss
            # del out

            if loss is None:
                print("[WARN] skip batch: no valid target tokens")
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                raise RuntimeError(f"loss 非法: {loss.item()}, valid_tokens={valid_tokens}")

            # backward + grad clip
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)  # unscale 后才能 clip
            else:
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)

            bad_grad = False
            for name, p in model.projector.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print(f"[BAD GRAD] {name}")
                    bad_grad = True
                    break

            if not torch.isfinite(grad_norm):
                print(f"[BAD GRAD NORM] grad_norm={grad_norm}")
                bad_grad = True

            if bad_grad:
                optim.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()  # scaler 必须 update，即使跳过 step
                print(f"[BAD BATCH] step={step}, skipping optimizer.step()")

                for n, p in model.projector.named_parameters():
                    if not torch.isfinite(p).all():
                        raise RuntimeError(f"[BAD PARAM AFTER STEP] {n} 出现 NaN/Inf")
                continue

            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()

            # ── 日志 ───────────────────────────────────────────
            if step % 10 == 0:
                elapsed = time.time() - train_start_time
                h, m, s = int(elapsed//3600), int((elapsed%3600)//60), int(elapsed%60)
                valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                print(f"[TIME] {h:02d}:{m:02d}:{s:02d}  step={step}  loss={loss.item():.6f}  "
                      f"valid_tokens={valid_tokens}  lr={cur_lr:.2e}")

                if swanlab_run is not None:
                    import swanlab
                    swanlab.log({"train/loss": float(loss.item()), "train/step": step})

            if TRAIN.save_full_checkpoint and step > 0 and step % TRAIN.save_every_steps == 0:
                save_training_checkpoint(
                    model=model, optim=optim, scaler=scaler,
                    step=step, loss_value=loss.item(),
                    save_dir=Path(TRAIN.save_dir),
                )

            step += 1
            if step >= TRAIN.max_steps:
                break

            if torch.cuda.is_available() and step % 100 == 0:
                torch.cuda.empty_cache()

    # 8) 最终保存
    save_dir = Path(TRAIN.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    projector_path = save_dir / TRAIN.save_name
    torch.save(model.projector.state_dict(), projector_path)
    tokenizer_dir = save_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model.tokenizer.save_pretrained(tokenizer_dir)

    print(f"[OK] Saved projector to: {projector_path}")
    print(f"[OK] Saved tokenizer to: {tokenizer_dir}")

    if swanlab_run is not None:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
