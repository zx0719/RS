from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.configs.stage1 import PATHS, TRAIN
from src.dataset.stage1_caption_dataset import (
    MultiPtCaptionDataset,
    PtCaptionDataset,
    build_pt_collate_fn,
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
    torch.backends.cudnn.benchmark = True


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
            "stage": TRAIN.stage,
            "batch_size": TRAIN.batch_size,
            "lr": TRAIN.lr,
            "max_steps": TRAIN.max_steps,
            "max_length": TRAIN.max_length,
            "num_image_tokens": TRAIN.num_image_tokens,
            "fp16": TRAIN.fp16,
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


def save_training_checkpoint(model, optim, scaler, step: int, loss_value: float, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"checkpoint_step_{step:06d}.pt"
    ckpt = {
        "step": step,
        "loss": float(loss_value),
        "projector": model.projector.state_dict(),
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "train_config": vars(TRAIN),
        "path_config": vars(PATHS),
        "image_token": model.image_token,
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
    start_step = int(ckpt.get("step", 0))
    # Stage B 切换时 param_groups 数量变化，optimizer state 不兼容，只加载 projector 权重
    projector_only = bool(getattr(TRAIN, "resume_projector_only", False))
    if not projector_only and "optimizer" in ckpt:
        try:
            optim.load_state_dict(ckpt["optimizer"])
            if scaler is not None and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
        except ValueError as e:
            print(f"[WARN] optimizer state 加载失败（{e}），仅加载 projector 权重，optimizer 重置")
    else:
        print(f"[INFO] resume_projector_only=True，optimizer 重置")
    print(f"[INFO] resumed from step={start_step}")
    return start_step


def warn_projector_param_explosion(model: SarQwenVLForCausalLM, step: int) -> None:
    if not bool(getattr(TRAIN, "debug_param_check_after_step", True)):
        return

    abs_mean_th = float(getattr(TRAIN, "debug_param_absmean_threshold", 1.0))
    abs_max_th = float(getattr(TRAIN, "debug_param_absmax_threshold", 100.0))
    issues = []

    for name, param in model.projector.named_parameters():
        pd = param.detach().float()
        finite = torch.isfinite(pd)
        if not finite.all():
            issues.append(
                f"{name}(non-finite, finite_ratio={float(finite.float().mean().item()):.6f})"
            )
            continue

        abs_mean = float(pd.abs().mean().item())
        abs_max = float(pd.abs().max().item())
        if abs_mean > abs_mean_th or abs_max > abs_max_th:
            issues.append(f"{name}(abs_mean={abs_mean:.6e}, abs_max={abs_max:.6e})")

    if issues:
        preview = "; ".join(issues[:3])
        if len(issues) > 3:
            preview += f"; ... and {len(issues) - 3} more"
        print(
            f"[WARN] projector parameter explosion risk after step={step}: {preview} "
            f"(thresholds: abs_mean>{abs_mean_th:.6e} or abs_max>{abs_max_th:.6e})"
        )


def forward_pt(
    model: SarQwenVLForCausalLM,
    sar_feats: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    batch_meta: dict | None = None,
) -> Optional[torch.Tensor]:
    main_device = model.main_device
    debug_nan = bool(getattr(TRAIN, "debug_nan", False))
    feat_warn_th = float(getattr(TRAIN, "debug_feat_absmax_threshold", 1e4))
    embed_warn_th = float(getattr(TRAIN, "debug_embed_absmax_threshold", 1e4))

    def _meta_str() -> str:
        if not batch_meta:
            return ""
        keys = ["sources", "image_ids", "pt_paths"]
        parts = []
        for key in keys:
            value = batch_meta.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                show = list(value[:2])
                suffix = f"...(n={len(value)})" if len(value) > 2 else ""
                parts.append(f"{key}={show}{suffix}")
            else:
                parts.append(f"{key}={value}")
        return " | " + " | ".join(parts) if parts else ""

    def _tensor_stat_str(name: str, tensor: torch.Tensor) -> str:
        x = tensor.detach().float()
        finite = torch.isfinite(x)
        finite_ratio = float(finite.float().mean().item())
        if finite.any():
            valid = x[finite]
            abs_mean = float(valid.abs().mean().item())
            abs_max = float(valid.abs().max().item())
            min_v = float(valid.min().item())
            max_v = float(valid.max().item())
        else:
            abs_mean = float("nan")
            abs_max = float("nan")
            min_v = float("nan")
            max_v = float("nan")
        return (
            f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
            f"finite_ratio={finite_ratio:.6f}, abs_mean={abs_mean:.6e}, "
            f"abs_max={abs_max:.6e}, min={min_v:.6e}, max={max_v:.6e}"
        )

    patch_tokens = sar_feats.to(main_device)
    if not torch.isfinite(patch_tokens).all():
        raise RuntimeError(
            "sar_feats 含 NaN/Inf\n"
            + _tensor_stat_str("patch_tokens", patch_tokens)
            + _meta_str()
        )

    patch_abs_max = float(patch_tokens.detach().abs().max().item())
    if debug_nan and patch_abs_max > feat_warn_th:
        print("[WARN] patch_tokens abs_max too large:", f"{patch_abs_max:.6e}", _meta_str())

    if debug_nan:
        for name, param in model.projector.named_parameters():
            if not torch.isfinite(param).all():
                raise RuntimeError(
                    f"projector 参数在 forward 前已损坏: {name}\n"
                    + _tensor_stat_str(f"param[{name}]", param)
                    + _meta_str()
                )

    with torch.amp.autocast("cuda", enabled=False):
        sar_token_embeds_fp32 = model.projector(patch_tokens.float())

    model._check_projector_dim(sar_token_embeds_fp32)

    if not torch.isfinite(sar_token_embeds_fp32).all():
        raise RuntimeError(
            "projector 原始输出出现 NaN/Inf\n"
            + _tensor_stat_str("patch_tokens", patch_tokens)
            + "\n"
            + _tensor_stat_str("sar_token_embeds_fp32_raw", sar_token_embeds_fp32)
            + _meta_str()
        )

    target_scale = float(model._emb_scale)
    cur_scale = sar_token_embeds_fp32.abs().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    if not torch.isfinite(cur_scale).all():
        raise RuntimeError(
            "cur_scale 出现 NaN/Inf\n"
            + _tensor_stat_str("cur_scale", cur_scale)
            + _meta_str()
        )

    sar_token_embeds_fp32 = sar_token_embeds_fp32 / (cur_scale + 1e-6) * target_scale
    sar_token_embeds_fp32 = torch.clamp(
        sar_token_embeds_fp32,
        min=-3.0 * target_scale,
        max=3.0 * target_scale,
    )
    if not torch.isfinite(sar_token_embeds_fp32).all():
        raise RuntimeError(
            "sar_token_embeds 在 scale/clamp 后出现 NaN/Inf\n"
            + _tensor_stat_str("sar_token_embeds_fp32", sar_token_embeds_fp32)
            + _meta_str()
        )

    embed_abs_max = float(sar_token_embeds_fp32.detach().abs().max().item())
    if debug_nan and embed_abs_max > embed_warn_th:
        print("[WARN] sar_token_embeds_fp32 abs_max too large:", f"{embed_abs_max:.6e}", _meta_str())

    sar_token_embeds = sar_token_embeds_fp32.to(model.lm_emb_dtype)
    lm_dev = model.lm_input_device
    input_ids = input_ids.to(lm_dev)
    attention_mask = attention_mask.to(lm_dev)
    labels = labels.to(lm_dev)
    sar_token_embeds = sar_token_embeds.to(lm_dev)

    # ── Diagnose text embedding NaN before inject ──
    if debug_nan:
        emb_table = model.llm_backbone.get_input_embeddings().weight
        if not torch.isfinite(emb_table).all():
            nan_rows = torch.nonzero(~torch.isfinite(emb_table).all(dim=-1)).view(-1)
            raise RuntimeError(
                f"Qwen embedding table 含 NaN/Inf，行索引: {nan_rows[:20].tolist()}" + _meta_str()
            )
        vocab_size = emb_table.shape[0]
        bad_ids = input_ids[(input_ids >= 0) & (input_ids >= vocab_size)]
        if bad_ids.numel() > 0:
            raise RuntimeError(
                f"input_ids 含越界 token id (vocab={vocab_size}): {bad_ids[:10].tolist()}" + _meta_str()
            )
        text_embeds_check = emb_table[input_ids.clamp(0, vocab_size - 1)]
        if not torch.isfinite(text_embeds_check).all():
            bad_pos = torch.nonzero(~torch.isfinite(text_embeds_check).all(dim=-1))
            bad_tok = input_ids[bad_pos[:, 0], bad_pos[:, 1]]
            raise RuntimeError(
                f"text_embeds 含 NaN/Inf，对应 token ids: {bad_tok[:10].tolist()}" + _meta_str()
            )

    inputs_embeds = model._inject_sar_embeds(input_ids, sar_token_embeds)
    if not torch.isfinite(inputs_embeds).all():
        raise RuntimeError("inputs_embeds 出现 NaN/Inf" + _meta_str())

    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    outputs = model.llm_backbone(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state
    logits = model.lm_head(hidden_states)

    if not torch.isfinite(logits).all():
        raise RuntimeError("logits 出现 NaN/Inf" + _meta_str())

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if shift_labels.ne(-100).sum() == 0:
        return None

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def build_pt_datasets_and_collate(model: SarQwenVLForCausalLM):
    if TRAIN.stage.strip().lower() != "stage1_mixed_caption":
        raise ValueError(
            f"stage1 训练仅支持 stage=stage1_mixed_caption，当前: {TRAIN.stage}"
        )

    collate_fn = build_pt_collate_fn(
        tokenizer=model.tokenizer,
        image_token=model.image_token,
        num_image_tokens=TRAIN.num_image_tokens,
        max_length=TRAIN.max_length,
    )

    def _make(root, pt_json, max_s):
        return PtCaptionDataset(root=root, pt_json=pt_json, max_samples=max_s)

    datasets = {}
    if TRAIN.mixed_weight_sarlang > 0:
        datasets["sarlang"] = _make(PATHS.sarlang_root, PATHS.sarlang_pt_train_json, PATHS.sarlang_max_samples)
    if TRAIN.mixed_weight_sarcap > 0:
        datasets["sarcap"] = _make(PATHS.sarcap_root, PATHS.sarcap_pt_train_json, PATHS.sarcap_max_samples)
    if TRAIN.mixed_weight_sartext > 0:
        datasets["sartext"] = _make(PATHS.sartext_root, PATHS.sartext_pt_train_json, PATHS.sartext_max_samples)
    if TRAIN.mixed_weight_fsarcap > 0:
        datasets["fsarcap"] = _make(PATHS.fsarcap_root, PATHS.fsarcap_pt_train_json, PATHS.fsarcap_max_samples)
    if not datasets:
        raise ValueError("没有启用任何数据集，请至少打开一个数据集")

    train_ds = MultiPtCaptionDataset(
        datasets=datasets,
        weights={
            "sarlang":  TRAIN.mixed_weight_sarlang,
            "sarcap":   TRAIN.mixed_weight_sarcap,
            "sartext":  TRAIN.mixed_weight_sartext,
            "fsarcap":  TRAIN.mixed_weight_fsarcap,
        },
        seed=TRAIN.mixed_seed,
    )
    return train_ds, collate_fn


@torch.no_grad()
def evaluate_loss(model, data_loader) -> Optional[float]:
    if data_loader is None:
        return None
    model.eval()
    total_loss, total_count = 0.0, 0
    for batch in data_loader:
        if batch is None:
            continue
        loss = forward_pt(
            model=model,
            sar_feats=batch["sar_feats"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            batch_meta={
                "sources": batch.get("sources"),
                "image_ids": batch.get("image_ids"),
                "pt_paths": batch.get("pt_paths"),
            },
        )
        if loss is not None:
            bs = batch["input_ids"].size(0)
            total_loss += float(loss.item()) * bs
            total_count += bs
    model.train()
    return total_loss / total_count if total_count > 0 else None


def _build_projector(llm_hidden: int, device: str) -> nn.Module:
    """Build projector based on config: BridgeGuidedProjector if bridge_ckpt set, else TokenLinearProjector."""
    bridge_ckpt = getattr(TRAIN, "bridge_ckpt", None)
    if bridge_ckpt:
        ckpt_path = Path(bridge_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"bridge_ckpt 不存在: {ckpt_path}")
        dim_hidden = int(getattr(TRAIN, "bridge_dim_hidden", 1024))
        proj = BridgeGuidedProjector(
            dim_sar=768,
            dim_qwen=llm_hidden,
            dim_hidden=dim_hidden,
            bridge_ckpt_path=str(ckpt_path),
        ).to(device)
        stage_b = bool(getattr(TRAIN, "stage_b_unfreeze_bridge", False))
        if stage_b:
            proj.unfreeze_bridge()
            print(f"[INFO] Using BridgeGuidedProjector (Stage B: bridge unfrozen, 0.1x lr)")
        else:
            proj.freeze_bridge()
            print(f"[INFO] Using BridgeGuidedProjector (Stage A: bridge frozen)")
        return proj
    else:
        print("[INFO] Using TokenLinearProjector (no bridge_ckpt set)")
        return TokenLinearProjector(in_dim=768, llm_hidden_size=llm_hidden).to(device)


class Stage1Trainer:
    def __init__(self) -> None:
        set_seed(9982443)
        self.device = pick_device(TRAIN.device)
        self.main_device = TRAIN.main_device if str(self.device) != "cpu" else "cpu"
        self.swanlab_run = init_swanlab()
        self.train_start_time = 0.0

        self._print_runtime_info()
        self.model = self._build_model()
        self.train_loader = self._build_train_loader()
        # Stage B: bridge 用 0.1x lr，residual+gate 用全 lr
        stage_b = bool(getattr(TRAIN, "stage_b_unfreeze_bridge", False))
        if stage_b and hasattr(self.model.projector, "bridge"):
            param_groups = [
                {"params": list(self.model.projector.bridge.parameters()), "lr": TRAIN.lr * 0.1},
                {"params": [p for n, p in self.model.projector.named_parameters()
                            if not n.startswith("bridge")], "lr": TRAIN.lr},
            ]
            self.optimizer = torch.optim.AdamW(param_groups, lr=TRAIN.lr)
            print(f"[INFO] Stage B optimizer: bridge lr={TRAIN.lr*0.1:.2e}, others lr={TRAIN.lr:.2e}")
        else:
            trainable = [p for p in self.model.projector.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable, lr=TRAIN.lr)
        print(f"[INFO] Optimizer: trainable param tensors in projector")
        self.use_amp = TRAIN.fp16 and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        if self.use_amp:
            print("[INFO] AMP GradScaler enabled (fp16 training)")

        self.step = maybe_resume_training(self.model, self.optimizer, self.scaler)

    def _print_runtime_info(self) -> None:
        print("torch.cuda.is_available() =", torch.cuda.is_available())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  gpu[{i}] = {torch.cuda.get_device_name(i)}")
        print("picked device =", self.device)
        print("[INFO] stage1 模式：跳过 SARCLIP encoder，直接加载预提取 .pt 特征")

    def _build_model(self) -> SarQwenVLForCausalLM:
        llm_hidden = infer_qwen3_vl_text_hidden_size(PATHS.qwen_path)
        print(f"[INFO] qwen3-vl text hidden size = {llm_hidden}")

        if TRAIN.num_image_tokens != 195:
            raise ValueError(
                f"TRAIN.num_image_tokens={TRAIN.num_image_tokens}，"
                "但预提取特征固定为 195 tokens，请在 config 中设置 num_image_tokens=195"
            )

        projector = _build_projector(llm_hidden, self.main_device)
        qwen_device_map = "cpu" if str(self.device) == "cpu" else (
            TRAIN.qwen_device_map if TRAIN.use_multi_gpu else None
        )
        model = SarQwenVLForCausalLM(
            qwen_path=PATHS.qwen_path,
            projector=projector,
            device=self.main_device,
            torch_dtype=torch.float16 if TRAIN.fp16 else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=qwen_device_map,
            gradient_checkpointing=TRAIN.gradient_checkpointing,
        )
        print(f"[INFO] model.llm_hidden_size = {model.llm_hidden_size}")
        print(f"[INFO] projector device = {self.main_device}, lm_input_device = {model.lm_input_device}")
        maybe_load_projector(model)
        return model

    def _build_train_loader(self) -> DataLoader:
        train_ds, collate_fn = build_pt_datasets_and_collate(self.model)
        print(f"[INFO] stage = {TRAIN.stage}")
        print(f"[INFO] train dataset size = {len(train_ds)}")
        return DataLoader(
            train_ds,
            batch_size=TRAIN.batch_size,
            shuffle=TRAIN.shuffle,
            num_workers=TRAIN.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(TRAIN.num_workers > 0 and TRAIN.persistent_workers),
            prefetch_factor=TRAIN.prefetch_factor if TRAIN.num_workers > 0 else None,
            collate_fn=collate_fn,
            drop_last=TRAIN.drop_last,
        )

    def _current_lr(self, step: int) -> float:
        warmup_steps = 500
        if step < warmup_steps:
            return TRAIN.lr * float(step + 1) / float(warmup_steps)
        # cosine decay: lr_min = 1e-6
        progress = (step - warmup_steps) / max(1, TRAIN.max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_min = 1e-6
        return lr_min + (TRAIN.lr - lr_min) * cosine

    def _log_train_step(self, step: int, loss: torch.Tensor, batch, cur_lr: float) -> None:
        if step % 10 != 0:
            return
        elapsed = time.time() - self.train_start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
        print(
            f"[TIME] {h:02d}:{m:02d}:{s:02d}  step={step}  loss={float(loss.item()):.6f}  "
            f"valid_tokens={valid_tokens}  lr={cur_lr:.2e}"
        )
        if self.swanlab_run is not None:
            self.swanlab_run.log({"train/loss": float(loss.item()), "train/step": step, "train/lr": cur_lr})

    def save_final_artifacts(self) -> None:
        save_dir = Path(TRAIN.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        projector_path = save_dir / TRAIN.save_name
        torch.save(self.model.projector.state_dict(), projector_path)
        tokenizer_dir = save_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.model.tokenizer.save_pretrained(tokenizer_dir)
        print(f"[OK] Saved projector to: {projector_path}")
        print(f"[OK] Saved tokenizer to: {tokenizer_dir}")

    def train(self) -> None:
        self.model.train()
        proj_trainable = sum(p.numel() for p in self.model.projector.parameters() if p.requires_grad)
        vl_trainable = sum(p.numel() for p in self.model.vl_model.parameters() if p.requires_grad)
        print(f"[INFO] trainable projector params = {proj_trainable}")
        print(f"[INFO] trainable vl params = {vl_trainable} (期望 0)")

        self.train_start_time = time.time()
        while self.step < TRAIN.max_steps:
            for batch in self.train_loader:
                if batch is None:
                    print("[WARN] skip empty batch")
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                cur_lr = self._current_lr(self.step)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = cur_lr

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    loss = forward_pt(
                        model=self.model,
                        sar_feats=batch["sar_feats"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )

                if loss is None:
                    print("[WARN] skip batch: no valid target tokens")
                    continue
                if torch.isnan(loss) or torch.isinf(loss):
                    valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                    raise RuntimeError(f"loss 非法: {loss.item()}, valid_tokens={valid_tokens}")

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.projector.parameters(), TRAIN.grad_clip_norm)

                bad_grad = not torch.isfinite(grad_norm)
                if not bad_grad:
                    for name, param in self.model.projector.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            print(f"[BAD GRAD] {name}")
                            bad_grad = True
                            break

                if bad_grad:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler is not None:
                        self.scaler.update()
                    print(f"[BAD BATCH] step={self.step}, skipping optimizer.step()")
                    continue

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self._log_train_step(self.step, loss, batch, cur_lr)

                if TRAIN.save_full_checkpoint and self.step > 0 and self.step % TRAIN.save_every_steps == 0:
                    save_training_checkpoint(
                        model=self.model,
                        optim=self.optimizer,
                        scaler=self.scaler,
                        step=self.step,
                        loss_value=float(loss.item()),
                        save_dir=Path(TRAIN.save_dir),
                    )

                self.step += 1
                warn_projector_param_explosion(self.model, self.step)
                if self.step >= TRAIN.max_steps:
                    break
                if torch.cuda.is_available() and self.step % 100 == 0:
                    torch.cuda.empty_cache()

        self.save_final_artifacts()
        if self.swanlab_run is not None:
            try:
                self.swanlab_run.finish()
            except Exception:
                pass


def run_stage1() -> None:
    trainer = Stage1Trainer()
    trainer.train()


def main() -> None:
    run_stage1()


if __name__ == "__main__":
    main()
