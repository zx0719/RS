# debug_pt_pipeline.py
from __future__ import annotations

import os
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config_local import PATHS, TRAIN
from qwen3_sar_model import SarQwenVLForCausalLM
from mixed_dataset_pt import PtCaptionDataset, MultiPtCaptionDataset, build_pt_collate_fn


# =========================================================
# 1. 通用工具
# =========================================================

def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_stat_str(name: str, x: Optional[torch.Tensor]) -> str:
    if x is None:
        return f"{name}: None"

    with torch.no_grad():
        y = x.detach()
        total = y.numel()

        # 先统一做有限值统计
        if y.is_floating_point() or y.is_complex():
            finite = torch.isfinite(y)
            finite_cnt = int(finite.sum().item())
            nan_cnt = int(torch.isnan(y).sum().item())
            posinf_cnt = int(torch.isposinf(y).sum().item())
            neginf_cnt = int(torch.isneginf(y).sum().item())

            if finite_cnt > 0:
                z = y[finite]
                return (
                    f"{name}: shape={tuple(y.shape)}, dtype={y.dtype}, device={y.device}, "
                    f"finite={finite_cnt}/{total}, nan={nan_cnt}, +inf={posinf_cnt}, -inf={neginf_cnt}, "
                    f"min={float(z.min().item()):.6e}, max={float(z.max().item()):.6e}, "
                    f"mean={float(z.mean().item()):.6e}, abs_mean={float(z.abs().mean().item()):.6e}, "
                    f"abs_max={float(z.abs().max().item()):.6e}"
                )
            else:
                return (
                    f"{name}: shape={tuple(y.shape)}, dtype={y.dtype}, device={y.device}, "
                    f"finite=0/{total}, nan={nan_cnt}, +inf={posinf_cnt}, -inf={neginf_cnt}"
                )

        # 整型 / bool 张量走这里：转 float 只为了统计
        z = y.float()
        return (
            f"{name}: shape={tuple(y.shape)}, dtype={y.dtype}, device={y.device}, "
            f"min={float(z.min().item()):.6e}, max={float(z.max().item()):.6e}, "
            f"mean={float(z.mean().item()):.6e}, abs_mean={float(z.abs().mean().item()):.6e}, "
            f"abs_max={float(z.abs().max().item()):.6e}"
        )

def assert_finite(name: str, x: torch.Tensor, meta: str = "") -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} 出现 NaN/Inf\n{tensor_stat_str(name, x)}\n{meta}")


def batch_meta_str(batch: Dict[str, Any]) -> str:
    lines = []
    sources = batch.get("sources", [])
    prompts = batch.get("prompts", [])
    captions = batch.get("captions", [])
    pt_paths = batch.get("pt_paths", [])

    bs = len(prompts) if isinstance(prompts, list) else 0
    lines.append(f"batch_size={bs}")

    for i in range(min(bs, 3)):
        src = sources[i] if i < len(sources) else "NA"
        prm = prompts[i][:120].replace("\n", "\\n") if i < len(prompts) else ""
        cap = captions[i][:120].replace("\n", "\\n") if i < len(captions) else ""
        pth = pt_paths[i] if i < len(pt_paths) else "NA"
        lines.append(f"[sample {i}] source={src}")
        lines.append(f"[sample {i}] pt_path={pth}")
        lines.append(f"[sample {i}] prompt={prm}")
        lines.append(f"[sample {i}] caption={cap}")
    return "\n".join(lines)


def check_module_params(module: torch.nn.Module, title: str) -> None:
    print(f"\n===== PARAM CHECK: {title} =====")
    for name, p in module.named_parameters():
        print(tensor_stat_str(f"param[{name}]", p))


def check_module_grads(module: torch.nn.Module, title: str) -> None:
    print(f"\n===== GRAD CHECK: {title} =====")
    for name, p in module.named_parameters():
        if p.grad is None:
            print(f"grad[{name}]: None")
        else:
            print(tensor_stat_str(f"grad[{name}]", p.grad))


def patch_collate_add_pt_paths(collate_fn):
    def _wrapped(batch):
        out = collate_fn(batch)
        if out is None:
            return None
        good = [b for b in batch if not b.get("_bad_sample", False)]

        # 注意：sample_valid 过滤后，这里先简单同步截取
        # 因为我们做 debug，batch_size=1 最稳妥，基本不会有错位
        out["pt_paths"] = [b.get("pt_path", "NA") for b in good[: len(out["prompts"])]]
        return out
    return _wrapped


# =========================================================
# 2. 构建数据
# =========================================================

def build_debug_dataset_and_loader(model: SarQwenVLForCausalLM) -> DataLoader:
    # 为了排错，先只开一个数据集、单线程、batch=1
    datasets = {
        "sarlang": PtCaptionDataset(
            root=PATHS.sarlang_root,
            pt_json=PATHS.sarlang_pt_train_json,
            max_samples=512,
        )
    }
    weights = {"sarlang": 1.0}

    mixed_ds = MultiPtCaptionDataset(
        datasets=datasets,
        weights=weights,
        epoch_length=512,
        seed=1234,
    )

    collate_fn = build_pt_collate_fn(
        tokenizer=model.tokenizer,
        image_token=model.image_token,
        num_image_tokens=TRAIN.num_image_tokens,
        max_length=TRAIN.max_length,
    )
    collate_fn = patch_collate_add_pt_paths(collate_fn)

    loader = DataLoader(
        mixed_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return loader


# =========================================================
# 3. 更细粒度 forward（完全展开）
# =========================================================

def forward_pt_debug(
    model: SarQwenVLForCausalLM,
    batch: Dict[str, Any],
    cast_to_lm_dtype: bool = True,
):
    meta = batch_meta_str(batch)

    sar_feats = batch["sar_feats"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    print("\n===== INPUT BATCH =====")
    print(meta)
    print(tensor_stat_str("sar_feats", sar_feats))
    print(tensor_stat_str("input_ids", input_ids))
    print(tensor_stat_str("attention_mask", attention_mask))
    print(tensor_stat_str("labels", labels))

    main_device = model.main_device
    patch_tokens = sar_feats.to(main_device)

    assert_finite("patch_tokens", patch_tokens, meta=meta)

    # 1) projector: 强制 fp32
    with torch.amp.autocast("cuda", enabled=False):
        sar_token_embeds_fp32 = model.projector(patch_tokens.float())

    print("\n===== AFTER PROJECTOR =====")
    print(tensor_stat_str("sar_token_embeds_fp32_raw", sar_token_embeds_fp32))
    assert_finite("sar_token_embeds_fp32_raw", sar_token_embeds_fp32, meta=meta)

    model._check_projector_dim(sar_token_embeds_fp32)

    # 2) scale
    target_scale = float(model._emb_scale)
    cur_scale = sar_token_embeds_fp32.abs().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)

    print(tensor_stat_str("cur_scale", cur_scale))
    assert_finite("cur_scale", cur_scale, meta=meta)

    sar_token_embeds_fp32 = sar_token_embeds_fp32 / (cur_scale + 1e-6) * target_scale
    print(tensor_stat_str("sar_token_embeds_fp32_scaled", sar_token_embeds_fp32))
    assert_finite("sar_token_embeds_fp32_scaled", sar_token_embeds_fp32, meta=meta)

    # 3) clamp
    sar_token_embeds_fp32 = torch.clamp(
        sar_token_embeds_fp32,
        min=-3.0 * target_scale,
        max=3.0 * target_scale,
    )
    print(tensor_stat_str("sar_token_embeds_fp32_clamped", sar_token_embeds_fp32))
    assert_finite("sar_token_embeds_fp32_clamped", sar_token_embeds_fp32, meta=meta)

    # 4) cast
    if cast_to_lm_dtype:
        sar_token_embeds = sar_token_embeds_fp32.to(model.lm_emb_dtype)
    else:
        sar_token_embeds = sar_token_embeds_fp32

    print(tensor_stat_str("sar_token_embeds_casted", sar_token_embeds))
    assert_finite("sar_token_embeds_casted", sar_token_embeds, meta=meta)

    # 5) move to lm device
    lm_dev = model.lm_input_device
    input_ids = input_ids.to(lm_dev)
    attention_mask = attention_mask.to(lm_dev)
    labels = labels.to(lm_dev)
    sar_token_embeds = sar_token_embeds.to(lm_dev)

    # 6) inject
    inputs_embeds = model._inject_sar_embeds(input_ids, sar_token_embeds)
    print("\n===== AFTER INJECT =====")
    print(tensor_stat_str("inputs_embeds", inputs_embeds))
    assert_finite("inputs_embeds", inputs_embeds, meta=meta)

    # 7) backbone
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    outputs = model.llm_backbone(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state
    print("\n===== AFTER LLM =====")
    print(tensor_stat_str("hidden_states", hidden_states))
    assert_finite("hidden_states", hidden_states, meta=meta)

    # 8) lm_head
    logits = model.lm_head(hidden_states)
    print(tensor_stat_str("logits", logits))
    assert_finite("logits", logits, meta=meta)

    # 9) loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    valid_targets = int(shift_labels.ne(-100).sum().item())
    print(f"valid_targets={valid_targets}")

    if valid_targets <= 0:
        raise RuntimeError(f"当前 batch 没有有效监督 token\n{meta}")

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    print(tensor_stat_str("loss", loss.detach().view(1)))
    assert_finite("loss", loss.detach(), meta=meta)

    return loss


# =========================================================
# 4. 单步反向 + 参数检查
# =========================================================

def run_one_debug_step(
    model: SarQwenVLForCausalLM,
    loader: DataLoader,
    lr: float = 1e-5,
    cast_to_lm_dtype: bool = True,
):
    optim = torch.optim.AdamW(model.projector.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    check_module_params(model.projector, "before forward")

    batch = next(iter(loader))
    if batch is None:
        raise RuntimeError("collate_fn 返回 None，当前 batch 全被过滤了")

    optim.zero_grad(set_to_none=True)

    loss = forward_pt_debug(model, batch, cast_to_lm_dtype=cast_to_lm_dtype)

    print("\n===== BACKWARD =====")
    loss.backward()

    check_module_grads(model.projector, "after backward")

    # 检查 grad finite
    for name, p in model.projector.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise RuntimeError(f"梯度出现 NaN/Inf: {name}\n{tensor_stat_str(f'grad[{name}]', p.grad)}")

    total_norm = torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_norm=1.0)
    print(f"grad_total_norm={float(total_norm):.6e}")

    optim.step()

    check_module_params(model.projector, "after optimizer.step")

    for name, p in model.projector.named_parameters():
        if not torch.isfinite(p).all():
            raise RuntimeError(f"optimizer.step 后参数损坏: {name}\n{tensor_stat_str(f'param[{name}]', p)}")

    print("\n[OK] 单步 forward/backward/step 完成，未出现 NaN/Inf")


# =========================================================
# 5. main
# =========================================================

def main():
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    set_seed(1234)

    # ===== 为了排错，强烈建议先手动改配置 =====
    # config_local.py 里临时改成：
    # TRAIN.fp16 = False
    # TRAIN.use_multi_gpu = False
    # TRAIN.main_device = "cuda:0"
    # TRAIN.device = "cuda:0"

    from sarclip_module import TokenLinearProjector
    from train_pt import infer_qwen3_vl_text_hidden_size

    main_device = TRAIN.main_device
    llm_hidden = infer_qwen3_vl_text_hidden_size(PATHS.qwen_path)
    print(f"[INFO] qwen3-vl text hidden size = {llm_hidden}")

    projector = TokenLinearProjector(
        in_dim=768,
        llm_hidden_size=llm_hidden,
    ).to(main_device)

    qwen_device_map = "cpu" if str(main_device) == "cpu" else (
        TRAIN.qwen_device_map if TRAIN.use_multi_gpu else None
    )

    model = SarQwenVLForCausalLM(
        qwen_path=PATHS.qwen_path,
        projector=projector,
        device=main_device,
        torch_dtype=torch.float16 if TRAIN.fp16 else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=qwen_device_map,
        gradient_checkpointing=TRAIN.gradient_checkpointing,
    )
    loader = build_debug_dataset_and_loader(model)

    # 第一轮：不经过 lm_emb_dtype cast，纯查 projector/注入/Qwen 是否稳定
    print("\n================ ROUND 1: 不 cast 到 lm_emb_dtype ================")
    run_one_debug_step(model, loader, lr=TRAIN.lr, cast_to_lm_dtype=False)

    # 第二轮：恢复 cast，查是不是 fp16/bf16 cast 触发
    print("\n================ ROUND 2: cast 到 lm_emb_dtype ================")
    run_one_debug_step(model, loader, lr=TRAIN.lr, cast_to_lm_dtype=True)


if __name__ == "__main__":
    main()