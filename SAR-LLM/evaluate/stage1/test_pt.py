"""
test_pt.py
==========
Stage1 旧评测实现。

说明：
- 推荐直接使用 `test.py`，它已经是统一入口
- 本文件保留为兼容实现与内部函数模块

对测试集进行完整评测，三项指标：
  1. Test Loss  ── 重用 forward_pt，衡量模型对 ground truth 文本的建模能力
  2. BLEU-4     ── n-gram 精度，caption 领域标准指标
  3. ROUGE-L    ── 最长公共子序列 F1，对召回率更敏感

依赖：
  pip install nltk rouge_score
  python -c "import nltk; nltk.download('punkt_tab')"

用法：
  # 只算 Test Loss（快）
  python test_pt.py --projector /path/to/projector.pt --loss_only

  # 完整评测，所有数据集，各取 500 条生成样本
  python test_pt.py --projector /path/to/projector.pt

  # 只跑指定数据集
  python test_pt.py --projector /path/to/projector.pt --datasets sarlang,sartext

  # 全量生成（慢）
  python test_pt.py --projector /path/to/projector.pt --max_gen_samples -1
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2] / "train" / "stage1"
PROJECT_DIR = Path(os.environ.get("SARCLIP_PROJECT_DIR", str(DEFAULT_PROJECT_DIR)))
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONUNBUFFERED", "1")  # 避免 background 模式下 stdout 缓冲
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config_local import PATHS, TRAIN                                              # noqa: E402
from common.vllm_backend import (                                                   # noqa: E402
    PromptEmbeddingBuilder,
    VLLMTextGenerator,
    release_cuda_resources,
    resolve_generation_backend,
)
from sarclip_module import TokenLinearProjector                                    # noqa: E402
from qwen3_sar_model import SarQwenVLForCausalLM, infer_qwen3_vl_text_hidden_size  # noqa: E402
from mixed_dataset_pt import PtCaptionDataset, build_pt_collate_fn                 # noqa: E402
from train_pt import forward_pt                                                    # noqa: E402


def _pick_inference_dtype() -> torch.dtype:
    if getattr(TRAIN, "bf16", False):
        return torch.bfloat16
    if getattr(TRAIN, "fp16", False):
        return torch.float16
    return torch.float32


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m{sec:04.1f}s"


def _emit_progress(progress_callback: Optional[Callable[[str], None]], msg: str) -> None:
    if progress_callback is not None:
        progress_callback(msg)
    else:
        print(msg)


def _prepare_sar_token_embeds_for_lm(
    model: SarQwenVLForCausalLM,
    sar_feats: torch.Tensor,
) -> torch.Tensor:
    """projector → scale-norm → cast to LLM embedding dtype."""
    device = model.main_device

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(amp_device, enabled=False):
        sar_token_embeds_fp32 = model.projector(sar_feats.to(device).float())

    target_scale = float(model._emb_scale)
    cur_scale = sar_token_embeds_fp32.abs().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    sar_token_embeds_fp32 = sar_token_embeds_fp32 / (cur_scale + 1e-6) * target_scale
    sar_token_embeds_fp32 = torch.clamp(
        sar_token_embeds_fp32,
        min=-3.0 * target_scale,
        max=3.0 * target_scale,
    )

    # 先搬到目标设备（单卡时为 no-op），再 cast 到 LLM embedding dtype（通常 bf16）
    return sar_token_embeds_fp32.to(model.lm_input_device).to(model.lm_emb_dtype)


# ──────────────────────────────────────────────────────────
# 1. 评测指标
# ──────────────────────────────────────────────────────────

def _compute_bleu(references: List[str], hypotheses: List[str], weights) -> float:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    refs = [[ref.split()] for ref in references]
    hyps = [hyp.split() for hyp in hypotheses]
    sf = SmoothingFunction().method1
    return float(corpus_bleu(refs, hyps, weights=weights, smoothing_function=sf))


def compute_bleu_scores(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    return {
        "bleu1": _compute_bleu(references, hypotheses, weights=(1.0, 0.0, 0.0, 0.0)),
        "bleu2": _compute_bleu(references, hypotheses, weights=(0.5, 0.5, 0.0, 0.0)),
        "bleu3": _compute_bleu(references, hypotheses, weights=(1 / 3, 1 / 3, 1 / 3, 0.0)),
        "bleu4": _compute_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)),
    }


def compute_rougeL(references: List[str], hypotheses: List[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)]
    return float(sum(scores) / len(scores)) if scores else 0.0


def _strip_image_tag(text: str) -> str:
    return str(text).strip().replace("<image>", "").strip()


def _role_prefix(role: str, role_prefix_map: Optional[Dict[str, str]] = None) -> str:
    role_prefix_map = role_prefix_map or {
        "user": "User", "assistant": "Assistant", "system": "System",
    }
    role = str(role).strip().lower()
    return role_prefix_map.get(role, role.capitalize() or "User")


def _build_caption_prompt(
    image_tokens_str: str,
    prompt_text: str,
    user_role: str = "user",
    assistant_role: str = "assistant",
    role_prefix_map: Optional[Dict[str, str]] = None,
) -> str:
    return (
        f"{image_tokens_str}\n"
        f"{_role_prefix(user_role, role_prefix_map)}: {_strip_image_tag(prompt_text)}\n"
        f"{_role_prefix(assistant_role, role_prefix_map)}: "
    )


# ──────────────────────────────────────────────────────────
# 2. Test Loss
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test_loss(model: SarQwenVLForCausalLM, data_loader: DataLoader) -> Optional[float]:
    """在测试集上计算平均交叉熵 loss，复用 train_pt.forward_pt。"""
    model.eval()
    total_loss, total_count = 0.0, 0

    for i, batch in enumerate(data_loader):
        if batch is None:
            continue
        try:
            loss = forward_pt(
                model          = model,
                sar_feats      = batch["sar_feats"],
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                labels         = batch["labels"],
            )
        except RuntimeError as e:
            print(f"  [WARN] batch {i} forward error: {e}, skipping")
            continue

        if loss is not None:
            bs = batch["input_ids"].size(0)
            total_loss  += float(loss.item()) * bs
            total_count += bs

        if (i + 1) % 50 == 0:
            done = total_count
            print(f"  [loss] processed {done} samples...")

    return total_loss / total_count if total_count > 0 else None


# ──────────────────────────────────────────────────────────
# 3. 推理生成（BLEU / ROUGE）
# ──────────────────────────────────────────────────────────

def _build_inference_collate(tokenizer, image_token: str, num_image_tokens: int, max_length: int):
    """推理专用 collate：左侧 padding，避免 decoder-only 批量生成从 pad 位置续写。"""
    image_tokens_str = " ".join([image_token] * num_image_tokens)

    def _collate(batch):
        good = [b for b in batch if not b.get("_bad_sample", False)]
        if not good:
            return None

        prompts = [
            _build_caption_prompt(
                image_tokens_str=image_tokens_str,
                prompt_text=b["prompt"],
                user_role=b.get("user_role", "user"),
                assistant_role=b.get("assistant_role", "assistant"),
            )
            for b in good
        ]
        captions  = [str(b["caption"]).strip() for b in good]
        sar_feats = torch.stack([b["sar_feat"] for b in good], dim=0)
        image_ids = [str(b.get("image_id", "")) for b in good]
        pt_paths = [str(b.get("pt_path", "")) for b in good]
        sources = [str(b.get("_source", "")) for b in good]

        old_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        try:
            tok = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = old_padding_side
        return {
            "sar_feats":      sar_feats,
            "input_ids":      tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "captions":       captions,
            "prompts":        prompts,
            "image_ids":      image_ids,
            "pt_paths":       pt_paths,
            "sources":        sources,
        }

    return _collate


@torch.inference_mode()
def generate_batch(
    model:          SarQwenVLForCausalLM,
    sar_feats:      torch.Tensor,   # (B, 195, 768)
    prompt_ids:     torch.Tensor,   # (B, L_prompt)
    prompt_mask:    torch.Tensor,   # (B, L_prompt)
    max_new_tokens: int = 50,
) -> Dict[str, object]:
    """
    使用 llm_causal.generate() 加速推理，替代手写 greedy decode。
    """
    lm_dev = model.lm_input_device
    prompt_ids       = prompt_ids.to(lm_dev)
    prompt_mask      = prompt_mask.to(lm_dev)
    sar_token_embeds = _prepare_sar_token_embeds_for_lm(model, sar_feats)
    inputs_embeds    = model._inject_sar_embeds(prompt_ids, sar_token_embeds)

    bsz = prompt_ids.shape[0]

    out_ids = model.vl_model.generate(
        inputs_embeds      = inputs_embeds,
        attention_mask     = prompt_mask,
        max_new_tokens     = max_new_tokens,
        do_sample          = False,
        pad_token_id       = model.tokenizer.pad_token_id,
        eos_token_id       = model.tokenizer.eos_token_id,
    )

    # 当用 inputs_embeds 传入 prompt 时，generate 返回的 out_ids 只含新生成的 token
    new_ids = out_ids

    texts = [model.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in new_ids]
    token_lengths = [int((ids != model.tokenizer.pad_token_id).sum()) for ids in new_ids]
    decode_steps  = int(new_ids.shape[1])
    return {
        "texts": texts,
        "token_lengths": token_lengths,
        "decode_steps": decode_steps,
        "batch_size": bsz,
    }


@torch.inference_mode()
def evaluate_generation(
    model:          SarQwenVLForCausalLM,
    data_loader:    DataLoader,
    dataset_name:   Optional[str] = None,
    max_new_tokens: int = 200,
    max_samples:    Optional[int] = 500,
    progress_every: int = 20,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict:
    model.eval()
    all_refs: List[str] = []
    all_hyps: List[str] = []
    prompt_examples: List[str] = []
    total_loader_wait = 0.0
    total_generate_time = 0.0
    total_decode_steps = 0
    total_output_tokens = 0
    total_batches = 0
    loop_start = time.time()
    prev_batch_end = loop_start
    sample_records: List[Dict[str, Any]] = []

    for i, batch in enumerate(data_loader):
        batch_ready_time = time.time()
        total_loader_wait += batch_ready_time - prev_batch_end
        if batch is None:
            prev_batch_end = time.time()
            continue
        if max_samples is not None and len(all_hyps) >= max_samples:
            break
        try:
            gen_start = time.time()
            gen_out = generate_batch(
                model          = model,
                sar_feats      = batch["sar_feats"],
                prompt_ids     = batch["input_ids"],
                prompt_mask    = batch["attention_mask"],
                max_new_tokens = max_new_tokens,
            )
            gen_elapsed = time.time() - gen_start
        except RuntimeError as e:
            _emit_progress(progress_callback, f"  [WARN] batch {i} generate error: {e}, skipping")
            prev_batch_end = time.time()
            continue

        batch_refs = batch["captions"]
        batch_hyps = list(gen_out["texts"])
        batch_prompts = batch.get("prompts", [])
        batch_image_ids = list(batch.get("image_ids", []))
        batch_pt_paths = list(batch.get("pt_paths", []))
        batch_sources = list(batch.get("sources", []))
        batch_token_lengths = list(gen_out["token_lengths"])
        total_generate_time += gen_elapsed
        total_decode_steps += int(gen_out["decode_steps"])
        total_output_tokens += sum(batch_token_lengths)
        total_batches += 1

        if max_samples is not None:
            remaining = max_samples - len(all_hyps)
            if remaining <= 0:
                break
            batch_refs = batch_refs[:remaining]
            batch_hyps = batch_hyps[:remaining]
            batch_prompts = batch_prompts[:remaining]
            batch_image_ids = batch_image_ids[:remaining]
            batch_pt_paths = batch_pt_paths[:remaining]
            batch_sources = batch_sources[:remaining]
            batch_token_lengths = batch_token_lengths[:remaining]

        if not prompt_examples and batch_prompts:
            prompt_examples = [batch_prompts[0]]

        all_refs.extend(batch_refs)
        all_hyps.extend(batch_hyps)
        for idx, (ref, hyp) in enumerate(zip(batch_refs, batch_hyps)):
            source_value = batch_sources[idx] if idx < len(batch_sources) else ""
            source_value = source_value or dataset_name or "unknown"
            sample_records.append({
                "sample_index": len(sample_records) + 1,
                "dataset": dataset_name or source_value,
                "image_id": batch_image_ids[idx] if idx < len(batch_image_ids) else "",
                "pt_path": batch_pt_paths[idx] if idx < len(batch_pt_paths) else "",
                "source": source_value,
                "prompt": batch_prompts[idx] if idx < len(batch_prompts) else "",
                "reference": ref,
                "hypothesis": hyp,
                "output_tokens": batch_token_lengths[idx] if idx < len(batch_token_lengths) else None,
            })

        if progress_every > 0 and ((i + 1) % progress_every == 0):
            elapsed = time.time() - loop_start
            samples_done = len(all_hyps)
            samples_per_sec = samples_done / max(elapsed, 1e-6)
            tokens_per_sec = total_output_tokens / max(total_generate_time, 1e-6)
            avg_tokens = total_output_tokens / max(samples_done, 1)
            avg_decode_steps = total_decode_steps / max(total_batches, 1)
            _emit_progress(
                progress_callback,
                "  [gen] "
                f"batch={i + 1} "
                f"samples={samples_done} "
                f"elapsed={_format_seconds(elapsed)} "
                f"gen={_format_seconds(total_generate_time)} "
                f"loader_wait={_format_seconds(total_loader_wait)} "
                f"samples/s={samples_per_sec:.2f} "
                f"tokens/s={tokens_per_sec:.2f} "
                f"avg_out_tokens={avg_tokens:.1f} "
                f"avg_decode_steps={avg_decode_steps:.1f}"
            )

        if max_samples is not None and len(all_hyps) >= max_samples:
            break

        prev_batch_end = time.time()

    if not all_refs:
        return {
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "rougeL": 0.0,
            "num_samples": 0,
            "examples": [],
            "prompt_examples": [],
            "sample_records": [],
            "elapsed_sec": 0.0,
            "loader_wait_sec": total_loader_wait,
            "generate_sec": total_generate_time,
            "samples_per_sec": 0.0,
            "tokens_per_sec": 0.0,
            "avg_output_tokens": 0.0,
        }

    elapsed_total = time.time() - loop_start
    bleu_scores = compute_bleu_scores(all_refs, all_hyps)

    return {
        **bleu_scores,
        "rougeL":      compute_rougeL(all_refs, all_hyps),
        "num_samples": len(all_refs),
        "examples":    list(zip(all_refs[:3], all_hyps[:3])),
        "prompt_examples": prompt_examples,
        "sample_records": sample_records,
        "elapsed_sec": elapsed_total,
        "loader_wait_sec": total_loader_wait,
        "generate_sec": total_generate_time,
        "samples_per_sec": len(all_refs) / max(elapsed_total, 1e-6),
        "tokens_per_sec": total_output_tokens / max(total_generate_time, 1e-6),
        "avg_output_tokens": total_output_tokens / max(len(all_refs), 1),
    }


@torch.inference_mode()
def evaluate_generation_vllm(
    prompt_builder: PromptEmbeddingBuilder,
    projector: TokenLinearProjector,
    generator: VLLMTextGenerator,
    data_loader: DataLoader,
    dataset_name: Optional[str] = None,
    max_new_tokens: int = 200,
    max_samples: Optional[int] = 500,
    progress_every: int = 20,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict:
    projector.eval()
    projector_device = next(projector.parameters()).device

    all_refs: List[str] = []
    all_hyps: List[str] = []
    prompt_examples: List[str] = []
    total_loader_wait = 0.0
    total_generate_time = 0.0
    total_output_tokens = 0
    total_batches = 0
    loop_start = time.time()
    prev_batch_end = loop_start
    sample_records: List[Dict[str, Any]] = []

    for i, batch in enumerate(data_loader):
        batch_ready_time = time.time()
        total_loader_wait += batch_ready_time - prev_batch_end
        if batch is None:
            prev_batch_end = time.time()
            continue
        if max_samples is not None and len(all_hyps) >= max_samples:
            break

        try:
            prompt_embeds = prompt_builder.build_prompt_embeds(
                prompt_ids= batch["input_ids"],
                attention_mask=batch["attention_mask"],
                sar_feats=batch["sar_feats"],
                projector=projector,
                projector_device=projector_device,
            )
            gen_start = time.time()
            gen_out = generator.generate(
                prompt_embeds=prompt_embeds,
                max_new_tokens=max_new_tokens,
            )
            gen_elapsed = time.time() - gen_start
        except RuntimeError as e:
            _emit_progress(progress_callback, f"  [WARN] batch {i} vLLM generate error: {e}, skipping")
            prev_batch_end = time.time()
            continue

        batch_refs = batch["captions"]
        batch_hyps = list(gen_out["texts"])
        batch_prompts = batch.get("prompts", [])
        batch_image_ids = list(batch.get("image_ids", []))
        batch_pt_paths = list(batch.get("pt_paths", []))
        batch_sources = list(batch.get("sources", []))
        batch_token_lengths = list(gen_out["token_lengths"])
        total_generate_time += gen_elapsed
        total_output_tokens += sum(batch_token_lengths)
        total_batches += 1

        if max_samples is not None:
            remaining = max_samples - len(all_hyps)
            if remaining <= 0:
                break
            batch_refs = batch_refs[:remaining]
            batch_hyps = batch_hyps[:remaining]
            batch_prompts = batch_prompts[:remaining]
            batch_image_ids = batch_image_ids[:remaining]
            batch_pt_paths = batch_pt_paths[:remaining]
            batch_sources = batch_sources[:remaining]
            batch_token_lengths = batch_token_lengths[:remaining]

        if not prompt_examples and batch_prompts:
            prompt_examples = [batch_prompts[0]]

        all_refs.extend(batch_refs)
        all_hyps.extend(batch_hyps)
        for idx, (ref, hyp) in enumerate(zip(batch_refs, batch_hyps)):
            source_value = batch_sources[idx] if idx < len(batch_sources) else ""
            source_value = source_value or dataset_name or "unknown"
            sample_records.append({
                "sample_index": len(sample_records) + 1,
                "dataset": dataset_name or source_value,
                "image_id": batch_image_ids[idx] if idx < len(batch_image_ids) else "",
                "pt_path": batch_pt_paths[idx] if idx < len(batch_pt_paths) else "",
                "source": source_value,
                "prompt": batch_prompts[idx] if idx < len(batch_prompts) else "",
                "reference": ref,
                "hypothesis": hyp,
                "output_tokens": batch_token_lengths[idx] if idx < len(batch_token_lengths) else None,
            })

        if progress_every > 0 and ((i + 1) % progress_every == 0):
            elapsed = time.time() - loop_start
            samples_done = len(all_hyps)
            samples_per_sec = samples_done / max(elapsed, 1e-6)
            tokens_per_sec = total_output_tokens / max(total_generate_time, 1e-6)
            avg_tokens = total_output_tokens / max(samples_done, 1)
            _emit_progress(
                progress_callback,
                "  [gen:vllm] "
                f"batch={i + 1} "
                f"samples={samples_done} "
                f"elapsed={_format_seconds(elapsed)} "
                f"gen={_format_seconds(total_generate_time)} "
                f"loader_wait={_format_seconds(total_loader_wait)} "
                f"samples/s={samples_per_sec:.2f} "
                f"tokens/s={tokens_per_sec:.2f} "
                f"avg_out_tokens={avg_tokens:.1f}"
            )

        if max_samples is not None and len(all_hyps) >= max_samples:
            break

        prev_batch_end = time.time()

    if not all_refs:
        return {
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "rougeL": 0.0,
            "num_samples": 0,
            "examples": [],
            "prompt_examples": [],
            "sample_records": [],
            "elapsed_sec": 0.0,
            "loader_wait_sec": total_loader_wait,
            "generate_sec": total_generate_time,
            "samples_per_sec": 0.0,
            "tokens_per_sec": 0.0,
            "avg_output_tokens": 0.0,
        }

    elapsed_total = time.time() - loop_start
    bleu_scores = compute_bleu_scores(all_refs, all_hyps)
    return {
        **bleu_scores,
        "rougeL": compute_rougeL(all_refs, all_hyps),
        "num_samples": len(all_refs),
        "examples": list(zip(all_refs[:3], all_hyps[:3])),
        "prompt_examples": prompt_examples,
        "sample_records": sample_records,
        "elapsed_sec": elapsed_total,
        "loader_wait_sec": total_loader_wait,
        "generate_sec": total_generate_time,
        "samples_per_sec": len(all_refs) / max(elapsed_total, 1e-6),
        "tokens_per_sec": total_output_tokens / max(total_generate_time, 1e-6),
        "avg_output_tokens": total_output_tokens / max(len(all_refs), 1),
    }


# ──────────────────────────────────────────────────────────
# 4. 模型加载
# ──────────────────────────────────────────────────────────

def load_model(projector_path: str, device: str = "cuda:0") -> SarQwenVLForCausalLM:
    """加载 SarQwenVLForCausalLM，所有组件放在同一设备上。

    Args:
        projector_path: projector 权重文件路径（.pt 纯权重或含 'projector' key 的完整 checkpoint）
        device: 推理设备，如 "cuda:0"、"cuda:1"、"cpu"
    """
    inference_dtype = _pick_inference_dtype()
    llm_hidden = infer_qwen3_vl_text_hidden_size(PATHS.qwen_path)
    print(f"[INFO] device          = {device}")
    print(f"[INFO] inference dtype = {inference_dtype}")
    print(f"[INFO] qwen3-vl text hidden size = {llm_hidden}")

    projector = TokenLinearProjector(in_dim=768, llm_hidden_size=llm_hidden).to(device)
    model = SarQwenVLForCausalLM(
        qwen_path              = PATHS.qwen_path,
        projector              = projector,
        device                 = device,
        torch_dtype            = inference_dtype,
        trust_remote_code      = True,
        low_cpu_mem_usage      = True,
        device_map             = device,   # Qwen 和 projector 放在同一卡
        gradient_checkpointing = False,
    )

    ckpt_path = Path(projector_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"projector 权重不存在: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "projector" in state:
        state = state["projector"]
    model.projector.load_state_dict(state, strict=True)
    print(f"[OK] Loaded projector from: {ckpt_path}")

    model.eval()
    # 刷新 stdout，避免 background 模式下缓冲导致输出不可见
    sys.stdout.flush()
    return model


def load_projector_only(projector_path: str, device: str = "cuda:0") -> TokenLinearProjector:
    llm_hidden = infer_qwen3_vl_text_hidden_size(PATHS.qwen_path)
    projector = TokenLinearProjector(in_dim=768, llm_hidden_size=llm_hidden).to(device)

    ckpt_path = Path(projector_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"projector 权重不存在: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "projector" in state:
        state = state["projector"]
    projector.load_state_dict(state, strict=True)
    projector.eval()
    print(f"[OK] Loaded projector for vLLM from: {ckpt_path}")
    sys.stdout.flush()
    return projector


def load_vllm_caption_backend(args) -> tuple[PromptEmbeddingBuilder, TokenLinearProjector, VLLMTextGenerator]:
    inference_dtype = _pick_inference_dtype()
    prompt_builder = PromptEmbeddingBuilder(
        qwen_path=PATHS.qwen_path,
        image_token="<sar>",
        torch_dtype=inference_dtype,
    )
    projector = load_projector_only(args.projector, device=args.device)
    generator = VLLMTextGenerator(
        model_path=PATHS.qwen_path,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        dtype=args.vllm_dtype,
        enforce_eager=args.vllm_enforce_eager,
    )
    return prompt_builder, projector, generator


def _log_generation_metrics(log: Callable[[str], None], gen_metrics: Dict, elapsed_sec: float) -> None:
    log(f"  BLEU-1     = {gen_metrics['bleu1']:.4f}")
    log(f"  BLEU-2     = {gen_metrics['bleu2']:.4f}")
    log(f"  BLEU-3     = {gen_metrics['bleu3']:.4f}")
    log(f"  BLEU-4     = {gen_metrics['bleu4']:.4f}")
    log(f"  ROUGE-L    = {gen_metrics['rougeL']:.4f}")
    log(f"  评测样本数 = {gen_metrics['num_samples']}  ({elapsed_sec:.0f}s)")
    log(
        "  生成性能   = "
        f"samples/s {gen_metrics['samples_per_sec']:.2f} | "
        f"tokens/s {gen_metrics['tokens_per_sec']:.2f} | "
        f"avg_out_tokens {gen_metrics['avg_output_tokens']:.1f} | "
        f"gen {_format_seconds(gen_metrics['generate_sec'])} | "
        f"loader_wait {_format_seconds(gen_metrics['loader_wait_sec'])}"
    )
    if gen_metrics.get("prompt_examples"):
        log("\n  首条真实 prompt:")
        log(gen_metrics["prompt_examples"][0])

def _log_sample_records(
    log: Callable[[str], None],
    sample_records: List[Dict[str, Any]],
    print_prompt: bool = False,
) -> None:
    if not sample_records:
        log("\n  [WARN] 无样本记录可输出")
        return

    log("\n  全量样本结果:")
    for rec in sample_records:
        log(
            f"  [{rec['sample_index']}] "
            f"dataset={rec.get('dataset', '') or 'NA'} "
            f"image_id={rec.get('image_id', '') or 'NA'} "
            f"source={rec.get('source', '') or 'NA'} "
            f"tokens={rec.get('output_tokens', 'NA')}"
        )
        if rec.get("pt_path"):
            log(f"      PT : {rec['pt_path']}")
        if print_prompt and rec.get("prompt"):
            log(f"      Prompt: {rec['prompt']}")
        log(f"      Ref: {rec.get('reference', '')}")
        log(f"      Hyp: {rec.get('hypothesis', '')}")
        log()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────
# 5. 数据集构建（评测时跑全部有 test json 的数据集）
# ──────────────────────────────────────────────────────────

ALL_DATASETS = {
    "sarlang":  lambda: PtCaptionDataset(root=PATHS.sarlang_root,  pt_json=PATHS.sarlang_pt_test_json),
    "sartext":  lambda: PtCaptionDataset(root=PATHS.sartext_root,  pt_json=PATHS.sartext_pt_test_json),
    "sarcap":   lambda: PtCaptionDataset(root=PATHS.sarcap_root,   pt_json=PATHS.sarcap_pt_test_json),
    "fsarcap":  lambda: PtCaptionDataset(root=PATHS.fsarcap_root,  pt_json=PATHS.fsarcap_pt_test_json),
}


def build_test_datasets(wanted: Optional[List[str]] = None) -> Dict[str, PtCaptionDataset]:
    """
    默认加载 config 里 mixed_weight > 0 的全部数据集。
    传入 wanted 列表时只加载指定的（不受 mixed_weight 限制）。
    """
    datasets: Dict[str, PtCaptionDataset] = {}

    if wanted:
        for name in wanted:
            if name not in ALL_DATASETS:
                raise ValueError(f"未知数据集: {name!r}，可选: {list(ALL_DATASETS)}")
            datasets[name] = ALL_DATASETS[name]()
    else:
        # 默认：按训练时权重决定评测哪些
        weight_map = {
            "sarlang": TRAIN.mixed_weight_sarlang,
            "sartext":  TRAIN.mixed_weight_sartext,
            "sarcap":   TRAIN.mixed_weight_sarcap,
            "fsarcap":  TRAIN.mixed_weight_fsarcap,
        }
        for name, w in weight_map.items():
            if w > 0:
                datasets[name] = ALL_DATASETS[name]()

    if not datasets:
        raise ValueError("没有可用的测试数据集，请检查 --datasets 参数或 config 中 mixed_weight_* 设置")

    return datasets


# ──────────────────────────────────────────────────────────
# 6. main
# ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="test_pt.py：SAR caption 完整评测")
    parser.add_argument("--projector", type=str, required=True,
                        help="projector 权重路径（.pt 纯权重 或 完整 checkpoint）")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备，如 cuda:0 / cuda:1 / cpu")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="推理 batch size")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="生成时最大新 token 数")
    parser.add_argument("--max_gen_samples", type=int, default=500,
                        help="每个数据集 BLEU/ROUGE 评测的最大样本数；-1 表示全量")
    parser.add_argument("--loss_only", action="store_true",
                        help="只计算 Test Loss，跳过生成")
    parser.add_argument("--gen_only", action="store_true",
                        help="只计算 BLEU/ROUGE，跳过 Test Loss")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集名，e.g. sarlang,sartext；默认跑 config 里启用的全部")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num_workers")
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
    parser.add_argument("--vllm_dtype", type=str, default="auto",
                        help="vLLM dtype，如 auto / float16 / bfloat16")
    parser.add_argument("--vllm_enforce_eager", action="store_true",
                        help="为 vLLM 打开 enforce_eager，便于兼容性排查")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.loss_only and args.gen_only:
        raise ValueError("--loss_only 与 --gen_only 不能同时开启")

    max_gen = None if args.max_gen_samples == -1 else args.max_gen_samples
    wanted = [s.strip() for s in args.datasets.split(",")] if args.datasets else None
    gen_backend = resolve_generation_backend(args.gen_backend)

    # ── 日志文件 ──────────────────────────────────────────
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"test_pt_{ts}.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    log(f"[INFO] log  → {log_path}")
    log(f"[INFO] projector → {args.projector}")
    log(f"[INFO] device    → {args.device}")
    log(f"[INFO] datasets  → {args.datasets or '(config defaults)'}")
    log(f"[INFO] loss_only = {args.loss_only}  max_gen_samples = {args.max_gen_samples}")
    log(f"[INFO] progress_every = {args.progress_every}")
    log(f"[INFO] generation backend = {gen_backend}")
    if gen_backend == "vllm":
        log("[INFO] vLLM 选卡建议通过 CUDA_VISIBLE_DEVICES 控制；--device 主要约束 projector/HF 路径")

    # ── 构建测试数据集 ────────────────────────────────────
    test_datasets = build_test_datasets(wanted)
    log(f"[INFO] datasets to evaluate: {list(test_datasets)}")
    all_results: Dict[str, Dict] = {name: {} for name in test_datasets}

    model: Optional[SarQwenVLForCausalLM] = None
    try:
        if not args.gen_only:
            log(f"\n{'='*60}")
            log("  阶段 1/2: Test Loss")
            log(f"{'='*60}")

            model = load_model(args.projector, device=args.device)
            collate_loss = build_pt_collate_fn(
                tokenizer=model.tokenizer,
                image_token=model.image_token,
                num_image_tokens=TRAIN.num_image_tokens,
                max_length=TRAIN.max_length,
            )

            for ds_name, ds in test_datasets.items():
                log(f"\n{'='*60}")
                log(f"  [Loss] 数据集: {ds_name}   test size = {len(ds)}")
                log(f"{'='*60}")
                t0 = time.time()
                loss_loader = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=collate_loss,
                    pin_memory=torch.cuda.is_available(),
                )
                test_loss = evaluate_test_loss(model, loss_loader)
                all_results[ds_name]["test_loss"] = test_loss
                loss_str = f"{test_loss:.4f}" if test_loss is not None else "N/A"
                log(f"  Test Loss = {loss_str}  ({time.time() - t0:.0f}s)")
        else:
            log("\n[INFO] 跳过 Test Loss（--gen_only）")
            for result in all_results.values():
                result["test_loss"] = None

        if args.loss_only:
            log("\n[INFO] 跳过生成（--loss_only）")
        else:
            if gen_backend == "vllm" and model is not None:
                log("\n[INFO] 准备切换到 vLLM 生成：释放 transformers 模型显存")
                release_cuda_resources(model)
                model = None

            log(f"\n{'='*60}")
            log(f"  阶段 2/2: Generation ({gen_backend})")
            log(f"{'='*60}")

            if gen_backend == "hf":
                if model is None:
                    model = load_model(args.projector, device=args.device)
                collate_gen = _build_inference_collate(
                    tokenizer=model.tokenizer,
                    image_token=model.image_token,
                    num_image_tokens=TRAIN.num_image_tokens,
                    max_length=TRAIN.max_length,
                )
                for ds_name, ds in test_datasets.items():
                    log(f"\n{'='*60}")
                    log(f"  [Gen:{gen_backend}] 数据集: {ds_name}   test size = {len(ds)}")
                    log(f"{'='*60}")
                    t1 = time.time()
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
                        max_new_tokens=args.max_new_tokens,
                        max_samples=max_gen,
                        progress_every=args.progress_every,
                        progress_callback=log,
                    )
                    all_results[ds_name].update(gen_metrics)
                    _log_generation_metrics(log, gen_metrics, time.time() - t1)
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
                        log(f"  [Gen:{gen_backend}] 数据集: {ds_name}   test size = {len(ds)}")
                        log(f"{'='*60}")
                        t1 = time.time()
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
                            max_new_tokens=args.max_new_tokens,
                            max_samples=max_gen,
                            progress_every=args.progress_every,
                            progress_callback=log,
                        )
                        all_results[ds_name].update(gen_metrics)
                        _log_generation_metrics(log, gen_metrics, time.time() - t1)
                finally:
                    release_cuda_resources(prompt_builder, projector, generator)
    finally:
        if model is not None:
            release_cuda_resources(model)

    # ── 汇总 ─────────────────────────────────────────────
    log(f"\n{'='*60}")
    log("  汇总结果")
    log(f"{'='*60}")
    if args.loss_only:
        header = f"{'数据集':<12}  {'Test Loss':>10}"
        log(header)
        log("-" * len(header))
        for ds_name, r in all_results.items():
            ls = f"{r['test_loss']:.4f}" if r.get("test_loss") is not None else "   N/A"
            log(f"{ds_name:<12}  {ls:>10}")
    else:
        header = (
            f"{'数据集':<12}  {'Test Loss':>10}  {'BLEU-1':>8}  {'BLEU-2':>8}  "
            f"{'BLEU-3':>8}  {'BLEU-4':>8}  {'ROUGE-L':>8}  {'样本数':>8}"
        )
        log(header)
        log("-" * len(header))
        for ds_name, r in all_results.items():
            ls = f"{r['test_loss']:.4f}"  if r.get("test_loss")    is not None else "   N/A"
            b1 = f"{r['bleu1']:.4f}"      if r.get("bleu1")        is not None else "   N/A"
            b2 = f"{r['bleu2']:.4f}"      if r.get("bleu2")        is not None else "   N/A"
            b3 = f"{r['bleu3']:.4f}"      if r.get("bleu3")        is not None else "   N/A"
            b4 = f"{r['bleu4']:.4f}"      if r.get("bleu4")        is not None else "   N/A"
            rs = f"{r['rougeL']:.4f}"     if r.get("rougeL")       is not None else "   N/A"
            ns = str(r.get("num_samples", "-"))
            log(
                f"{ds_name:<12}  {ls:>10}  {b1:>8}  {b2:>8}  "
                f"{b3:>8}  {b4:>8}  {rs:>8}  {ns:>8}"
            )

    log(f"\n[INFO] 完整日志已保存至: {log_path}")
    log_file.close()


if __name__ == "__main__":
    print("[INFO] `test_pt.py` 已降级为兼容入口；推荐改用 `test.py`。")
    main()
