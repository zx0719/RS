"""
test_stage2.py
==============
Stage2 VQA 评测脚本。

指标：
  1. Test Loss  ── 交叉熵 loss
  2. BLEU-4     ── n-gram 精度
  3. ROUGE-L    ── 最长公共子序列 F1
  4. Exact Match（EM）── 完全匹配率（适合短答案 VQA）

依赖：
  pip install nltk rouge_score
  python -c "import nltk; nltk.download('punkt_tab')"

用法：
  # 只算 Test Loss
  python test_stage2.py --projector /path/to/projector_stage2.pt --loss_only

  # 完整评测（Loss + BLEU-4 + ROUGE-L + EM）
  python test_stage2.py --projector /path/to/projector_stage2.pt

  # 同时加载 LoRA adapter
  python test_stage2.py --projector /path/to/projector_stage2.pt \
                        --lora_adapter /path/to/lora_adapter

  # 全量测试集
  python test_stage2.py --projector /path/to/projector_stage2.pt --max_gen_samples -1
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

STAGE2_TRAIN_DIR = Path(__file__).resolve().parent.parent.parent / "train" / "stage2"
STAGE1_DIR = Path(__file__).resolve().parent.parent.parent / "train" / "stage1"
EVAL_ROOT = Path(__file__).resolve().parent.parent
for d in [str(EVAL_ROOT), str(STAGE2_TRAIN_DIR), str(STAGE1_DIR)]:
    if d not in sys.path:
        sys.path.append(d)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from common.vllm_backend import (                                                 # noqa: E402
    PromptEmbeddingBuilder,
    VLLMTextGenerator,
    build_lora_request,
    release_cuda_resources,
    resolve_generation_backend,
)
from config_local import PATHS, TRAIN                                              # noqa: E402
from qwen3_sar_model import SarQwenVLForCausalLM, infer_qwen3_vl_text_hidden_size  # noqa: E402
from sarclip_module import TokenLinearProjector                                    # noqa: E402
from train_stage2 import forward_vqa                                               # noqa: E402
from vqa_dataset_pt import PtVqaDataset, build_vqa_collate_fn                      # noqa: E402


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


def _resolve_main_device(device: Optional[str]) -> str:
    if device:
        return device
    if getattr(TRAIN, "use_multi_gpu", False):
        return str(getattr(TRAIN, "main_device", "cuda:0"))
    return str(getattr(TRAIN, "device", "cuda:0"))


def _resolve_qwen_device_map(device: Optional[str]):
    if device:
        return device
    if getattr(TRAIN, "use_multi_gpu", False):
        return getattr(TRAIN, "qwen_device_map", None)
    return _resolve_main_device(device)


def compute_bleu4(references: List[str], hypotheses: List[str]) -> float:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    refs = [[ref.split()] for ref in references]
    hyps = [hyp.split() for hyp in hypotheses]
    sf = SmoothingFunction().method1
    return float(corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf))


def compute_rougeL(references: List[str], hypotheses: List[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_exact_match(references: List[str], hypotheses: List[str]) -> float:
    matches = sum(
        ref.strip().lower() == hyp.strip().lower()
        for ref, hyp in zip(references, hypotheses)
    )
    return float(matches) / len(references) if references else 0.0


@torch.no_grad()
def evaluate_test_loss(model: SarQwenVLForCausalLM, data_loader: DataLoader) -> Optional[float]:
    model.eval()
    total_loss, total_count = 0.0, 0

    for i, batch in enumerate(data_loader):
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
            print(f"[WARN] batch {i} forward error: {e}, skipping")
            continue

        if loss is not None:
            bs = batch["input_ids"].size(0)
            total_loss += float(loss.item()) * bs
            total_count += bs

        if (i + 1) % 50 == 0:
            print(f"  [loss eval] processed {(i + 1) * data_loader.batch_size} samples...")

    model.train()
    return total_loss / total_count if total_count > 0 else None


def _build_stage2_inference_collate(
    tokenizer,
    image_token: str = "<sar>",
    num_image_tokens: int = 195,
    max_length: int = 1024,
) -> Callable:
    image_tokens_str = " ".join([image_token] * num_image_tokens)

    def _collate(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        good = [b for b in batch if not b.get("_bad_sample", False)]
        if not good:
            return None

        sar_feats = torch.stack([b["sar_feat"] for b in good], dim=0)
        prompts: List[str] = []
        references: List[str] = []

        for b in good:
            turns = b.get("turns", [])
            if not turns:
                turns = [(b.get("prompt", ""), b.get("caption", ""))]

            parts = []
            for i, (user_text, asst_text) in enumerate(turns):
                if i == 0:
                    parts.append(f"{image_tokens_str}\nUser: {user_text}\nAssistant: ")
                else:
                    parts.append(f"User: {user_text}\nAssistant: ")
                if i < len(turns) - 1:
                    parts[-1] = parts[-1] + asst_text + "\n"

            prompts.append("".join(parts))
            references.append(str(turns[-1][1]).strip())

        tok = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "sar_feats": sar_feats,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "references": references,
            "prompts": prompts,
        }

    return _collate


@torch.no_grad()
def generate_batch(
    model: SarQwenVLForCausalLM,
    sar_feats: torch.Tensor,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    max_new_tokens: int = 200,
) -> Dict[str, object]:
    main_device = model.main_device

    patch_tokens = sar_feats.to(main_device)
    amp_device = "cuda" if main_device.type == "cuda" else "cpu"
    with torch.amp.autocast(amp_device, enabled=False):
        sar_token_embeds = model.projector(patch_tokens.float())

    target_scale = model._emb_scale
    cur_scale = sar_token_embeds.abs().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    sar_token_embeds = sar_token_embeds / (cur_scale + 1e-6) * target_scale
    sar_token_embeds = torch.clamp(
        sar_token_embeds,
        min=-3.0 * target_scale,
        max=3.0 * target_scale,
    )
    sar_token_embeds = sar_token_embeds.to(model.lm_emb_dtype)

    lm_dev = model.lm_input_device
    prompt_ids = prompt_ids.to(lm_dev)
    prompt_mask = prompt_mask.to(lm_dev)
    sar_token_embeds = sar_token_embeds.to(lm_dev)

    inputs_embeds = model._inject_sar_embeds(prompt_ids, sar_token_embeds)

    output_ids = model.vl_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prompt_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )

    texts = [str(x).strip() for x in model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)]
    token_lengths = [int((ids != model.tokenizer.pad_token_id).sum()) for ids in output_ids]
    return {
        "texts": texts,
        "token_lengths": token_lengths,
    }


@torch.no_grad()
def evaluate_generation(
    model: SarQwenVLForCausalLM,
    data_loader: DataLoader,
    max_new_tokens: int = 200,
    max_samples: Optional[int] = 500,
    progress_every: int = 20,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict:
    model.eval()
    all_refs: List[str] = []
    all_hyps: List[str] = []
    prompt_examples: List[str] = []
    total_loader_wait = 0.0
    total_generate_time = 0.0
    total_output_tokens = 0
    loop_start = time.time()
    prev_batch_end = loop_start

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
            generated = generate_batch(
                model=model,
                sar_feats=batch["sar_feats"],
                prompt_ids=batch["input_ids"],
                prompt_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
            gen_elapsed = time.time() - gen_start
        except RuntimeError as e:
            _emit_progress(progress_callback, f"  [WARN] batch {i} generate error: {e}, skipping")
            prev_batch_end = time.time()
            continue

        batch_refs = list(batch["references"])
        batch_hyps = list(generated["texts"])
        batch_prompts = list(batch.get("prompts", []))
        batch_token_lengths = list(generated["token_lengths"])
        total_generate_time += gen_elapsed
        total_output_tokens += sum(batch_token_lengths)

        if max_samples is not None:
            remaining = max_samples - len(all_hyps)
            if remaining <= 0:
                break
            batch_refs = batch_refs[:remaining]
            batch_hyps = batch_hyps[:remaining]
            batch_prompts = batch_prompts[:remaining]
            batch_token_lengths = batch_token_lengths[:remaining]

        if not prompt_examples and batch_prompts:
            prompt_examples = [batch_prompts[0]]

        all_refs.extend(batch_refs)
        all_hyps.extend(batch_hyps)

        if progress_every > 0 and ((i + 1) % progress_every == 0):
            elapsed = time.time() - loop_start
            samples_done = len(all_hyps)
            samples_per_sec = samples_done / max(elapsed, 1e-6)
            tokens_per_sec = total_output_tokens / max(total_generate_time, 1e-6)
            avg_tokens = total_output_tokens / max(samples_done, 1)
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
                f"avg_out_tokens={avg_tokens:.1f}"
            )

        if max_samples is not None and len(all_hyps) >= max_samples:
            break

        prev_batch_end = time.time()

    model.train()

    if not all_refs:
        return {
            "bleu4": 0.0,
            "rougeL": 0.0,
            "exact_match": 0.0,
            "num_samples": 0,
            "examples": [],
            "prompt_examples": [],
            "elapsed_sec": 0.0,
            "loader_wait_sec": total_loader_wait,
            "generate_sec": total_generate_time,
            "samples_per_sec": 0.0,
            "tokens_per_sec": 0.0,
            "avg_output_tokens": 0.0,
        }

    elapsed_total = time.time() - loop_start
    return {
        "bleu4": compute_bleu4(all_refs, all_hyps),
        "rougeL": compute_rougeL(all_refs, all_hyps),
        "exact_match": compute_exact_match(all_refs, all_hyps),
        "num_samples": len(all_refs),
        "examples": list(zip(all_refs[:3], all_hyps[:3])),
        "prompt_examples": prompt_examples,
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
    max_new_tokens: int = 200,
    max_samples: Optional[int] = 500,
    progress_every: int = 20,
    progress_callback: Optional[Callable[[str], None]] = None,
    lora_request=None,
) -> Dict:
    projector.eval()
    projector_device = next(projector.parameters()).device

    all_refs: List[str] = []
    all_hyps: List[str] = []
    prompt_examples: List[str] = []
    total_loader_wait = 0.0
    total_generate_time = 0.0
    total_output_tokens = 0
    loop_start = time.time()
    prev_batch_end = loop_start

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
                prompt_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                sar_feats=batch["sar_feats"],
                projector=projector,
                projector_device=projector_device,
            )
            gen_start = time.time()
            generated = generator.generate(
                prompt_embeds=prompt_embeds,
                max_new_tokens=max_new_tokens,
                lora_request=lora_request,
            )
            gen_elapsed = time.time() - gen_start
        except RuntimeError as e:
            _emit_progress(progress_callback, f"  [WARN] batch {i} vLLM generate error: {e}, skipping")
            prev_batch_end = time.time()
            continue

        batch_refs = list(batch["references"])
        batch_hyps = list(generated["texts"])
        batch_prompts = list(batch.get("prompts", []))
        batch_token_lengths = list(generated["token_lengths"])
        total_generate_time += gen_elapsed
        total_output_tokens += sum(batch_token_lengths)

        if max_samples is not None:
            remaining = max_samples - len(all_hyps)
            if remaining <= 0:
                break
            batch_refs = batch_refs[:remaining]
            batch_hyps = batch_hyps[:remaining]
            batch_prompts = batch_prompts[:remaining]
            batch_token_lengths = batch_token_lengths[:remaining]

        if not prompt_examples and batch_prompts:
            prompt_examples = [batch_prompts[0]]

        all_refs.extend(batch_refs)
        all_hyps.extend(batch_hyps)

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
            "bleu4": 0.0,
            "rougeL": 0.0,
            "exact_match": 0.0,
            "num_samples": 0,
            "examples": [],
            "prompt_examples": [],
            "elapsed_sec": 0.0,
            "loader_wait_sec": total_loader_wait,
            "generate_sec": total_generate_time,
            "samples_per_sec": 0.0,
            "tokens_per_sec": 0.0,
            "avg_output_tokens": 0.0,
        }

    elapsed_total = time.time() - loop_start
    return {
        "bleu4": compute_bleu4(all_refs, all_hyps),
        "rougeL": compute_rougeL(all_refs, all_hyps),
        "exact_match": compute_exact_match(all_refs, all_hyps),
        "num_samples": len(all_refs),
        "examples": list(zip(all_refs[:3], all_hyps[:3])),
        "prompt_examples": prompt_examples,
        "elapsed_sec": elapsed_total,
        "loader_wait_sec": total_loader_wait,
        "generate_sec": total_generate_time,
        "samples_per_sec": len(all_refs) / max(elapsed_total, 1e-6),
        "tokens_per_sec": total_output_tokens / max(total_generate_time, 1e-6),
        "avg_output_tokens": total_output_tokens / max(len(all_refs), 1),
    }


def load_model(
    projector_path: str,
    device: Optional[str] = None,
    lora_adapter: Optional[str] = None,
) -> SarQwenVLForCausalLM:
    main_device = _resolve_main_device(device)
    qwen_device_map = _resolve_qwen_device_map(device)
    inference_dtype = _pick_inference_dtype()
    llm_hidden = infer_qwen3_vl_text_hidden_size(PATHS.qwen_path)
    print(f"[INFO] projector device = {main_device}")
    print(f"[INFO] qwen device_map  = {qwen_device_map}")
    print(f"[INFO] inference dtype = {inference_dtype}")
    print(f"[INFO] qwen3-vl text hidden size = {llm_hidden}")

    projector = TokenLinearProjector(in_dim=768, llm_hidden_size=llm_hidden).to(main_device)
    model = SarQwenVLForCausalLM(
        qwen_path=PATHS.qwen_path,
        projector=projector,
        device=main_device,
        torch_dtype=inference_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=qwen_device_map,
        gradient_checkpointing=False,
    )

    ckpt_path = Path(projector_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"projector 权重不存在: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "projector" in state:
        state = state["projector"]
    model.projector.load_state_dict(state, strict=True)
    print(f"[OK] Loaded projector from: {ckpt_path}")

    if lora_adapter:
        lora_dir = Path(lora_adapter)
        if not lora_dir.exists():
            raise FileNotFoundError(f"LoRA adapter 目录不存在: {lora_dir}")
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("需要安装 peft: pip install peft") from e
        model.llm_causal = PeftModel.from_pretrained(model.llm_causal, str(lora_dir))
        print(f"[OK] Loaded LoRA adapter from: {lora_dir}")

    model.eval()
    sys.stdout.flush()
    return model


def load_projector_only(projector_path: str, device: Optional[str] = None) -> TokenLinearProjector:
    llm_hidden = infer_qwen3_vl_text_hidden_size(PATHS.qwen_path)
    main_device = _resolve_main_device(device)
    projector = TokenLinearProjector(in_dim=768, llm_hidden_size=llm_hidden).to(main_device)

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


ALL_DATASETS = {
    "sarvqa": lambda: PtVqaDataset(root=PATHS.sarvqa_root, pt_json=PATHS.sarvqa_pt_test_json),
    "sartext": lambda: PtVqaDataset(root=PATHS.sartext_root, pt_json=PATHS.sartext_pt_test_json),
    "sarlang_vqa": lambda: PtVqaDataset(root=PATHS.sarlang_vqa_root, pt_json=PATHS.sarlang_vqa_pt_test_json),
}


def build_test_datasets(wanted: Optional[List[str]] = None) -> Dict[str, PtVqaDataset]:
    datasets: Dict[str, PtVqaDataset] = {}
    if wanted:
        for name in wanted:
            if name not in ALL_DATASETS:
                raise ValueError(f"未知数据集: {name!r}，可选: {list(ALL_DATASETS)}")
            datasets[name] = ALL_DATASETS[name]()
        return datasets

    weight_map = {
        "sarvqa": TRAIN.mixed_weight_sarvqa,
        "sartext": TRAIN.mixed_weight_sartext,
        "sarlang_vqa": TRAIN.mixed_weight_sarlang_vqa,
    }
    for name, weight in weight_map.items():
        if weight > 0:
            datasets[name] = ALL_DATASETS[name]()

    if not datasets:
        raise ValueError("没有任何测试数据集，请检查 --datasets 或 config_local.py 中 mixed_weight_* 的设置")
    return datasets


def _log_generation_metrics(log: Callable[[str], None], gen_metrics: Dict, elapsed_sec: float) -> None:
    log(f"  BLEU-4      = {gen_metrics['bleu4']:.4f}")
    log(f"  ROUGE-L     = {gen_metrics['rougeL']:.4f}")
    log(f"  Exact Match = {gen_metrics['exact_match']:.4f}")
    log(f"  评测样本数  = {gen_metrics['num_samples']}  ({elapsed_sec:.0f}s)")
    log(
        "  生成性能    = "
        f"samples/s {gen_metrics['samples_per_sec']:.2f} | "
        f"tokens/s {gen_metrics['tokens_per_sec']:.2f} | "
        f"avg_out_tokens {gen_metrics['avg_output_tokens']:.1f} | "
        f"gen {_format_seconds(gen_metrics['generate_sec'])} | "
        f"loader_wait {_format_seconds(gen_metrics['loader_wait_sec'])}"
    )
    if gen_metrics.get("prompt_examples"):
        log("\n  首条真实 prompt:")
        log(gen_metrics["prompt_examples"][0])

    log("\n  样本示例（前 3 条）:")
    for j, (ref, hyp) in enumerate(gen_metrics.get("examples", [])):
        log(f"  [{j+1}] Ref: {ref[:120]}")
        log(f"       Hyp: {hyp[:120]}")
        log()


def parse_args():
    parser = argparse.ArgumentParser(description="test_stage2.py：SAR VQA 评测")
    parser.add_argument("--projector", type=str, required=True,
                        help="stage2 projector 权重路径（.pt 文件或完整 checkpoint）")
    parser.add_argument("--lora_adapter", type=str, default=None,
                        help="LoRA adapter 目录（save_pretrained 保存的目录），不传则不加载 LoRA")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备；不传则沿用 stage2 config（支持多卡 loss）")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="推理 batch size")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="生成时最大新 token 数")
    parser.add_argument("--max_gen_samples", type=int, default=500,
                        help="BLEU/ROUGE/EM 评测的最大样本数；-1 表示全量")
    parser.add_argument("--loss_only", action="store_true",
                        help="只计算 Test Loss，跳过生成")
    parser.add_argument("--gen_only", action="store_true",
                        help="只计算生成指标，跳过 Test Loss")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集名，e.g. sarvqa,sartext")
    parser.add_argument("--num_workers", type=int, default=2,
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
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16",
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

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"test_stage2_{ts}.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    log(f"[INFO] log        → {log_path}")
    log(f"[INFO] projector  → {args.projector}")
    log(f"[INFO] lora       → {args.lora_adapter or '(none)'}")
    log(f"[INFO] device     → {args.device or '(config defaults)'}")
    log(f"[INFO] datasets   → {args.datasets or '(config defaults)'}")
    log(f"[INFO] generation backend = {gen_backend}")
    log(f"[INFO] max_gen_samples = {args.max_gen_samples}  max_new_tokens = {args.max_new_tokens}")
    log(f"[INFO] progress_every = {args.progress_every}")
    if gen_backend == "vllm":
        log("[INFO] vLLM 选卡建议通过 CUDA_VISIBLE_DEVICES 控制；--device 主要约束 projector/HF 路径")

    test_datasets = build_test_datasets(wanted)
    log(f"[INFO] datasets to evaluate: {list(test_datasets)}")
    all_results: Dict[str, Dict] = {name: {} for name in test_datasets}

    model: Optional[SarQwenVLForCausalLM] = None
    try:
        if not args.gen_only:
            log(f"\n{'='*60}")
            log("  阶段 1/2: Test Loss")
            log(f"{'='*60}")

            model = load_model(
                projector_path=args.projector,
                device=args.device,
                lora_adapter=args.lora_adapter,
            )
            collate_loss = build_vqa_collate_fn(
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
                    model = load_model(
                        projector_path=args.projector,
                        device=args.device,
                        lora_adapter=args.lora_adapter,
                    )
                collate_gen = _build_stage2_inference_collate(
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
                prompt_builder = PromptEmbeddingBuilder(
                    qwen_path=PATHS.qwen_path,
                    image_token="<sar>",
                    torch_dtype=_pick_inference_dtype(),
                )
                projector = load_projector_only(args.projector, device=args.device)
                generator = VLLMTextGenerator(
                    model_path=PATHS.qwen_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                    max_model_len=args.vllm_max_model_len,
                    dtype=args.vllm_dtype,
                    enforce_eager=args.vllm_enforce_eager,
                    enable_lora=bool(args.lora_adapter),
                )
                lora_request = build_lora_request(args.lora_adapter) if args.lora_adapter else None
                try:
                    collate_gen = _build_stage2_inference_collate(
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
                            lora_request=lora_request,
                        )
                        all_results[ds_name].update(gen_metrics)
                        _log_generation_metrics(log, gen_metrics, time.time() - t1)
                finally:
                    release_cuda_resources(prompt_builder, projector, generator)
    finally:
        if model is not None:
            release_cuda_resources(model)

    log(f"\n{'='*60}")
    log("  汇总结果")
    log(f"{'='*60}")
    header = f"{'数据集':<14}  {'Test Loss':>10}  {'BLEU-4':>8}  {'ROUGE-L':>8}  {'EM':>8}  {'样本数':>8}"
    log(header)
    log("-" * len(header))
    for ds_name, r in all_results.items():
        loss_str = f"{r['test_loss']:.4f}" if r.get("test_loss") is not None else "  N/A  "
        bleu_str = f"{r['bleu4']:.4f}" if r.get("bleu4") is not None else "  N/A  "
        rouge_str = f"{r['rougeL']:.4f}" if r.get("rougeL") is not None else "  N/A  "
        em_str = f"{r['exact_match']:.4f}" if r.get("exact_match") is not None else "  N/A  "
        n_str = str(r.get("num_samples", "-"))
        log(f"{ds_name:<14}  {loss_str:>10}  {bleu_str:>8}  {rouge_str:>8}  {em_str:>8}  {n_str:>8}")

    log(f"\n[INFO] 完整日志已保存至: {log_path}")
    log_file.close()


if __name__ == "__main__":
    main()
