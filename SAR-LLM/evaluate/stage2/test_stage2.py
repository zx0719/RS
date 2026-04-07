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
  python test_stage2.py --projector /path/to/projector_stage2.pt \\
                        --lora_adapter /path/to/lora_adapter

  # 全量测试集
  python test_stage2.py --projector /path/to/projector_stage2.pt --max_gen_samples -1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

STAGE2_TRAIN_DIR = Path(__file__).resolve().parent.parent.parent / "train" / "stage2"
STAGE1_DIR       = Path(__file__).resolve().parent.parent.parent / "train" / "stage1"
for d in [str(STAGE2_TRAIN_DIR), str(STAGE1_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config_local import PATHS, TRAIN                          # noqa: E402
from sarclip_module import TokenLinearProjector                # noqa: E402
from qwen3_sar_model import SarQwenVLForCausalLM, infer_qwen3_vl_text_hidden_size  # noqa: E402
from vqa_dataset_pt import (                                   # noqa: E402
    PtVqaDataset,
    build_vqa_collate_fn,
    build_vqa_inference_collate,
)
from train_stage2 import forward_vqa                           # noqa: E402


# ──────────────────────────────────────────────────────────
# 1. 评测指标
# ──────────────────────────────────────────────────────────

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
    """完全匹配率（忽略大小写和首尾空格），适合短答案 VQA。"""
    matches = sum(
        ref.strip().lower() == hyp.strip().lower()
        for ref, hyp in zip(references, hypotheses)
    )
    return float(matches) / len(references) if references else 0.0


# ──────────────────────────────────────────────────────────
# 2. Test Loss
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test_loss(model: SarQwenVLForCausalLM, data_loader: DataLoader) -> Optional[float]:
    model.eval()
    total_loss, total_count = 0.0, 0

    for i, batch in enumerate(data_loader):
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
            print(f"[WARN] batch {i} forward error: {e}, skipping")
            continue

        if loss is not None:
            bs = batch["input_ids"].size(0)
            total_loss  += float(loss.item()) * bs
            total_count += bs

        if (i + 1) % 50 == 0:
            print(f"  [loss eval] processed {(i+1) * data_loader.batch_size} samples...")

    model.train()
    return total_loss / total_count if total_count > 0 else None


# ──────────────────────────────────────────────────────────
# 3. 推理生成
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def generate_batch(
    model:          SarQwenVLForCausalLM,
    sar_feats:      torch.Tensor,
    prompt_ids:     torch.Tensor,
    prompt_mask:    torch.Tensor,
    max_new_tokens: int = 200,
) -> List[str]:
    main_device = model.main_device

    patch_tokens = sar_feats.to(main_device)
    with torch.amp.autocast("cuda", enabled=False):
        sar_token_embeds = model.projector(patch_tokens.float())

    target_scale     = model._emb_scale
    cur_scale        = sar_token_embeds.abs().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    sar_token_embeds = sar_token_embeds / (cur_scale + 1e-6) * target_scale
    sar_token_embeds = torch.clamp(sar_token_embeds,
                                   min=-3.0 * target_scale, max=3.0 * target_scale)
    sar_token_embeds = sar_token_embeds.to(model.lm_emb_dtype)

    lm_dev       = model.lm_input_device
    prompt_ids   = prompt_ids.to(lm_dev)
    prompt_mask  = prompt_mask.to(lm_dev)
    sar_token_embeds = sar_token_embeds.to(lm_dev)

    inputs_embeds = model._inject_sar_embeds(prompt_ids, sar_token_embeds)

    output_ids = model.llm_causal.generate(
        inputs_embeds  = inputs_embeds,
        attention_mask = prompt_mask,
        max_new_tokens = max_new_tokens,
        do_sample      = False,
        eos_token_id   = model.tokenizer.eos_token_id,
        pad_token_id   = model.tokenizer.eos_token_id,
    )

    return model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


@torch.no_grad()
def evaluate_generation(
    model:          SarQwenVLForCausalLM,
    data_loader:    DataLoader,
    max_new_tokens: int = 200,
    max_samples:    Optional[int] = 500,
) -> Dict:
    model.eval()
    all_refs: List[str] = []
    all_hyps: List[str] = []

    for i, batch in enumerate(data_loader):
        if batch is None:
            continue
        try:
            generated = generate_batch(
                model          = model,
                sar_feats      = batch["sar_feats"],
                prompt_ids     = batch["input_ids"],
                prompt_mask    = batch["attention_mask"],
                max_new_tokens = max_new_tokens,
            )
        except RuntimeError as e:
            print(f"[WARN] batch {i} generate error: {e}, skipping")
            continue

        all_refs.extend(batch["references"])
        all_hyps.extend(generated)

        if (i + 1) % 20 == 0:
            print(f"  [gen eval] generated {len(all_hyps)} samples...")

        if max_samples and len(all_hyps) >= max_samples:
            break

    model.train()

    if not all_refs:
        return {"bleu4": 0.0, "rougeL": 0.0, "exact_match": 0.0, "num_samples": 0}

    return {
        "bleu4":       compute_bleu4(all_refs, all_hyps),
        "rougeL":      compute_rougeL(all_refs, all_hyps),
        "exact_match": compute_exact_match(all_refs, all_hyps),
        "num_samples": len(all_refs),
        "examples":    list(zip(all_refs[:3], all_hyps[:3])),
    }


# ──────────────────────────────────────────────────────────
# 4. main
# ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="test_stage2.py：SAR VQA 评测")
    parser.add_argument("--projector", type=str, required=True,
                        help="stage2 projector 权重路径（.pt 文件或完整 checkpoint）")
    parser.add_argument("--lora_adapter", type=str, default=None,
                        help="LoRA adapter 目录（save_pretrained 保存的目录），不传则不加载 LoRA")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--max_gen_samples", type=int, default=500,
                        help="BLEU/ROUGE/EM 评测的最大样本数；-1 表示全量")
    parser.add_argument("--loss_only", action="store_true",
                        help="只计算 Test Loss，跳过生成")
    return parser.parse_args()


def main():
    args = parse_args()
    max_gen = None if args.max_gen_samples == -1 else args.max_gen_samples

    # ── 加载模型 ──────────────────────────────────────────
    main_device = TRAIN.main_device
    llm_hidden  = infer_qwen3_vl_text_hidden_size(PATHS.qwen_path)
    print(f"[INFO] qwen3-vl text hidden size = {llm_hidden}")

    projector = TokenLinearProjector(in_dim=768, llm_hidden_size=llm_hidden).to(main_device)

    qwen_device_map = TRAIN.qwen_device_map if TRAIN.use_multi_gpu else None
    model = SarQwenVLForCausalLM(
        qwen_path              = PATHS.qwen_path,
        projector              = projector,
        device                 = main_device,
        torch_dtype            = torch.float16 if TRAIN.fp16 else torch.float32,
        trust_remote_code      = True,
        low_cpu_mem_usage      = True,
        device_map             = qwen_device_map,
        gradient_checkpointing = False,
    )

    # 加载 projector 权重
    ckpt_path = Path(args.projector)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"projector 权重不存在: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "projector" in state:
        state = state["projector"]
    model.projector.load_state_dict(state, strict=True)
    print(f"[OK] Loaded projector from: {ckpt_path}")

    # 加载 LoRA adapter（可选）
    if args.lora_adapter:
        lora_dir = Path(args.lora_adapter)
        if not lora_dir.exists():
            raise FileNotFoundError(f"LoRA adapter 目录不存在: {lora_dir}")
        try:
            from peft import PeftModel
            model.llm_causal = PeftModel.from_pretrained(model.llm_causal, str(lora_dir))
            print(f"[OK] Loaded LoRA adapter from: {lora_dir}")
        except ImportError:
            raise ImportError("需要安装 peft: pip install peft")

    model.eval()

    # ── 构建测试数据集 ────────────────────────────────────
    test_datasets: Dict[str, PtVqaDataset] = {}

    if TRAIN.mixed_weight_sarvqa > 0:
        test_datasets["sarvqa"] = PtVqaDataset(
            root    = PATHS.sarvqa_root,
            pt_json = PATHS.sarvqa_pt_test_json,
        )
    if TRAIN.mixed_weight_sartext > 0:
        test_datasets["sartext"] = PtVqaDataset(
            root    = PATHS.sartext_root,
            pt_json = PATHS.sartext_pt_test_json,
        )
    if TRAIN.mixed_weight_sarlang_vqa > 0:
        test_datasets["sarlang_vqa"] = PtVqaDataset(
            root    = PATHS.sarlang_vqa_root,
            pt_json = PATHS.sarlang_vqa_pt_test_json,
        )

    if not test_datasets:
        raise ValueError("没有任何测试数据集，请检查 config_local.py 中 mixed_weight_* 的设置")

    collate_loss = build_vqa_collate_fn(
        tokenizer        = model.tokenizer,
        image_token      = model.image_token,
        num_image_tokens = TRAIN.num_image_tokens,
        max_length       = TRAIN.max_length,
    )
    collate_gen = build_vqa_inference_collate(
        tokenizer        = model.tokenizer,
        image_token      = model.image_token,
        num_image_tokens = TRAIN.num_image_tokens,
        max_length       = TRAIN.max_length,
    )

    # ── 逐数据集评测 ──────────────────────────────────────
    all_results: Dict[str, Dict] = {}

    for ds_name, ds in test_datasets.items():
        print(f"\n{'='*60}")
        print(f"  评测数据集: {ds_name}   test size = {len(ds)}")
        print(f"{'='*60}")

        result = {}

        # 1. Test Loss
        print("\n[1/3] 计算 Test Loss ...")
        loss_loader = DataLoader(
            ds,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = 2,
            collate_fn  = collate_loss,
            pin_memory  = torch.cuda.is_available(),
        )
        test_loss = evaluate_test_loss(model, loss_loader)
        result["test_loss"] = test_loss
        print(f"  Test Loss = {test_loss:.4f}" if test_loss is not None else "  Test Loss = N/A")

        if not args.loss_only:
            # 2. BLEU-4 + ROUGE-L + EM
            print(f"\n[2/3] 生成回答并计算 BLEU-4 / ROUGE-L / EM ...")
            print(f"      (max_gen_samples={max_gen}, max_new_tokens={args.max_new_tokens})")
            gen_loader = DataLoader(
                ds,
                batch_size  = args.batch_size,
                shuffle     = False,
                num_workers = 2,
                collate_fn  = collate_gen,
                pin_memory  = torch.cuda.is_available(),
            )
            gen_metrics = evaluate_generation(
                model          = model,
                data_loader    = gen_loader,
                max_new_tokens = args.max_new_tokens,
                max_samples    = max_gen,
            )
            result.update(gen_metrics)

            print(f"  BLEU-4      = {gen_metrics['bleu4']:.4f}")
            print(f"  ROUGE-L     = {gen_metrics['rougeL']:.4f}")
            print(f"  Exact Match = {gen_metrics['exact_match']:.4f}")
            print(f"  评测样本数  = {gen_metrics['num_samples']}")

            print(f"\n[3/3] 样本示例（前 3 条）:")
            for j, (ref, hyp) in enumerate(gen_metrics.get("examples", [])):
                print(f"  [{j+1}] Ref: {ref[:120]}")
                print(f"       Hyp: {hyp[:120]}")
                print()

        all_results[ds_name] = result

    # ── 汇总 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  汇总结果")
    print(f"{'='*60}")
    header = f"{'数据集':<14}  {'Test Loss':>10}  {'BLEU-4':>8}  {'ROUGE-L':>8}  {'EM':>8}  {'样本数':>8}"
    print(header)
    print("-" * len(header))
    for ds_name, r in all_results.items():
        loss_str  = f"{r['test_loss']:.4f}"    if r.get("test_loss")    is not None else "  N/A  "
        bleu_str  = f"{r['bleu4']:.4f}"        if r.get("bleu4")        is not None else "  N/A  "
        rouge_str = f"{r['rougeL']:.4f}"       if r.get("rougeL")       is not None else "  N/A  "
        em_str    = f"{r['exact_match']:.4f}"  if r.get("exact_match")  is not None else "  N/A  "
        n_str     = str(r.get("num_samples", "-"))
        print(f"{ds_name:<14}  {loss_str:>10}  {bleu_str:>8}  {rouge_str:>8}  {em_str:>8}  {n_str:>8}")


if __name__ == "__main__":
    main()
