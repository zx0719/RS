"""
test_pt.py
==========
对测试集进行评测，三项指标：
  1. Test Loss  ── 重用 forward_pt，衡量模型对 ground truth 文本的建模能力
  2. BLEU-4     ── n-gram 精度，caption 领域标准指标（需 pip install nltk）
  3. ROUGE-L    ── 最长公共子序列 F1，对召回率更敏感（需 pip install rouge_score）

依赖安装：
  pip install nltk rouge_score
  python -c "import nltk; nltk.download('punkt_tab')"

用法：
  # 只算 Test Loss（快，不需要生成文本）
  python test_pt.py --projector /path/to/projector.pt --loss_only

  # 完整评测（Loss + BLEU-4 + ROUGE-L，在 500 条样本上）
  python test_pt.py --projector /path/to/projector.pt

  # 完整评测，在全量测试集上（很慢，生成 16000 条文本）
  python test_pt.py --projector /path/to/projector.pt --max_gen_samples -1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path("/home/qianwentao/SARClip/SARCLIP+QwenVL-pt")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config_local import PATHS, TRAIN
from sarclip_module import TokenLinearProjector
from qwen3_sar_model import SarQwenVLForCausalLM, infer_qwen3_vl_text_hidden_size
from mixed_dataset_pt import PtCaptionDataset, build_pt_collate_fn
from train_pt import forward_pt  # 复用 forward_pt，不重复造轮子


# ──────────────────────────────────────────────────────────
# 1. 评测指标
# ──────────────────────────────────────────────────────────

def compute_bleu4(references: List[str], hypotheses: List[str]) -> float:
    """
    BLEU-4：衡量 4-gram 精确率。
    - 值域 0~1，越高越好；0.3+ 算 caption 任务里表现不错
    - 使用 SmoothingFunction.method1 避免短句时出现 0 值
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    refs = [[ref.split()] for ref in references]   # corpus_bleu 要求 list[list[list]]
    hyps = [hyp.split() for hyp in hypotheses]
    sf = SmoothingFunction().method1
    return float(corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf))


def compute_rougeL(references: List[str], hypotheses: List[str]) -> float:
    """
    ROUGE-L：基于最长公共子序列的 F1。
    - 比 BLEU 对词序变化更宽容
    - 值域 0~1，越高越好
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    return float(sum(scores) / len(scores)) if scores else 0.0


# ──────────────────────────────────────────────────────────
# 2. Test Loss（直接复用 forward_pt）
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test_loss(model: SarQwenVLForCausalLM, data_loader: DataLoader) -> Optional[float]:
    """
    在测试集上计算平均交叉熵 loss。
    复用 train_pt.forward_pt，label 掩码逻辑与训练完全一致。
    """
    model.eval()
    total_loss, total_count = 0.0, 0

    for i, batch in enumerate(data_loader):
        if batch is None:
            continue
        try:
            out = forward_pt(
                model          = model,
                sar_feats      = batch["sar_feats"],
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                labels         = batch["labels"],
            )
        except RuntimeError as e:
            print(f"[WARN] batch {i} forward error: {e}, skipping")
            continue

        if out.loss is not None:
            bs = batch["input_ids"].size(0)
            total_loss  += float(out.loss.item()) * bs
            total_count += bs

        if (i + 1) % 50 == 0:
            print(f"  [loss eval] processed {(i+1) * data_loader.batch_size} samples...")

    model.train()
    return total_loss / total_count if total_count > 0 else None


# ──────────────────────────────────────────────────────────
# 3. 推理生成（用于 BLEU/ROUGE）
# ──────────────────────────────────────────────────────────

def _build_inference_collate(tokenizer, image_token: str, num_image_tokens: int, max_length: int):
    """
    推理专用 collate_fn：只 tokenize prompt（不含 caption），
    用于 generate() 的输入。同时返回 ground truth caption 用于指标计算。
    """
    image_tokens_str = " ".join([image_token] * num_image_tokens)

    def _collate(batch):
        good = [b for b in batch if not b.get("_bad_sample", False)]
        if not good:
            return None

        prompts = [
            f"{image_tokens_str}\nUser: {str(b['prompt']).strip()}\nAssistant: "
            for b in good
        ]
        captions   = [str(b["caption"]).strip() for b in good]
        sar_feats  = torch.stack([b["sar_feat"] for b in good], dim=0)

        tok = tokenizer(
            prompts,
            padding        = True,
            truncation     = True,
            max_length     = max_length,
            return_tensors = "pt",
        )
        return {
            "sar_feats":      sar_feats,
            "input_ids":      tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "captions":       captions,   # ground truth，用于计算 BLEU/ROUGE
        }

    return _collate


@torch.no_grad()
def generate_batch(
    model:          SarQwenVLForCausalLM,
    sar_feats:      torch.Tensor,   # (B, 195, 768)
    prompt_ids:     torch.Tensor,   # (B, L_prompt)
    prompt_mask:    torch.Tensor,   # (B, L_prompt)
    max_new_tokens: int = 200,
) -> List[str]:
    """
    对一个 batch 的 prompt 做贪心解码，返回生成的文本列表。

    注意：传入 inputs_embeds 时，generate() 返回的 output_ids 是新生成的 token（不含 prompt），
    可直接 batch_decode。
    """
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
        inputs_embeds   = inputs_embeds,
        attention_mask  = prompt_mask,
        max_new_tokens  = max_new_tokens,
        do_sample       = False,                      # 贪心解码，结果确定性更好
        eos_token_id    = model.tokenizer.eos_token_id,
        pad_token_id    = model.tokenizer.eos_token_id,
    )

    texts = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return texts


@torch.no_grad()
def evaluate_generation(
    model:          SarQwenVLForCausalLM,
    data_loader:    DataLoader,
    max_new_tokens: int = 200,
    max_samples:    Optional[int] = 500,
) -> Dict:
    """
    生成 caption 并计算 BLEU-4 / ROUGE-L。
    max_samples=None 表示跑全量（很慢，全量 16000 条约需数小时）。
    推荐先用 max_samples=500 快速验证。
    """
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

        all_refs.extend(batch["captions"])
        all_hyps.extend(generated)

        if (i + 1) % 20 == 0:
            print(f"  [gen eval] generated {len(all_hyps)} samples...")

        if max_samples and len(all_hyps) >= max_samples:
            break

    model.train()

    if not all_refs:
        return {"bleu4": 0.0, "rougeL": 0.0, "num_samples": 0}

    return {
        "bleu4":       compute_bleu4(all_refs, all_hyps),
        "rougeL":      compute_rougeL(all_refs, all_hyps),
        "num_samples": len(all_refs),
        "examples":    list(zip(all_refs[:3], all_hyps[:3])),  # 前 3 条对比，便于肉眼检查
    }


# ──────────────────────────────────────────────────────────
# 4. main
# ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="test_pt.py：SAR caption 测试评测")
    parser.add_argument("--projector", type=str, required=True,
                        help="训练好的 projector 权重路径，支持 .pt（纯权重）或完整 checkpoint")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="推理 batch size，比训练小一些减少显存压力")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="生成时最大新 token 数")
    parser.add_argument("--max_gen_samples", type=int, default=500,
                        help="BLEU/ROUGE 评测的最大样本数；-1 表示全量（很慢）")
    parser.add_argument("--loss_only", action="store_true",
                        help="只计算 Test Loss，跳过生成（速度快 10x 以上）")
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

    # 加载 projector 权重（支持纯 state_dict 或完整 checkpoint）
    ckpt_path = Path(args.projector)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"projector 权重不存在: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "projector" in state:
        state = state["projector"]   # 完整 checkpoint
    model.projector.load_state_dict(state, strict=True)
    print(f"[OK] Loaded projector from: {ckpt_path}")

    model.eval()

    # ── 构建测试数据集 ────────────────────────────────────
    # 哪个数据集在训练时权重 > 0，就在测试时评测哪个
    test_datasets: Dict[str, PtCaptionDataset] = {}

    if TRAIN.mixed_weight_sarlang > 0:
        test_datasets["sarlang"] = PtCaptionDataset(
            root     = PATHS.sarlang_root,
            pt_json  = PATHS.sarlang_pt_test_json,
        )
    if TRAIN.mixed_weight_sartext > 0:
        test_datasets["sartext"] = PtCaptionDataset(
            root     = PATHS.sartext_root,
            pt_json  = PATHS.sartext_pt_test_json,
        )
    if TRAIN.mixed_weight_sarcap > 0:
        test_datasets["sarcap"] = PtCaptionDataset(
            root     = PATHS.sarcap_root,
            pt_json  = PATHS.sarcap_pt_test_json,
        )
    if TRAIN.mixed_weight_fsarcap > 0:
        test_datasets["fsarcap"] = PtCaptionDataset(
            root     = PATHS.fsarcap_root,
            pt_json  = PATHS.fsarcap_pt_test_json,
        )

    if not test_datasets:
        raise ValueError("没有任何测试数据集，请检查 config_local.py 中 mixed_weight_* 的设置")

    # loss 评测的 collate（含 labels）
    collate_loss = build_pt_collate_fn(
        tokenizer        = model.tokenizer,
        image_token      = model.image_token,
        num_image_tokens = TRAIN.num_image_tokens,
        max_length       = TRAIN.max_length,
    )
    # 推理生成的 collate（仅 prompt）
    collate_gen = _build_inference_collate(
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
            batch_size   = args.batch_size,
            shuffle      = False,
            num_workers  = 2,
            collate_fn   = collate_loss,
            pin_memory   = torch.cuda.is_available(),
        )
        test_loss = evaluate_test_loss(model, loss_loader)
        result["test_loss"] = test_loss
        print(f"  Test Loss = {test_loss:.4f}" if test_loss is not None else "  Test Loss = N/A")

        if not args.loss_only:
            # 2. BLEU-4 + ROUGE-L
            print(f"\n[2/3] 生成 caption 并计算 BLEU-4 / ROUGE-L ...")
            print(f"      (max_gen_samples={max_gen}, max_new_tokens={args.max_new_tokens})")
            gen_loader = DataLoader(
                ds,
                batch_size   = args.batch_size,
                shuffle      = False,
                num_workers  = 2,
                collate_fn   = collate_gen,
                pin_memory   = torch.cuda.is_available(),
            )
            gen_metrics = evaluate_generation(
                model          = model,
                data_loader    = gen_loader,
                max_new_tokens = args.max_new_tokens,
                max_samples    = max_gen,
            )
            result.update(gen_metrics)

            print(f"  BLEU-4     = {gen_metrics['bleu4']:.4f}")
            print(f"  ROUGE-L    = {gen_metrics['rougeL']:.4f}")
            print(f"  评测样本数 = {gen_metrics['num_samples']}")

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
    header = f"{'数据集':<12}  {'Test Loss':>10}  {'BLEU-4':>8}  {'ROUGE-L':>8}  {'样本数':>8}"
    print(header)
    print("-" * len(header))
    for ds_name, r in all_results.items():
        loss_str  = f"{r['test_loss']:.4f}" if r.get("test_loss") is not None else "  N/A  "
        bleu_str  = f"{r['bleu4']:.4f}"     if r.get("bleu4")     is not None else "  N/A  "
        rouge_str = f"{r['rougeL']:.4f}"    if r.get("rougeL")    is not None else "  N/A  "
        n_str     = str(r.get("num_samples", "-"))
        print(f"{ds_name:<12}  {loss_str:>10}  {bleu_str:>8}  {rouge_str:>8}  {n_str:>8}")


if __name__ == "__main__":
    main()
