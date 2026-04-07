"""
test_pt.py
==========
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
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(os.environ.get("SARCLIP_PROJECT_DIR", "/home/qianwentao/SARClip/SARCLIP+QwenVL-pt"))
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config_local import PATHS, TRAIN                                              # noqa: E402
from sarclip_module import TokenLinearProjector                                    # noqa: E402
from qwen3_sar_model import SarQwenVLForCausalLM, infer_qwen3_vl_text_hidden_size  # noqa: E402
from mixed_dataset_pt import PtCaptionDataset, build_pt_collate_fn                # noqa: E402
from train_pt import forward_pt                                                    # noqa: E402


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
    """推理专用 collate：只 tokenize prompt，同时返回 ground truth caption。"""
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

        tok = tokenizer(
            prompts,
            padding       = True,
            truncation    = True,
            max_length    = max_length,
            return_tensors= "pt",
        )
        return {
            "sar_feats":      sar_feats,
            "input_ids":      tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "captions":       captions,
            "prompts":        prompts,
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
    手写 greedy decode：
      - llm_backbone（Qwen3VLTextModel）+ lm_head 逐步生成
      - 与 train_pt.forward_pt 完全一致的 embed 路径
      - 避免 vl_model.generate 对 inputs_embeds 的兼容问题
    """
    main_device = model.main_device

    with torch.amp.autocast("cuda", enabled=False):
        sar_token_embeds = model.projector(sar_feats.to(main_device).float())

    target_scale     = float(model._emb_scale)
    cur_scale        = sar_token_embeds.abs().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    sar_token_embeds = sar_token_embeds / (cur_scale + 1e-6) * target_scale
    sar_token_embeds = torch.clamp(sar_token_embeds,
                                   min=-3.0 * target_scale, max=3.0 * target_scale)
    sar_token_embeds = sar_token_embeds.to(model.lm_emb_dtype)

    lm_dev = model.lm_input_device
    prompt_ids       = prompt_ids.to(lm_dev)
    prompt_mask      = prompt_mask.to(lm_dev)
    sar_token_embeds = sar_token_embeds.to(lm_dev)

    # 第一步：用完整 prompt 的 inputs_embeds 跑 prefill，拿到 kv-cache
    inputs_embeds = model._inject_sar_embeds(prompt_ids, sar_token_embeds)
    position_ids  = (prompt_mask.long().cumsum(-1) - 1).clamp(min=0)

    out = model.llm_backbone(
        inputs_embeds  = inputs_embeds,
        attention_mask = prompt_mask,
        position_ids   = position_ids,
        use_cache      = True,
        return_dict    = True,
    )
    past_key_values = out.past_key_values
    next_logits     = model.lm_head(out.last_hidden_state[:, -1, :])  # (B, V)

    bsz        = prompt_ids.shape[0]
    eos_id     = model.tokenizer.eos_token_id
    cur_len    = prompt_ids.shape[1]
    generated  = [[] for _ in range(bsz)]          # 每个样本已生成的 token ids
    finished   = [False] * bsz
    cur_mask   = prompt_mask                        # (B, cur_len)

    for _ in range(max_new_tokens):
        next_token = next_logits.argmax(dim=-1)     # (B,)

        for i in range(bsz):
            if not finished[i]:
                tok_i = int(next_token[i].item())
                if tok_i == eos_id:
                    finished[i] = True
                else:
                    generated[i].append(tok_i)

        if all(finished):
            break

        # 下一步：只输入刚生成的 token embedding
        next_embeds = model.llm_backbone.get_input_embeddings()(
            next_token.unsqueeze(1)   # (B, 1)
        )                             # (B, 1, H)
        cur_len  += 1
        cur_mask  = torch.cat([cur_mask, torch.ones(bsz, 1, dtype=cur_mask.dtype, device=lm_dev)], dim=1)
        pos       = (cur_mask.long().cumsum(-1) - 1)[:, -1:]   # (B, 1)

        out = model.llm_backbone(
            inputs_embeds   = next_embeds,
            attention_mask  = cur_mask,
            position_ids    = pos,
            past_key_values = past_key_values,
            use_cache       = True,
            return_dict     = True,
        )
        past_key_values = out.past_key_values
        next_logits     = model.lm_head(out.last_hidden_state[:, -1, :])

    texts = [model.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]
    return texts


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
    prompt_examples: List[str] = []

    for i, batch in enumerate(data_loader):
        if batch is None:
            continue
        if max_samples is not None and len(all_hyps) >= max_samples:
            break
        try:
            generated = generate_batch(
                model          = model,
                sar_feats      = batch["sar_feats"],
                prompt_ids     = batch["input_ids"],
                prompt_mask    = batch["attention_mask"],
                max_new_tokens = max_new_tokens,
            )
        except RuntimeError as e:
            print(f"  [WARN] batch {i} generate error: {e}, skipping")
            continue

        batch_refs = batch["captions"]
        batch_hyps = generated
        batch_prompts = batch.get("prompts", [])

        if max_samples is not None:
            remaining = max_samples - len(all_hyps)
            if remaining <= 0:
                break
            batch_refs = batch_refs[:remaining]
            batch_hyps = batch_hyps[:remaining]
            batch_prompts = batch_prompts[:remaining]

        if not prompt_examples and batch_prompts:
            prompt_examples = [batch_prompts[0]]

        all_refs.extend(batch_refs)
        all_hyps.extend(batch_hyps)

        if (i + 1) % 20 == 0:
            print(f"  [gen] generated {len(all_hyps)} samples...")

        if max_samples is not None and len(all_hyps) >= max_samples:
            break

    if not all_refs:
        return {"bleu4": 0.0, "rougeL": 0.0, "num_samples": 0, "examples": [], "prompt_examples": []}

    return {
        "bleu4":       compute_bleu4(all_refs, all_hyps),
        "rougeL":      compute_rougeL(all_refs, all_hyps),
        "num_samples": len(all_refs),
        "examples":    list(zip(all_refs[:3], all_hyps[:3])),
        "prompt_examples": prompt_examples,
    }


# ──────────────────────────────────────────────────────────
# 4. 模型加载
# ──────────────────────────────────────────────────────────

def load_model(projector_path: str) -> SarQwenVLForCausalLM:
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

    ckpt_path = Path(projector_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"projector 权重不存在: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "projector" in state:
        state = state["projector"]
    model.projector.load_state_dict(state, strict=True)
    print(f"[OK] Loaded projector from: {ckpt_path}")

    model.eval()
    return model


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
    parser.add_argument("--batch_size", type=int, default=4,
                        help="推理 batch size")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="生成时最大新 token 数")
    parser.add_argument("--max_gen_samples", type=int, default=500,
                        help="每个数据集 BLEU/ROUGE 评测的最大样本数；-1 表示全量")
    parser.add_argument("--loss_only", action="store_true",
                        help="只计算 Test Loss，跳过生成")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集名，e.g. sarlang,sartext；默认跑 config 里启用的全部")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader num_workers")
    return parser.parse_args()


def main():
    args     = parse_args()
    max_gen  = None if args.max_gen_samples == -1 else args.max_gen_samples
    wanted   = [s.strip() for s in args.datasets.split(",")] if args.datasets else None

    # ── 日志文件 ──────────────────────────────────────────
    log_dir  = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"test_pt_{ts}.log"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    log(f"[INFO] log  → {log_path}")
    log(f"[INFO] projector → {args.projector}")
    log(f"[INFO] datasets  → {args.datasets or '(config defaults)'}")
    log(f"[INFO] loss_only = {args.loss_only}  max_gen_samples = {args.max_gen_samples}")

    # ── 加载模型 ──────────────────────────────────────────
    model = load_model(args.projector)

    # ── 构建测试数据集 ────────────────────────────────────
    test_datasets = build_test_datasets(wanted)
    log(f"[INFO] datasets to evaluate: {list(test_datasets)}")

    # collate
    collate_loss = build_pt_collate_fn(
        tokenizer        = model.tokenizer,
        image_token      = model.image_token,
        num_image_tokens = TRAIN.num_image_tokens,
        max_length       = TRAIN.max_length,
    )
    collate_gen = _build_inference_collate(
        tokenizer        = model.tokenizer,
        image_token      = model.image_token,
        num_image_tokens = TRAIN.num_image_tokens,
        max_length       = TRAIN.max_length,
    )

    # ── 逐数据集评测 ──────────────────────────────────────
    all_results: Dict[str, Dict] = {}

    for ds_name, ds in test_datasets.items():
        log(f"\n{'='*60}")
        log(f"  评测数据集: {ds_name}   test size = {len(ds)}")
        log(f"{'='*60}")

        result: Dict = {}
        t0 = time.time()

        # 1. Test Loss
        log("\n[1/3] 计算 Test Loss ...")
        loss_loader = DataLoader(
            ds,
            batch_size   = args.batch_size,
            shuffle      = False,
            num_workers  = args.num_workers,
            collate_fn   = collate_loss,
            pin_memory   = torch.cuda.is_available(),
        )
        test_loss = evaluate_test_loss(model, loss_loader)
        result["test_loss"] = test_loss
        loss_str = f"{test_loss:.4f}" if test_loss is not None else "N/A"
        log(f"  Test Loss = {loss_str}  ({time.time()-t0:.0f}s)")

        if not args.loss_only:
            # 2. BLEU-4 + ROUGE-L
            log(f"\n[2/3] 生成 caption 并计算 BLEU-4 / ROUGE-L ...")
            log(f"      (max_gen_samples={max_gen}, max_new_tokens={args.max_new_tokens})")
            t1 = time.time()
            gen_loader = DataLoader(
                ds,
                batch_size   = args.batch_size,
                shuffle      = False,
                num_workers  = args.num_workers,
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

            log(f"  BLEU-4     = {gen_metrics['bleu4']:.4f}")
            log(f"  ROUGE-L    = {gen_metrics['rougeL']:.4f}")
            log(f"  评测样本数 = {gen_metrics['num_samples']}  ({time.time()-t1:.0f}s)")
            if gen_metrics.get("prompt_examples"):
                log("\n  首条真实 prompt:")
                log(gen_metrics["prompt_examples"][0])

            log(f"\n[3/3] 样本示例（前 3 条）:")
            for j, (ref, hyp) in enumerate(gen_metrics.get("examples", [])):
                log(f"  [{j+1}] Ref: {ref}")
                log(f"       Hyp: {hyp}")
                log()
        else:
            log("\n[2/3] 跳过生成（--loss_only）")
            log("\n[3/3] 跳过样本示例（--loss_only）")

        all_results[ds_name] = result

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
        header = f"{'数据集':<12}  {'Test Loss':>10}  {'BLEU-4':>8}  {'ROUGE-L':>8}  {'样本数':>8}"
        log(header)
        log("-" * len(header))
        for ds_name, r in all_results.items():
            ls = f"{r['test_loss']:.4f}"  if r.get("test_loss")    is not None else "   N/A"
            bs = f"{r['bleu4']:.4f}"      if r.get("bleu4")        is not None else "   N/A"
            rs = f"{r['rougeL']:.4f}"     if r.get("rougeL")       is not None else "   N/A"
            ns = str(r.get("num_samples", "-"))
            log(f"{ds_name:<12}  {ls:>10}  {bs:>8}  {rs:>8}  {ns:>8}")

    log(f"\n[INFO] 完整日志已保存至: {log_path}")
    log_file.close()


if __name__ == "__main__":
    main()
