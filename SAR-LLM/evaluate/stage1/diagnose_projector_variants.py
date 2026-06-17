import argparse
import csv
import datetime as dt
import gc
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

import test_pt
from test_pt import (
    TRAIN,
    TokenLinearProjector,
    _build_inference_collate,
    _prepare_sar_token_embeds_for_lm,
    build_pt_collate_fn,
    build_test_datasets,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose projector variants on a fixed mini-batch.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="sarlang,sartext",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--samples_per_dataset",
        type=int,
        default=8,
        help="Number of good samples to keep per dataset.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=48,
        help="Max generated tokens per sample.",
    )
    parser.add_argument(
        "--first_tokens",
        type=int,
        default=20,
        help="How many generated tokens to expose in the report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260408,
        help="Seed used for random-projector initialization.",
    )
    parser.add_argument(
        "--checkpoint_030000",
        type=str,
        default="/mnt/data/qianwentao/checkpoints_sarqwen/checkpoint_step_030000.pt",
    )
    parser.add_argument(
        "--checkpoint_final",
        type=str,
        default="/mnt/data/qianwentao/checkpoints_sarqwen/sar_projector_stage1_sarcap.pt",
    )
    return parser.parse_args()


def log(msg: str = "") -> None:
    print(msg, flush=True)


def select_good_samples(dataset, ds_name: str, num_samples: int) -> List[Dict]:
    picked: List[Dict] = []
    dropped = 0
    for idx in range(len(dataset)):
        item = dict(dataset[idx])
        if item.get("_bad_sample", False):
            dropped += 1
            continue
        item["_source"] = ds_name
        item["_picked_index"] = idx
        picked.append(item)
        if len(picked) >= num_samples:
            break

    if len(picked) < num_samples:
        raise RuntimeError(
            f"{ds_name}: only found {len(picked)} good samples, need {num_samples}. dropped_bad={dropped}"
        )
    log(f"[INFO] {ds_name}: selected {len(picked)} good samples (dropped_bad_before_pick={dropped})")
    return picked


def build_fixed_batches(model, samples: List[Dict]) -> Tuple[Dict, Dict]:
    collate_loss = build_pt_collate_fn(
        tokenizer=model.tokenizer,
        image_token=model.image_token,
        num_image_tokens=TRAIN.num_image_tokens,
        max_length=TRAIN.max_length,
    )
    collate_gen = _build_inference_collate(
        tokenizer=model.tokenizer,
        image_token=model.image_token,
        num_image_tokens=TRAIN.num_image_tokens,
        max_length=TRAIN.max_length,
    )
    loss_batch = collate_loss(samples)
    gen_batch = collate_gen(samples)
    if loss_batch is None or gen_batch is None:
        raise RuntimeError("Fixed sample batch unexpectedly collapsed during collate.")
    return loss_batch, gen_batch


def slice_batch(batch: Dict, index: int) -> Dict:
    out: Dict = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value[index:index + 1]
        elif isinstance(value, list):
            out[key] = value[index:index + 1]
        else:
            out[key] = value
    return out


def generate_batch_with_ids(
    model,
    sar_feats: torch.Tensor,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    max_new_tokens: int,
) -> Dict[str, object]:
    lm_dev = model.lm_input_device
    prompt_ids = prompt_ids.to(lm_dev)
    prompt_mask = prompt_mask.to(lm_dev)
    sar_token_embeds = _prepare_sar_token_embeds_for_lm(model, sar_feats)

    inputs_embeds = model._inject_sar_embeds(prompt_ids, sar_token_embeds)
    position_ids = (prompt_mask.long().cumsum(-1) - 1).clamp(min=0)

    out = model.llm_backbone(
        inputs_embeds=inputs_embeds,
        attention_mask=prompt_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = out.past_key_values
    next_logits = model.lm_head(out.last_hidden_state[:, -1, :])

    bsz = prompt_ids.shape[0]
    eos_id = model.tokenizer.eos_token_id
    generated: List[List[int]] = [[] for _ in range(bsz)]
    finished = [False] * bsz
    cur_mask = prompt_mask

    for _ in range(max_new_tokens):
        next_token = next_logits.argmax(dim=-1)
        for i in range(bsz):
            if finished[i]:
                continue
            tok_i = int(next_token[i].item())
            if tok_i == eos_id:
                finished[i] = True
            else:
                generated[i].append(tok_i)

        if all(finished):
            break

        next_embeds = model.llm_backbone.get_input_embeddings()(next_token.unsqueeze(1))
        cur_mask = torch.cat(
            [cur_mask, torch.ones(bsz, 1, dtype=cur_mask.dtype, device=lm_dev)],
            dim=1,
        )
        pos = (cur_mask.long().cumsum(-1) - 1)[:, -1:]
        out = model.llm_backbone(
            inputs_embeds=next_embeds,
            attention_mask=cur_mask,
            position_ids=pos,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = out.past_key_values
        next_logits = model.lm_head(out.last_hidden_state[:, -1, :])

    texts = [model.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]
    token_texts = [model.tokenizer.convert_ids_to_tokens(ids) for ids in generated]
    return {
        "texts": texts,
        "token_ids": generated,
        "token_texts": token_texts,
    }


@torch.inference_mode()
def compute_per_sample_teacher_forcing(
    model,
    sar_feats: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, object]:
    lm_dev = model.lm_input_device
    input_ids = input_ids.to(lm_dev)
    attention_mask = attention_mask.to(lm_dev)
    labels = labels.to(lm_dev)
    sar_token_embeds = _prepare_sar_token_embeds_for_lm(model, sar_feats)
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
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid_mask = shift_labels.ne(-100)
    token_counts = valid_mask.sum(dim=1)
    per_sample_loss = (loss_flat * valid_mask).sum(dim=1) / token_counts.clamp(min=1)
    return {
        "per_sample_loss": per_sample_loss.detach().float().cpu().tolist(),
        "token_counts": token_counts.detach().cpu().tolist(),
    }


def with_random_projector(model, seed: int):
    original_state = {k: v.detach().cpu().clone() for k, v in model.projector.state_dict().items()}
    llm_hidden = model.llm_hidden_size
    torch.manual_seed(seed)
    random.seed(seed)
    fresh = TokenLinearProjector(in_dim=768, llm_hidden_size=llm_hidden)
    model.projector.load_state_dict(fresh.state_dict(), strict=True)
    model.projector.to(model.main_device)
    model.projector.eval()
    return original_state


def restore_projector(model, state_dict: Dict[str, torch.Tensor]) -> None:
    model.projector.load_state_dict(state_dict, strict=True)
    model.projector.to(model.main_device)
    model.projector.eval()


def summarize_text_patterns(texts: List[str]) -> Dict[str, int]:
    summary = {
        "arabic_mentions": 0,
        "black_image_mentions": 0,
        "zero_number_collapse": 0,
        "featureless_mentions": 0,
        "starts_with_numbered_list": 0,
    }
    for text in texts:
        s = text.strip().lower()
        if "arabic" in s:
            summary["arabic_mentions"] += 1
        if "black image" in s:
            summary["black_image_mentions"] += 1
        if "featureless" in s or "no discernible" in s:
            summary["featureless_mentions"] += 1
        if re.fullmatch(r"[0-9. ]{12,}", s):
            summary["zero_number_collapse"] += 1
        if re.match(r"^\d+\.", s):
            summary["starts_with_numbered_list"] += 1
    return summary


def extract_first_tokens(token_texts: List[List[str]]) -> Dict[str, object]:
    first = [tokens[0] for tokens in token_texts if tokens]
    counter = Counter(first)
    return {
        "top_first_tokens": counter.most_common(5),
        "empty_generations": sum(1 for tokens in token_texts if not tokens),
    }


def run_variant(
    model,
    dataset_name: str,
    variant_name: str,
    loss_batch: Dict,
    gen_batch: Dict,
    max_new_tokens: int,
    first_tokens: int,
    use_zero_input: bool = False,
) -> Tuple[Dict, List[Dict]]:
    t0 = time.time()
    rows: List[Dict] = []
    num_samples = len(gen_batch["captions"])
    for idx in range(num_samples):
        loss_one = slice_batch(loss_batch, idx)
        gen_one = slice_batch(gen_batch, idx)
        loss_sar_feats = loss_one["sar_feats"]
        gen_sar_feats = gen_one["sar_feats"]
        if use_zero_input:
            loss_sar_feats = torch.zeros_like(loss_sar_feats)
            gen_sar_feats = torch.zeros_like(gen_sar_feats)

        tf = compute_per_sample_teacher_forcing(
            model=model,
            sar_feats=loss_sar_feats,
            input_ids=loss_one["input_ids"],
            attention_mask=loss_one["attention_mask"],
            labels=loss_one["labels"],
        )
        gen = generate_batch_with_ids(
            model=model,
            sar_feats=gen_sar_feats,
            prompt_ids=gen_one["input_ids"],
            prompt_mask=gen_one["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
        sample_loss = float(tf["per_sample_loss"][0])
        token_count = int(tf["token_counts"][0])
        tok_ids = gen["token_ids"][0]
        tok_texts = gen["token_texts"][0]
        hyp = gen["texts"][0]
        ref = gen_one["captions"][0]
        prompt = gen_one["prompts"][0]
        image_id = loss_one["image_ids"][0]
        pt_path = loss_one["pt_paths"][0]
        rows.append(
            {
                "dataset": dataset_name,
                "variant": variant_name,
                "sample_rank": idx + 1,
                "image_id": image_id,
                "pt_path": pt_path,
                "teacher_forcing_loss": sample_loss,
                "target_token_count": token_count,
                "gen_token_count": len(tok_ids),
                "first_token_id": int(tok_ids[0]) if tok_ids else None,
                "first_token": tok_texts[0] if tok_texts else "",
                "first_tokens": tok_texts[:first_tokens],
                "prompt": prompt,
                "reference": ref,
                "hypothesis": hyp,
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0

    texts = [r["hypothesis"] for r in rows]
    token_texts = [r["first_tokens"] for r in rows]
    first_token_summary = extract_first_tokens(token_texts)
    summary = {
        "dataset": dataset_name,
        "variant": variant_name,
        "num_samples": len(rows),
        "avg_teacher_forcing_loss": sum(r["teacher_forcing_loss"] for r in rows) / max(len(rows), 1),
        "avg_target_token_count": sum(r["target_token_count"] for r in rows) / max(len(rows), 1),
        "avg_gen_token_count": sum(r["gen_token_count"] for r in rows) / max(len(rows), 1),
        "elapsed_sec": elapsed,
    }
    summary.update(first_token_summary)
    summary.update(summarize_text_patterns(texts))
    return summary, rows


def main() -> None:
    args = parse_args()
    datasets_wanted = [x.strip() for x in args.datasets.split(",") if x.strip()]
    out_dir = Path(__file__).parent / "logs"
    out_dir.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"diagnose_projector_variants_{ts}.json"
    csv_path = out_dir / f"diagnose_projector_variants_{ts}.csv"
    log_path = out_dir / f"diagnose_projector_variants_{ts}.log"

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def tee(msg: str = "") -> None:
        log(msg)
        log_file.write(msg + "\n")

    tee(f"[INFO] log_path            = {log_path}")
    tee(f"[INFO] json_path           = {json_path}")
    tee(f"[INFO] csv_path            = {csv_path}")
    tee(f"[INFO] datasets            = {datasets_wanted}")
    tee(f"[INFO] samples_per_dataset = {args.samples_per_dataset}")
    tee(f"[INFO] max_new_tokens      = {args.max_new_tokens}")
    tee(f"[INFO] first_tokens        = {args.first_tokens}")
    tee(f"[INFO] seed                = {args.seed}")
    tee(f"[INFO] checkpoint_030000   = {args.checkpoint_030000}")
    tee(f"[INFO] checkpoint_final    = {args.checkpoint_final}")

    test_datasets = build_test_datasets(datasets_wanted)
    fixed_samples: Dict[str, List[Dict]] = {}

    bootstrap_model = load_model(args.checkpoint_final)
    for ds_name in datasets_wanted:
        fixed_samples[ds_name] = select_good_samples(
            dataset=test_datasets[ds_name],
            ds_name=ds_name,
            num_samples=args.samples_per_dataset,
        )
    del bootstrap_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_rows: List[Dict] = []
    all_summaries: List[Dict] = []

    final_model = load_model(args.checkpoint_final)
    for ds_name in datasets_wanted:
        samples = fixed_samples[ds_name]
        loss_batch, gen_batch = build_fixed_batches(final_model, samples)

        tee(f"\n{'=' * 72}")
        tee(f"[DATASET] {ds_name} | variants on final/random baselines")
        tee(f"{'=' * 72}")

        summary, rows = run_variant(
            model=final_model,
            dataset_name=ds_name,
            variant_name="zero_input_final",
            loss_batch=loss_batch,
            gen_batch=gen_batch,
            max_new_tokens=args.max_new_tokens,
            first_tokens=args.first_tokens,
            use_zero_input=True,
        )
        all_summaries.append(summary)
        all_rows.extend(rows)
        tee(f"[SUMMARY] {summary}")

        random_state = with_random_projector(final_model, seed=args.seed)
        try:
            summary, rows = run_variant(
                model=final_model,
                dataset_name=ds_name,
                variant_name="random_projector",
                loss_batch=loss_batch,
                gen_batch=gen_batch,
                max_new_tokens=args.max_new_tokens,
                first_tokens=args.first_tokens,
            )
        finally:
            restore_projector(final_model, random_state)
        all_summaries.append(summary)
        all_rows.extend(rows)
        tee(f"[SUMMARY] {summary}")

        summary, rows = run_variant(
            model=final_model,
            dataset_name=ds_name,
            variant_name="final_real_sar",
            loss_batch=loss_batch,
            gen_batch=gen_batch,
            max_new_tokens=args.max_new_tokens,
            first_tokens=args.first_tokens,
        )
        all_summaries.append(summary)
        all_rows.extend(rows)
        tee(f"[SUMMARY] {summary}")

    del final_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    step30_model = load_model(args.checkpoint_030000)
    for ds_name in datasets_wanted:
        samples = fixed_samples[ds_name]
        loss_batch, gen_batch = build_fixed_batches(step30_model, samples)
        tee(f"\n{'=' * 72}")
        tee(f"[DATASET] {ds_name} | checkpoint_step_030000")
        tee(f"{'=' * 72}")
        summary, rows = run_variant(
            model=step30_model,
            dataset_name=ds_name,
            variant_name="checkpoint_step_030000",
            loss_batch=loss_batch,
            gen_batch=gen_batch,
            max_new_tokens=args.max_new_tokens,
            first_tokens=args.first_tokens,
        )
        all_summaries.append(summary)
        all_rows.extend(rows)
        tee(f"[SUMMARY] {summary}")

    del step30_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    by_dataset = defaultdict(list)
    for summary in all_summaries:
        by_dataset[summary["dataset"]].append(summary)

    for ds_name, rows in by_dataset.items():
        ranked = sorted(rows, key=lambda x: x["avg_teacher_forcing_loss"])
        tee(f"\n[LOSS RANK] dataset={ds_name}")
        for row in ranked:
            tee(
                f"  - {row['variant']}: avg_loss={row['avg_teacher_forcing_loss']:.4f} "
                f"| avg_gen_tokens={row['avg_gen_token_count']:.1f} "
                f"| first_tokens={row['top_first_tokens']}"
            )

    payload = {
        "created_at": ts,
        "args": vars(args),
        "summaries": all_summaries,
        "rows": all_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "dataset",
        "variant",
        "sample_rank",
        "image_id",
        "pt_path",
        "teacher_forcing_loss",
        "target_token_count",
        "gen_token_count",
        "first_token_id",
        "first_token",
        "first_tokens",
        "prompt",
        "reference",
        "hypothesis",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            row = dict(row)
            row["first_tokens"] = " | ".join(row["first_tokens"])
            writer.writerow(row)

    tee(f"\n[INFO] saved json = {json_path}")
    tee(f"[INFO] saved csv  = {csv_path}")
    tee(f"[INFO] saved log  = {log_path}")


if __name__ == "__main__":
    main()
