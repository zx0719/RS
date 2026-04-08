"""
inspect_first_token_logits.py
=============================

排查生成阶段第一步 logits 是否已经塌缩到某些固定 token。

默认强制使用本地仓库的 train/stage1 代码，而不是 /home/qianwentao/... 的外部目录。

示例：
  python inspect_first_token_logits.py \
    --projector /mnt/data/qianwentao/checkpoints_sarqwen/sar_projector_stage1_sarcap.pt \
    --dataset sartext \
    --sample-index 0 \
    --topk 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_DIR = SCRIPT_DIR.parent.parent / "train" / "stage1"
VALID_DATASETS = ("sarlang", "sartext", "sarcap", "fsarcap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect first-step logits for a SAR caption sample.")
    parser.add_argument("--projector", type=str, required=True, help="Projector 权重路径。")
    parser.add_argument("--dataset", type=str, default="sartext", choices=VALID_DATASETS, help="要检查的数据集。")
    parser.add_argument("--sample-index", type=int, default=0, help="数据集样本下标。")
    parser.add_argument("--topk", type=int, default=20, help="打印前 k 个 token。")
    parser.add_argument(
        "--project-dir",
        type=str,
        default=str(DEFAULT_PROJECT_DIR),
        help="本地训练代码目录，默认使用仓库内的 train/stage1。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="可选覆盖 tokenizer 的 max_length，默认使用 config 中 TRAIN.max_length。",
    )
    return parser.parse_args()


def configure_import_paths(project_dir: Path) -> None:
    os.environ["SARCLIP_PROJECT_DIR"] = str(project_dir)
    for path in (SCRIPT_DIR, project_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def format_token_text(text: str) -> str:
    return repr(text.replace("\n", "\\n").replace("\t", "\\t"))


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"project_dir 不存在: {project_dir}")

    configure_import_paths(project_dir)

    import torch
    import test_pt
    from config_local import TRAIN

    if args.topk <= 0:
        raise ValueError("--topk 必须 > 0")

    model = test_pt.load_model(args.projector)
    datasets = test_pt.build_test_datasets([args.dataset])
    dataset = datasets[args.dataset]

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index 越界: {args.sample_index}, dataset size={len(dataset)}")

    sample = dataset[args.sample_index]
    if sample.get("_bad_sample", False):
        raise RuntimeError(f"样本不可用: {sample}")

    max_length = args.max_length or TRAIN.max_length
    collate = test_pt._build_inference_collate(
        tokenizer=model.tokenizer,
        image_token=model.image_token,
        num_image_tokens=TRAIN.num_image_tokens,
        max_length=max_length,
    )
    batch = collate([sample])
    if batch is None:
        raise RuntimeError("collate 返回 None")

    sar_feats = batch["sar_feats"]
    prompt_ids = batch["input_ids"]
    prompt_mask = batch["attention_mask"]
    main_device = model.main_device

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            sar_token_embeds = model.projector(sar_feats.to(main_device).float())

        target_scale = float(model._emb_scale)
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
        position_ids = (prompt_mask.long().cumsum(-1) - 1).clamp(min=0)

        out = model.llm_backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=prompt_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        next_logits = model.lm_head(out.last_hidden_state[:, -1, :])[0]
        next_probs = torch.softmax(next_logits, dim=-1)

        k = min(args.topk, next_logits.shape[-1])
        top_vals, top_ids = torch.topk(next_logits, k=k)
        top_probs = next_probs[top_ids]

    print("=" * 72)
    print("First-Step Logits Inspection")
    print("=" * 72)
    print(f"project_dir : {project_dir}")
    print(f"dataset     : {args.dataset}")
    print(f"sample_index: {args.sample_index}")
    print(f"image_id    : {sample.get('image_id', '<unknown>')}")
    print(f"pt_path     : {sample.get('pt_path', '<missing>')}")
    print()
    print("Reference:")
    print(sample.get("caption", ""))
    print()
    print("Prompt:")
    print(batch["prompts"][0])
    print()
    print(
        f"Greedy first token: id={int(top_ids[0])} token={format_token_text(model.tokenizer.decode([int(top_ids[0])], skip_special_tokens=False))}"
    )
    print()
    print(f"Top-{k} first-token logits:")
    for rank, (token_id, logit, prob) in enumerate(zip(top_ids.tolist(), top_vals.tolist(), top_probs.tolist()), 1):
        token_text = model.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        print(
            f"{rank:>2}. id={int(token_id):>8}  "
            f"logit={float(logit):>10.4f}  "
            f"prob={float(prob):>10.6f}  "
            f"token={format_token_text(token_text)}"
        )


if __name__ == "__main__":
    main()
