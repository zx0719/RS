from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
STAGE1_CONFIG = REPO_ROOT / "train" / "stage1" / "config_local.py"
STAGE2_CONFIG = REPO_ROOT / "train" / "stage2" / "config_local.py"

from src.model.qwen3_vl_direct import (  # noqa: E402
    DirectQwen3VLSample as EvalSample,
    DirectQwen3VLVLLM,
)


def _load_py_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON top-level must be a list: {path}")
    return [x for x in data if isinstance(x, dict)]


def _strip_image_tag(text: str) -> str:
    return str(text).replace("<image>", "").strip()


def _existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        try:
            if path.exists() and path.is_file():
                return path
        except OSError:
            continue
    return None


def _resolve_image_path(sample: Dict[str, Any], root: Path) -> Optional[str]:
    candidates: List[Path] = []

    raw_paths: List[str] = []
    images = sample.get("images")
    if isinstance(images, list):
        raw_paths.extend(str(x) for x in images if x)
    image = sample.get("image")
    if image:
        raw_paths.append(str(image))

    for raw in raw_paths:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else root / p)

    pt_path = str(sample.get("pt_path", "") or "")
    if pt_path:
        pt = Path(pt_path)
        pt_stem_rel = pt.with_suffix("")
        suffixes: List[str] = []
        for raw in raw_paths:
            suffix = Path(raw).suffix
            if suffix and suffix.lower() not in {x.lower() for x in suffixes}:
                suffixes.append(suffix)
        suffixes.extend([".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"])

        # SARTEXT cache layout:
        #   .../pt_cache/<subset>/<name>.pt -> .../Image/<subset>/<name>.<ext>
        marker = f"{os.sep}pt_cache{os.sep}"
        if marker in pt_path:
            prefix, rel = pt_path.split(marker, 1)
            base = Path(prefix) / "Image" / Path(rel).with_suffix("")
            candidates.extend(base.with_suffix(suffix) for suffix in suffixes)

        # Generic fallback when the .pt and image share the same stem nearby.
        candidates.extend(pt_stem_rel.with_suffix(suffix) for suffix in suffixes)

    found = _existing_path(candidates)
    return str(found) if found is not None else None


def _first_user_assistant_pair(messages: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(messages, list):
        return None
    for i in range(len(messages) - 1):
        u = messages[i]
        a = messages[i + 1]
        if not isinstance(u, dict) or not isinstance(a, dict):
            continue
        if str(u.get("role", "")).lower() != "user":
            continue
        if str(a.get("role", "")).lower() != "assistant":
            continue
        prompt = _strip_image_tag(str(u.get("content", "")))
        answer = str(a.get("content", "")).strip()
        if prompt and answer:
            return prompt, answer
    return None


def _parse_turns(sample: Dict[str, Any]) -> List[Tuple[str, str]]:
    turns: List[Tuple[str, str]] = []

    messages = sample.get("messages")
    if isinstance(messages, list):
        i = 0
        while i < len(messages) - 1:
            u = messages[i]
            a = messages[i + 1]
            if (
                isinstance(u, dict)
                and isinstance(a, dict)
                and str(u.get("role", "")).lower() == "user"
                and str(a.get("role", "")).lower() == "assistant"
            ):
                prompt = _strip_image_tag(str(u.get("content", "")))
                answer = str(a.get("content", "")).strip()
                if prompt and answer:
                    turns.append((prompt, answer))
                i += 2
            else:
                i += 1
        return turns

    conversations = sample.get("conversations")
    if isinstance(conversations, list):
        i = 0
        while i < len(conversations) - 1:
            u = conversations[i]
            a = conversations[i + 1]
            if (
                isinstance(u, dict)
                and isinstance(a, dict)
                and str(u.get("from", "")).lower() in {"human", "user"}
                and str(a.get("from", "")).lower() in {"gpt", "assistant"}
            ):
                prompt = _strip_image_tag(str(u.get("value", "")))
                answer = str(a.get("value", "")).strip()
                if prompt and answer:
                    turns.append((prompt, answer))
                i += 2
            else:
                i += 1

    return turns


def _sample_id(sample: Dict[str, Any], image_path: Optional[str]) -> str:
    if sample.get("id"):
        return str(sample["id"])
    if image_path:
        return Path(image_path).stem
    for key in ("image", "pt_path"):
        if sample.get(key):
            return Path(str(sample[key])).stem
    images = sample.get("images")
    if isinstance(images, list) and images:
        return Path(str(images[0])).stem
    return ""


def _build_stage1_samples(
    dataset_names: Optional[List[str]],
    log: Callable[[str], None],
) -> Tuple[str, Dict[str, List[EvalSample]]]:
    cfg = _load_py_module(STAGE1_CONFIG, "qwen3_vl_direct_stage1_config")
    paths = cfg.PATHS
    train = cfg.TRAIN
    model_path = str(paths.qwen_path)

    defs = {
        "sarlang": (Path(paths.sarlang_root), paths.sarlang_pt_test_json, train.mixed_weight_sarlang),
        "sartext": (Path(paths.sartext_root), paths.sartext_pt_test_json, train.mixed_weight_sartext),
        "sarcap": (Path(paths.sarcap_root), paths.sarcap_pt_test_json, train.mixed_weight_sarcap),
        "fsarcap": (Path(paths.fsarcap_root), paths.fsarcap_pt_test_json, train.mixed_weight_fsarcap),
    }
    wanted = dataset_names or [name for name, (_, _, weight) in defs.items() if float(weight) > 0]
    out: Dict[str, List[EvalSample]] = {}

    for name in wanted:
        if name not in defs:
            raise ValueError(f"Unknown stage1 dataset {name!r}; choices: {list(defs)}")
        root, rel_json, _ = defs[name]
        json_path = root / rel_json
        if not json_path.exists():
            log(f"[WARN] skip {name}: missing json {json_path}")
            continue
        rows = _read_json(json_path)
        samples: List[EvalSample] = []
        missing_image = 0
        bad_text = 0
        for row in rows:
            pair = _first_user_assistant_pair(row.get("messages"))
            if pair is None:
                bad_text += 1
                continue
            image_path = _resolve_image_path(row, root)
            if image_path is None:
                missing_image += 1
                continue
            prompt, reference = pair
            samples.append(
                EvalSample(
                    dataset=name,
                    sample_id=_sample_id(row, image_path),
                    image_path=image_path,
                    prompt=prompt,
                    reference=reference,
                )
            )
        log(
            f"[INFO] stage1/{name}: loaded={len(samples)} "
            f"raw={len(rows)} missing_image={missing_image} bad_text={bad_text}"
        )
        if samples:
            out[name] = samples
    return model_path, out


def _build_stage2_samples(
    dataset_names: Optional[List[str]],
    log: Callable[[str], None],
) -> Tuple[str, Dict[str, List[EvalSample]]]:
    cfg = _load_py_module(STAGE2_CONFIG, "qwen3_vl_direct_stage2_config")
    paths = cfg.PATHS
    train = cfg.TRAIN
    model_path = str(paths.qwen_path)

    defs = {
        "sarvqa": (Path(paths.sarvqa_root), paths.sarvqa_pt_test_json, train.mixed_weight_sarvqa),
        "sartext": (Path(paths.sartext_root), paths.sartext_pt_test_json, train.mixed_weight_sartext),
        "sarlang_vqa": (
            Path(paths.sarlang_vqa_root),
            paths.sarlang_vqa_pt_test_json,
            train.mixed_weight_sarlang_vqa,
        ),
    }
    wanted = dataset_names or [name for name, (_, _, weight) in defs.items() if float(weight) > 0]
    out: Dict[str, List[EvalSample]] = {}

    for name in wanted:
        if name not in defs:
            raise ValueError(f"Unknown stage2 dataset {name!r}; choices: {list(defs)}")
        root, rel_json, _ = defs[name]
        json_path = root / rel_json
        if not json_path.exists():
            log(f"[WARN] skip {name}: missing json {json_path}")
            continue
        rows = _read_json(json_path)
        samples: List[EvalSample] = []
        missing_image = 0
        bad_turns = 0
        for row in rows:
            turns = _parse_turns(row)
            if not turns:
                bad_turns += 1
                continue
            image_path = _resolve_image_path(row, root)
            if image_path is None:
                missing_image += 1
                continue
            samples.append(
                EvalSample(
                    dataset=name,
                    sample_id=_sample_id(row, image_path),
                    image_path=image_path,
                    prompt=turns[-1][0],
                    reference=turns[-1][1],
                    turns=turns,
                )
            )
        log(
            f"[INFO] stage2/{name}: loaded={len(samples)} "
            f"raw={len(rows)} missing_image={missing_image} bad_turns={bad_turns}"
        )
        if samples:
            out[name] = samples
    return model_path, out


def _compute_bleu(references: List[str], hypotheses: List[str], weights) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    refs = [[ref.split()] for ref in references]
    hyps = [hyp.split() for hyp in hypotheses]
    return float(corpus_bleu(refs, hyps, weights=weights, smoothing_function=SmoothingFunction().method1))


def _compute_bleu_scores(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    return {
        "bleu1": _compute_bleu(references, hypotheses, (1.0, 0.0, 0.0, 0.0)),
        "bleu2": _compute_bleu(references, hypotheses, (0.5, 0.5, 0.0, 0.0)),
        "bleu3": _compute_bleu(references, hypotheses, (1 / 3, 1 / 3, 1 / 3, 0.0)),
        "bleu4": _compute_bleu(references, hypotheses, (0.25, 0.25, 0.25, 0.25)),
    }


def _compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)]
    return float(sum(scores) / len(scores)) if scores else 0.0


def _compute_exact_match(references: List[str], hypotheses: List[str]) -> float:
    if not references:
        return 0.0
    matches = sum(ref.strip().lower() == hyp.strip().lower() for ref, hyp in zip(references, hypotheses))
    return float(matches) / len(references)


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m{sec:04.1f}s"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _evaluate_dataset(
    stage: str,
    dataset_name: str,
    samples: Sequence[EvalSample],
    generator: DirectQwen3VLVLLM,
    batch_size: int,
    max_new_tokens: int,
    max_samples: Optional[int],
    progress_every: int,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    refs: List[str] = []
    hyps: List[str] = []
    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    total_generate_sec = 0.0
    total_tokens = 0
    start = time.time()

    batch_id = 0
    for offset in range(0, len(samples), batch_size):
        if max_samples is not None and len(records) >= max_samples:
            break
        batch = list(samples[offset : offset + batch_size])
        if max_samples is not None:
            remaining = max_samples - len(records)
            batch = batch[:remaining]
        batch_id += 1

        try:
            batch_records, batch_skipped, gen_sec = generator.generate_batch(
                batch,
                stage=stage,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            if len(batch) <= 1:
                log(f"  [WARN] batch={batch_id} vLLM error, skipped 1 sample: {e}")
                skipped.extend(
                    {
                        "dataset": x.dataset,
                        "sample_id": x.sample_id,
                        "image_path": x.image_path,
                        "reason": repr(e),
                    }
                    for x in batch
                )
                continue

            log(
                f"  [WARN] batch={batch_id} vLLM error, retrying {len(batch)} samples one-by-one: {e}"
            )
            batch_records = []
            batch_skipped = []
            gen_sec = 0.0
            for sample in batch:
                try:
                    one_records, one_skipped, one_sec = generator.generate_batch(
                        [sample],
                        stage=stage,
                        max_new_tokens=max_new_tokens,
                    )
                    batch_records.extend(one_records)
                    batch_skipped.extend(one_skipped)
                    gen_sec += one_sec
                except Exception as one_e:
                    batch_skipped.append(
                        {
                            "dataset": sample.dataset,
                            "sample_id": sample.sample_id,
                            "image_path": sample.image_path,
                            "reason": repr(one_e),
                        }
                    )

        total_generate_sec += gen_sec
        skipped.extend(batch_skipped)

        for rec in batch_records:
            rec["sample_index"] = len(records) + 1
            records.append(rec)
            refs.append(str(rec["reference"]))
            hyps.append(str(rec["hypothesis"]))
            total_tokens += int(rec.get("output_tokens") or 0)

        if progress_every > 0 and batch_id % progress_every == 0:
            elapsed = time.time() - start
            log(
                "  [gen:qwen3vl-vllm] "
                f"batch={batch_id} samples={len(records)} skipped={len(skipped)} "
                f"elapsed={_format_seconds(elapsed)} gen={_format_seconds(total_generate_sec)} "
                f"samples/s={len(records) / max(elapsed, 1e-6):.2f} "
                f"tokens/s={total_tokens / max(total_generate_sec, 1e-6):.2f}"
            )

    elapsed = time.time() - start
    base = {
        "num_samples": len(records),
        "num_skipped": len(skipped),
        "sample_records": records,
        "skipped_records": skipped,
        "examples": [(r, h) for r, h in zip(refs[:3], hyps[:3])],
        "elapsed_sec": elapsed,
        "generate_sec": total_generate_sec,
        "samples_per_sec": len(records) / max(elapsed, 1e-6),
        "tokens_per_sec": total_tokens / max(total_generate_sec, 1e-6),
        "avg_output_tokens": total_tokens / max(len(records), 1),
    }
    if not refs:
        if stage == "stage1":
            return {**base, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "rougeL": 0.0}
        return {**base, "bleu4": 0.0, "rougeL": 0.0, "exact_match": 0.0}

    if stage == "stage1":
        return {
            **base,
            **_compute_bleu_scores(refs, hyps),
            "rougeL": _compute_rouge_l(refs, hyps),
        }
    return {
        **base,
        "bleu4": _compute_bleu(refs, hyps, (0.25, 0.25, 0.25, 0.25)),
        "rougeL": _compute_rouge_l(refs, hyps),
        "exact_match": _compute_exact_match(refs, hyps),
    }


def _log_metrics(stage: str, log: Callable[[str], None], metrics: Dict[str, Any]) -> None:
    if stage == "stage1":
        log(f"  BLEU-1      = {metrics['bleu1']:.4f}")
        log(f"  BLEU-2      = {metrics['bleu2']:.4f}")
        log(f"  BLEU-3      = {metrics['bleu3']:.4f}")
        log(f"  BLEU-4      = {metrics['bleu4']:.4f}")
        log(f"  ROUGE-L     = {metrics['rougeL']:.4f}")
    else:
        log(f"  BLEU-4      = {metrics['bleu4']:.4f}")
        log(f"  ROUGE-L     = {metrics['rougeL']:.4f}")
        log(f"  Exact Match = {metrics['exact_match']:.4f}")
    log(
        "  performance = "
        f"samples={metrics['num_samples']} skipped={metrics['num_skipped']} | "
        f"samples/s {metrics['samples_per_sec']:.2f} | "
        f"tokens/s {metrics['tokens_per_sec']:.2f} | "
        f"avg_out_tokens {metrics['avg_output_tokens']:.1f} | "
        f"gen {_format_seconds(metrics['generate_sec'])}"
    )
    if metrics.get("examples"):
        log("  examples:")
        for i, (ref, hyp) in enumerate(metrics["examples"], start=1):
            log(f"  [{i}] Ref: {str(ref)[:160]}")
            log(f"      Hyp: {str(hyp)[:160]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Qwen3-VL-4B-Instruct vLLM evaluation on SAR-LLM stage1/stage2 data.")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--model_path", type=str, default=None, help="Qwen3-VL model path. Defaults to the stage config qwen_path.")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names. Defaults follow mixed_weight_* in config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Default: 64 for both stage1 and stage2 on GPU1.")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_gen_samples", type=int, default=-1, help="-1 means full dataset.")
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--vllm_max_model_len", type=int, default=6144)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--min_pixels", type=int, default=None, help="Optional Qwen-VL image processor min_pixels.")
    parser.add_argument("--max_pixels", type=int, default=262144, help="Optional Qwen-VL image processor max_pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_arg = [x.strip() for x in args.datasets.split(",") if x.strip()] if args.datasets else None
    max_samples = None if args.max_gen_samples == -1 else int(args.max_gen_samples)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (50 if args.stage == "stage1" else 200)
    batch_size = args.batch_size if args.batch_size is not None else 64

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"direct_qwen3vl_{args.stage}_{ts}"
    out_root = Path(__file__).resolve().parent
    log_dir = out_root / "logs"
    output_dir = out_root / "outputs" / run_name
    metrics_dir = out_root / "metrics" / run_name
    for d in (log_dir, output_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg: str = "") -> None:
        print(msg)
        log_file.write(msg + "\n")

    try:
        if args.stage == "stage1":
            default_model_path, datasets = _build_stage1_samples(datasets_arg, log)
        else:
            default_model_path, datasets = _build_stage2_samples(datasets_arg, log)

        model_path = args.model_path or default_model_path
        if not datasets:
            raise RuntimeError("No evaluable datasets were loaded.")

        log(f"[INFO] log        -> {log_path}")
        log(f"[INFO] stage      -> {args.stage}")
        log(f"[INFO] model      -> {model_path}")
        log(f"[INFO] datasets   -> {list(datasets)}")
        log(f"[INFO] batch_size -> {batch_size}")
        log(f"[INFO] max_gen_samples={args.max_gen_samples} max_new_tokens={max_new_tokens}")
        log(
            "[INFO] vLLM      -> "
            f"tp={args.vllm_tensor_parallel_size}, "
            f"gpu_mem={args.vllm_gpu_memory_utilization}, "
            f"max_model_len={args.vllm_max_model_len}, "
            f"dtype={args.vllm_dtype}, "
            f"max_pixels={args.max_pixels}"
        )
        log("[INFO] GPU binding is intentionally controlled by the shell script via CUDA_VISIBLE_DEVICES.")

        generator = DirectQwen3VLVLLM(
            model_path=model_path,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            dtype=args.vllm_dtype,
            enforce_eager=args.vllm_enforce_eager,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

        all_results: Dict[str, Dict[str, Any]] = {}
        for dataset_name, samples in datasets.items():
            log(f"\n{'=' * 60}")
            log(f"  [{args.stage}] dataset={dataset_name} size={len(samples)}")
            log(f"{'=' * 60}")
            metrics = _evaluate_dataset(
                stage=args.stage,
                dataset_name=dataset_name,
                samples=samples,
                generator=generator,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                max_samples=max_samples,
                progress_every=args.progress_every,
                log=log,
            )
            all_results[dataset_name] = metrics
            _log_metrics(args.stage, log, metrics)

            _write_jsonl(output_dir / f"{dataset_name}.jsonl", metrics["sample_records"])
            if metrics["skipped_records"]:
                _write_jsonl(output_dir / f"{dataset_name}_skipped.jsonl", metrics["skipped_records"])

            slim_metrics = {k: v for k, v in metrics.items() if k not in {"sample_records", "skipped_records"}}
            _write_json(metrics_dir / f"{dataset_name}.json", {"dataset": dataset_name, "metrics": slim_metrics})

        log(f"\n{'=' * 60}")
        log("  Summary")
        log(f"{'=' * 60}")
        if args.stage == "stage1":
            header = f"{'dataset':<14} {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-3':>8} {'BLEU-4':>8} {'ROUGE-L':>8} {'N':>8} {'skip':>8}"
            log(header)
            log("-" * len(header))
            for name, r in all_results.items():
                log(
                    f"{name:<14} {r['bleu1']:>8.4f} {r['bleu2']:>8.4f} {r['bleu3']:>8.4f} "
                    f"{r['bleu4']:>8.4f} {r['rougeL']:>8.4f} {r['num_samples']:>8} {r['num_skipped']:>8}"
                )
        else:
            header = f"{'dataset':<14} {'BLEU-4':>8} {'ROUGE-L':>8} {'EM':>8} {'N':>8} {'skip':>8}"
            log(header)
            log("-" * len(header))
            for name, r in all_results.items():
                log(
                    f"{name:<14} {r['bleu4']:>8.4f} {r['rougeL']:>8.4f} {r['exact_match']:>8.4f} "
                    f"{r['num_samples']:>8} {r['num_skipped']:>8}"
                )

        _write_json(
            metrics_dir / "summary.json",
            {
                "stage": args.stage,
                "model_path": model_path,
                "outputs_dir": str(output_dir),
                "generation_backend": "vllm_direct_qwen3vl",
                "run_config": {
                    "batch_size": batch_size,
                    "max_gen_samples": args.max_gen_samples,
                    "max_new_tokens": max_new_tokens,
                    "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
                    "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                    "vllm_max_model_len": args.vllm_max_model_len,
                    "vllm_dtype": args.vllm_dtype,
                    "vllm_enforce_eager": args.vllm_enforce_eager,
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                },
                "datasets": {
                    name: {k: v for k, v in r.items() if k not in {"sample_records", "skipped_records", "examples"}}
                    for name, r in all_results.items()
                },
            },
        )
        log(f"\n[INFO] outputs -> {output_dir}")
        log(f"[INFO] metrics -> {metrics_dir}")
        log(f"[INFO] log     -> {log_path}")
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
