#!/usr/bin/env python3
"""
run_batch_test.py — 批量测试多张图像，记录各阶段耗时和显存占用
"""
import argparse
import os
import sys, time, json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

from modules.report.collab_config import collaborative_generator_kwargs, resolve_collaborative_config

WEIGHTS   = "/mnt/data/zhuxiang/SAR_experiments/runs/v4-multiclass/weights/best.pt"
OUTPUT_DIR = Path(__file__).parent / "output"
DEVICE    = "0"

# 测试图像：3张小图(256px) + 3张大图(2048px)
TEST_IMAGES = [
    ("/mnt/data/zhuxiang/SAR_experiments/datasets/v4-multiclass/val/images/ms_04866.jpg",  "小图-港口A"),
    ("/mnt/data/zhuxiang/SAR_experiments/datasets/v4-multiclass/val/images/ms_010000.jpg", "小图-目标B"),
    ("/mnt/data/zhuxiang/SAR_experiments/datasets/v4-multiclass/val/images/ms_010045.jpg", "小图-目标C"),
    ("/mnt/data/zhuxiang/SAR_experiments/datasets/v4-multiclass/val/images/ms_014553.jpg", "大图-港口D"),
    ("/mnt/data/zhuxiang/SAR_experiments/datasets/v4-multiclass/val/images/ms_015315.jpg", "大图-目标E"),
    ("/mnt/data/zhuxiang/SAR_experiments/datasets/v4-multiclass/val/images/ms_015589.jpg", "大图-目标F"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-file", default=None,
                   help="可选 .env 配置文件，用于加载协同模型参数")
    return p.parse_args()

def gpu_mem_mb() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            dev = int(DEVICE)
            alloc = torch.cuda.memory_allocated(dev) / 1024**2
            reserv = torch.cuda.memory_reserved(dev) / 1024**2
            return f"{alloc:.0f}MB alloc / {reserv:.0f}MB reserved"
    except Exception:
        pass
    return "N/A"

def run_one(image_path: str, region: str, detector, assembler, gen):
    from modules.detector.class_map import CLASS_MAP
    from modules.report.generator import ReportGenerator

    timings = {}
    result = {}

    # ── Stage 1: Detection ──────────────────────────────────────────────
    t0 = time.perf_counter()
    objects, vis_path = detector.detect(
        image_path, save_vis=True, vis_dir=str(OUTPUT_DIR / "batch_vis")
    )
    timings["detection"] = time.perf_counter() - t0
    result["n_objects"] = len(objects)
    result["vis_path"]  = vis_path

    # ── Stage 2: Evidence Package ────────────────────────────────────────
    t0 = time.perf_counter()
    counts: dict[str, int] = {}
    for obj in objects:
        code = obj["class"]["code"]
        counts[code] = counts.get(code, 0) + 1
    by_class = []
    for code, cnt in counts.items():
        desc = next((v for v in CLASS_MAP.values() if v["code"] == code), None)
        by_class.append({"code": code,
                          "name_cn": desc["name_cn"] if desc else code,
                          "super_class": desc["super_class"] if desc else "unknown",
                          "count": cnt})
    stats = {"by_class": by_class, "totals": {"objects": len(objects)}}

    now = datetime.now(timezone.utc).isoformat()
    pkg_id = f"sar-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    package = {
        "schema_version": "1.0.0", "package_id": pkg_id,
        "task_type": "intel_brief", "status": "READY_FOR_NLG",
        "input": {"image_uri": f"file://{Path(image_path).resolve()}",
                  "metadata": {"satellite": "SAR卫星", "sensor": "SAR",
                               "acquisition_time": now[:10], "gsd_m": None,
                               "polarization": "未知"},
                  "task": {"region_name": region, "region_type": "harbor"}},
        "scene": {"scene_type": "harbor", "geo_bounds": None, "environment": {}},
        "objects": objects, "statistics": stats,
        "attachments": {},
        "report": {}, "quality": {}, "errors": [],
    }
    if vis_path:
        package["attachments"]["annotated_image"] = {"uri": f"file://{vis_path}"}
    timings["evidence"] = time.perf_counter() - t0

    # ── Stage 3: NLG ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    package = gen.generate(package)
    timings["nlg"] = time.perf_counter() - t0

    # ── Stage 4: Word ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    docx_path = assembler.assemble(
        evidence_package=package,
        output_dir=str(OUTPUT_DIR),
        draft_mode=True,
        include_image=True,
    )
    package["status"] = "DOCX_RENDERED"
    timings["docx"] = time.perf_counter() - t0

    result["timings"] = timings
    result["docx_path"] = docx_path
    result["total"] = sum(timings.values())
    result["gpu_mem"] = gpu_mem_mb()
    return result


def main():
    args = parse_args()
    from PIL import Image as _PILImage
    from modules.detector import DetectorTool
    from modules.detector.class_map import CLASS_MAP as FULL_CLASS_MAP
    from modules.report.collaborative import CollaborativeReportGenerator
    from modules.report.docx_assembler import DocxAssembler

    (OUTPUT_DIR / "batch_vis").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SAR 批量检测 + 报告生成测试")
    print(f"  权重: {Path(WEIGHTS).name}")
    print(f"  设备: GPU {DEVICE}")
    print("=" * 70)

    # 预热：加载模型（计入冷启动时间）
    t_load = time.perf_counter()
    detector = DetectorTool(
        model_path=WEIGHTS,
        class_map=FULL_CLASS_MAP,
        score_thresh=0.25,
        device=DEVICE,
        tile_size=640,
        tile_overlap=128,
        tile_threshold=1280,
    )
    cache_dir = str(OUTPUT_DIR / ".report_cache")
    config = resolve_collaborative_config(cache_dir=cache_dir, env_file=args.env_file)
    gen       = CollaborativeReportGenerator(**collaborative_generator_kwargs(config))
    assembler = DocxAssembler()
    t_load = time.perf_counter() - t_load
    print(f"\n[模型加载] {t_load:.2f}s   显存: {gpu_mem_mb()}\n")

    all_results = []
    for img_path, region in TEST_IMAGES:
        if not Path(img_path).exists():
            print(f"  [跳过] 文件不存在: {img_path}")
            continue

        w, h = _PILImage.open(img_path).size
        print(f"── {region}  ({Path(img_path).name}, {w}×{h}) ──")

        t_total = time.perf_counter()
        res = run_one(img_path, region, detector, assembler, gen)
        res["image"] = Path(img_path).name
        res["region"] = region
        res["size"] = f"{w}×{h}"
        all_results.append(res)

        t = res["timings"]
        print(f"  目标数:  {res['n_objects']}")
        print(f"  检测:    {t['detection']:.2f}s")
        print(f"  证据包:  {t['evidence']:.3f}s")
        print(f"  NLG:     {t['nlg']:.3f}s")
        print(f"  Word:    {t['docx']:.2f}s")
        print(f"  合计:    {res['total']:.2f}s")
        print(f"  显存:    {res['gpu_mem']}")
        print(f"  Word:    {res['docx_path']}")
        print()

    # ── 汇总 ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  汇总")
    print("=" * 70)
    print(f"  {'图像':<22} {'尺寸':<12} {'目标':>4}  {'检测':>6}  {'Word':>6}  {'合计':>6}")
    print(f"  {'-'*22} {'-'*12} {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}")
    for r in all_results:
        t = r["timings"]
        print(f"  {r['region']:<22} {r['size']:<12} {r['n_objects']:>4}  "
              f"{t['detection']:>5.2f}s  {t['docx']:>5.2f}s  {r['total']:>5.2f}s")

    if all_results:
        avg_total = sum(r["total"] for r in all_results) / len(all_results)
        print(f"\n  平均总耗时: {avg_total:.2f}s  (不含模型加载 {t_load:.2f}s)")
        print(f"  最终显存:   {gpu_mem_mb()}")

    print("=" * 70)
    print(f"\n  Word 文件保存在: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
