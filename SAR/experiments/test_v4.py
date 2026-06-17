#!/usr/bin/env python3
"""
test_v4.py — v4-multiclass 模型测试脚本

用法：
    python test_v4.py [--weights PATH] [--data PATH] [--split val] [--save-vis]

默认使用 last.pt，对 v4-multiclass val 集做评估，输出报告到 output/v4_test_report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ── 路径常量 ────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "/mnt/data/zhuxiang/SAR_experiments/runs/v4-multiclass/weights/last.pt"
DEFAULT_DATA    = "/mnt/data/zhuxiang/SAR_experiments/datasets/v4-multiclass/dataset.yaml"
OUTPUT_DIR      = Path(__file__).parent / "output"

# ── 类别名（与 dataset.yaml 严格对应）──────────────────────────────────────
CLASS_NAMES = {0: "ship", 1: "aircraft", 2: "tank", 3: "bridge", 4: "harbor"}
CLASS_CN    = {0: "舰船", 1: "飞机", 2: "坦克/装甲车", 3: "桥梁", 4: "港口"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--data",    default=DEFAULT_DATA)
    p.add_argument("--split",   default="val", choices=["val", "train"])
    p.add_argument("--save-vis", action="store_true",
                   help="保存前 20 张检测可视化图")
    p.add_argument("--device",  default="0")
    p.add_argument("--batch",   type=int, default=16)
    return p.parse_args()


def run_val(args):
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[ERROR] 权重文件不存在: {weights}")
        sys.exit(1)

    print(f"[INFO] 加载模型: {weights}")
    model = YOLO(str(weights))

    print(f"[INFO] 开始验证 (split={args.split}, batch={args.batch}, device={args.device}) ...")
    metrics = model.val(
        data=args.data,
        split=args.split,
        batch=args.batch,
        device=args.device,
        verbose=True,
        save_json=True,
    )
    return model, metrics


def run_vis(model, args, n=20):
    """对 val 集前 n 张图做推理并保存可视化。"""
    import yaml
    from modules.detector.visualize import save_annotated
    from modules.detector.class_map import CLASS_MAP

    with open(args.data) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(args.data).parent
    img_dir = data_root / args.split / "images"
    images = sorted(img_dir.glob("*.jpg"))[:n] + sorted(img_dir.glob("*.png"))[:n]
    images = images[:n]

    vis_out = OUTPUT_DIR / "v4_vis"
    vis_out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 可视化 {len(images)} 张图 → {vis_out}")
    for img_path in images:
        results = model.predict(str(img_path), device=args.device, verbose=False)
        objects = []
        for r in results:
            if r.obb is None:
                continue
            for box in r.obb:
                cls_id = int(box.cls.item())
                desc = CLASS_MAP.get(cls_id, {"code": "unknown", "name_cn": "未知",
                                              "super_class": "unknown", "priority": "LOW",
                                              "status": "ACTIVE"})
                xywhr = box.xywhr[0].cpu().numpy()
                objects.append({
                    "object_id": f"obj-{cls_id:02d}",
                    "class": desc,
                    "score": {"confidence": float(box.conf.item())},
                    "geometry": {
                        "pixel": {
                            "center_x": float(xywhr[0]),
                            "center_y": float(xywhr[1]),
                            "width":    float(xywhr[2]),
                            "height":   float(xywhr[3]),
                            "angle_deg": float(xywhr[4]),
                        }
                    },
                })
        if objects:
            save_annotated(str(img_path), objects, output_dir=str(vis_out))

    print(f"[INFO] 可视化完成，共 {len(images)} 张")


def build_report(metrics, args) -> str:
    """生成 Markdown 报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    weights = Path(args.weights)

    # 从 metrics 对象提取数据
    box = metrics.box
    map50    = float(box.map50)
    map5095  = float(box.map)
    precision = float(box.mp)
    recall    = float(box.mr)

    # 逐类 AP50
    per_class_ap50 = box.ap50  # numpy array, len = nc
    per_class_ap   = box.ap    # numpy array

    lines = [
        f"# v4-multiclass 模型测试报告",
        f"",
        f"**生成时间**: {now}  ",
        f"**权重文件**: `{weights}`  ",
        f"**评估集**: {args.split}  ",
        f"",
        f"## 整体指标",
        f"",
        f"| 指标 | 值 |",
        f"|------|----|",
        f"| mAP@50 | **{map50:.4f}** |",
        f"| mAP@50-95 | {map5095:.4f} |",
        f"| Precision | {precision:.4f} |",
        f"| Recall | {recall:.4f} |",
        f"",
        f"## 逐类 AP50",
        f"",
        f"| ID | 类别 | 中文名 | AP50 | AP50-95 |",
        f"|----|------|--------|------|---------|",
    ]

    for i in range(len(per_class_ap50)):
        name = CLASS_NAMES.get(i, f"class{i}")
        cn   = CLASS_CN.get(i, "—")
        ap50_val = float(per_class_ap50[i])
        ap_val   = float(per_class_ap[i])
        lines.append(f"| {i} | {name} | {cn} | {ap50_val:.4f} | {ap_val:.4f} |")

    lines += [
        f"",
        f"## 训练进度（来自 results.csv）",
        f"",
    ]

    # 读取 results.csv
    results_csv = Path("/mnt/data/zhuxiang/SAR_experiments/runs/v4-multiclass/results.csv")
    if results_csv.exists():
        import csv
        rows = list(csv.DictReader(open(results_csv)))
        if rows:
            lines.append(f"| Epoch | mAP50 | mAP50-95 | box_loss | cls_loss | 耗时(s) |")
            lines.append(f"|-------|-------|----------|----------|----------|---------|")
            for row in rows:
                ep   = row.get("epoch", "?").strip()
                m50  = float(row.get("metrics/mAP50(B)", 0))
                m95  = float(row.get("metrics/mAP50-95(B)", 0))
                bloss = float(row.get("train/box_loss", 0))
                closs = float(row.get("train/cls_loss", 0))
                t    = float(row.get("time", 0))
                lines.append(f"| {ep} | {m50:.4f} | {m95:.4f} | {bloss:.4f} | {closs:.4f} | {t:.0f} |")
    else:
        lines.append("_results.csv 未找到_")

    lines += [
        f"",
        f"## 说明",
        f"",
        f"- 模型训练中（last.pt 为最新 checkpoint，best.pt 为历史最优）",
        f"- 类别 5-13 为 RESERVED，当前训练集不含，推理输出"预留-XX"",
        f"- 可视化图保存在 `output/v4_vis/`（如使用 --save-vis）",
    ]

    return "\n".join(lines)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, metrics = run_val(args)

    if args.save_vis:
        run_vis(model, args)

    report = build_report(metrics, args)

    report_path = OUTPUT_DIR / "v4_test_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[INFO] 报告已保存: {report_path}")
    print("\n" + "="*60)
    print(report)


if __name__ == "__main__":
    main()
