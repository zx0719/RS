"""
train.py — YOLOv8-OBB training script for SAR ship/aircraft detection (M2)

CLI usage
---------
    python train.py --data dataset.yaml --epochs 100 --imgsz 1024 --model yolov8m-obb.pt

    # Resume a previous run
    python train.py --data dataset.yaml --resume --model runs/detect/train/weights/last.pt

    # Fine-tune with higher resolution and more augmentation
    python train.py --data dataset.yaml --epochs 50 --imgsz 1280 --model yolov8l-obb.pt \\
        --batch 8 --lr0 1e-4 --device cuda:0

Output
------
Trained weights are saved to `runs/detect/<name>/weights/best.pt`.

Notes
-----
- Requires: pip install ultralytics
- The `dataset.yaml` must follow YOLO-OBB format; use dataset_utils.py to convert DOTA annotations.
- Class ordering in dataset.yaml must match class_map.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-OBB on SAR ship/aircraft dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file (YOLO-OBB format).",
    )

    # ── Model ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m-obb.pt",
        help=(
            "Pretrained YOLOv8-OBB weights to start from. "
            "Use a local .pt file to resume training."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from --model checkpoint (passes resume=True to ultralytics).",
    )

    # ── Training hyper-parameters ──────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size (square).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 = auto).")
    parser.add_argument("--lr0", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=1e-2, help="Final LR factor (lr0 * lrf).")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Optimizer weight decay.")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="Warmup epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without improvement). 0 = disabled.",
    )

    # ── Augmentation ───────────────────────────────────────────────────────
    parser.add_argument("--hsv_h", type=float, default=0.015, help="HSV-hue augmentation.")
    parser.add_argument("--hsv_s", type=float, default=0.7, help="HSV-saturation augmentation.")
    parser.add_argument("--hsv_v", type=float, default=0.4, help="HSV-value augmentation.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability.")
    parser.add_argument("--flipud", type=float, default=0.5, help="Vertical flip probability.")
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augmentation probability (SAR benefits from mosaic).",
    )

    # ── Hardware / output ──────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device.")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker threads.")
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Output directory root.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Experiment name (subdirectory under --project).",
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Allow overwriting an existing experiment directory.",
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=-1,
        help="Save checkpoint every N epochs (-1 = save only best/last).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -- Import check --------------------------------------------------------
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        print(
            "ERROR: ultralytics is not installed.\n"
            "Install it with:  pip install ultralytics",
            file=sys.stderr,
        )
        sys.exit(1)

    from ultralytics import YOLO  # type: ignore[import-untyped]

    # -- Validate paths ------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset YAML not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # -- Load model ----------------------------------------------------------
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # -- Build training kwargs -----------------------------------------------
    train_kwargs: dict = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "mosaic": args.mosaic,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "save_period": args.save_period,
        "resume": args.resume,
        "verbose": True,
    }

    # -- Train ---------------------------------------------------------------
    print("Starting training with parameters:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")

    results = model.train(**train_kwargs)

    # -- Report --------------------------------------------------------------
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nTraining complete. Best weights saved to: {best_weights}")
    else:
        print(f"\nTraining complete. Check {args.project}/{args.name}/ for outputs.")

    # Print final metrics summary if available
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print("\nFinal metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
