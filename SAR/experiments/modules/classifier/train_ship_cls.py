#!/usr/bin/env python3
"""
train_ship_cls.py — Train the ResNet-18 ship fine-grained classifier (M2a, offline).

Reads data_script_run detection data:
  - labels/ → YOLO HBB format: class_id cx cy w h
  - metadata/samples.jsonl → fine_class (ship subtype name)
  - images/ → SAR tiles

Extracts ship ROIs from detection labels where coarse_class is ship-related,
trains a 16-class ResNet-18 classifier.

Usage
-----
    python train_ship_cls.py \
        --data-root /path/to/data_script_run \
        --output weights/ship_cls_resnet18.pt \
        --epochs 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure modules/ package is importable regardless of working directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODULES_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _MODULES_DIR.parent
for _path in (_SCRIPT_DIR, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

try:
    from .ship_classifier import SHIP_SUBTYPES
except ImportError:
    from ship_classifier import SHIP_SUBTYPES

CHIP_SIZE = 128
DEBUG_MAX_EXAMPLES = 20

_SHIP_CLASS_ID_TO_CODE = {
    1: "military_auxiliary",
    2: "combat_ship",
    3: "liquid_cargo",
    4: "harbor_service",
    5: "bulk_carrier",
    6: "other_vessel",
    7: "survey_vessel",
    10: "amphibious",
    11: "container_ship",
    12: "engineering_vessel",
    13: "fishing_vessel",
    14: "tug_boat",
    15: "passenger_ship",
    16: "ro_ro_ship",
    19: "sailing_vessel",
    20: "research_vessel",
}

# Coarse classes that are ship-related
_SHIP_COARSE = {"舰船", "军用辅助舰船", "作战舰船", "液货船", "港务船", "散货船",
                "舰船_其他", "调查船", "两栖舰船", "集装箱船", "工程船", "渔船",
                "拖船", "客船", "滚装船", "帆船", "研究船",
                "ship", "vessel", "destroyer", "frigate", "carrier",
                "replenishment", "amphibious", "cargo", "tanker", "fishing"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ShipChipDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], augment: bool = True):
        self.samples = samples
        self.transform = self._build_transform(augment)

    @staticmethod
    def _build_transform(augment: bool):
        if augment:
            return T.Compose([
                T.ToPILImage(),
                T.RandomRotation(degrees=180),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.Resize((CHIP_SIZE, CHIP_SIZE)),
                T.ToTensor(),
            ])
        return T.Compose([T.ToPILImage(), T.Resize((CHIP_SIZE, CHIP_SIZE)), T.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        img = img.astype(np.float32) / 255.0
        return self.transform(img), label


def describe_transform(transform: Any) -> list[str]:
    if hasattr(transform, "transforms"):
        parts = []
        for t in transform.transforms:
            attrs = []
            for key in ("size", "p", "degrees", "translate", "scale"):
                if hasattr(t, key):
                    attrs.append(f"{key}={getattr(t, key)}")
            joined = ", ".join(attrs)
            parts.append(f"{type(t).__name__}({joined})" if joined else type(t).__name__)
        return parts
    return [repr(transform)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def collect_ship_chips(data_root: str) -> tuple[list, list]:
    """Extract ship ROI chips from detection data.

    For each detection with a ship-related coarse_class:
      1. Read bbox from YOLO label
      2. Crop from image
      3. Assign fine_class label (mapped to SHIP_SUBTYPES index)
    """
    root = Path(data_root)
    det_img_dir = root / "detection" / "train" / "images"
    det_lbl_dir = root / "detection" / "train" / "labels"
    det_meta = root / "detection" / "metadata" / "samples.jsonl"

    if not det_img_dir.is_dir():
        print(f"[ERROR] Image dir not found: {det_img_dir}")
        sys.exit(1)

    # Build image_name → samples.jsonl record
    img_meta: dict[str, dict] = {}
    if det_meta.exists():
        with open(det_meta) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] Skipping malformed JSON line in {det_meta}")
                    continue
                img_name = Path(rec["image_path"]).name
                img_meta[img_name] = rec

    # Build fine_class → SHIP_SUBTYPES index mapping
    # Support both English codes and Chinese names via reverse lookup
    fine_to_idx: dict[str, int] = {}
    for i, name in enumerate(SHIP_SUBTYPES):
        fine_to_idx[name] = i
    # Add Chinese name → index mapping
    try:
        from modules.class_labels import _CODE_TO_CN
        _cn_to_code = {v: k for k, v in _CODE_TO_CN.items()}
        for cn_name, code in _cn_to_code.items():
            if code in fine_to_idx:
                fine_to_idx[cn_name] = fine_to_idx[code]
    except ImportError:
        pass

    import cv2

    chips: list[tuple[Path, int]] = []
    temp_dir = Path("temp_ship_chips")
    temp_dir.mkdir(exist_ok=True)
    debug_examples: list[str] = []

    img_files = sorted(det_img_dir.glob("*.jpg")) + sorted(det_img_dir.glob("*.png"))
    print(f"[ShipCls] Scanning {len(img_files)} images for ship instances...")

    for img_path in img_files:
        img_name = img_path.name
        lbl_path = det_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        # Read YOLO labels (HBB: class_id cx cy w h)
        lines = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if 5 <= len(parts) <= 9:  # HBB(5) or OBB(9)
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                    lines.append((cls_id, cx, cy, w, h))

        if not lines:
            continue

        # Get instance metadata from samples.jsonl
        meta = img_meta.get(img_name, {})
        instances = meta.get("instances", [])

        # Load image once per file
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        H, W = image.shape

        for idx, (cls_id, cx, cy, w, h) in enumerate(lines):
            # Get fine_class from matching instance
            fine_class = None
            coarse_class = ""
            if idx < len(instances):
                inst = instances[idx]
                coarse_class = inst.get("coarse_class", "")
                fine_class = inst.get("fine_class")

            # Prefer explicit detection class ids.
            class_code = _SHIP_CLASS_ID_TO_CODE.get(int(cls_id))
            is_ship = class_code is not None

            # Fallback to metadata-based coarse/fine labels if needed.
            if not is_ship:
                if coarse_class in _SHIP_COARSE:
                    is_ship = True
                elif fine_class and fine_class in _SHIP_COARSE:
                    is_ship = True

            if not is_ship:
                continue

            # Map from class id first, then metadata fallback.
            target_idx = fine_to_idx.get(class_code) if class_code else None
            if target_idx is None:
                target_idx = fine_to_idx.get(fine_class) if fine_class else None
            if target_idx is None:
                target_idx = fine_to_idx.get(coarse_class)
            if target_idx is None:
                target_idx = fine_to_idx["other_vessel"]

            # Crop bbox (denormalize)
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            chip_path = temp_dir / f"{img_path.stem}_{idx}.png"
            cv2.imwrite(str(chip_path), crop)
            chips.append((chip_path, target_idx))
            if len(debug_examples) < DEBUG_MAX_EXAMPLES:
                debug_examples.append(
                    f"img={img_name} det_idx={idx} cls_id={cls_id} "
                    f"class_code={class_code or '-'} "
                    f"coarse={coarse_class or '-'} fine={fine_class or '-'} "
                    f"target={SHIP_SUBTYPES[target_idx]}"
                )

    print(f"[ShipCls] Extracted {len(chips)} ship chips")

    # Count per class
    counts = np.zeros(len(SHIP_SUBTYPES), dtype=int)
    for _, lbl in chips:
        counts[lbl] += 1
    for i, name in enumerate(SHIP_SUBTYPES):
        if counts[i] > 0:
            print(f"  {name}: {counts[i]}")
    print(f"[ShipCls][DEBUG] Sample label mappings (first {len(debug_examples)}):")
    for line in debug_examples:
        print(f"  {line}")

    # Split
    rng = np.random.RandomState(42)
    train, val = [], []
    for lbl in range(len(SHIP_SUBTYPES)):
        lbl_samples = [c for c in chips if c[1] == lbl]
        if not lbl_samples:
            continue
        rng.shuffle(lbl_samples)
        n_val = max(1, int(len(lbl_samples) * 0.15))
        if len(lbl_samples) - n_val < 1:
            n_val = max(0, len(lbl_samples) - 1)
        train.extend(lbl_samples[n_val:])
        val.extend(lbl_samples[:n_val])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def compute_class_weights(samples: list, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label in samples:
        counts[label] += 1
    counts = np.maximum(counts, 1)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_counts(samples: list, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in samples:
        counts[label] += 1
    return counts


def build_balanced_sampler(samples: list, num_classes: int, power: float = 1.0) -> WeightedRandomSampler:
    counts = np.maximum(compute_class_counts(samples, num_classes), 1)
    sample_weights = [1.0 / float(counts[label]) ** power for _, label in samples]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(samples),
        replacement=True,
    )


def build_model(num_classes: int = 16) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {"loss": total_loss / len(loader.dataset), "accuracy": correct / total}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    pred_hist = np.bincount(all_preds, minlength=len(SHIP_SUBTYPES)).tolist()
    label_hist = np.bincount(all_labels, minlength=len(SHIP_SUBTYPES)).tolist()
    confusion = np.zeros((len(SHIP_SUBTYPES), len(SHIP_SUBTYPES)), dtype=np.int64)
    for true, pred in zip(all_labels, all_preds):
        confusion[true, pred] += 1
    f1_scores = []
    confusion_items = []
    per_class_recall = []
    for true_idx in range(len(SHIP_SUBTYPES)):
        tp = int(confusion[true_idx, true_idx])
        fn = int(confusion[true_idx].sum() - tp)
        fp = int(confusion[:, true_idx].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        if label_hist[true_idx] > 0:
            per_class_recall.append((rec, SHIP_SUBTYPES[true_idx], label_hist[true_idx]))
        for pred_idx in range(len(SHIP_SUBTYPES)):
            if true_idx == pred_idx or confusion[true_idx, pred_idx] == 0:
                continue
            confusion_items.append(
                (int(confusion[true_idx, pred_idx]), f"{SHIP_SUBTYPES[true_idx]}->{SHIP_SUBTYPES[pred_idx]}")
            )
    confusion_items.sort(reverse=True)
    per_class_recall.sort(key=lambda x: x[0])
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": correct / total,
        "macro_f1": float(sum(f1_scores) / len(f1_scores)),
        "pred_hist": pred_hist,
        "label_hist": label_hist,
        "top_confusions": confusion_items[:8],
        "worst_recalls": per_class_recall[:5],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Ship Fine-grained Classifier")
    parser.add_argument("--data-root", required=True,
                        help="Path to data_script_run directory")
    parser.add_argument("--output", "--output-weights", dest="output", default="ship_cls_resnet18.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--sampler-power", type=float, default=1.0)
    parser.add_argument("--no-balanced-sampler", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[ShipCls] Device: {device}")
    print(f"[ShipCls] Target classes ({len(SHIP_SUBTYPES)}): {SHIP_SUBTYPES}")

    train_samples, val_samples = collect_ship_chips(args.data_root)
    if not train_samples:
        print("[ERROR] No ship chips found!")
        sys.exit(1)
    print(f"[ShipCls] Train: {len(train_samples)}, Val: {len(val_samples)}")

    class_weights = compute_class_weights(train_samples, len(SHIP_SUBTYPES)).to(device)
    train_class_counts = compute_class_counts(train_samples, len(SHIP_SUBTYPES))
    val_class_counts = compute_class_counts(val_samples, len(SHIP_SUBTYPES))
    print(f"[ShipCls] Class weights: {class_weights.tolist()}")
    print(f"[ShipCls] Train class counts: {train_class_counts.tolist()}")
    print(f"[ShipCls] Val class counts: {val_class_counts.tolist()}")

    train_ds = ShipChipDataset(train_samples, augment=True)
    val_ds = ShipChipDataset(val_samples, augment=False)
    print(f"[ShipCls] Train augmentations: {describe_transform(train_ds.transform)}")
    print(f"[ShipCls] Val augmentations: {describe_transform(val_ds.transform)}")
    train_sampler = None
    if not args.no_balanced_sampler:
        train_sampler = build_balanced_sampler(train_samples, len(SHIP_SUBTYPES), power=args.sampler_power)
        print(f"[ShipCls] Using WeightedRandomSampler(power={args.sampler_power})")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=2,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(len(SHIP_SUBTYPES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_acc, best_state, patience_counter = 0.0, None, 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = validate(model, val_loader, device)
        acc = metrics["accuracy"]
        scheduler.step()
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={metrics['loss']:.4f} | val_acc={acc:.4f} | val_f1={metrics['macro_f1']:.4f}"
        )
        print(f"[ShipCls][VAL] pred_hist={metrics['pred_hist']} label_hist={metrics['label_hist']}")
        if metrics["top_confusions"]:
            print(f"[ShipCls][VAL] top_confusions={metrics['top_confusions']}")
        if metrics["worst_recalls"]:
            print(f"[ShipCls][VAL] worst_recalls={metrics['worst_recalls']}")
        improved = best_state is None or (acc - best_acc) > args.min_delta
        if improved:
            best_acc, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch >= args.min_epochs and patience_counter >= args.patience:
            print(f"[ShipCls] Early stopping at epoch {epoch}")
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state or model.state_dict(),
                "classes": SHIP_SUBTYPES, "chip_size": CHIP_SIZE,
                "best_acc": best_acc}, str(out_path))
    print(f"[ShipCls] Saved to {out_path} (acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
