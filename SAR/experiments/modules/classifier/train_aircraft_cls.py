#!/usr/bin/env python3
"""
train_aircraft_cls.py — Train the ResNet-18 aircraft fine-grained classifier (M2b, offline).

Same data_script_run format as train_ship_cls.py but extracts aircraft ROIs.

Usage
-----
    python train_aircraft_cls.py \
        --data-root /path/to/data_script_run \
        --output weights/aircraft_cls_resnet18.pt \
        --epochs 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    from .aircraft_classifier import AIRCRAFT_SUBTYPES
except ImportError:
    from aircraft_classifier import AIRCRAFT_SUBTYPES

CHIP_SIZE = 128
DEBUG_MAX_EXAMPLES = 20

_AIRCRAFT_CLASS_ID_TO_CODE = {
    8: "combat_aircraft",
    9: "transport_aircraft",
    17: "combat_support_aircraft",
    18: "helicopter",
    21: "other_aircraft",
}

_AIRCRAFT_COARSE = {"飞机", "作战飞机", "运输机", "作战支援飞机", "直升机", "飞机_其他",
                    "aircraft", "fighter", "bomber", "transport", "helicopter",
                    "aew", "awacs"}


class AircraftChipDataset(Dataset):
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


def describe_transform(transform):
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


def collect_aircraft_chips(data_root: str) -> tuple[list, list]:
    root = Path(data_root)
    det_img_dir = root / "detection" / "train" / "images"
    det_lbl_dir = root / "detection" / "train" / "labels"
    det_meta = root / "detection" / "metadata" / "samples.jsonl"

    if not det_img_dir.is_dir():
        print(f"[ERROR] Image dir not found: {det_img_dir}")
        sys.exit(1)

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

    fine_to_idx: dict[str, int] = {n: i for i, n in enumerate(AIRCRAFT_SUBTYPES)}
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
    temp_dir = Path("temp_aircraft_chips")
    temp_dir.mkdir(exist_ok=True)
    debug_examples: list[str] = []

    img_files = sorted(det_img_dir.glob("*.jpg")) + sorted(det_img_dir.glob("*.png"))
    print(f"[AircraftCls] Scanning {len(img_files)} images for aircraft instances...")

    for img_path in img_files:
        lbl_path = det_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

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

        meta = img_meta.get(img_path.name, {})
        instances = meta.get("instances", [])

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        H, W = image.shape

        for idx, (cls_id, cx, cy, w, h) in enumerate(lines):
            fine_class = None
            coarse_class = ""
            if idx < len(instances):
                inst = instances[idx]
                coarse_class = inst.get("coarse_class", "")
                fine_class = inst.get("fine_class")

            class_code = _AIRCRAFT_CLASS_ID_TO_CODE.get(int(cls_id))
            is_aircraft = class_code is not None
            if not is_aircraft and coarse_class in _AIRCRAFT_COARSE:
                is_aircraft = True
            if not is_aircraft and fine_class and fine_class in _AIRCRAFT_COARSE:
                is_aircraft = True

            if not is_aircraft:
                continue

            target_idx = fine_to_idx.get(class_code) if class_code else None
            if target_idx is None:
                target_idx = fine_to_idx.get(fine_class) if fine_class else None
            if target_idx is None:
                target_idx = fine_to_idx.get(coarse_class)
            if target_idx is None:
                target_idx = fine_to_idx["other_aircraft"]

            x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
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
                    f"img={img_path.name} det_idx={idx} cls_id={cls_id} "
                    f"class_code={class_code or '-'} "
                    f"coarse={coarse_class or '-'} fine={fine_class or '-'} "
                    f"target={AIRCRAFT_SUBTYPES[target_idx]}"
                )

    print(f"[AircraftCls] Extracted {len(chips)} aircraft chips")
    counts = np.zeros(len(AIRCRAFT_SUBTYPES), dtype=int)
    for _, lbl in chips:
        counts[lbl] += 1
    for i, name in enumerate(AIRCRAFT_SUBTYPES):
        if counts[i] > 0:
            print(f"  {name}: {counts[i]}")
    print(f"[AircraftCls][DEBUG] Sample label mappings (first {len(debug_examples)}):")
    for line in debug_examples:
        print(f"  {line}")

    rng = np.random.RandomState(42)
    train, val = [], []
    for lbl in range(len(AIRCRAFT_SUBTYPES)):
        lbl_samples = [c for c in chips if c[1] == lbl]
        if not lbl_samples:
            continue
        rng.shuffle(lbl_samples)
        n_val = max(1, int(len(lbl_samples) * 0.15))
        if len(lbl_samples) - n_val < 1:
            n_val = max(0, len(lbl_samples) - 1)
        train.extend(lbl_samples[n_val:])
        val.extend(lbl_samples[:n_val])
    rng.shuffle(train); rng.shuffle(val)
    return train, val


def compute_class_weights(samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, l in samples:
        counts[l] += 1
    counts = np.maximum(counts, 1)
    w = 1.0 / np.sqrt(counts)
    return torch.tensor(w / w.sum() * num_classes, dtype=torch.float32)


def compute_class_counts(samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, l in samples:
        counts[l] += 1
    return counts


def build_balanced_sampler(samples, num_classes, power=1.0):
    counts = np.maximum(compute_class_counts(samples, num_classes), 1)
    sample_weights = [1.0 / float(counts[label]) ** power for _, label in samples]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(samples),
        replacement=True,
    )


def build_model(num_classes=5):
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss = 0.0
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {"loss": total_loss / len(loader.dataset), "accuracy": correct / total}


@torch.no_grad()
def validate(model, loader, device):
    model.eval(); correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
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
    pred_hist = np.bincount(all_preds, minlength=len(AIRCRAFT_SUBTYPES)).tolist()
    label_hist = np.bincount(all_labels, minlength=len(AIRCRAFT_SUBTYPES)).tolist()
    confusion = np.zeros((len(AIRCRAFT_SUBTYPES), len(AIRCRAFT_SUBTYPES)), dtype=np.int64)
    for true, pred in zip(all_labels, all_preds):
        confusion[true, pred] += 1
    f1_scores = []
    confusion_items = []
    per_class_recall = []
    for true_idx in range(len(AIRCRAFT_SUBTYPES)):
        tp = int(confusion[true_idx, true_idx])
        fn = int(confusion[true_idx].sum() - tp)
        fp = int(confusion[:, true_idx].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        if label_hist[true_idx] > 0:
            per_class_recall.append((rec, AIRCRAFT_SUBTYPES[true_idx], label_hist[true_idx]))
        for pred_idx in range(len(AIRCRAFT_SUBTYPES)):
            if true_idx == pred_idx or confusion[true_idx, pred_idx] == 0:
                continue
            confusion_items.append(
                (int(confusion[true_idx, pred_idx]), f"{AIRCRAFT_SUBTYPES[true_idx]}->{AIRCRAFT_SUBTYPES[pred_idx]}")
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
    parser = argparse.ArgumentParser(description="Train Aircraft Fine-grained Classifier")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", "--output-weights", dest="output", default="aircraft_cls_resnet18.pt")
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
    print(f"[AircraftCls] Device: {device}")
    print(f"[AircraftCls] Target classes ({len(AIRCRAFT_SUBTYPES)}): {AIRCRAFT_SUBTYPES}")

    train_samples, val_samples = collect_aircraft_chips(args.data_root)
    if not train_samples:
        print("[ERROR] No aircraft chips found!"); sys.exit(1)
    print(f"[AircraftCls] Train: {len(train_samples)}, Val: {len(val_samples)}")

    class_weights = compute_class_weights(train_samples, len(AIRCRAFT_SUBTYPES)).to(device)
    train_class_counts = compute_class_counts(train_samples, len(AIRCRAFT_SUBTYPES))
    val_class_counts = compute_class_counts(val_samples, len(AIRCRAFT_SUBTYPES))
    print(f"[AircraftCls] Class weights: {class_weights.tolist()}")
    print(f"[AircraftCls] Train class counts: {train_class_counts.tolist()}")
    print(f"[AircraftCls] Val class counts: {val_class_counts.tolist()}")
    train_ds = AircraftChipDataset(train_samples, augment=True)
    val_ds = AircraftChipDataset(val_samples, augment=False)
    print(f"[AircraftCls] Train augmentations: {describe_transform(train_ds.transform)}")
    print(f"[AircraftCls] Val augmentations: {describe_transform(val_ds.transform)}")
    train_sampler = None
    if not args.no_balanced_sampler:
        train_sampler = build_balanced_sampler(train_samples, len(AIRCRAFT_SUBTYPES), power=args.sampler_power)
        print(f"[AircraftCls] Using WeightedRandomSampler(power={args.sampler_power})")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=2,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(len(AIRCRAFT_SUBTYPES)).to(device)
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
        print(f"[AircraftCls][VAL] pred_hist={metrics['pred_hist']} label_hist={metrics['label_hist']}")
        if metrics["top_confusions"]:
            print(f"[AircraftCls][VAL] top_confusions={metrics['top_confusions']}")
        if metrics["worst_recalls"]:
            print(f"[AircraftCls][VAL] worst_recalls={metrics['worst_recalls']}")
        improved = best_state is None or (acc - best_acc) > args.min_delta
        if improved:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch >= args.min_epochs and patience_counter >= args.patience:
            print(f"[AircraftCls] Early stopping at epoch {epoch}"); break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state or model.state_dict(),
                "classes": AIRCRAFT_SUBTYPES, "chip_size": CHIP_SIZE,
                "best_acc": best_acc}, str(out_path))
    print(f"[AircraftCls] Saved to {out_path} (acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
