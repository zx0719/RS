#!/usr/bin/env python3
"""
train_gate.py — Train the ResNet-18 scene gate classifier (offline).

Reads the data_script_run directory structure:
  detection/   → train/images/ + train/labels/ (YOLO HBB format)
  segmentation/ → train/masks/ + train/polygons/ + train/metadata/samples.jsonl

Trains a 3-class SAR scene classifier: harbor / airport / none.

Usage
-----
    python train_gate.py \
        --data-root /path/to/data_script_run \
        --output weights/gate_resnet18.pt \
        --epochs 50 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCENE_CLASSES = ["harbor", "airport", "none"]
INPUT_SIZE = 512
DEBUG_MAX_EXAMPLES = 20

# Coarse class name → gate label
_HARBOR_KEYWORDS = {"港口", "harbor", "dock", "pier", "berth", "军港", "码头"}
_AIRPORT_KEYWORDS = {"机场", "airport", "airbase", "airfield", "跑道", "停机坪",
                     "滑行道", "机库", "机棚", "塔台", "弹药库", "端保险道", "联络道",
                     "飞机掩蔽库", "runway", "taxiway", "apron", "hangar"}
_HARBOR_CLASS_ID = 0
_AIRPORT_CLASS_ID = 22


def _determine_scene_label(seg_json_path: Path) -> int:
    """Check segmentation polygons to determine if scene is harbor/airport/none."""
    if not seg_json_path.exists():
        return 2  # none
    with open(seg_json_path) as f:
        polygons = json.load(f)
    if not polygons:
        return 2
    if isinstance(polygons, dict):
        polygons = [polygons]

    # Prefer explicit class ids when available.
    class_ids: set[int] = set()
    for p in polygons:
        if not isinstance(p, dict):
            continue
        raw_class_id = p.get("class_id", p.get("class__id"))
        if raw_class_id is None:
            continue
        try:
            class_ids.add(int(raw_class_id))
        except (TypeError, ValueError):
            continue
    if _AIRPORT_CLASS_ID in class_ids:
        return 1
    if _HARBOR_CLASS_ID in class_ids:
        return 0

    for p in polygons:
        if not isinstance(p, dict):
            continue
        coarse = str(p.get("coarse_class", ""))
        fine = str(p.get("fine_class", ""))
        # Token-based matching (avoid substring false positives like "port" in "airport")
        tokens = set((coarse + " " + fine).lower().split())
        # Check airport first (more specific), then harbor
        for kw in _AIRPORT_KEYWORDS:
            if kw in tokens:
                return 1  # airport
        for kw in _HARBOR_KEYWORDS:
            if kw in tokens and kw not in _AIRPORT_KEYWORDS:
                return 0  # harbor
    return 2  # none


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SceneGateDataset(Dataset):
    """Load SAR scene tiles and their gate labels from data_script_run structure."""

    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform or self._default_transform()

    @staticmethod
    def _default_transform():
        return T.Compose([
            T.ToPILImage(),
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
        ])

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
# Data loading — reads the actual data_script_run layout
# ---------------------------------------------------------------------------
def collect_samples(data_root: str) -> tuple[list, list]:
    """Walk the data_script_run directory and collect (image_path, label) pairs.

    Strategy:
      1. Read `manifests/detection_train.txt` for the image list.
      2. For each detection image, find its source_tif from samples.jsonl.
      3. Look up the corresponding segmentation polygon file.
      4. Determine the scene label from the polygon's coarse_class.
    """
    root = Path(data_root)
    det_dir = root / "detection" / "train"
    seg_poly_dir = root / "segmentation" / "train" / "polygons"
    det_meta = root / "detection" / "metadata" / "samples.jsonl"

    # Build image → source_tif mapping from samples.jsonl
    img_to_source: dict[str, str] = {}
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
                source = rec.get("source_tif", "")
                img_to_source[img_name] = source

    # Collect image paths
    img_dir = det_dir / "images"
    if not img_dir.is_dir():
        print(f"[ERROR] Image directory not found: {img_dir}")
        sys.exit(1)

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_paths:
        img_paths = sorted(img_dir.glob("*.bmp"))
    print(f"[Gate] Found {len(img_paths)} detection images")

    # Determine scene label for each image
    samples: list[tuple[Path, int]] = []
    label_counts = {0: 0, 1: 0, 2: 0}
    debug_examples: list[str] = []

    for img_path in img_paths:
        img_name = img_path.name
        source_tif = img_to_source.get(img_name, "")
        seg_json = None
        if source_tif:
            # Find segmentation polygon for this source
            source_stem = Path(source_tif).stem
            seg_json = seg_poly_dir / f"{source_stem}.json"
            label = _determine_scene_label(seg_json)
        else:
            label = 2  # none — no segmentation data

        samples.append((img_path, label))
        label_counts[label] += 1
        if len(debug_examples) < DEBUG_MAX_EXAMPLES:
            debug_examples.append(
                f"img={img_name} source_tif={source_tif or '-'} "
                f"seg_json={(seg_json.name if seg_json else '-')} "
                f"label={SCENE_CLASSES[label]}"
            )

    print(f"[Gate] Label distribution: harbor={label_counts[0]}, "
          f"airport={label_counts[1]}, none={label_counts[2]}")
    print(f"[Gate][DEBUG] Sample label mappings (first {len(debug_examples)}):")
    for line in debug_examples:
        print(f"  {line}")

    # Split train/val (stratified by label)
    rng = np.random.RandomState(42)
    train, val = [], []
    for lbl in range(3):
        lbl_samples = [s for s in samples if s[1] == lbl]
        rng.shuffle(lbl_samples)
        n_val = max(1, int(len(lbl_samples) * 0.15))
        # Ensure at least 1 sample stays in training set
        if len(lbl_samples) - n_val < 1:
            n_val = max(0, len(lbl_samples) - 1)
        train.extend(lbl_samples[n_val:])
        val.extend(lbl_samples[:n_val])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def compute_class_weights(samples: list) -> torch.Tensor:
    counts = np.zeros(len(SCENE_CLASSES), dtype=np.float64)
    for _, label in samples:
        counts[label] += 1
    counts = np.maximum(counts, 1)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * len(SCENE_CLASSES)
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int = 3) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
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
    # Pure-numpy macro F1 (no sklearn dependency)
    f1_scores = []
    for c in range(len(SCENE_CLASSES)):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == l == c)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != c and l == c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
    f1 = sum(f1_scores) / len(f1_scores)
    pred_hist = np.bincount(all_preds, minlength=len(SCENE_CLASSES)).tolist()
    label_hist = np.bincount(all_labels, minlength=len(SCENE_CLASSES)).tolist()
    confusion = np.zeros((len(SCENE_CLASSES), len(SCENE_CLASSES)), dtype=np.int64)
    for true, pred in zip(all_labels, all_preds):
        confusion[true, pred] += 1
    confusion_items = []
    for true_idx in range(len(SCENE_CLASSES)):
        for pred_idx in range(len(SCENE_CLASSES)):
            if true_idx == pred_idx or confusion[true_idx, pred_idx] == 0:
                continue
            confusion_items.append(
                (int(confusion[true_idx, pred_idx]), f"{SCENE_CLASSES[true_idx]}->{SCENE_CLASSES[pred_idx]}")
            )
    confusion_items.sort(reverse=True)
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": correct / total,
        "macro_f1": float(f1),
        "pred_hist": pred_hist,
        "label_hist": label_hist,
        "top_confusions": confusion_items[:5],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Scene Gate classifier")
    parser.add_argument("--data-root", required=True,
                        help="Path to data_script_run directory")
    parser.add_argument("--output", "--output-weights", dest="output", default="gate_resnet18.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--sampler-power", type=float, default=1.0)
    parser.add_argument("--no-balanced-sampler", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Gate] Device: {device}")
    print(f"[Gate] Classes: {SCENE_CLASSES}")
    print(f"[Gate] Data root: {args.data_root}")

    train_samples, val_samples = collect_samples(args.data_root)
    print(f"[Gate] Train: {len(train_samples)}, Val: {len(val_samples)}")

    class_weights = compute_class_weights(train_samples).to(device)
    train_class_counts = compute_class_counts(train_samples, len(SCENE_CLASSES))
    val_class_counts = compute_class_counts(val_samples, len(SCENE_CLASSES))
    print(f"[Gate] Class weights: {class_weights.tolist()}")
    print(f"[Gate] Train class counts: {train_class_counts.tolist()}")
    print(f"[Gate] Val class counts: {val_class_counts.tolist()}")

    train_ds = SceneGateDataset(train_samples)
    val_ds = SceneGateDataset(val_samples)
    print(f"[Gate] Train augmentations: {describe_transform(train_ds.transform)}")
    print(f"[Gate] Val augmentations: {describe_transform(val_ds.transform)}")
    train_sampler = None
    if not args.no_balanced_sampler:
        train_sampler = build_balanced_sampler(train_samples, len(SCENE_CLASSES), power=args.sampler_power)
        print(f"[Gate] Using WeightedRandomSampler(power={args.sampler_power})")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=2,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1, best_state, patience_counter = 0.0, None, 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = validate(model, val_loader, device)
        scheduler.step()
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={metrics['loss']:.4f} | val_acc={metrics['accuracy']:.4f} | "
            f"val_f1={metrics['macro_f1']:.4f}"
        )
        print(f"[Gate][VAL] pred_hist={metrics['pred_hist']} label_hist={metrics['label_hist']}")
        if metrics["top_confusions"]:
            print(f"[Gate][VAL] top_confusions={metrics['top_confusions']}")
        improved = best_state is None or (metrics["macro_f1"] - best_f1) > args.min_delta
        if improved:
            best_f1 = metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch >= args.min_epochs and patience_counter >= args.patience:
            print(f"[Gate] Early stopping at epoch {epoch}")
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state or model.state_dict(),
        "classes": SCENE_CLASSES,
        "input_size": INPUT_SIZE,
        "best_f1": best_f1,
    }, str(out_path))
    print(f"[Gate] Saved to {out_path} (F1={best_f1:.4f})")


if __name__ == "__main__":
    main()
