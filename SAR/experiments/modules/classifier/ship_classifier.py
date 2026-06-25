"""
ship_classifier.py — M2a: ResNet-18 fine-grained ship subtype classifier.

Classifies cropped SAR ship ROIs into 16 subtypes.

Usage
-----
    from modules.classifier import ShipClassifier

    clf = ShipClassifier("weights/ship_cls_resnet18.pt")
    clf.classify_batch("scene.tif", ship_objects)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .roi_crop import batch_crop_obb_roi
except ImportError:
    from roi_crop import batch_crop_obb_roi  # fallback for standalone scripts

# ---------------------------------------------------------------------------
# Conditional torch imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torchvision
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# 16 ship subtypes (matching user's 23-class system)
# ---------------------------------------------------------------------------

SHIP_SUBTYPES: list[str] = [
    "military_auxiliary",   # 1  军用辅助舰船
    "combat_ship",          # 2  作战舰船
    "liquid_cargo",         # 3  液货船
    "harbor_service",       # 4  港务船
    "bulk_carrier",         # 5  散货船
    "other_vessel",         # 6  舰船_其他
    "survey_vessel",        # 7  调查船
    "amphibious",           # 10 两栖舰船
    "container_ship",       # 11 集装箱船
    "engineering_vessel",   # 12 工程船
    "fishing_vessel",       # 13 渔船
    "tug_boat",             # 14 拖船
    "passenger_ship",       # 15 客船
    "ro_ro_ship",           # 16 滚装船
    "sailing_vessel",       # 19 帆船
    "research_vessel",      # 20 研究船
]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_resnet18_ship(num_classes: int = 16) -> nn.Module:
    """Build a ResNet-18 for single-channel SAR ship classification."""
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ShipClassifier:
    """ResNet-18 fine-grained ship subtype classifier (M2a).

    Parameters
    ----------
    model_path  : Path to trained weights (.pt).
    device      : PyTorch device string.
    score_thresh: Minimum confidence to accept fine-grained label.
    chip_size   : ROI chip size (square pixels).
    """

    CLASSIFIER_VERSION: str = "resnet18-ship-16class-v1.0"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        score_thresh: float = 0.3,
        chip_size: int = 128,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required: pip install torch torchvision")

        self.model_path = Path(model_path)
        self.device = device
        self.score_thresh = score_thresh
        self.chip_size = chip_size

        self._model = _build_resnet18_ship(num_classes=len(SHIP_SUBTYPES))
        state = torch.load(str(self.model_path), map_location=device, weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self._model.load_state_dict(state, strict=True)
        self._model.to(device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_batch(
        self,
        image_uri: str | Path,
        objects: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Classify a batch of ship objects from the same source image.

        Crops each ship's OBB ROI, runs inference, and mutates each object
        dict in-place to update ``class.code``, ``score.confidence``, and
        ``evidence.classifier_version``.

        Objects whose classification confidence is below ``score_thresh``
        retain their coarse "ship" label.

        Parameters
        ----------
        image_uri : Path to the source SAR image.
        objects   : List of Evidence Package object dicts (all super_class="ship").

        Returns
        -------
        The mutated objects list.
        """
        if not objects:
            return objects

        # 1. Batch crop all ship ROIs
        chips = batch_crop_obb_roi(image_uri, objects, chip_size=self.chip_size)

        # 2. Stack into tensor
        tensor = np.stack(chips, axis=0)  # [N, H, W]
        tensor = torch.from_numpy(tensor).unsqueeze(1).to(self.device)  # [N, 1, H, W]

        # 3. Inference
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # [N, 16]

        pred_ids = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(pred_ids)), pred_ids]

        # 4. Update objects in-place
        for i, obj in enumerate(objects):
            conf = float(confidences[i])
            pred = int(pred_ids[i])

            if conf >= self.score_thresh:
                subtype = SHIP_SUBTYPES[pred]
                obj["class"]["code"] = subtype
                obj["class"]["name_cn"] = subtype  # will be resolved later by class_labels
                obj["score"]["confidence"] = round(conf, 6)
                obj["score"]["classifier_confidence"] = round(conf, 6)

            obj.setdefault("evidence", {})
            obj["evidence"]["classifier_version"] = self.CLASSIFIER_VERSION

        return objects

    def classify_single(
        self,
        image_uri: str | Path,
        obj: dict[str, Any],
    ) -> dict[str, Any]:
        """Classify a single ship object (convenience wrapper)."""
        return self.classify_batch(image_uri, [obj])[0]
