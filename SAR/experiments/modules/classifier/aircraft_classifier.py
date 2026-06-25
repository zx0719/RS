"""
aircraft_classifier.py — M2b: ResNet-18 fine-grained aircraft subtype classifier.

Classifies cropped SAR aircraft ROIs into 5 subtypes.

Usage
-----
    from modules.classifier import AircraftClassifier

    clf = AircraftClassifier("weights/aircraft_cls_resnet18.pt")
    clf.classify_batch("scene.tif", aircraft_objects)
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
# 5 aircraft subtypes (matching user's 23-class system)
# ---------------------------------------------------------------------------

AIRCRAFT_SUBTYPES: list[str] = [
    "combat_aircraft",       # 8  作战飞机
    "transport_aircraft",    # 9  运输机
    "combat_support_aircraft",  # 17 作战支援飞机
    "helicopter",            # 18 直升机
    "other_aircraft",        # 21 飞机_其他
]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_resnet18_aircraft(num_classes: int = 5) -> nn.Module:
    """Build a ResNet-18 for single-channel SAR aircraft classification."""
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AircraftClassifier:
    """ResNet-18 fine-grained aircraft subtype classifier (M2b).

    Parameters
    ----------
    model_path  : Path to trained weights (.pt).
    device      : PyTorch device string.
    score_thresh: Minimum confidence to accept fine-grained label.
    chip_size   : ROI chip size (square pixels).
    """

    CLASSIFIER_VERSION: str = "resnet18-aircraft-5class-v1.0"

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

        self._model = _build_resnet18_aircraft(num_classes=len(AIRCRAFT_SUBTYPES))
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
        """Classify a batch of aircraft objects from the same source image.

        Crops each aircraft's OBB ROI, runs inference, and mutates each
        object dict in-place.

        Parameters
        ----------
        image_uri : Path to the source SAR image.
        objects   : List of Evidence Package object dicts (all super_class="aircraft").

        Returns
        -------
        The mutated objects list.
        """
        if not objects:
            return objects

        chips = batch_crop_obb_roi(image_uri, objects, chip_size=self.chip_size)

        tensor = np.stack(chips, axis=0)
        tensor = torch.from_numpy(tensor).unsqueeze(1).to(self.device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        pred_ids = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(pred_ids)), pred_ids]

        for i, obj in enumerate(objects):
            conf = float(confidences[i])
            pred = int(pred_ids[i])

            if conf >= self.score_thresh:
                subtype = AIRCRAFT_SUBTYPES[pred]
                obj["class"]["code"] = subtype
                obj["class"]["name_cn"] = subtype
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
        """Classify a single aircraft object (convenience wrapper)."""
        return self.classify_batch(image_uri, [obj])[0]
