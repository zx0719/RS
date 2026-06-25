"""
scene_gate.py — Branch B Gate: ResNet-18 3-class scene classifier.

Determines whether the input SAR image contains a harbor or airport,
triggering FastSAM segmentation only when needed.

Classes: harbor / airport / none

Usage
-----
    from modules.classifier import SceneGate

    gate = SceneGate("weights/gate_resnet18.pt")
    result = gate.predict("scene.tif")
    # {"scene_class": "harbor", "confidence": 0.87, "all_probs": {...}}
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional torch / torchvision imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torchvision
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn(
        "torch / torchvision are not installed. SceneGate requires PyTorch.",
        ImportWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def _build_resnet18_gate(num_classes: int = 3) -> nn.Module:
    """Build a ResNet-18 for single-channel SAR scene classification."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required: pip install torch torchvision")

    model = torchvision.models.resnet18(weights=None)
    # Replace first conv for single-channel (grayscale SAR)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Replace FC head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENE_CLASSES: list[str] = ["harbor", "airport", "none"]

# ID → Evidence code mapping
_SCENE_ID_TO_CODE: dict[int, str] = {0: "harbor", 1: "airport", 2: "none"}
_CODE_TO_SCENE_ID: dict[str, int] = {"harbor": 0, "airport": 1, "none": 2}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SceneGate:
    """ResNet-18 scene classifier gate for Branch B.

    Determines whether to trigger FastSAM by classifying the input SAR
    image as harbor / airport / none.

    Parameters
    ----------
    model_path       : Path to trained ResNet-18 weights (.pt).
    device           : PyTorch device string.
    confidence_thresh: Minimum confidence for harbor/airport trigger.
    input_size       : Image resize dimension (square).
    """

    GATE_VERSION: str = "resnet18-scene-gate-v1.0"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        confidence_thresh: float = 0.5,
        input_size: int = 512,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required: pip install torch torchvision")

        self.model_path = Path(model_path)
        self.device = device
        self.confidence_thresh = confidence_thresh
        self.input_size = input_size

        # Build model and load weights
        self._model = _build_resnet18_gate(num_classes=len(SCENE_CLASSES))
        state = torch.load(str(self.model_path), map_location=device, weights_only=False)
        # Handle training checkpoint dict vs raw state_dict
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self._model.load_state_dict(state, strict=True)
        self._model.to(device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_uri: str | Path) -> dict[str, Any]:
        """Classify a single SAR image.

        Parameters
        ----------
        image_uri : Path to the SAR image file.

        Returns
        -------
        dict with keys:
            scene_class   : "harbor" | "airport" | "none"
            confidence    : float (0-1)
            all_probs     : {"harbor": p, "airport": p, "none": p}
            gate_decision : bool (True = trigger FastSAM)
            model_version : str
        """
        import cv2

        # Load and preprocess
        img = cv2.imread(str(image_uri), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_uri}")

        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0

        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        tensor = tensor.to(self.device)

        # Inference
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])

        scene_class = _SCENE_ID_TO_CODE.get(pred_id, "none")

        # Apply threshold: if best is harbor/airport but below threshold, default to none
        if scene_class in ("harbor", "airport") and confidence < self.confidence_thresh:
            scene_class = "none"
            confidence = float(probs[_CODE_TO_SCENE_ID["none"]])

        gate_decision = scene_class in ("harbor", "airport")

        return {
            "scene_class": scene_class,
            "confidence": confidence,
            "all_probs": {
                "harbor": float(probs[_CODE_TO_SCENE_ID["harbor"]]),
                "airport": float(probs[_CODE_TO_SCENE_ID["airport"]]),
                "none": float(probs[_CODE_TO_SCENE_ID["none"]]),
            },
            "gate_decision": gate_decision,
            "model_version": self.GATE_VERSION,
        }
