"""
classifier package — Fine-grained SAR target classifiers (Branch A, M2 stage).

Modules
-------
roi_crop            : OBB ROI extraction from detection results.
scene_gate          : ResNet-18 3-class scene gate (harbor / airport / none).
ship_classifier     : ResNet-18 16-class ship subtype classifier.
aircraft_classifier : ResNet-18 5-class aircraft subtype classifier.

Training scripts (offline, for air-gapped machine)
---------------------------------------------------
train_gate.py        : Train the scene gate classifier.
train_ship_cls.py    : Train the ship fine-grained classifier.
train_aircraft_cls.py: Train the aircraft fine-grained classifier.
"""

from .roi_crop import crop_obb_roi, batch_crop_obb_roi
from .scene_gate import SceneGate
from .ship_classifier import ShipClassifier
from .aircraft_classifier import AircraftClassifier

__all__ = [
    "crop_obb_roi",
    "batch_crop_obb_roi",
    "SceneGate",
    "ShipClassifier",
    "AircraftClassifier",
]
