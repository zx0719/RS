"""
modules/detector — M2 Detection module

Public API
----------
    DetectorTool   Main inference class.  Import and use directly:

        from modules.detector import DetectorTool

        tool = DetectorTool("runs/detect/weights/best.pt", device="cuda:0")
        objects = tool.detect("scene.tif")   # returns list[dict] for Evidence Package

    MockDetector   Synthetic detector for testing (no ultralytics required):

        from modules.detector import MockDetector

        tool = MockDetector(n_ships=2, n_aircraft=1, seed=42)
        objects = tool.detect("scene.tif")

Also re-exports the class map utilities for downstream use:

        from modules.detector import get_class_descriptor, CLASS_MAP
"""

from .detector import DetectorTool
from .mock_detector import MockDetector
from .class_map import CLASS_MAP, UNKNOWN_CLASS, get_class_descriptor

__all__ = [
    "DetectorTool",
    "MockDetector",
    "CLASS_MAP",
    "UNKNOWN_CLASS",
    "get_class_descriptor",
]
