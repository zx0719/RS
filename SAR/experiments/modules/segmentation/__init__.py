"""
segmentation package — Area segmentation for harbor/airport (Branch B, M3).

Modules
-------
fastsam_segmenter : FastSAM zero-shot segmentation + area estimation.
"""

from .fastsam_segmenter import FastSAMSegmenter

__all__ = ["FastSAMSegmenter"]
