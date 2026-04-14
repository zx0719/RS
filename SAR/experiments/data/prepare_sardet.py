#!/usr/bin/env python3
"""
prepare_sardet.py — STUB for SARDet_100K → YOLO-OBB conversion.

STATUS: STUB — full implementation pending OBB annotation availability.

SARDet_100K format overview
---------------------------
- Path:   /mnt/data/mm_data/SAR/SARDet_100K/
- Format: COCO JSON (instances_*.json), horizontal bounding boxes only (HBB).
- Classes: ship, aircraft, car, tank, bridge  (5 classes)
- COCO bbox format: [x_min, y_min, width, height]  (NOT oriented)

Why this is not yet implemented
--------------------------------
SARDet_100K currently provides only axis-aligned (horizontal) bounding boxes.
YOLOv8-OBB requires oriented bounding boxes (OBB) with a rotation angle.

Two future options:
  A. Use SARDet as an HBB dataset with YOLOv8-det (not OBB model).
     → Simply convert COCO bbox to YOLO HBB format: class cx cy w h  (normalised).
     → Fast to implement, no angle information.

  B. Obtain or generate OBB annotations for SARDet (e.g. via SAM-assisted
     re-annotation, or by using a pretrained OBB model to pseudo-label).
     → Required for mixing with SSDD-OBB in a single OBB training run.

Stub function signatures are provided below so the interface is agreed before
the data is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SARDET_CLASSES: list[str] = ["ship", "aircraft", "car", "tank", "bridge"]

SARDET_DEFAULT_ROOT = Path(
    "/mnt/data/mm_data/SAR/SARDet_100K"
)

# ---------------------------------------------------------------------------
# Stub functions
# ---------------------------------------------------------------------------


def convert_sardet_to_yolo_obb(
    sardet_root: str | Path,
    output_root: str | Path,
    split: str = "train",
) -> dict[str, int]:
    """Convert SARDet_100K annotations to YOLO-OBB format.

    THIS FUNCTION IS A STUB.  It will raise NotImplementedError until
    OBB annotations are available for SARDet_100K.

    Parameters
    ----------
    sardet_root:  Root of the SARDet_100K dataset.
                  Expected layout::

                      sardet_root/
                          annotations/
                              instances_train.json
                              instances_val.json
                              instances_test.json
                          images/
                              train/
                              val/
                              test/

    output_root:  Destination root for YOLO-OBB dataset.
    split:        One of "train", "val", "test".

    Returns
    -------
    dict with keys n_images, n_objects, output_dir.

    Raises
    ------
    NotImplementedError  Always, until OBB annotations are available.
    """
    raise NotImplementedError(
        "SARDet_100K OBB conversion is not yet implemented.\n"
        "SARDet currently provides HBB annotations only.  "
        "See module docstring for options."
    )


def convert_sardet_to_yolo_hbb(
    sardet_root: str | Path,
    output_root: str | Path,
    split: str = "train",
) -> dict[str, int]:
    """Convert SARDet_100K COCO annotations to YOLO HBB format (non-OBB).

    THIS FUNCTION IS A STUB.  Signature is provided for future implementation.

    YOLO HBB label line format::

        <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

    Parameters
    ----------
    sardet_root:  Root of the SARDet_100K dataset.
    output_root:  Destination root for the YOLO-HBB dataset.
    split:        One of "train", "val", "test".

    Returns
    -------
    dict with keys n_images, n_objects, output_dir.

    Raises
    ------
    NotImplementedError  Always, pending implementation.
    """
    raise NotImplementedError(
        "SARDet_100K HBB conversion is not yet implemented."
    )


def generate_sardet_yaml(
    output_root: str | Path,
    yaml_path: str | Path | None = None,
    obb: bool = False,
) -> Path:
    """Write a YOLO dataset YAML for SARDet_100K.

    THIS FUNCTION IS A STUB.

    Parameters
    ----------
    output_root:  Root of the converted dataset.
    yaml_path:    Output path; defaults to {output_root}/dataset.yaml.
    obb:          True if the dataset was converted to OBB format.

    Returns
    -------
    Path of the written YAML file (stub: raises NotImplementedError).
    """
    raise NotImplementedError(
        "generate_sardet_yaml is not yet implemented."
    )


# ---------------------------------------------------------------------------
# CLI entry-point (stub)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        "prepare_sardet.py is a STUB and does not perform any conversion yet.\n"
        "See the module docstring for details on SARDet_100K format and\n"
        "the planned implementation approach.",
        file=sys.stderr,
    )
    sys.exit(1)
