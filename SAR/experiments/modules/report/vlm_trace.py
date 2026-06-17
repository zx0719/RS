"""
Helpers for recording auditable VLM scene-description provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .vlm_describer import VLMDescriber


def attach_vlm_scene_description(
    package: dict,
    *,
    description: str,
    describer: VLMDescriber,
    image_path: str | Path,
) -> None:
    """Attach a VLM scene description and its provenance trace to evidence."""
    scene = package.setdefault("scene", {})
    scene["scene_description"] = description
    if hasattr(describer, "trace_metadata"):
        trace = describer.trace_metadata(
            image_path=image_path,
            status="used",
            description=description,
        )
    else:
        trace = {
            "source": "vlm_v1",
            "mode": "api",
            "model_name": getattr(describer, "model_name", None),
            "base_url": getattr(describer, "base_url", None),
            "image_path": str(image_path),
            "status": "used",
            "description_present": bool(description),
        }
    scene["scene_description_trace"] = trace
