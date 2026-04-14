"""
class_map.py — YOLO class index → Evidence Package class descriptor

Defines the mapping from integer class indices (as output by the YOLOv8-OBB model)
to the standardised class fields required by the Evidence Package schema v1.0.

The ordering MUST match the class order in your dataset.yaml `names:` list.
Edit the list below to reflect the exact class ordering your model was trained on.
"""

from typing import TypedDict


class ClassDescriptor(TypedDict):
    code: str          # Evidence Package class code
    name_cn: str       # Chinese display name
    super_class: str   # "ship" | "aircraft"
    priority: str      # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"


# ---------------------------------------------------------------------------
# Primary mapping: YOLO index → ClassDescriptor
#
# Default ordering follows a typical ship-first, aircraft-second dataset.
# Adjust indices to match your dataset.yaml exactly.
# ---------------------------------------------------------------------------

CLASS_MAP: dict[int, ClassDescriptor] = {
    # ── Ships ──────────────────────────────────────────────────────────────
    0: {
        "code": "carrier",
        "name_cn": "航空母舰",
        "super_class": "ship",
        "priority": "CRITICAL",
    },
    1: {
        "code": "destroyer",
        "name_cn": "驱逐舰",
        "super_class": "ship",
        "priority": "HIGH",
    },
    2: {
        "code": "frigate",
        "name_cn": "护卫舰",
        "super_class": "ship",
        "priority": "HIGH",
    },
    3: {
        "code": "replenishment",
        "name_cn": "补给舰",
        "super_class": "ship",
        "priority": "MEDIUM",
    },
    4: {
        "code": "amphibious",
        "name_cn": "两栖舰",
        "super_class": "ship",
        "priority": "HIGH",
    },
    5: {
        "code": "other_vessel",
        "name_cn": "其他舰船",
        "super_class": "ship",
        "priority": "LOW",
    },
    # ── Aircraft ───────────────────────────────────────────────────────────
    6: {
        "code": "fighter",
        "name_cn": "战斗机",
        "super_class": "aircraft",
        "priority": "HIGH",
    },
    7: {
        "code": "bomber",
        "name_cn": "轰炸机",
        "super_class": "aircraft",
        "priority": "CRITICAL",
    },
    8: {
        "code": "transport",
        "name_cn": "运输机",
        "super_class": "aircraft",
        "priority": "MEDIUM",
    },
    9: {
        "code": "aew",
        "name_cn": "预警机",
        "super_class": "aircraft",
        "priority": "HIGH",
    },
    10: {
        "code": "helicopter",
        "name_cn": "直升机",
        "super_class": "aircraft",
        "priority": "MEDIUM",
    },
    11: {
        "code": "other_aircraft",
        "name_cn": "其他飞机",
        "super_class": "aircraft",
        "priority": "LOW",
    },
}

# Fallback descriptor used when the model produces an unknown class index.
UNKNOWN_CLASS: ClassDescriptor = {
    "code": "other_vessel",
    "name_cn": "未知目标",
    "super_class": "ship",
    "priority": "LOW",
}


def get_class_descriptor(class_index: int) -> ClassDescriptor:
    """Return the ClassDescriptor for a given YOLO class index.

    Falls back to UNKNOWN_CLASS if the index is not in CLASS_MAP.
    """
    return CLASS_MAP.get(class_index, UNKNOWN_CLASS)
