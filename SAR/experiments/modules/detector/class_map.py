"""
class_map.py — YOLO class index → Evidence Package class descriptor

v5 2 类方案（M1 粗检测，当前主线）：
  ID  类别          数据来源                  状态
  ──────────────────────────────────────────────────────
  0   ship          SARDet_100K + SSDD + ...   ACTIVE
  1   aircraft      SARDet_100K + SAR-air     ACTIVE

v4 5 类方案（历史，保留兼容）：
  ID  类别          数据来源                  状态
  ──────────────────────────────────────────────────────
  0   ship          SSDD + SAR-Ship-Dataset   ACTIVE
  1   aircraft      SARDet_100K + SAR-air     ACTIVE
  2   tank          SARDet_100K               DEPRECATED
  3   bridge        SARDet_100K + MSAR        DEPRECATED
  4   harbor        SARDet_100K               DEPRECATED
  5-13 (RESERVED)

harbor/airport 不再由 YOLO 检测，由分支 B 的 Gate + FastSAM 独立处理。
"""

from typing import TypedDict


class ClassDescriptor(TypedDict):
    code: str          # Evidence Package class code
    name_cn: str       # Chinese display name
    super_class: str   # "ship" | "aircraft" | "ground" | "infrastructure"
    priority: str      # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    status: str        # "ACTIVE" | "RESERVED"


# ---------------------------------------------------------------------------
# v4 固定类别映射 — ID 与 dataset.yaml names 顺序严格对应
# ---------------------------------------------------------------------------

CLASS_MAP: dict[int, ClassDescriptor] = {
    # ── 已激活类别（有训练数据）────────────────────────────────────────────
    0: {
        "code": "ship",
        "name_cn": "舰船",
        "super_class": "ship",
        "priority": "HIGH",
        "status": "ACTIVE",
    },
    1: {
        "code": "aircraft",
        "name_cn": "飞机",
        "super_class": "aircraft",
        "priority": "HIGH",
        "status": "ACTIVE",
    },
    2: {
        "code": "tank",
        "name_cn": "坦克/装甲车",
        "super_class": "ground",
        "priority": "HIGH",
        "status": "ACTIVE",
    },
    3: {
        "code": "bridge",
        "name_cn": "桥梁",
        "super_class": "infrastructure",
        "priority": "MEDIUM",
        "status": "ACTIVE",
    },
    4: {
        "code": "harbor",
        "name_cn": "港口",
        "super_class": "infrastructure",
        "priority": "HIGH",
        "status": "ACTIVE",
    },
    # ── 预留类别（待补充标注数据后激活）───────────────────────────────────
    5: {
        "code": "runway",
        "name_cn": "跑道",
        "super_class": "infrastructure",
        "priority": "HIGH",
        "status": "RESERVED",
    },
    6: {
        "code": "taxiway",
        "name_cn": "滑行道",
        "super_class": "infrastructure",
        "priority": "MEDIUM",
        "status": "RESERVED",
    },
    7: {
        "code": "apron",
        "name_cn": "停机坪",
        "super_class": "infrastructure",
        "priority": "MEDIUM",
        "status": "RESERVED",
    },
    8: {
        "code": "hangar",
        "name_cn": "机库",
        "super_class": "infrastructure",
        "priority": "HIGH",
        "status": "RESERVED",
    },
    9: {
        "code": "shelter",
        "name_cn": "飞机掩蔽库",
        "super_class": "infrastructure",
        "priority": "HIGH",
        "status": "RESERVED",
    },
    10: {
        "code": "tower",
        "name_cn": "塔台",
        "super_class": "infrastructure",
        "priority": "MEDIUM",
        "status": "RESERVED",
    },
    11: {
        "code": "ammo_depot",
        "name_cn": "弹药库",
        "super_class": "infrastructure",
        "priority": "CRITICAL",
        "status": "RESERVED",
    },
    12: {
        "code": "stopway",
        "name_cn": "端保险道",
        "super_class": "infrastructure",
        "priority": "LOW",
        "status": "RESERVED",
    },
    13: {
        "code": "liaison",
        "name_cn": "联络道",
        "super_class": "infrastructure",
        "priority": "LOW",
        "status": "RESERVED",
    },
}

# 未知类别兜底
UNKNOWN_CLASS: ClassDescriptor = {
    "code": "unknown",
    "name_cn": "未知目标",
    "super_class": "unknown",
    "priority": "LOW",
    "status": "ACTIVE",
}


def get_class_descriptor(class_index: int) -> ClassDescriptor:
    """Return the ClassDescriptor for a given YOLO class index."""
    desc = CLASS_MAP.get(class_index)
    if desc is None:
        return UNKNOWN_CLASS
    if desc["status"] == "RESERVED":
        return {
            "code": desc["code"],
            "name_cn": f"预留-{desc['name_cn']}",
            "super_class": desc["super_class"],
            "priority": desc["priority"],
            "status": "RESERVED",
        }
    return desc


def get_active_classes() -> dict[int, ClassDescriptor]:
    """返回所有 ACTIVE 状态的类别（已有训练数据）。"""
    return {k: v for k, v in CLASS_MAP.items() if v["status"] == "ACTIVE"}


# ---------------------------------------------------------------------------
# v5 2 类方案 — M1 粗检测 (当前主线)
# ---------------------------------------------------------------------------

V5_2CLASS_MAP: dict[int, ClassDescriptor] = {
    0: {
        "code": "ship",
        "name_cn": "舰船",
        "super_class": "ship",
        "priority": "HIGH",
        "status": "ACTIVE",
    },
    1: {
        "code": "aircraft",
        "name_cn": "飞机",
        "super_class": "aircraft",
        "priority": "HIGH",
        "status": "ACTIVE",
    },
}

# ---------------------------------------------------------------------------
# 历史兼容：v1/v2 单类模型的映射（推理时传入 class_map 参数使用）
# ---------------------------------------------------------------------------

# v1-ssdd-ship: class 0 → ship
SSDD_CLASS_MAP: dict[int, ClassDescriptor] = {
    0: CLASS_MAP[0],  # ship
}

# v2-sardet-aircraft: class 0 → aircraft
SARDET_AIRCRAFT_CLASS_MAP: dict[int, ClassDescriptor] = {
    0: CLASS_MAP[1],  # aircraft
}
