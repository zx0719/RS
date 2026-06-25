"""
class_labels.py — Shared class-code and super-class naming helpers.

This module keeps human-readable Chinese labels consistent across:
  - evidence statistics
  - prompt construction
  - report table generation
"""

from __future__ import annotations

from typing import Any


_CODE_TO_CN: dict[str, str] = {
    # ── Area targets (Branch B: FastSAM) ──────────────────────────────────
    "harbor": "港口",
    "airport": "机场",
    # ── Ship subtypes 16类 (Branch A: M2a) ─────────────────────────────────
    "military_auxiliary": "军用辅助舰船",
    "combat_ship": "作战舰船",
    "liquid_cargo": "液货船",
    "harbor_service": "港务船",
    "bulk_carrier": "散货船",
    "other_vessel": "舰船_其他",
    "survey_vessel": "调查船",
    "amphibious": "两栖舰船",
    "container_ship": "集装箱船",
    "engineering_vessel": "工程船",
    "fishing_vessel": "渔船",
    "tug_boat": "拖船",
    "passenger_ship": "客船",
    "ro_ro_ship": "滚装船",
    "sailing_vessel": "帆船",
    "research_vessel": "研究船",
    "ship": "舰船",
    # ── Aircraft subtypes 5类 (Branch A: M2b) ──────────────────────────────
    "combat_aircraft": "作战飞机",
    "transport_aircraft": "运输机",
    "combat_support_aircraft": "作战支援飞机",
    "helicopter": "直升机",
    "other_aircraft": "飞机_其他",
    "aircraft": "飞机",
    # ── Legacy / deprecated (aliases → different names to avoid overwriting) ──
    "carrier": "航空母舰",
    "destroyer": "驱逐舰",
    "frigate": "护卫舰",
    "replenishment": "补给舰",
    "fighter": "战斗机",
    "bomber": "轰炸机",
    "transport": "运输机(legacy)",
    "aew": "预警机",
    "tank": "坦克/装甲车",
    "bridge": "桥梁",
    "runway": "跑道",
    "taxiway": "滑行道",
    "apron": "停机坪",
    "hangar": "机库",
    "shelter": "飞机掩蔽库",
    "tower": "塔台",
    "ammo_depot": "弹药库",
    "stopway": "端保险道",
    "liaison": "联络道",
    "unknown": "未知目标",
}

_SUPER_CLASS_TO_CN: dict[str, str] = {
    "ship": "舰船",
    "aircraft": "飞机",
    "ground": "地面目标",
    "infrastructure": "基础设施",
    "area": "区域目标",
    "unknown": "未知目标",
}

_CODE_TO_SUPER_CLASS: dict[str, str] = {
    # ── Area targets ──────────────────────────────────────────────────────
    "harbor": "area",
    "airport": "area",
    # ── Ship subtypes 16类 ─────────────────────────────────────────────────
    "military_auxiliary": "ship",
    "combat_ship": "ship",
    "liquid_cargo": "ship",
    "harbor_service": "ship",
    "bulk_carrier": "ship",
    "other_vessel": "ship",
    "survey_vessel": "ship",
    "amphibious": "ship",
    "container_ship": "ship",
    "engineering_vessel": "ship",
    "fishing_vessel": "ship",
    "tug_boat": "ship",
    "passenger_ship": "ship",
    "ro_ro_ship": "ship",
    "sailing_vessel": "ship",
    "research_vessel": "ship",
    "ship": "ship",
    # ── Aircraft subtypes 5类 ──────────────────────────────────────────────
    "combat_aircraft": "aircraft",
    "transport_aircraft": "aircraft",
    "combat_support_aircraft": "aircraft",
    "helicopter": "aircraft",
    "other_aircraft": "aircraft",
    "aircraft": "aircraft",
    # ── Legacy / deprecated ────────────────────────────────────────────────
    "carrier": "ship",
    "destroyer": "ship",
    "frigate": "ship",
    "replenishment": "ship",
    "fighter": "aircraft",
    "bomber": "aircraft",
    "transport": "aircraft",
    "aew": "aircraft",
    "tank": "ground",
    "bridge": "infrastructure",
    "runway": "infrastructure",
    "taxiway": "infrastructure",
    "apron": "infrastructure",
    "hangar": "infrastructure",
    "shelter": "infrastructure",
    "tower": "infrastructure",
    "ammo_depot": "infrastructure",
    "stopway": "infrastructure",
    "liaison": "infrastructure",
}


def infer_super_class(code: str, super_class: str | None = None) -> str:
    """Infer a stable super-class label from code and optional input."""
    if super_class:
        return str(super_class)
    return _CODE_TO_SUPER_CLASS.get(str(code), "unknown")


def get_super_class_name_cn(super_class: str | None) -> str:
    """Return Chinese display name for a super-class."""
    key = str(super_class or "unknown")
    return _SUPER_CLASS_TO_CN.get(key, key)


def get_class_name_cn(
    code: str | None,
    name_cn: str | None = None,
    super_class: str | None = None,
) -> str:
    """Return a stable Chinese label for a class descriptor.

    Priority:
      1. known ``code`` mapping
      2. known ``name_cn`` that is actually a code-like English token
      3. already-Chinese ``name_cn``
      4. super-class display name
      5. ``未知目标``
    """
    code_key = str(code or "").strip()
    if code_key in _CODE_TO_CN:
        return _CODE_TO_CN[code_key]

    candidate = str(name_cn or "").strip()
    if candidate:
        candidate_key = candidate.lower().replace(" ", "_")
        if candidate_key in _CODE_TO_CN:
            return _CODE_TO_CN[candidate_key]
        if _contains_cjk(candidate):
            return candidate

    inferred_super = infer_super_class(code_key, super_class)
    return get_super_class_name_cn(inferred_super)


def normalize_by_class_list(by_class: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a normalized copy of ``statistics.by_class``."""
    normalized: list[dict[str, Any]] = []
    for item in by_class:
        code = str(item.get("code", "unknown"))
        super_class = infer_super_class(code, item.get("super_class"))
        normalized_item = dict(item)
        normalized_item["code"] = code
        normalized_item["super_class"] = super_class
        normalized_item["name_cn"] = get_class_name_cn(
            code,
            item.get("name_cn"),
            super_class,
        )
        normalized.append(normalized_item)
    return normalized


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)

