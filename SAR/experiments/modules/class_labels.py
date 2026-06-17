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
    "carrier": "航空母舰",
    "destroyer": "驱逐舰",
    "frigate": "护卫舰",
    "replenishment": "补给舰",
    "amphibious": "两栖舰",
    "other_vessel": "其他舰船",
    "ship": "舰船",
    "fighter": "战斗机",
    "bomber": "轰炸机",
    "transport": "运输机",
    "aew": "预警机",
    "helicopter": "直升机",
    "other_aircraft": "其他飞机",
    "aircraft": "飞机",
    "tank": "坦克/装甲车",
    "bridge": "桥梁",
    "harbor": "港口",
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
    "unknown": "未知目标",
}

_CODE_TO_SUPER_CLASS: dict[str, str] = {
    "carrier": "ship",
    "destroyer": "ship",
    "frigate": "ship",
    "replenishment": "ship",
    "amphibious": "ship",
    "other_vessel": "ship",
    "ship": "ship",
    "fighter": "aircraft",
    "bomber": "aircraft",
    "transport": "aircraft",
    "aew": "aircraft",
    "helicopter": "aircraft",
    "other_aircraft": "aircraft",
    "aircraft": "aircraft",
    "tank": "ground",
    "bridge": "infrastructure",
    "harbor": "infrastructure",
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

