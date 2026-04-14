"""
prompt_templates.py — SAR情报通报 LLM Prompt 模板

为 M5 文本生成模块提供：
  - build_system_prompt()  -> str
  - build_user_prompt(evidence: dict) -> str
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _fmt_acquisition_time(iso_str: str) -> str:
    """将 ISO 8601 时间字符串转换为中文日期，例如 '2025年5月14日'。"""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return f"{dt.year}年{dt.month}月{dt.day}日"
    except Exception:
        return iso_str


def _build_class_list_text(by_class: list[dict]) -> str:
    """将 by_class 统计列表格式化为中文枚举字符串。

    例如：驱逐舰2艘、护卫舰1艘、战斗机2架
    """
    ship_units = {
        "carrier", "destroyer", "frigate", "replenishment",
        "amphibious", "other_vessel",
    }
    parts: list[str] = []
    for item in by_class:
        count = item.get("count", 0)
        if count == 0:
            continue
        name_cn = item.get("name_cn", item.get("code", "未知"))
        code = item.get("code", "")
        unit = "艘" if code in ship_units else "架"
        parts.append(f"{name_cn}{count}{unit}")
    return "、".join(parts) if parts else "无"


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """返回系统提示词。"""
    return (
        "你是一名军事情报分析助手，负责根据SAR卫星侦察数据生成标准军事情报通报正文。\n"
        "\n"
        "## 核心规则（违反则输出无效）\n"
        "1. **禁止幻觉**：所有数量、类别、坐标必须严格来自 <EVIDENCE> 字段，\n"
        "   不得凭想象添加任何证据中未出现的目标类别或数量。\n"
        "2. **禁止自行推断数字**：totals 与 by_class 是唯一可信数量来源。\n"
        "3. **保守措辞**：若 evidence 中标注 needs_caution=true，\n"
        "   必须对相关目标使用'疑似'或'初步判断'等保守措辞。\n"
        "4. **格式**：输出为纯中文通报正文段落，不含 Markdown 标题，\n"
        "   不含多余解释，以'据'字开头，以句号结尾。\n"
        "5. **语言风格**：简洁、正式、军事情报行文风格。\n"
        "6. **长度**：100～300字之间。\n"
        "\n"
        "## 段落结构建议\n"
        "- 首句：卫星/传感器、成像日期、侦察区域、总体目标数量汇总。\n"
        "- 中段：按目标类别逐一列出数量（严格依赖 by_class）。\n"
        "- 末句：目标空间分布概述（来自 spatial_summary.distribution）。\n"
        "\n"
        "重要：只输出正文段落，不要包含任何其他内容。"
    )


def build_user_prompt(evidence: dict) -> str:
    """将结构化证据注入模板，返回用户侧提示词。

    Parameters
    ----------
    evidence:
        完整的 Evidence Package 字典（status 必须为 READY_FOR_NLG）。

    Returns
    -------
    str
        格式化后的用户提示词字符串。
    """
    # ---------- 提取字段，失败时给出安全默认值 ----------
    inp = evidence.get("input", {})
    metadata = inp.get("metadata", {})
    mission = inp.get("mission", {})
    scene = evidence.get("scene", {})
    statistics = evidence.get("statistics", {})
    totals = statistics.get("totals", {})
    by_class = statistics.get("by_class", [])
    spatial_summary = statistics.get("spatial_summary", {})
    confidence_summary = statistics.get("confidence_summary", {})
    quality = evidence.get("quality", {})

    satellite = metadata.get("satellite", "未知卫星")
    acq_time_raw = metadata.get("acquisition_time", "")
    acq_date_cn = _fmt_acquisition_time(acq_time_raw) if acq_time_raw else "未知日期"
    region_name = mission.get("region_name", "未知区域")
    scene_type_cn = scene.get("scene_type_cn", "")

    all_objects = totals.get("all_objects", 0)
    ships = totals.get("ships", 0)
    aircraft = totals.get("aircraft", 0)

    class_list_text = _build_class_list_text(by_class)
    distribution = spatial_summary.get("distribution", "分布情况不详")

    review_required = confidence_summary.get("review_required_count", 0)
    needs_caution = review_required > 0

    # ---------- 序列化 by_class 供 LLM 参考 ----------
    by_class_json = json.dumps(by_class, ensure_ascii=False, indent=2)

    # ---------- 组装提示词 ----------
    caution_note = (
        "\n【重要】evidence 中存在待审核目标（review_required_count > 0），"
        "请对相关目标使用'疑似'或'初步判断'等保守措辞。"
        if needs_caution
        else ""
    )

    prompt = (
        f"请根据以下 <EVIDENCE> 生成SAR卫星情报通报正文。{caution_note}\n"
        "\n"
        "<EVIDENCE>\n"
        f"卫星：{satellite}\n"
        f"成像日期：{acq_date_cn}\n"
        f"侦察区域：{region_name}"
        + (f"（{scene_type_cn}）" if scene_type_cn else "")
        + "\n"
        f"目标总数：{all_objects}个（舰船{ships}艘，飞机{aircraft}架）\n"
        f"按类别统计（严格以此为准，禁止修改）：\n{by_class_json}\n"
        f"中文列表：{class_list_text}\n"
        f"空间分布：{distribution}\n"
        f"需保守措辞：{'是' if needs_caution else '否'}\n"
        "</EVIDENCE>\n"
        "\n"
        "请直接输出通报正文，不要有任何额外说明。"
    )
    return prompt
