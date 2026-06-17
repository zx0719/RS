"""
prompt_templates.py — SAR情报通报 LLM Prompt 模板

为 M5 文本生成模块提供：
  - build_system_prompt()  -> str
  - build_system_prompt_variants(index) -> str
  - build_user_prompt(evidence: dict) -> str
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from typing import Any

from modules.class_labels import get_class_name_cn, normalize_by_class_list
from .large_scene import (
    ROUTE_LARGE,
    ROUTE_SMALL,
    ROUTE_TEMPLATE,
    build_evidence_digest,
    choose_generation_route,
    is_large_scene,
)


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
    for item in normalize_by_class_list(by_class):
        count = item.get("count", 0)
        if count == 0:
            continue
        name_cn = get_class_name_cn(
            item.get("code"),
            item.get("name_cn"),
            item.get("super_class"),
        )
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


def build_system_prompt_variants(index: int | None = None) -> str:
    """返回多种风格的系统提示词之一。

    Parameters
    ----------
    index:
        变体索引（0-3）。为 None 时随机选取。

    Variant 0 — 标准军事情报通报风格（同 build_system_prompt）
    Variant 1 — 简洁事实风格，去除套话
    Variant 2 — 强调空间分布与战术态势评估
    Variant 3 — 包含不确定性语言（置信度较低时使用）
    """
    _variants = [
        # Variant 0: standard (same as build_system_prompt)
        (
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
        ),
        # Variant 1: concise, facts-only, no filler phrases
        (
            "你是一名军事情报分析员，任务是将SAR卫星侦察数据转化为简洁的情报通报。\n"
            "\n"
            "## 要求\n"
            "1. 严格基于 <EVIDENCE> 中的数字，不得添加或更改任何目标数量与类别。\n"
            "2. 文风简洁直接，只陈述事实，不加修饰性语言。\n"
            "3. 不使用套话，如\u300c具有较高情报价值\u300d、\u300c图像质量良好\u300d等。\n"
            "4. 输出纯中文正文段落，以日期或卫星信息开头，以句号结尾。\n"
            "5. 长度：80～200字。\n"
            "\n"
            "## 结构\n"
            "- 第一句：侦察时间、卫星、区域及目标总数。\n"
            "- 后续句：分类列出各目标数量。\n"
            "- 末句：目标位置或分布情况，一句话概括。\n"
            "\n"
            "只输出正文，无需任何解释。"
        ),
        # Variant 2: emphasizes spatial distribution and tactical assessment
        (
            "你是一名SAR图像情报分析专家，专注于目标空间分布与战术态势研判。\n"
            "\n"
            "## 写作规则\n"
            "1. 数量与类别必须与 <EVIDENCE> 完全一致，禁止添加未见目标。\n"
            "2. 重点描述目标的空间位置关系、集群态势及战术意义。\n"
            "3. 语言正式、专业，符合军事情报通报规范。\n"
            "4. 输出纯中文段落，以侦察事实开头，以态势研判结尾。\n"
            "5. 长度：120～300字。\n"
            "\n"
            "## 段落结构\n"
            "- 首句：卫星、时间、区域、发现目标总概。\n"
            "- 中段：各类目标数量及空间分布特征。\n"
            "- 末句：综合态势研判，建议持续关注方向。\n"
            "\n"
            "直接输出正文段落。"
        ),
        # Variant 3: includes uncertainty language when confidence is lower
        (
            "你是一名军事情报分析助手，处理置信度有限的SAR侦察数据。\n"
            "\n"
            "## 核心规则\n"
            "1. 数量与类别必须严格来自 <EVIDENCE>，禁止任何推断性添加。\n"
            "2. 对于置信度不足的目标，使用'疑似'、'初步判断'、'待核实'等措辞。\n"
            "3. 对于高置信度目标，可使用'确认'、'识别'等肯定性措辞。\n"
            "4. 语言应体现不确定性与审慎态度，避免武断结论。\n"
            "5. 输出纯中文通报正文，以'据'字或时间开头，以句号结尾。\n"
            "6. 长度：100～280字。\n"
            "\n"
            "## 结构\n"
            "- 首句：侦察背景（卫星、时间、区域）。\n"
            "- 中段：逐类描述目标，标注置信程度。\n"
            "- 末句：综合研判，指出待核实事项。\n"
            "\n"
            "只输出正文。"
        ),
    ]

    if index is None:
        index = random.randint(0, len(_variants) - 1)
    idx = int(index) % len(_variants)
    return _variants[idx]


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
    by_class = normalize_by_class_list(statistics.get("by_class", []))
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
    scene_description = scene.get("scene_description", "")

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
    scene_line = f"侦察区域：{region_name}" + (f"（{scene_type_cn}）" if scene_type_cn else "")
    scene_desc_line = (
        f"场景整体描述（必须融入正文，不可改写数量）：{scene_description}\n"
        if scene_description
        else ""
    )

    prompt = (
        f"请根据以下 <EVIDENCE> 生成SAR卫星情报通报正文。{caution_note}\n"
        "\n"
        "<EVIDENCE>\n"
        f"卫星：{satellite}\n"
        f"成像日期：{acq_date_cn}\n"
        f"{scene_line}\n"
        f"目标总数：{all_objects}个（舰船{ships}艘，飞机{aircraft}架）\n"
        f"按类别统计（严格以此为准，禁止修改）：\n{by_class_json}\n"
        f"中文列表：{class_list_text}\n"
        f"空间分布：{distribution}\n"
        f"{scene_desc_line}"
        f"需保守措辞：{'是' if needs_caution else '否'}\n"
        "</EVIDENCE>\n"
        "\n"
        "请直接输出通报正文，不要有任何额外说明。"
    )
    return prompt


def build_prompt_payload(evidence: dict) -> dict[str, str]:
    """Build system/user prompts with route metadata.

    Large-scene cases use a compact digest-oriented user prompt.
    """
    report_context = evidence.get("report_context", {})
    route = report_context.get("generation_route")
    if not route:
        route = choose_generation_route(evidence) if is_large_scene(evidence) else ROUTE_SMALL
    system_prompt = build_system_prompt()
    if is_large_scene(evidence):
        digest = build_evidence_digest(evidence)
        user_prompt = _build_large_scene_user_prompt(digest, route)
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "route": route,
            "mode": "large_scene",
        }

    return {
        "system_prompt": system_prompt,
        "user_prompt": build_user_prompt(evidence),
        "route": route,
        "mode": "default",
    }


def build_refine_prompt_payload(
    evidence: dict,
    draft_body: str,
    route: str = ROUTE_LARGE,
) -> dict[str, str]:
    """Build a prompt for large-model refinement of a small-model/template draft."""
    system_prompt = (
        "你是一名军事情报通报审校助手。你的任务不是重新发明事实，"
        "而是在严格不改变数量、类别和分布事实的前提下，"
        "把给定草稿润色成更稳定、专业、紧凑的正式中文通报正文。"
    )

    if is_large_scene(evidence):
        digest = build_evidence_digest(evidence)
        payload = {
            "route": route,
            "large_scene_digest": digest,
            "draft_body": draft_body,
        }
        user_prompt = (
            "请根据以下大图证据摘要与现有草稿，进行审校式重写。\n"
            "要求：\n"
            "1. 严格保持数量、类别、区域和空间分布事实不变。\n"
            "2. 可以优化句式、压缩重复、提升专业性。\n"
            "3. 不得新增摘要中不存在的目标类型、数量或结论。\n"
            "4. 输出纯中文正文，以'据'字开头，以句号结尾。\n\n"
            "<REFINE_INPUT>\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
            "</REFINE_INPUT>\n\n"
            "请直接输出重写后的正文。"
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "route": route,
            "mode": "large_scene_refine",
        }

    user_prompt = (
        "请根据以下事实证据和现有草稿进行审校式重写。\n"
        "要求：严格保持数量与类别不变，只优化表达。\n\n"
        "<EVIDENCE>\n"
        f"{build_user_prompt(evidence)}\n"
        "</EVIDENCE>\n\n"
        "<DRAFT>\n"
        f"{draft_body}\n"
        "</DRAFT>\n\n"
        "请直接输出重写后的正文。"
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "route": route,
        "mode": "default_refine",
    }


def _build_large_scene_user_prompt(digest: dict[str, Any], route: str) -> str:
    global_summary = digest.get("global_summary", {})
    class_digest = digest.get("class_digest", [])
    cluster_summary = digest.get("cluster_summary", {})
    quality_flags = digest.get("quality_flags", {})
    representatives = digest.get("representative_targets", [])
    scene_description = digest.get("global_summary", {}).get("scene_description", "")

    route_hint = {
        ROUTE_TEMPLATE: "当前案例结构简单，可用保守、短文本表述。",
        ROUTE_SMALL: "当前案例适合常规摘要式生成，保持事实压缩和措辞稳定。",
        ROUTE_LARGE: "当前案例复杂度较高，请优先处理多类别、多区域和不确定性。",
    }.get(route, "请保持事实压缩和措辞稳定。")

    payload = {
        "route": route,
        "route_hint": route_hint,
        "global_summary": global_summary,
        "class_digest": class_digest,
        "cluster_summary": cluster_summary,
        "representative_targets": representatives,
        "quality_flags": quality_flags,
        "scene_description": scene_description,
    }

    return (
        "请根据以下 <LARGE_SCENE_DIGEST> 生成SAR大图情报通报正文。\n"
        "要求：\n"
        "1. 只使用 digest 中给出的数量、类别和分布信息。\n"
        "2. 不要复述所有代表目标，仅用它们辅助把握场景。\n"
        "3. 对 review_required_count > 0 的案例使用保守措辞。\n"
        "4. 数量单位必须固定：舰船用“艘”，飞机用“架”，坦克/装甲车等地面目标用“辆”或“个”，桥梁/港口等基础设施用“处”。\n"
        "5. 必须逐项写出 class_digest 中每个 count > 0 的类别及数量，格式示例：舰船18艘、飞机6架、坦克/装甲车6辆。\n"
        "6. 如果 scene_description 非空，必须将该场景描述自然融入正文，但不得据此改写数量和类别。\n"
        "7. 输出纯中文正文，以'据'字开头，以句号结尾。\n"
        "8. 长度控制在120~260字。\n\n"
        "<LARGE_SCENE_DIGEST>\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "</LARGE_SCENE_DIGEST>\n\n"
        "请直接输出正文，不要添加任何解释。"
    )
