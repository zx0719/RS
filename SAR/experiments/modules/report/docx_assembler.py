"""
docx_assembler.py — M6 Word 文档组装模块

class DocxAssembler:
    assemble(evidence_package, output_dir, template_path=None) -> str

文档结构：
  标题页（标题、副标题、日期）
  正文
  附件1：侦察图（若有）
  附件2：组成分布统计表
  附件3：装备分布统计表
  页脚：卫星/传感器/成像时间/pipeline_run_id + 自动生成标识（审核版）

支持两种模式：
  - 审核版（draft_mode=True）：页脚含"【自动生成草稿，待人工审核】"
  - 正式版（draft_mode=False）：无自动生成标识
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from docx import Document  # type: ignore
    from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore
    from docx.oxml.ns import qn  # type: ignore
    from docx.shared import Cm, Pt, RGBColor  # type: ignore
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

# ---------------------------------------------------------------------------
# 默认模板路径
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE_PATH = "/home/zhuxiang/RS/SAR/文档/成品.docx"

# ---------------------------------------------------------------------------
# 中文字段列头定义
# ---------------------------------------------------------------------------

_COMPONENT_TABLE_HEADERS = ["序号", "目标大类", "目标子类", "经度", "纬度"]
_EQUIPMENT_TABLE_HEADERS = ["序号", "装备类型", "经度", "纬度", "置信度", "审核状态"]


def _uri_to_local_path(uri: str) -> str | None:
    """将 file:// URI 转换为本地文件路径，非 file:// 时返回 None。"""
    if uri and uri.startswith("file://"):
        return uri[len("file://"):]
    return None


class DocxAssembler:
    """M6 Word 文档组装器。

    使用 python-docx 将 Evidence Package 中的 report 字段组装为 .docx 文件。
    """

    TEMPLATE_VERSION = "brief-template-v1"

    def assemble(
        self,
        evidence_package: dict,
        output_dir: str,
        template_path: str | None = None,
        draft_mode: bool = True,
    ) -> str:
        """组装 Word 文档，返回输出文件的绝对路径。

        Parameters
        ----------
        evidence_package:
            状态为 REPORT_DRAFTED 或 READY_FOR_NLG 的 Evidence Package。
        output_dir:
            输出目录路径（不存在时自动创建）。
        template_path:
            可选的 .docx 模板文件路径。若为 None，从空白文档开始。
        draft_mode:
            True = 审核版（页脚含自动生成标识）；False = 正式版。

        Returns
        -------
        str
            生成的 .docx 文件绝对路径。

        Raises
        ------
        ImportError
            若 python-docx 未安装。
        """
        if not _DOCX_AVAILABLE:
            raise ImportError(
                "python-docx 未安装，请执行：pip install python-docx"
            )

        os.makedirs(output_dir, exist_ok=True)

        # 若未指定模板，尝试使用默认模板
        if template_path is None and Path(DEFAULT_TEMPLATE_PATH).exists():
            template_path = DEFAULT_TEMPLATE_PATH

        # 提取字段
        pkg = evidence_package
        report = pkg.get("report", {})
        statistics = pkg.get("statistics", {})
        objects = pkg.get("objects", [])
        attachments = pkg.get("attachments", {})
        inp = pkg.get("input", {})
        metadata = inp.get("metadata", {})
        trace = pkg.get("trace", {})

        title = report.get("title", "航天通报")
        subtitle = report.get("subtitle", "SAR目标监测通报")
        report_date = report.get("report_date", "")
        body = report.get("body", "（正文待生成）")
        tables = report.get("tables", {})
        component_table = tables.get("component_table", [])
        equipment_table = tables.get("equipment_table", [])

        # 若 tables 为空，尝试从 objects 重建（防御性）
        if not component_table and not equipment_table and objects:
            from .table_builder import TableBuilder
            tb = TableBuilder()
            component_table = tb.build_component_table(objects)
            equipment_table = tb.build_equipment_table(objects)

        # 构建输出文件名
        package_id = pkg.get("package_id", "report")
        safe_id = re.sub(r"[^\w\-]", "_", package_id)
        draft_suffix = "_draft" if draft_mode else "_final"
        filename = f"{safe_id}{draft_suffix}.docx"
        output_path = str(Path(output_dir) / filename)

        # 创建文档
        if template_path and Path(template_path).exists():
            doc = Document(template_path)
            # 清除正文内容，保留模板的样式、页眉和页脚
            self._clear_body_content(doc)
        else:
            doc = Document()

        # 页面边距（2.5cm 四边）
        self._set_margins(doc)

        # ---------- 标题页 ----------
        self._add_title_section(doc, title, subtitle, report_date)

        # ---------- 正文 ----------
        self._add_body_section(doc, body)

        # ---------- 附件1：侦察图 ----------
        annotated = attachments.get("annotated_image", {})
        ann_uri = annotated.get("uri", "")
        ann_path = _uri_to_local_path(ann_uri) if ann_uri else None
        self._add_attachment_image(doc, ann_path)

        # ---------- 附件2：组成分布统计表 ----------
        self._add_component_table(doc, component_table)

        # ---------- 附件3：装备分布统计表 ----------
        self._add_equipment_table(doc, equipment_table)

        # ---------- 页脚 ----------
        satellite = metadata.get("satellite", "")
        sensor = metadata.get("sensor", "SAR")
        acq_time = metadata.get("acquisition_time", "")
        pipeline_run_id = trace.get("pipeline_run_id", "")
        self._add_footer(doc, satellite, sensor, acq_time, pipeline_run_id, draft_mode)

        doc.save(output_path)
        return output_path

    # ------------------------------------------------------------------
    # 内部组装方法
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_body_content(doc: Any) -> None:
        """清除文档正文中的所有段落和表格，保留样式、页眉、页脚。

        通过直接操作 XML body 元素来移除正文节点，同时保留
        sectPr（节属性，含页眉/页脚引用）。
        """
        body = doc.element.body
        # 收集所有要删除的子节点（段落 w:p 和表格 w:tbl），保留 w:sectPr
        to_remove = [
            child for child in body
            if child.tag != qn("w:sectPr")
        ]
        for child in to_remove:
            body.remove(child)

    @staticmethod
    def _set_margins(doc: Any) -> None:
        """设置页面边距为 2.5cm。"""
        for section in doc.sections:
            section.top_margin = Cm(2.5)
            section.bottom_margin = Cm(2.5)
            section.left_margin = Cm(2.8)
            section.right_margin = Cm(2.8)

    @staticmethod
    def _add_title_section(
        doc: Any, title: str, subtitle: str, report_date: str
    ) -> None:
        """添加标题、副标题、日期。"""
        # 大标题
        p_title = doc.add_paragraph()
        p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p_title.add_run(title)
        run.bold = True
        run.font.size = Pt(22)

        # 副标题
        p_sub = doc.add_paragraph()
        p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_sub = p_sub.add_run(subtitle)
        run_sub.bold = True
        run_sub.font.size = Pt(16)

        # 日期
        if report_date:
            p_date = doc.add_paragraph()
            p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run_date = p_date.add_run(report_date)
            run_date.font.size = Pt(12)

        doc.add_paragraph()  # 空行

    @staticmethod
    def _add_body_section(doc: Any, body: str) -> None:
        """添加正文段落。"""
        heading = doc.add_paragraph()
        h_run = heading.add_run("一、侦察情况")
        h_run.bold = True
        h_run.font.size = Pt(14)

        p = doc.add_paragraph()
        # 正文首行缩进2字符
        p.paragraph_format.first_line_indent = Pt(24)
        run = p.add_run(body)
        run.font.size = Pt(12)

        doc.add_paragraph()

    @staticmethod
    def _add_attachment_image(doc: Any, image_path: str | None) -> None:
        """添加附件1：侦察图。"""
        heading = doc.add_paragraph()
        h_run = heading.add_run("附件1：侦察图像")
        h_run.bold = True
        h_run.font.size = Pt(13)

        if image_path and Path(image_path).exists():
            try:
                doc.add_picture(image_path, width=Cm(14))
                p = doc.paragraphs[-1]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            except Exception:
                p = doc.add_paragraph()
                p.add_run(f"[图像加载失败：{image_path}]").italic = True
        else:
            p = doc.add_paragraph()
            p.add_run("（侦察图像暂无或路径不可访问）").italic = True

        doc.add_paragraph()

    @staticmethod
    def _add_component_table(doc: Any, rows: list[dict]) -> None:
        """添加附件2：组成分布统计表。"""
        heading = doc.add_paragraph()
        h_run = heading.add_run("附件2：目标组成分布统计表")
        h_run.bold = True
        h_run.font.size = Pt(13)

        headers = _COMPONENT_TABLE_HEADERS
        n_cols = len(headers)

        table = doc.add_table(rows=1 + len(rows), cols=n_cols)
        table.style = "Table Grid"

        # 表头
        hdr_cells = table.rows[0].cells
        for i, h in enumerate(headers):
            hdr_cells[i].text = h
            run = hdr_cells[i].paragraphs[0].runs[0] if hdr_cells[i].paragraphs[0].runs else hdr_cells[i].paragraphs[0].add_run(h)
            run.bold = True
            run.font.size = Pt(11)

        # 数据行
        for row_idx, row_data in enumerate(rows, start=1):
            cells = table.rows[row_idx].cells
            cells[0].text = str(row_data.get("seq", row_idx))
            cells[1].text = str(row_data.get("target_type", ""))
            cells[2].text = str(row_data.get("sub_type", ""))
            lon = row_data.get("lon")
            lat = row_data.get("lat")
            cells[3].text = f"{lon:.6f}" if lon is not None else "—"
            cells[4].text = f"{lat:.6f}" if lat is not None else "—"
            for cell in cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(10)

        if not rows:
            doc.add_paragraph("（暂无目标数据）").italic = True

        doc.add_paragraph()

    @staticmethod
    def _add_equipment_table(doc: Any, rows: list[dict]) -> None:
        """添加附件3：装备分布统计表。"""
        heading = doc.add_paragraph()
        h_run = heading.add_run("附件3：装备分布统计表")
        h_run.bold = True
        h_run.font.size = Pt(13)

        headers = _EQUIPMENT_TABLE_HEADERS
        n_cols = len(headers)

        table = doc.add_table(rows=1 + len(rows), cols=n_cols)
        table.style = "Table Grid"

        # 表头
        hdr_cells = table.rows[0].cells
        for i, h in enumerate(headers):
            hdr_cells[i].text = h
            run = hdr_cells[i].paragraphs[0].runs[0] if hdr_cells[i].paragraphs[0].runs else hdr_cells[i].paragraphs[0].add_run(h)
            run.bold = True
            run.font.size = Pt(11)

        # 数据行
        for row_idx, row_data in enumerate(rows, start=1):
            cells = table.rows[row_idx].cells
            cells[0].text = str(row_data.get("seq", row_idx))
            cells[1].text = str(row_data.get("equipment_type", ""))
            lon = row_data.get("lon")
            lat = row_data.get("lat")
            cells[2].text = f"{lon:.6f}" if lon is not None else "—"
            cells[3].text = f"{lat:.6f}" if lat is not None else "—"
            conf = row_data.get("confidence")
            cells[4].text = f"{conf:.3f}" if conf is not None else "—"
            cells[5].text = str(row_data.get("review_status", "—"))
            for cell in cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(10)

        if not rows:
            doc.add_paragraph("（暂无装备数据）").italic = True

        doc.add_paragraph()

    @staticmethod
    def _add_footer(
        doc: Any,
        satellite: str,
        sensor: str,
        acq_time: str,
        pipeline_run_id: str,
        draft_mode: bool,
    ) -> None:
        """在文档每节的页脚中添加元数据信息。"""
        for section in doc.sections:
            footer = section.footer
            # 清除默认内容
            for para in footer.paragraphs:
                for run in para.runs:
                    run.text = ""

            if not footer.paragraphs:
                footer.add_paragraph()

            p = footer.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            parts: list[str] = []
            if satellite:
                parts.append(f"卫星：{satellite}")
            if sensor:
                parts.append(f"传感器：{sensor}")
            if acq_time:
                parts.append(f"成像时间：{acq_time}")
            if pipeline_run_id:
                parts.append(f"流水线：{pipeline_run_id}")

            footer_text = " | ".join(parts)
            if draft_mode:
                footer_text += "  【自动生成草稿，待人工审核】"

            run = p.add_run(footer_text)
            run.font.size = Pt(9)
            if draft_mode:
                run.font.color.rgb = RGBColor(0xAA, 0x00, 0x00)


# ---------------------------------------------------------------------------
# 顶层便捷函数（可直接调用，无需实例化）
# ---------------------------------------------------------------------------

def assemble_docx(
    evidence_package: dict,
    output_dir: str,
    template_path: str | None = None,
    draft_mode: bool = True,
) -> str:
    """便捷函数，直接调用 DocxAssembler.assemble()。"""
    return DocxAssembler().assemble(
        evidence_package,
        output_dir,
        template_path=template_path,
        draft_mode=draft_mode,
    )
