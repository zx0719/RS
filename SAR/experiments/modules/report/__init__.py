"""
modules/report — M5 NLG 文本生成 + M6 Word 文档组装

公开导出：
    ReportGenerator       — M5：基于 Evidence Package 调用 LLM 生成通报正文（OpenAI-compatible API）
    LocalModelGenerator   — M5：本地 Transformers 模型版本（离线推理）
    DocxAssembler         — M6：将 Evidence Package 组装为标准 .docx 文件
    ReportPipeline        — M5+M6 联合流水线（推荐入口）
    DEFAULT_TEMPLATE_PATH — 默认 Word 模板路径

快速使用（API 模式）：
    from modules.report import ReportPipeline

    pipeline = ReportPipeline(
        model_name="Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    result_pkg = pipeline.run(
        evidence_package=pkg,
        output_dir="/data/output",
        draft_mode=True,
    )

快速使用（本地模型模式）：
    from modules.report import ReportPipeline

    pipeline = ReportPipeline.from_local_model(
        model_path="/mnt/data/zhuxiang/Qwen/Qwen3-4B",
        device="cuda",
    )
    result_pkg = pipeline.run(evidence_package=pkg, output_dir="/data/output")
    print(result_pkg["status"])          # DOCX_RENDERED
    print(result_pkg["report"]["body"])  # 正文
    print(result_pkg["report"]["docx"]["uri"])  # file:///data/output/...docx
"""

from .docx_assembler import DEFAULT_TEMPLATE_PATH, DocxAssembler, assemble_docx
from .generator import LocalModelGenerator, ReportGenerator
from .pipeline import ReportPipeline
from .prompt_templates import build_system_prompt, build_user_prompt
from .table_builder import TableBuilder

__all__ = [
    "ReportGenerator",
    "LocalModelGenerator",
    "DocxAssembler",
    "ReportPipeline",
    "TableBuilder",
    "build_system_prompt",
    "build_user_prompt",
    "assemble_docx",
    "DEFAULT_TEMPLATE_PATH",
]
