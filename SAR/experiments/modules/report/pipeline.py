"""
pipeline.py — M5+M6 联合入口

ReportPipeline.run(evidence_package, output_dir, ...) -> dict

将 ReportGenerator (M5) 与 DocxAssembler (M6) 串联，
输入状态为 READY_FOR_NLG 的 Evidence Package，
输出状态为 DOCX_RENDERED 的 Evidence Package，
同时将 report.docx.uri 写入包中。
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .collaborative import CollaborativeReportGenerator
from .docx_assembler import DocxAssembler
from .generator import LocalModelGenerator, ReportGenerator

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE = "/home/zhuxiang/RS/SAR/文档/成品.docx"


class ReportPipeline:
    """M5+M6 联合流水线。

    Parameters
    ----------
    generator_kwargs:
        传递给 ReportGenerator.__init__ 的关键字参数。
    """

    def __init__(self, **generator_kwargs: object) -> None:
        self._generator = ReportGenerator(**generator_kwargs)  # type: ignore[arg-type]
        self._assembler = DocxAssembler()

    @classmethod
    def from_local_model(
        cls,
        model_path: str,
        **kwargs: object,
    ) -> "ReportPipeline":
        """使用本地 Transformers 模型创建 ReportPipeline 实例。

        Parameters
        ----------
        model_path:
            本地模型目录，例如 "/mnt/data/zhuxiang/Qwen/Qwen3-4B"。
        **kwargs:
            传递给 LocalModelGenerator.__init__ 的其他关键字参数
            （device, max_new_tokens, temperature 等）。

        Returns
        -------
        ReportPipeline
            使用 LocalModelGenerator 的流水线实例。
        """
        instance = cls.__new__(cls)
        instance._generator = LocalModelGenerator(model_path, **kwargs)  # type: ignore[arg-type]
        instance._assembler = DocxAssembler()
        return instance

    @classmethod
    def from_collaborative_models(
        cls,
        small_model_name: str,
        small_base_url: str | None = None,
        small_model_path: str | None = None,
        large_model_name: str | None = None,
        large_base_url: str | None = None,
        large_model_path: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        cache_dir: str | None = None,
        local_device: str = "auto",
        local_max_new_tokens: int = 2048,
        require_gpu: bool = False,
        allow_template_fallback: bool = True,
    ) -> "ReportPipeline":
        instance = cls.__new__(cls)
        instance._generator = CollaborativeReportGenerator(
            small_model_name=small_model_name,
            small_base_url=small_base_url,
            small_model_path=small_model_path,
            large_model_name=large_model_name,
            large_base_url=large_base_url,
            large_model_path=large_model_path,
            api_key=api_key,
            timeout=timeout,
            cache_dir=cache_dir,
            local_device=local_device,
            local_max_new_tokens=local_max_new_tokens,
            require_gpu=require_gpu,
            allow_template_fallback=allow_template_fallback,
        )
        instance._assembler = DocxAssembler()
        return instance

    def run(
        self,
        evidence_package: dict,
        output_dir: str,
        template_path: str | None = None,
        draft_mode: bool = True,
    ) -> dict:
        """执行完整的 M5→M6 流水线。

        Parameters
        ----------
        evidence_package:
            状态为 READY_FOR_NLG 的 Evidence Package 字典。
        output_dir:
            .docx 输出目录（不存在时自动创建）。
        template_path:
            可选的 Word 模板文件路径。若为 None 且
            ``/home/zhuxiang/RS/SAR/文档/成品.docx`` 存在，则自动使用该文件。
        draft_mode:
            True = 审核版；False = 正式版。

        Returns
        -------
        dict
            状态推进至 DOCX_RENDERED 的 Evidence Package，
            report.docx.uri 已填写。
        """
        # 若未指定模板，尝试使用默认模板
        if template_path is None and Path(_DEFAULT_TEMPLATE).exists():
            template_path = _DEFAULT_TEMPLATE
            logger.info("使用默认模板：%s", template_path)

        # M5：文本生成
        logger.info("M5: 开始生成通报正文...")
        pkg = self._generator.generate(evidence_package)
        logger.info("M5: 正文生成完成，status=%s", pkg.get("status"))

        # M6：文档组装
        logger.info("M6: 开始组装 Word 文档...")
        docx_path = self._assembler.assemble(
            pkg,
            output_dir=output_dir,
            template_path=template_path,
            draft_mode=draft_mode,
        )
        logger.info("M6: Word 文档已输出至 %s", docx_path)

        # 填充 docx 信息，推进状态
        report = pkg.setdefault("report", {})
        report["docx"] = {
            "uri": f"file://{docx_path}",
            "template_version": DocxAssembler.TEMPLATE_VERSION,
        }
        pkg["status"] = "DOCX_RENDERED"
        pkg["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

        return pkg
