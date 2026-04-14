"""
generator.py — M5 文本生成模块

class ReportGenerator:
    __init__(model_name, base_url, api_key)
    generate(evidence_package: dict) -> dict   # 返回更新后的 Evidence Package

class LocalModelGenerator:
    __init__(model_path, device, max_new_tokens)
    generate(evidence_package: dict) -> dict   # 本地 Transformers 推理版本
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .prompt_templates import (
    _build_class_list_text,
    _fmt_acquisition_time,
    build_system_prompt,
    build_user_prompt,
)
from .table_builder import TableBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部常量
# ---------------------------------------------------------------------------

_SHIP_UNITS = {
    "carrier", "destroyer", "frigate", "replenishment",
    "amphibious", "other_vessel",
}


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """M5 文本生成器。

    调用 OpenAI-compatible API 生成通报正文，并执行后验证。
    若 LLM 调用失败或后验证不通过，自动 fallback 到模板填充。

    Parameters
    ----------
    model_name:
        LLM 模型名称，例如 "Qwen2.5-7B-Instruct"。
    base_url:
        OpenAI-compatible API 端点，例如 "http://localhost:8000/v1"。
        为 None 时直接走 fallback 模板路径（离线/测试用途）。
    api_key:
        API 鉴权密钥，无鉴权时传 "EMPTY" 或 None。
    timeout:
        HTTP 请求超时秒数，默认 60。
    """

    def __init__(
        self,
        model_name: str = "Qwen2.5-7B-Instruct",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        self._table_builder = TableBuilder()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def generate(self, evidence_package: dict) -> dict:
        """生成通报正文，返回填充了 report 字段的新 Evidence Package。

        Parameters
        ----------
        evidence_package:
            状态为 READY_FOR_NLG 的 Evidence Package 字典。

        Returns
        -------
        dict
            已填充 report.body / report.body_sections / report.tables /
            report.title / report.subtitle / report.report_date 的
            新 Evidence Package（状态推进至 REPORT_DRAFTED）。

        Raises
        ------
        ValueError
            若 evidence_package 的 status 不是 READY_FOR_NLG。
        """
        pkg = deepcopy(evidence_package)

        status = pkg.get("status", "")
        if status != "READY_FOR_NLG":
            raise ValueError(
                f"Evidence Package 状态应为 READY_FOR_NLG，实际为 {status!r}"
            )

        # 提取关键字段
        statistics = pkg.get("statistics", {})
        objects = pkg.get("objects", [])

        # 尝试 LLM 生成
        body: str | None = None
        source_tag = "template_v1"

        if self.base_url is not None:
            try:
                body = self._call_llm(pkg)
                source_tag = "llm_v1"
                logger.info("LLM 生成成功，执行后验证...")
                ok, reason = self._post_validate(body, statistics)
                if not ok:
                    logger.warning("后验证失败：%s，切换到模板 fallback", reason)
                    body = None
                    source_tag = "template_v1"
            except Exception as exc:
                logger.warning("LLM 调用失败：%s，切换到模板 fallback", exc)
                body = None

        if body is None:
            body = self._template_fallback(pkg)

        # 构建表格（程序化，不经过 LLM）
        component_table = self._table_builder.build_component_table(objects)
        equipment_table = self._table_builder.build_equipment_table(objects)

        # 填充 report 字段
        inp = pkg.get("input", {})
        metadata = inp.get("metadata", {})
        mission = inp.get("mission", {})

        acq_time_raw = metadata.get("acquisition_time", "")
        report_date_cn = _fmt_acquisition_time(acq_time_raw) if acq_time_raw else ""
        region_name = mission.get("region_name", "未知区域")

        report = pkg.setdefault("report", {})
        report["title"] = report.get("title") or "航天通报"
        report["subtitle"] = report.get("subtitle") or f"{region_name}SAR目标监测通报"
        report["report_date"] = report.get("report_date") or report_date_cn
        report["body"] = body
        report["body_sections"] = [
            {
                "section_name": "summary",
                "content": body,
                "source": source_tag,
            }
        ]
        report["tables"] = {
            "component_table": component_table,
            "equipment_table": equipment_table,
        }

        # 推进状态
        pkg["status"] = "REPORT_DRAFTED"
        pkg["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

        return pkg

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    def _call_llm(self, evidence: dict) -> str:
        """调用 OpenAI-compatible Chat Completions API，返回生成的正文字符串。"""
        try:
            import openai  # type: ignore
        except ImportError:
            # fallback: 用 requests 手写
            return self._call_llm_requests(evidence)

        client = openai.OpenAI(
            base_url=f"{self.base_url}",
            api_key=self.api_key,
        )
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(evidence)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
            timeout=self.timeout,
        )
        return response.choices[0].message.content.strip()

    def _call_llm_requests(self, evidence: dict) -> str:
        """纯 requests 实现的 OpenAI-compatible 调用（openai SDK 不可用时使用）。"""
        import requests  # type: ignore

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(evidence)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    # 后验证
    # ------------------------------------------------------------------

    def _post_validate(self, body: str, statistics: dict) -> tuple[bool, str]:
        """验证 LLM 输出的正文中数字与 statistics 是否一致。

        Returns
        -------
        tuple[bool, str]
            (True, "") 表示通过；(False, reason) 表示未通过。
        """
        totals = statistics.get("totals", {})
        by_class = statistics.get("by_class", [])

        # 检查总舰船数
        ships = totals.get("ships", 0)
        aircraft = totals.get("aircraft", 0)

        # 简单数字提取：查找正文中所有"N艘"/"N架"
        ship_matches = re.findall(r"(\d+)\s*艘", body)
        aircraft_matches = re.findall(r"(\d+)\s*架", body)

        if ships > 0 and ship_matches:
            found_ships = sum(int(x) for x in ship_matches)
            # 正文中提到的单类数量之和可能超过总数（分别列举），
            # 只检查明确提到"共N艘"的数字
            total_ship_pattern = re.search(r"舰船\s*(\d+)\s*艘", body)
            if total_ship_pattern:
                mentioned = int(total_ship_pattern.group(1))
                if mentioned != ships:
                    return False, f"正文舰船总数 {mentioned} ≠ statistics.totals.ships {ships}"

        if aircraft > 0:
            total_aircraft_pattern = re.search(r"飞机\s*(\d+)\s*架", body)
            if total_aircraft_pattern:
                mentioned = int(total_aircraft_pattern.group(1))
                if mentioned != aircraft:
                    return False, f"正文飞机总数 {mentioned} ≠ statistics.totals.aircraft {aircraft}"

        # 检查 by_class 中每个类别的数量
        for cls_item in by_class:
            name_cn = cls_item.get("name_cn", "")
            count = cls_item.get("count", 0)
            code = cls_item.get("code", "")
            if not name_cn or count == 0:
                continue
            unit = "艘" if code in _SHIP_UNITS else "架"
            pattern = rf"{re.escape(name_cn)}\s*(\d+)\s*{unit}"
            match = re.search(pattern, body)
            if match:
                mentioned = int(match.group(1))
                if mentioned != count:
                    return False, (
                        f"正文 {name_cn} 数量 {mentioned} ≠ by_class 中 {count}"
                    )

        # 检查是否出现了 by_class 以外的类别名称
        known_names = {item.get("name_cn", "") for item in by_class}
        all_class_names = {
            "航母", "驱逐舰", "护卫舰", "综合补给舰", "两栖舰", "其他舰船",
            "战斗机", "轰炸机", "运输机", "预警机", "直升机", "其他飞机",
        }
        hallucinated = all_class_names - known_names
        for name in hallucinated:
            if name in body:
                return False, f"正文出现了 statistics.by_class 中不存在的类别：{name}"

        return True, ""

    # ------------------------------------------------------------------
    # Fallback 模板
    # ------------------------------------------------------------------

    def _template_fallback(self, evidence: dict) -> str:
        """当 LLM 不可用或后验证失败时，使用规则模板生成正文。"""
        inp = evidence.get("input", {})
        metadata = inp.get("metadata", {})
        mission = inp.get("mission", {})
        scene = evidence.get("scene", {})
        statistics = evidence.get("statistics", {})
        totals = statistics.get("totals", {})
        by_class = statistics.get("by_class", [])
        spatial_summary = statistics.get("spatial_summary", {})
        confidence_summary = statistics.get("confidence_summary", {})

        satellite = metadata.get("satellite", "侦察卫星")
        acq_time_raw = metadata.get("acquisition_time", "")
        date_cn = _fmt_acquisition_time(acq_time_raw) if acq_time_raw else "某日"
        region_name = mission.get("region_name", "目标区域")
        scene_type_cn = scene.get("scene_type_cn", "")

        all_objects = totals.get("all_objects", 0)
        ships = totals.get("ships", 0)
        aircraft = totals.get("aircraft", 0)

        needs_caution = confidence_summary.get("review_required_count", 0) > 0
        caution_prefix = "疑似" if needs_caution else ""

        # 首句：总体汇总
        summary_parts: list[str] = []
        if ships > 0:
            summary_parts.append(f"舰船{ships}艘")
        if aircraft > 0:
            summary_parts.append(f"飞机{aircraft}架")
        summary_str = "、".join(summary_parts) if summary_parts else f"军事目标{all_objects}个"

        scene_clause = f"{scene_type_cn}" if scene_type_cn else ""
        first_sentence = (
            f"据{satellite}卫星{date_cn}侦察，{region_name}"
            + (f"（{scene_clause}）" if scene_clause else "")
            + f"共{caution_prefix}发现{summary_str}。"
        )

        # 中段：按类别列举
        class_text = _build_class_list_text(by_class)
        middle_sentence = ""
        if class_text and class_text != "无":
            middle_sentence = f"主要包括{class_text}。"

        # 末句：空间分布
        distribution = spatial_summary.get("distribution", "")
        last_sentence = f"目标{distribution}。" if distribution else ""

        body = first_sentence + middle_sentence + last_sentence
        return body.strip()


# ---------------------------------------------------------------------------
# LocalModelGenerator
# ---------------------------------------------------------------------------


class LocalModelGenerator(ReportGenerator):
    """M5 本地模型文本生成器（基于 transformers 库，针对 Qwen3-4B 优化）。

    直接从本地磁盘加载 Hugging Face 格式的模型（如 Qwen3-4B），
    不依赖任何外部 API 服务。模型在首次调用 generate() 时才加载（懒加载），
    避免导入时 OOM。

    Parameters
    ----------
    model_path:
        本地模型目录，例如 "/mnt/data/zhuxiang/Qwen/Qwen3-4B"。
    device:
        推理设备，"auto" / "cpu" / "cuda" / "cuda:0" 等。
    max_new_tokens:
        最大生成 token 数，默认 512。
    enable_thinking:
        是否启用 Qwen3 思维链（CoT）。生产环境默认 False，
        设为 True 时模型会输出 <think>...</think> 块（调试用）。
    temperature:
        生成温度，默认 0.2。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
        temperature: float = 0.2,
    ) -> None:
        # 父类以 base_url=None 初始化（强制走 fallback，稍后会覆盖 _call_llm）
        super().__init__(model_name=str(model_path), base_url=None)
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        # 懒加载：__init__ 不触碰 GPU/transformers
        self._pipeline = None
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # VRAM 估算（静态方法，可在加载前调用）
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_vram_gb(model_path: str) -> float:
        """通过扫描 .safetensors 文件大小估算所需 VRAM（GB）。

        估算公式：所有 .safetensors 文件大小之和 × 1.2（激活开销系数）。
        若估算结果超过 10 GB，记录 WARNING 日志。

        Parameters
        ----------
        model_path:
            本地模型目录路径。

        Returns
        -------
        float
            估算所需 VRAM，单位 GB。
        """
        import os
        import glob as _glob

        pattern = os.path.join(model_path, "*.safetensors")
        files = _glob.glob(pattern)
        total_bytes = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        estimated_gb = total_bytes * 1.2 / (1024 ** 3)

        if estimated_gb > 10.0:
            logger.warning(
                "模型 VRAM 估算结果为 %.1f GB（>10 GB），请确认 GPU 显存充足。"
                "model_path=%s",
                estimated_gb,
                model_path,
            )
        else:
            logger.info(
                "模型 VRAM 估算：%.1f GB（%d 个 .safetensors 文件）",
                estimated_gb,
                len(files),
            )
        return estimated_gb

    # ------------------------------------------------------------------
    # 懒加载：首次推理时加载 tokenizer + model
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """首次调用时加载 AutoTokenizer + AutoModelForCausalLM。

        使用 AutoTokenizer / AutoModelForCausalLM 替代 pipeline()，
        以便精细控制 chat template 和 Qwen3 思维链开关。
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers 未安装，请执行：pip install transformers"
            ) from exc

        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "torch 未安装，请执行：pip install torch"
            ) from exc

        # 加载前打印 VRAM 估算（仅供参考，不阻塞）
        try:
            self.estimate_vram_gb(self.model_path)
        except Exception:
            pass  # 估算失败不影响加载

        logger.info("开始加载本地模型：%s（device=%s）", self.model_path, self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if self.device == "auto":
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = torch.float16
        elif "cuda" in self.device:
            load_kwargs["device_map"] = self.device
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        self._model.eval()
        # 兼容旧代码中对 _pipeline 的 None 检查
        self._pipeline = True
        logger.info("本地模型加载完成：%s", self.model_path)

    # ------------------------------------------------------------------
    # 静态工具：剥离 <think>...</think> 块
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """移除 Qwen3 输出中的 <think>...</think> 推理链块。"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # ------------------------------------------------------------------
    # 覆盖父类的 _call_llm，使用本地 tokenizer + model 推理
    # ------------------------------------------------------------------

    def _call_llm(self, evidence: dict) -> str:  # type: ignore[override]
        """使用本地 Qwen3 模型生成通报正文。

        - 使用 tokenizer.apply_chat_template 构造输入
        - enable_thinking=False 时在用户消息末尾追加 /no_think
        - 解码时只解码新生成的 token，跳过输入部分
        - 返回前剥离 <think>...</think> 块
        """
        if self._pipeline is None:
            self._load_pipeline()

        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError("torch 未安装，请执行：pip install torch") from exc

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(evidence)

        # Qwen3 思维链控制：生产环境追加 /no_think 跳过 CoT
        if not self.enable_thinking:
            user_prompt = user_prompt + " /no_think"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 使用 chat template 格式化输入（正确处理特殊 token）
        text = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(  # type: ignore[union-attr]
            text,
            return_tensors="pt",
        ).to(self._model.device)  # type: ignore[union-attr]

        with torch.no_grad():
            output_ids = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
            )

        # 只解码新生成的 token，跳过输入部分
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self._tokenizer.decode(  # type: ignore[union-attr]
            new_tokens,
            skip_special_tokens=True,
        )

        # 剥离 <think>...</think>（enable_thinking=True 时模型可能仍输出）
        return self._strip_thinking(raw_output)

    # ------------------------------------------------------------------
    # 覆盖父类 generate，确保走 LLM 路径（当 base_url 为 None 时父类跳过）
    # ------------------------------------------------------------------

    def generate(self, evidence_package: dict) -> dict:  # type: ignore[override]
        """使用本地模型生成通报正文。

        与父类逻辑相同，但强制尝试本地推理，失败时 fallback 到模板。
        """
        from copy import deepcopy

        pkg = deepcopy(evidence_package)
        status = pkg.get("status", "")
        if status != "READY_FOR_NLG":
            raise ValueError(
                f"Evidence Package 状态应为 READY_FOR_NLG，实际为 {status!r}"
            )

        statistics = pkg.get("statistics", {})
        objects = pkg.get("objects", [])

        body: str | None = None
        source_tag = "template_v1"

        try:
            body = self._call_llm(pkg)
            source_tag = "local_llm_v1"
            logger.info("本地模型生成成功，执行后验证...")
            ok, reason = self._post_validate(body, statistics)
            if not ok:
                logger.warning("后验证失败：%s，切换到模板 fallback", reason)
                body = None
                source_tag = "template_v1"
        except Exception as exc:
            logger.warning("本地模型调用失败：%s，切换到模板 fallback", exc)
            body = None

        if body is None:
            body = self._template_fallback(pkg)

        component_table = self._table_builder.build_component_table(objects)
        equipment_table = self._table_builder.build_equipment_table(objects)

        inp = pkg.get("input", {})
        metadata = inp.get("metadata", {})
        mission = inp.get("mission", {})

        acq_time_raw = metadata.get("acquisition_time", "")
        report_date_cn = _fmt_acquisition_time(acq_time_raw) if acq_time_raw else ""
        region_name = mission.get("region_name", "未知区域")

        report = pkg.setdefault("report", {})
        report["title"] = report.get("title") or "航天通报"
        report["subtitle"] = report.get("subtitle") or f"{region_name}SAR目标监测通报"
        report["report_date"] = report.get("report_date") or report_date_cn
        report["body"] = body
        report["body_sections"] = [
            {
                "section_name": "summary",
                "content": body,
                "source": source_tag,
            }
        ]
        report["tables"] = {
            "component_table": component_table,
            "equipment_table": equipment_table,
        }

        pkg["status"] = "REPORT_DRAFTED"
        from datetime import datetime, timezone
        pkg["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

        return pkg
