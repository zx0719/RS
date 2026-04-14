# modules/report — M5 NLG 文本生成 + M6 Word 文档组装

## 功能概述

本模块负责 SAR 情报通报系统流水线的最后两个阶段：

| 阶段 | 类 | 职责 |
|------|----|------|
| M5 | `ReportGenerator` | 读取结构化 Evidence Package，调用 LLM 生成通报正文；LLM 不可用时自动 fallback 到规则模板 |
| M6 | `DocxAssembler` | 将正文、附件图、统计表组装为 `.docx` 文件 |
| M5+M6 | `ReportPipeline` | 串联 M5→M6 的联合入口，推荐使用 |

**输入状态**：`READY_FOR_NLG`  
**输出状态**：`DOCX_RENDERED`

---

## 安装依赖

```bash
pip install python-docx
# LLM 调用（二选一）
pip install openai         # 推荐
# 或只使用内置 requests（无需额外安装）
```

---

## 快速上手

### 完整流水线（推荐）

```python
from modules.report import ReportPipeline
import json

# 加载状态为 READY_FOR_NLG 的 Evidence Package
with open("evidence_package.json", encoding="utf-8") as f:
    pkg = json.load(f)

pipeline = ReportPipeline(
    model_name="Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",   # vLLM / Ollama 等 OpenAI-compatible 端点
    api_key="EMPTY",
)

result = pipeline.run(
    evidence_package=pkg,
    output_dir="/data/output",
    draft_mode=True,     # True=审核版，False=正式版
)

print(result["status"])                    # DOCX_RENDERED
print(result["report"]["body"])            # 通报正文
print(result["report"]["docx"]["uri"])     # file:///data/output/sar-xxx_draft.docx
```

### 离线/无 LLM 模式（自动 fallback）

```python
pipeline = ReportPipeline()   # base_url=None，自动使用规则模板
result = pipeline.run(pkg, output_dir="/tmp/output")
```

### 单独使用 M5（只生成正文）

```python
from modules.report import ReportGenerator

gen = ReportGenerator(model_name="Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1")
pkg_with_body = gen.generate(pkg)
print(pkg_with_body["report"]["body"])
```

### 单独使用 M6（只组装 Word）

```python
from modules.report import DocxAssembler

assembler = DocxAssembler()
docx_path = assembler.assemble(
    evidence_package=pkg,
    output_dir="/data/output",
    draft_mode=False,   # 正式版
)
print(docx_path)
```

### 单独构建表格

```python
from modules.report import TableBuilder

tb = TableBuilder()
component_table = tb.build_component_table(pkg["objects"])
equipment_table = tb.build_equipment_table(pkg["objects"])
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `__init__.py` | 包入口，导出所有公开类 |
| `pipeline.py` | `ReportPipeline`：M5+M6 联合流水线 |
| `generator.py` | `ReportGenerator`：M5 文本生成，含 LLM 调用 + 后验证 + fallback |
| `prompt_templates.py` | Prompt 模板：`build_system_prompt()`、`build_user_prompt()` |
| `table_builder.py` | `TableBuilder`：从 `objects[]` 程序化生成附件2/3 表格 |
| `docx_assembler.py` | `DocxAssembler`：M6 Word 组装，支持审核版/正式版 |

---

## 硬约束（设计保证）

1. **禁止数量幻觉**：`generator.py` 的后验证会检查 LLM 输出中所有数字与 `statistics.totals` / `statistics.by_class` 是否一致，不一致时自动 fallback。
2. **禁止类别幻觉**：后验证检查正文是否出现了 `by_class` 以外的目标类别，一旦发现则 fallback。
3. **保守措辞**：若 `quality.confidence_summary.review_required_count > 0`，Prompt 强制要求使用"疑似"/"初步判断"措辞。
4. **表格程序化生成**：`build_component_table` / `build_equipment_table` 严格从 `objects[]` 读取，LLM 无法影响表格内容。

---

## Word 文档结构

```
标题（航天通报）
副标题（XXX SAR目标监测通报）
日期

一、侦察情况
  正文段落（首行缩进）

附件1：侦察图像
  [图片或"暂无"提示]

附件2：目标组成分布统计表
  序号 | 目标大类 | 目标子类 | 经度 | 纬度

附件3：装备分布统计表
  序号 | 装备类型 | 经度 | 纬度 | 置信度 | 审核状态

页脚：卫星 | 传感器 | 成像时间 | 流水线ID
      [审核版] 【自动生成草稿，待人工审核】（红色字）
```

---

## 接口版本

- 兼容 Evidence Schema v1.0.0
- 模板版本：`brief-template-v1`
