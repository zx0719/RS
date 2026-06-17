# modules/report — M5 NLG 文本生成 + M6 Word 文档组装

## 功能概述

本模块负责 SAR 情报通报系统流水线的最后两个阶段：

| 阶段 | 类 | 职责 |
|------|----|------|
| M5 | `ReportGenerator` | 读取结构化 Evidence Package，调用 LLM 生成通报正文；离线测试可启用规则模板 fallback |
| M5-协同 | `CollaborativeReportGenerator` | 负责 `small_llm / large_llm / large_refine / template` 协同生成 |
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

### 协同大小模型模式

```python
from modules.report import ReportPipeline

pipeline = ReportPipeline.from_collaborative_models(
    small_model_name="Qwen2.5-7B-Instruct",
    small_base_url="http://small-model-host:8000/v1",
    large_model_name="Qwen2.5-72B-Instruct",
    large_base_url="http://large-model-host:8100/v1",
    cache_dir="/tmp/report_cache",
)

result = pipeline.run(pkg, output_dir="/tmp/output")
```

### VLM 场景描述

VLM 只负责 `scene_description`，不参与数量和类别事实判断。主流水线、大图链路与模板 fallback 均可消费该字段。

### 离线/无 LLM 模式（自动 fallback）

```python
pipeline = ReportPipeline()   # base_url=None，自动使用规则模板
result = pipeline.run(pkg, output_dir="/tmp/output")
```

> 该模式只用于离线开发、单元测试或无模型环境下验证版式。生产/上星验收不得接受 `template_v1` 产物。

### 生产严格模式

上星前验收必须关闭模板兜底、要求本地模型 GPU trace，并强制 VLM 参与：

```bash
export SAR_SMALL_LLM_PATH=/mnt/data/zhuxiang/Qwen/Qwen3-4B
export SAR_LLM_DEVICE=cuda:0
export SAR_REQUIRE_GPU=1
export SAR_ALLOW_TEMPLATE_FALLBACK=0
export SAR_REQUIRE_VLM=1
export SAR_VLM_URL=http://127.0.0.1:8101/v1
bash scripts/run_gpu_small_model_acceptance.sh
```

最终发布闸门是 `scripts/validate_release_readiness.py`。它直接检查 evidence、DOCX、模型来源、GPU trace、VLM 描述使用情况和 `scene_description_trace`。
对于本地小模型验收，必须使用 `--require-local-gpu`，防止远端 API 模型服务被误认为本地 GPU 小模型产物。

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
| `collaborative.py` | `CollaborativeReportGenerator`：small/large/refine 协同生成 |
| `collab_config.py` | 协同配置、阈值、env-file 装配 |
| `large_scene.py` | 大图 digest、自动路由、route rationale |
| `prompt_templates.py` | Prompt 模板：`build_system_prompt()`、`build_user_prompt()` |
| `vlm_describer.py` | VLM 场景描述（scene-level only） |
| `table_builder.py` | `TableBuilder`：从 `objects[]` 程序化生成附件2/3 表格 |
| `docx_assembler.py` | `DocxAssembler`：M6 Word 组装，支持审核版/正式版；缺少 report.body 时直接失败 |

---

## 硬约束（设计保证）

1. **禁止数量幻觉**：`generator.py` 的后验证会检查 LLM 输出中所有数字与 `statistics.totals` / `statistics.by_class` 是否一致；严格模式下不一致直接失败，离线模式可 fallback。
2. **禁止类别幻觉**：后验证检查正文是否出现了 `by_class` 以外的目标类别；严格模式下不一致直接失败，离线模式可 fallback。
3. **保守措辞**：若 `quality.confidence_summary.review_required_count > 0`，Prompt 强制要求使用"疑似"/"初步判断"措辞。
4. **表格程序化生成**：`build_component_table` / `build_equipment_table` 严格从 `objects[]` 读取，LLM 无法影响表格内容。
5. **VLM 受限使用**：`scene_description` 仅作场景语义补充，不得改写数量和类别事实。

## 协同验收

真实 `small / large / VLM` 服务可用时，建议按如下顺序执行：

```bash
python scripts/check_collaborative_llm.py --env-file .env.collaborative.local
python scripts/run_collaborative_acceptance.py --env-file .env.collaborative.local --require-model-participation --require-gpu --require-local-gpu --require-vlm
python scripts/rerun_with_recommended_env.py --env-file .env.collaborative.recommended
```

关键产物：
- `output/large_scene_original_format/collaborative_acceptance_report.json`
- `output/large_scene_original_format/collaborative_validation_report.json`
- `output/large_scene_original_format/vlm_validation_report.json`
- `output/large_scene_original_format/release_readiness_report.json`
- `output/large_scene_original_format/collaborative_route_summary.json`
- `output/large_scene_original_format/collaborative_threshold_recommendation.json`

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
