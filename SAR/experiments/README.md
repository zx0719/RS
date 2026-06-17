# SAR图像军事目标情报通报自动生成系统

> 输入单景SAR卫星图像及其元数据，系统自动完成目标检测、坐标地理化、结构化证据生成、情报通报正文生成、Word文档组装。

**设计原则：专用感知工具 + 结构化证据 + small/large LLM 协同 + VLM 场景描述 + Word组装**，不做端到端自由生成式检测。

---

## 系统架构

```
输入：SAR图像（GeoTIFF/JPEG）+ 元数据 + 目标区域名称
  ↓
[M1] 预处理与元数据解析          modules/geo/preprocess.py
  ↓
[M2] YOLOv8-OBB 目标检测        modules/detector/
  ↓
[M3] GDAL 坐标地理化             modules/geo/geolocalize.py
  ↓
[M4] 证据融合（Evidence JSON）   modules/evidence/
  ↓
[M5] LLM 通报正文生成            modules/report/generator.py
  ↓
[M6] python-docx 文档组装        modules/report/docx_assembler.py
  ↓
[M7] 质量评估 & 幻觉检测         modules/eval/
  ↓
输出：标准通报.docx + 证据JSON + 标注图
```

---

## 快速开始

### 1. 安装环境

```bash
bash setup_uv.sh           # 创建轻量 uv dev 环境：运行时依赖 + pytest/ipykernel
```

按需安装重依赖：

```bash
bash setup_uv.sh geo       # + rasterio GeoTIFF 支持
bash setup_uv.sh det       # + ultralytics/YOLO 检测支持
bash setup_uv.sh llm       # + torch/transformers 本地 Qwen 推理
bash setup_uv.sh api       # + OpenAI-compatible API 客户端
bash setup_uv.sh all       # 全量依赖，不建议日常默认使用
```

旧 conda 入口 `setup_env.sh` 仍保留作兼容，但日常开发优先使用 `uv`。详见 [ENVIRONMENT.md](ENVIRONMENT.md)。

### 2. 运行端到端流水线

```bash
# 使用 MockDetector（无需模型权重）+ 规则模板（无需LLM）
python run_pipeline.py --image /path/to/image.tif --region 某某军港

# 使用本地 Qwen3-4B
python run_pipeline.py --image /path/to/image.tif --region 某某军港 \
    --llm-path /mnt/data/zhuxiang/Qwen/Qwen3-4B

# 使用真实检测权重 + API LLM
python run_pipeline.py --image /path/to/image.tif --region 某某军港 \
    --model runs/detect/ssdd_obb/weights/best.pt \
    --llm-url http://localhost:8000/v1 --final

# 使用协同大小模型
python run_pipeline.py --image /path/to/image.tif --region 某某军港 \
    --llm-url http://small-model-host:8000/v1 \
    --llm-model-name Qwen2.5-7B-Instruct \
    --large-llm-url http://large-model-host:8100/v1 \
    --large-llm-model-name Qwen2.5-72B-Instruct \
    --use-collaborative-llm
```

### 2.1 大图 docx 批处理接入协同模型

```bash
cp .env.collaborative.example .env.collaborative.local
# 按需填写：
#   SAR_SMALL_LLM_URL
#   SAR_LARGE_LLM_URL
#   SAR_VLM_URL

python scripts/build_original_large_scene_docx.py --env-file .env.collaborative.local
python scripts/build_large_scene_release_manifest.py
```

如已有 docx 和 evidence 只需要刷新“大模型参与元数据”，执行：

```bash
python scripts/refresh_large_scene_metadata.py --env-file .env.collaborative.local
python scripts/build_large_scene_release_manifest.py
```

### 2.2 协同验收与阈值调优

```bash
python scripts/check_collaborative_llm.py --env-file .env.collaborative.local
python scripts/run_collaborative_acceptance.py --env-file .env.collaborative.local
python scripts/sweep_collaborative_thresholds.py
python scripts/recommend_collaborative_thresholds.py
python scripts/write_recommended_env.py
python scripts/rerun_with_recommended_env.py
```

关键输出：
- `output/large_scene_original_format/collaborative_acceptance_report.json`
- `output/large_scene_original_format/collaborative_validation_report.json`
- `output/large_scene_original_format/vlm_validation_report.json`
- `output/large_scene_original_format/collaborative_route_summary.json`
- `output/large_scene_original_format/collaborative_threshold_recommendation.json`
- `.env.collaborative.recommended`

### 3. 运行测试

```bash
uv run pytest tests/ -v
# 76 passed, 1 skipped（Qwen3-4B路径挂载后自动解除skip）
```

---

## 目录结构

```
experiments/
├── run_pipeline.py              # 端到端CLI入口
├── setup_uv.sh                  # uv环境一键安装（推荐）
├── setup_env.sh                 # conda环境一键安装（兼容）
├── pyproject.toml               # uv/PEP 621依赖配置
├── requirements.txt             # 旧版完整依赖入口
├── setup.py                     # 包安装配置
├── PROJECT_STATUS.md            # 项目进度看板
│
├── modules/
│   ├── detector/                # M2: YOLOv8-OBB检测
│   │   ├── detector.py          # DetectorTool（推理接口）
│   │   ├── mock_detector.py     # MockDetector（测试用，无需权重）
│   │   ├── class_map.py         # 类别编码映射（12类）
│   │   ├── train.py             # 训练CLI
│   │   └── dataset_utils.py     # DOTA/SSDD→YOLO-OBB格式转换
│   │
│   ├── geo/                     # M1+M3: 预处理+地理化
│   │   ├── preprocess.py        # ImagePreprocessor（GeoTIFF/PIL fallback）
│   │   └── geolocalize.py       # GeoLocalizer（像素→WGS84）
│   │
│   ├── evidence/                # M4: 证据融合
│   │   ├── builder.py           # EvidenceBuilder（核心，统计纯程序化）
│   │   └── schema_validator.py  # Schema字段校验
│   │
│   ├── report/                  # M5+M6: 文本生成+文档组装
│   │   ├── generator.py         # ReportGenerator（API）/ LocalModelGenerator（本地）
│   │   ├── collaborative.py     # small/large/refine 协同生成
│   │   ├── collab_config.py     # 协同模型/阈值/VLM 配置装配
│   │   ├── prompt_templates.py  # LLM Prompt模板
│   │   ├── table_builder.py     # 统计表程序化生成
│   │   ├── docx_assembler.py    # Word文档组装（成品.docx模板）
│   │   ├── pipeline.py          # ReportPipeline（M5+M6串联）
│   │   ├── large_scene.py       # 大图 digest / 路由 / 解释
│   │   ├── vlm_describer.py     # VLM 场景整体描述
│   │   └── training_utils.py    # SFT训练辅助工具
│   │
│   └── eval/                    # M7: 质量评估
│       ├── consistency.py       # ConsistencyChecker（数量/类别/坐标一致性）
│       ├── hallucination.py     # HallucinationDetector（数字/类别幻觉检测）
│       └── quality_gate.py      # QualityGate（综合评判，填充quality块）
│
├── data/                        # 数据准备脚本（输出目录不入库）
│   ├── prepare_ssdd.py          # SSDD RBox VOC→YOLO-OBB转换
│   ├── prepare_sardet.py        # SARDet_100K转换（stub，二期）
│   ├── build_nlg_training_data.py  # FSAR-Cap→Evidence JSON→通报JSONL
│   ├── train_ssdd.sh            # YOLOv8m-OBB训练启动脚本
│   └── nlg_training/            # NLG训练数据输出目录（运行后生成）
│
└── tests/
    ├── conftest.py              # pytest fixtures
    ├── test_consistency.py      # 一致性检查单元测试（7）
    ├── test_hallucination.py    # 幻觉检测单元测试（11）
    ├── test_integration.py      # 集成测试（6）
    ├── test_e2e_smoke.py        # 端到端冒烟测试（4）
    ├── test_detector_mock.py    # MockDetector单元测试（7）
    ├── test_dataset_utils.py    # 数据集转换单元测试（14）
    └── test_nlg_training.py     # NLG训练数据单元测试（28）
```

---

## 核心设计原则

| 原则 | 说明 |
|------|------|
| **事实由工具给出** | 数量/类别/坐标等事实必须来自检测器和GDAL，LLM只负责语言组织 |
| **结构化证据防幻觉** | 所有关键事实以Evidence JSON传递，后验证确保LLM输出与证据一致 |
| **模块解耦** | 更换检测器/LLM/Word模板时不影响整体接口 |
| **优雅降级** | 无rasterio→PARTIAL_SUCCESS；无LLM→规则模板；无权重→MockDetector |

---

## 证据JSON Schema（v1.0）

状态流转：
```
RECEIVED → PREPROCESSED → DETECTED → GEOLOCATED → FUSED
→ READY_FOR_NLG → REPORT_DRAFTED → DOCX_RENDERED → REVIEW_PENDING → COMPLETED
```

目标类别：
- 舰船：`carrier` / `destroyer` / `frigate` / `replenishment` / `amphibious` / `other_vessel`
- 飞机：`fighter` / `bomber` / `transport` / `aew` / `helicopter` / `other_aircraft`

---

## 训练数据准备

```bash
# 1. 转换SSDD检测数据集（~928张，OBB格式）
python data/prepare_ssdd.py \
    --ssdd-root /mnt/data/mm_data/SAR/dection/SSDD/Official-SSDD-OPEN/RBox_SSDD/voc_style \
    --output-dir data/ssdd_yolo_obb

# 2. 生成NLG训练数据（从FSAR-Cap 50K图文对）
python data/build_nlg_training_data.py \
    --fsar-cap /mnt/data/mm_data/SAR/FSAR-Cap/FSAR-Captrain.json \
    --output data/nlg_training/train.jsonl \
    --max-samples 5000

# 3. 训练YOLOv8m-OBB检测器
bash data/train_ssdd.sh
```

---

## 依赖

| 包 | 用途 | 状态 |
|----|------|------|
| `python-docx` | Word文档组装 | ✅ 已安装 |
| `rasterio` | GeoTIFF坐标解析 | 需安装 |
| `ultralytics` | YOLOv8-OBB检测 | 需安装 |
| `transformers` + `torch` | 本地Qwen3-4B推理 | 需安装 |
| `openai` | API模式LLM调用 | 需安装 |

推荐安装：`bash setup_uv.sh`；旧 conda 安装：`bash setup_env.sh`

---

## 进度

详见 [PROJECT_STATUS.md](PROJECT_STATUS.md) | 部署要求与分辨率说明详见 [DEPLOYMENT.md](DEPLOYMENT.md)

| 阶段 | 状态 |
|------|------|
| 阶段0：接口与协议先行 | ✅ 完成 |
| 阶段1：Demo跑通（mock数据端到端） | ✅ 完成 |
| 阶段2：舰船正式版（真实GeoTIFF + YOLOv8-OBB） | 🔄 进行中 |
| 阶段3：飞机扩展版 | ⏳ 待开始 |
| 阶段4：增强版（VLM + 多时相 + LoRA） | 🔮 二期 |
