# SAR图像军事目标情报通报系统 — 项目进度

> 最后更新：2026-04-14  
> 测试状态：**76 passed, 1 skipped**（Qwen3-4B路径挂载后自动解除skip）

---

## 一、总体进度概览

| 阶段 | 内容 | 状态 |
|------|------|------|
| 阶段0 | 接口与协议先行（Schema + 模块框架） | ✅ 完成 |
| 阶段1 | Demo跑通（mock数据端到端） | ✅ 完成 |
| 阶段2 | 舰船正式版（真实GeoTIFF + YOLOv8-OBB） | 🔄 进行中 |
| 阶段3 | 飞机扩展版 | ⏳ 待开始 |
| 阶段4 | 增强版（VLM + 多时相 + LoRA） | 🔮 二期 |

---

## 二、模块完成状态

### M1 预处理 (`modules/geo/preprocess.py`)
**状态：** ✅ 框架完成，⚠️ 需安装rasterio才能读取GeoTIFF地理参考

| 功能 | 状态 | 备注 |
|------|------|------|
| GeoTIFF元数据解析 | ✅ | 需rasterio/GDAL |
| Pillow fallback（无geo库时读图像尺寸） | ✅ | 已实现，优雅降级 |
| 无地理参考时降级为PARTIAL_SUCCESS | ✅ | 不抛异常 |

---

### M2 目标检测 (`modules/detector/`)
**状态：** ✅ 框架完成，⚠️ 需安装ultralytics + 真实训练权重

| 功能 | 状态 | 备注 |
|------|------|------|
| YOLOv8-OBB推理接口 (`DetectorTool`) | ✅ | 需ultralytics |
| YOLO类别→schema编码映射 | ✅ | `class_map.py` |
| OBB四点多边形计算 | ✅ | |
| 训练脚本CLI (`train.py`) | ✅ | |
| DOTA→YOLO-OBB格式转换 | ✅ | `dataset_utils.py` |
| Mock检测器（无ultralytics可用） | ❌ | 待实现 |
| 实际训练权重 (`best.pt`) | 🔄 | SSDD数据集已就绪，转换脚本已写 |

---

### M3 坐标地理化 (`modules/geo/geolocalize.py`)
**状态：** ✅ 框架完成，⚠️ 需安装rasterio

| 功能 | 状态 | 备注 |
|------|------|------|
| 仿射变换像素→WGS84 | ✅ | 需rasterio |
| OBB四角坐标转换 | ✅ | |
| 物理尺寸估算（米） | ✅ | GSD × 像素宽高 |
| 无geo参考时PARTIAL_SUCCESS降级 | ✅ | |

---

### M4 证据融合 (`modules/evidence/`)
**状态：** ✅ 完成（无外部依赖）

| 功能 | 状态 | 备注 |
|------|------|------|
| Evidence Package构建器 | ✅ | `EvidenceBuilder.build()` |
| 统计纯程序化计算（不经LLM） | ✅ | |
| Haversine距离 + Union-Find聚类 | ✅ | |
| Schema字段校验 (`schema_validator.py`) | ✅ | |
| 数量一致性硬校验 | ✅ | 不一致抛ValueError |

---

### M5 文本生成 (`modules/report/generator.py`)
**状态：** ✅ 框架完成，⚠️ 需安装transformers

| 功能 | 状态 | 备注 |
|------|------|------|
| OpenAI-compatible API调用 (`ReportGenerator`) | ✅ | |
| 本地Qwen3-4B推理 (`LocalModelGenerator`) | ✅ | 需transformers |
| 规则模板fallback（LLM不可用时） | ✅ | 无依赖，可立即用 |
| 后验证（数字/类别一致性） | ✅ | |
| 保守措辞（低置信度时"疑似"） | ✅ | |

**当前测试路径：** 规则模板fallback（27个测试全通过）  
**待测试路径：** Qwen3-4B本地推理（需`pip install transformers accelerate`）

---

### M6 文档组装 (`modules/report/docx_assembler.py`)
**状态：** ✅ 完成

| 功能 | 状态 | 备注 |
|------|------|------|
| 标题/正文/附件结构组装 | ✅ | python-docx |
| 成品.docx模板自动加载 | ✅ | 默认路径已配置 |
| 模板清空正文保留页眉页脚 | ✅ | XML操作 |
| 附件2：组成分布统计表 | ✅ | 程序化生成 |
| 附件3：装备分布统计表 | ✅ | 程序化生成 |
| 审核版/正式版双输出 | ✅ | `draft_mode` 参数 |
| 附件1：侦察图插入 | ✅ | 图像路径可选 |

---

### M7 评估与质量控制 (`modules/eval/`)
**状态：** ✅ 完成

| 功能 | 状态 | 备注 |
|------|------|------|
| 数量一致性检查 | ✅ | |
| 类别一致性检查（list-of-dicts） | ✅ | |
| 坐标完整性检查 | ✅ | |
| NLG幻觉检测（数字/类别） | ✅ | |
| 年份数字过滤（避免误报） | ✅ | |
| format_compliant（接受report_date） | ✅ | |
| review_gate自动触发 | ✅ | |

---

### 流水线入口
**状态：** ✅ 完成

| 功能 | 状态 | 备注 |
|------|------|------|
| 主入口脚本 (`run_pipeline.py`) | ✅ | `python run_pipeline.py --image x.tif --region 某某军港` |
| Mock检测器 (`mock_detector.py`) | ✅ | 无需ultralytics，可复现 |
| 真实图像端到端Demo | ❌ | 需GeoTIFF + 权重 |

---

## 三、测试覆盖情况

```
tests/
├── test_consistency.py       7个单元测试  ✅ 全通过
├── test_hallucination.py    11个单元测试  ✅ 全通过
├── test_integration.py       6个集成测试  ✅ 全通过
├── test_e2e_smoke.py         4个冒烟测试  ✅ 3通过 1 skip
├── test_detector_mock.py     7个单元测试  ✅ 全通过
├── test_dataset_utils.py    14个单元测试  ✅ 全通过（SSDD转换）
└── test_nlg_training.py     28个单元测试  ✅ 全通过（NLG训练数据）
合计：77个测试，76通过，1 skip（Qwen3-4B路径挂载后自动解除）
```

---

## 四、环境依赖状态

| 包 | 用途 | 安装状态 | 安装命令 |
|----|------|----------|----------|
| `python-docx` | M6 Word组装 | ✅ 已安装 | — |
| `rasterio` | M1/M3 GeoTIFF解析 | ❌ 未安装 | `pip install rasterio` |
| `ultralytics` | M2 YOLOv8-OBB检测 | ❌ 未安装 | `pip install ultralytics` |
| `transformers` | M5 本地LLM推理 | ❌ 未安装 | `pip install transformers accelerate` |
| `torch` (CUDA) | transformers后端 | ❌ 未安装 | 见 `setup_env.sh` |
| `openai` | M5 API模式 | ❌ 未安装 | `pip install openai` |
| `Pillow` | M1 图像尺寸fallback | 待确认 | `pip install Pillow` |

**快速安装（推荐）：**
```bash
bash setup_env.sh       # 创建 conda env "sar-intel" 并安装所有依赖
conda activate sar-intel
pytest tests/ -v        # 验证全部通过
```

---

## 五、关键路径 & 下一步行动

### 立即可做（无阻塞）

- [ ] **运行 `setup_env.sh`** 安装依赖环境
- [x] ~~实现 `run_pipeline.py`~~ ✅ 已完成
- [x] ~~实现 MockDetector~~ ✅ 已完成
- [ ] **跑端到端冒烟**：`python run_pipeline.py --image 任意图片.jpg --region 测试区域`

### 依赖安装后可做

- [ ] **Qwen3-4B冒烟测试** — `pytest tests/test_e2e_smoke.py::test_local_qwen_generate`
- [ ] **真实GeoTIFF端到端** — 用一张GF-3或SSDD图像跑完整链路，验证5字段最小闭环
- [ ] **Prompt调优** — 根据实际生成结果调整 `prompt_templates.py` 的风格约束

### 需要数据/模型

- [ ] **SAR舰船OBB标注数据集** — 1000~1500张，DOTA格式
- [ ] **训练YOLOv8-OBB** — `python modules/detector/train.py --data dataset.yaml`
- [ ] **证据JSON→通报正文对** — 300~500份，用于Qwen3-4B SFT微调

---

## 六、项目文件结构

```
experiments/
├── CLAUDE.md                    # Agent团队配置与项目规范
├── PROJECT_STATUS.md            # 本文件
├── requirements.txt             # 完整依赖
├── requirements-dev.txt         # 开发依赖
├── setup.py                     # 包安装配置
├── setup_env.sh                 # conda环境一键安装脚本
│
├── modules/
│   ├── detector/                # M2: YOLOv8-OBB检测
│   │   ├── detector.py          # DetectorTool（推理接口）
│   │   ├── class_map.py         # 类别编码映射
│   │   ├── train.py             # 训练CLI
│   │   └── dataset_utils.py     # DOTA→YOLO-OBB转换
│   │
│   ├── geo/                     # M1+M3: 预处理+地理化
│   │   ├── preprocess.py        # ImagePreprocessor
│   │   └── geolocalize.py       # GeoLocalizer
│   │
│   ├── evidence/                # M4: 证据融合
│   │   ├── builder.py           # EvidenceBuilder（核心）
│   │   └── schema_validator.py  # Schema校验
│   │
│   ├── report/                  # M5+M6: 文本生成+文档组装
│   │   ├── generator.py         # ReportGenerator / LocalModelGenerator
│   │   ├── prompt_templates.py  # LLM Prompt模板
│   │   ├── table_builder.py     # 统计表程序化生成
│   │   ├── docx_assembler.py    # Word文档组装（成品.docx模板）
│   │   └── pipeline.py          # ReportPipeline（M5+M6串联）
│   │
│   └── eval/                    # M7: 质量评估
│       ├── consistency.py       # ConsistencyChecker
│       ├── hallucination.py     # HallucinationDetector
│       └── quality_gate.py      # QualityGate（综合评判）
│
└── tests/
    ├── conftest.py              # pytest fixtures
    ├── test_consistency.py      # 一致性单元测试
    ├── test_hallucination.py    # 幻觉检测单元测试
    ├── test_integration.py      # 集成测试
    └── test_e2e_smoke.py        # 端到端冒烟测试
```

---

## 七、一期验收指标（参考需求文档v2.0）

| 层级 | 指标 | 目标值 | 当前状态 |
|------|------|--------|----------|
| 检测 | mAP@0.5 (OBB) | > 0.70 | ⏳ 未开始训练 |
| 检测 | 单张推理 | < 200ms | ⏳ 未测试 |
| 证据 | JSON schema合规率 | 100% | ✅ schema_validator覆盖 |
| 证据 | 数量统计一致率 | 100% | ✅ 硬校验 |
| 证据 | 坐标转换成功率 | > 95% | ⏳ 需rasterio |
| 文本 | 数字一致率 | > 99% | ✅ 后验证+fallback |
| 文本 | 幻觉字段率 | < 1% | ✅ HallucinationDetector |
| 文本 | 模板合规率 | > 95% | ✅ docx组装结构完整 |
| 端到端 | 全流程耗时 | < 30s | ⏳ 未测试 |
| 端到端 | 可审计率 | 100% | ✅ 每条结论可追溯证据 |

---

## 八、数据集资源

路径：`/mnt/data/mm_data/`

### 检测训练数据

| 数据集 | 路径 | 图像数 | 类别 | 格式 | 用途 |
|--------|------|--------|------|------|------|
| SSDD RBox | `SAR/dection/SSDD/.../RBox_SSDD/voc_style/` | ~1100 | ship | VOC XML + OBB | **一期主训练集** |
| FUSAR-Ship 1.0 | `SAR/dection/高分辨率船只数据集FUSAR-Ship1.0/SAR/` | 多类型 | Cargo/Fishing/Tanker等 | 目录分类 | 细粒度类别映射参考 |
| SARDet_100K | `SAR/SARDet_100K/` | 94,493 | ship/aircraft/car/tank/bridge | COCO HBB | 二期扩充（需OBB转换） |
| SAR-Airport | `SAR/dection/星载SAR机场检测数据集(SAR-Airport-1.0)/` | — | airport | — | 机场场景飞机检测 |

### NLG训练数据

| 数据集 | 路径 | 样本数 | 格式 | 用途 |
|--------|------|--------|------|------|
| FSAR-Cap | `SAR/FSAR-Cap/FSAR-Captrain.json` | 50,000 | 英文图文对 | 生成Evidence JSON→报告训练对 |
| SARLANG-1M Caption | `SAR/SARLANG-1M/Text/Caption/train/` | 31,968 | 英文图文对 | 补充训练数据 |

### 训练数据准备流程

```bash
# 1. 转换SSDD为YOLO-OBB格式
python data/prepare_ssdd.py \
    --ssdd-root /mnt/data/mm_data/SAR/dection/SSDD/Official-SSDD-OPEN/RBox_SSDD/voc_style \
    --output-dir data/ssdd_yolo_obb

# 2. 生成NLG训练数据
python data/build_nlg_training_data.py \
    --fsar-cap /mnt/data/mm_data/SAR/FSAR-Cap/FSAR-Captrain.json \
    --output data/nlg_training/train.jsonl

# 3. 开始检测器训练
bash data/train_ssdd.sh
```
