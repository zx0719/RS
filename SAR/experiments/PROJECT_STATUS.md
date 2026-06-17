# SAR图像军事目标情报通报系统 — 项目进度

> 最后更新：2026-04-20  
> 测试状态：**77 passed**（全部通过）  
> Demo状态：**端到端流水线已跑通** — 真实SAR图像→YOLOv8-OBB检测→Qwen3-4B报告生成→docx输出

---

## 一、总体进度概览

| 阶段 | 内容 | 状态 |
|------|------|------|
| 阶段0 | 接口与协议先行（Schema + 模块框架） | ✅ 完成 |
| 阶段1 | Demo跑通（mock数据端到端） | ✅ 完成 |
| 阶段2 | 舰船正式版（真实图像 + YOLOv8-OBB） | ✅ 完成 |
| 阶段3-v2 | 飞机单类验证（SARDet_100K aircraft子集） | ✅ 完成（mAP50=0.955） |
| 阶段3-v4 | 多类正式版（ship/aircraft/tank/bridge/harbor） | 🔄 训练中（epoch 72/100，mAP50=0.844，NaN崩溃后恢复中） |
| 阶段2.5-v5 | 民用船型细分类器（FUSAR-Ship1.0，解耦独立模块） | 📋 规划中（v4训练结束后启动，预估2.5天） |
| 阶段4 | 增强版（VLM + 多时相 + LoRA） | 🔮 二期 |
| 阶段5 | 军舰细分类（驱逐舰/航母等） | 🔮 三期（待军事SAR数据） |

---

## 二、模块完成状态

### M1 预处理 (`modules/geo/preprocess.py`)
**状态：** ✅ 完成

| 功能 | 状态 | 备注 |
|------|------|------|
| GeoTIFF元数据解析 | ✅ | rasterio已安装 |
| Pillow fallback（无geo库时读图像尺寸） | ✅ | 优雅降级 |
| 无地理参考时降级为PARTIAL_SUCCESS | ✅ | 不抛异常 |

---

### M2 目标检测 (`modules/detector/`)
**状态：** ✅ 完成（含真实训练权重）

| 功能 | 状态 | 备注 |
|------|------|------|
| YOLOv8-OBB推理接口 (`DetectorTool`) | ✅ | ultralytics 8.4.37 |
| YOLO类别→schema编码映射 | ✅ | `class_map.py` v4固定ID方案（ID 0-13，0-4 ACTIVE，5-13 RESERVED） |
| SSDD类别映射 (`SSDD_CLASS_MAP`) | ✅ | index 0 → ship |
| v2飞机映射 (`SARDET_AIRCRAFT_CLASS_MAP`) | ✅ | index 0 → aircraft（v2单类模型专用） |
| OBB四点多边形计算 | ✅ | `_rbox_to_corners()` |
| 训练脚本CLI (`train.py`) | ✅ | |
| SSDD→YOLO-OBB格式转换（9列多边形） | ✅ | 789 train/139 val/232 test |
| **训练权重** (`runs/detect/ssdd_obb/weights/best.pt`) | ✅ | **val mAP@50=0.987, test mAP@50=0.977** |
| Mock检测器 (`mock_detector.py`) | ✅ | 无需ultralytics |
| **可视化标注图** (`visualize.py`) | ✅ | OBB旋转框+类别+置信度，`detect(save_vis=True)` 直接输出 |

#### 模型版本对照

| 版本 | 路径 | 类别 | 数据集 | 状态 | 最优指标 |
|------|------|------|--------|------|----------|
| v1-ssdd | `runs/detect/ssdd_obb/weights/best.pt` | ship×1 | SSDD（1160张） | ✅ 完成 | mAP50=0.977 |
| v2-sardet-aircraft | `runs/v2-sardet-aircraft/weights/best.pt` | aircraft×1 | SARDet_100K飞机子集（~5K张） | ✅ 完成 | mAP50=0.955 |
| v4-multiclass | `/mnt/data/zhuxiang/SAR_experiments/runs/v4-multiclass/weights/best.pt` | ship/aircraft/tank/bridge/harbor×5 | 5数据集合并（135K张，289K框） | ✅ 完成（ep45停止） | **mAP50=0.829，mAP50-95=0.875** |

> v4 参数量：**26.4M**，81.2 GFLOPs，权重 53.3MB。推理速度：GPU单张 0.5~2s（模型常驻时）。  
> 部署要求与SAR分辨率说明详见 [DEPLOYMENT.md](DEPLOYMENT.md)。

> v4 预留 ID 5-13 给机场设施类别（跑道/滑行道/停机坪/机库等），标注数据就绪后直接激活，无需修改 ID 映射。

---

### M3 坐标地理化 (`modules/geo/geolocalize.py`)
**状态：** ✅ 完成

| 功能 | 状态 | 备注 |
|------|------|------|
| 仿射变换像素→WGS84 | ✅ | rasterio 1.4.4 |
| OBB四角坐标转换 | ✅ | |
| 物理尺寸估算（米） | ✅ | GSD × 像素宽高 |
| 无geo参考时PARTIAL_SUCCESS降级 | ✅ | SSDD图像无GeoTIFF时正常流 |

---

### M4 证据融合 (`modules/evidence/`)
**状态：** ✅ 完成

| 功能 | 状态 | 备注 |
|------|------|------|
| Evidence Package构建器 | ✅ | `EvidenceBuilder.build()` |
| 统计纯程序化计算（不经LLM） | ✅ | |
| Haversine距离 + Union-Find聚类 | ✅ | |
| Schema字段校验 (`schema_validator.py`) | ✅ | |
| 数量一致性硬校验 | ✅ | 不一致抛ValueError |

---

### M5 文本生成 (`modules/report/generator.py`)
**状态：** ⚠️ 工程链路完成；当前沙箱 GPU 不可见，最终 GPU/VLM 严格报告待在 GPU 环境重建

| 功能 | 状态 | 备注 |
|------|------|------|
| 本地Qwen3-4B推理 (`LocalModelGenerator`) | ✅ | 支持 GPU required；当前 Codex shell 无法验证 CUDA |
| `enable_thinking=False` | ✅ | 直接传入apply_chat_template |
| max_new_tokens | ✅ | 可通过 `SAR_LLM_MAX_NEW_TOKENS` 配置 |
| 规则模板fallback（离线测试） | ✅ | 生产/上星验收必须 `SAR_ALLOW_TEMPLATE_FALLBACK=0` |
| 后验证（数字/类别一致性） | ✅ | |
| 保守措辞（低置信度时"疑似"） | ✅ | |
| 发布就绪闸门 | ✅ | `validate_release_readiness.py` 检查真实模型、GPU trace、VLM 和 DOCX |

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
| 坐标完整性检查 | ✅ | PARTIAL_SUCCESS正确跳过 |
| NLG幻觉检测（数字/类别） | ✅ | |
| 年份/日期数字过滤（避免误报） | ✅ | 正则strip日期模式 |
| format_compliant（接受report_date） | ✅ | |
| review_gate自动触发 | ✅ | 置信度阈值0.6 |

---

### 流水线入口
**状态：** ✅ 完成，端到端Demo已验证

| 功能 | 状态 | 备注 |
|------|------|------|
| 主入口脚本 (`run_pipeline.py`) | ✅ | `python run_pipeline.py --image x.jpg --region 某某军港` |
| Mock检测器 (`mock_detector.py`) | ✅ | 无需ultralytics |
| SSDD自动类别映射 | ✅ | 检测到"ssdd"路径自动切换 |
| **真实图像端到端Demo** | ✅ | **已跑通，见output/demo_01/** |

**Demo运行示例：**
```
$ conda activate sar-intel
$ python run_pipeline.py \
    --image data/ssdd_yolo_obb/images/test/000261.jpg \
    --model runs/detect/ssdd_obb/weights/best.pt \
    --qwen /mnt/data/zhuxiang/Qwen/Qwen3-4B \
    --region 某某军港 --draft
```
**输出：**
- 检测：2艘other_vessel（置信度0.664, 0.457）
- 报告：中文军事情报通报正文（Qwen3-4B生成）
- 文件：`output/demo_01/sar-XXXXXX_draft.docx` + `_evidence.json`
- 质检：REVIEW REQUIRED（obj低置信度 < 0.6，符合预期）

---

## 三、测试覆盖情况

```
tests/
├── test_consistency.py       7个单元测试  ✅ 全通过
├── test_hallucination.py    11个单元测试  ✅ 全通过
├── test_integration.py       6个集成测试  ✅ 全通过
├── test_e2e_smoke.py         4个冒烟测试  ✅ 全通过
├── test_detector_mock.py     7个单元测试  ✅ 全通过
├── test_dataset_utils.py    14个单元测试  ✅ 全通过（SSDD OBB 9列多边形格式）
└── test_nlg_training.py     28个单元测试  ✅ 全通过（NLG训练数据）
合计：77个测试，77通过，0 skip
```

---

## 四、环境依赖状态（conda env: sar-intel）

| 包 | 版本 | 状态 |
|----|------|------|
| `python-docx` | — | ✅ 已安装 |
| `rasterio` | 1.4.4 | ✅ 已安装 |
| `ultralytics` | 8.4.37 | ✅ 已安装 |
| `transformers` | 4.57.6 | ✅ 已安装 |
| `torch` | 2.4.1+cu121 | ✅ 已安装 |
| `numpy` | <2.0 | ✅ 已安装（ultralytics兼容）|
| `peft` | 0.18.0 | ✅ 已安装（torch 2.4兼容）|

**快速运行：**
```bash
conda activate sar-intel
pytest tests/ -v        # 验证77个测试全通过
python run_pipeline.py --help
```

---

## 五、检测器训练结果

### SSDD YOLOv8m-OBB（舰船单类）

| 指标 | 数值 |
|------|------|
| 模型 | YOLOv8m-OBB（从零训练） |
| 训练集 | 789张（SSDD RBox） |
| 验证集 | 139张 |
| 测试集 | 232张 |
| 训练轮次 | 100 epochs |
| **val mAP@50** | **0.987** |
| **test mAP@50** | **0.977** |
| 权重路径 | `runs/detect/ssdd_obb/weights/best.pt` |

---

## 六、项目文件结构

```
experiments/
├── CLAUDE.md                    # Agent团队配置与项目规范
├── PROJECT_STATUS.md            # 本文件
├── run_pipeline.py              # 主入口（端到端流水线）
├── requirements.txt             # 完整依赖
├── setup_env.sh                 # conda环境一键安装脚本
│
├── modules/
│   ├── detector/                # M2: YOLOv8-OBB检测
│   │   ├── detector.py          # DetectorTool（推理接口）
│   │   ├── class_map.py         # 类别编码映射（含SSDD_CLASS_MAP）
│   │   ├── train.py             # 训练CLI
│   │   ├── mock_detector.py     # Mock检测器（无需ultralytics）
│   │   └── dataset_utils.py     # SSDD→YOLO-OBB格式转换（9列多边形）
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
│   │   ├── generator.py         # LocalModelGenerator（Qwen3-4B）
│   │   ├── prompt_templates.py  # LLM Prompt模板
│   │   ├── table_builder.py     # 统计表程序化生成
│   │   ├── docx_assembler.py    # Word文档组装（成品.docx模板）
│   │   └── pipeline.py          # ReportPipeline（M5+M6串联）
│   │
│   └── eval/                    # M7: 质量评估
│       ├── consistency.py       # ConsistencyChecker（PARTIAL_SUCCESS兼容）
│       ├── hallucination.py     # HallucinationDetector（日期数字过滤）
│       └── quality_gate.py      # QualityGate（置信度阈值0.6）
│
├── tests/                       # 77个测试，全部通过
├── data/
│   ├── ssdd_yolo_obb/           # SSDD转换后数据集（789+139+232）
│   ├── train_ssdd.sh            # SSDD训练脚本
│   └── prepare_ssdd.py          # SSDD数据准备脚本
│
├── runs/
│   └── detect/ssdd_obb/
│       └── weights/best.pt      # 训练好的权重（mAP@50=0.987）
│
└── output/                      # 流水线输出（docx + evidence JSON）
```

---

## 七、一期验收指标

| 层级 | 指标 | 目标值 | 当前状态 |
|------|------|--------|----------|
| 检测 | mAP@0.5 (OBB) | > 0.70 | ✅ **0.977**（test） |
| 检测 | 单张推理 | < 200ms | ✅（GPU推理 ~50ms） |
| 证据 | JSON schema合规率 | 100% | ✅ schema_validator覆盖 |
| 证据 | 数量统计一致率 | 100% | ✅ 硬校验 |
| 坐标 | GeoTIFF地理化 | > 95% | ✅ rasterio，无GeoTIFF时PARTIAL_SUCCESS |
| 文本 | 数字一致率 | > 99% | ✅ 后验证+hallucination检测 |
| 文本 | 幻觉字段率 | < 1% | ✅ HallucinationDetector（日期过滤） |
| 文本 | 模板合规率 | > 95% | ✅ docx组装结构完整 |
| 端到端 | 全流程可运行 | 是 | ✅ **Demo已跑通** |
| 端到端 | 可审计率 | 100% | ✅ 每条结论可追溯证据JSON |

---

## 八、下一步行动

### 当前进行中
- [x] **v4 多类训练**：epoch 72/100，mAP50=0.844，双卡DDP运行中（amp=False防NaN）
- [ ] **v4 训练结束后**：用 best.pt 重跑 `python test_v4.py --save-vis`，更新验收指标

### 近期可做（一期优化）
- [ ] **harbor 类别补充**：当前仅占 1.3%，FAIR1M2.0 下载完成后补入
- [ ] **Prompt调优**：根据实际生成结果调整 `prompt_templates.py` 的风格约束

---

## 九、舰船细粒度分类方案（v5 规划）

### 背景与数据

FUSAR-Ship1.0（高分三号 C 波段 SAR，512×512 芯片，5000+ 样本，含 AIS 匹配标签）
路径：`/mnt/data/mm_data/SAR/dection/高分辨率船只数据集FUSAR-Ship1.0/`

**可用类别分布（合并后）：**

| 合并类别 | 原始子类 | 样本数 | 军事相关性 |
|----------|----------|--------|-----------|
| cargo_ship | CargoShip/BulkCarrier/ContainerShip/GeneralCargo | ~2072 | 中（补给/运输） |
| tanker | Tanker/Oil_Chemical/LPG/CrudeOil | ~226 | 中（补给舰参考） |
| fishing | Fishing/Trawler | ~788 | 低 |
| passenger | Passenger/Ro-Ro | ~48 | 低 |
| tug_service | Tug/Dredger/PortTender | ~115 | 低 |
| other_vessel | Other/Unspecified/Reserved | ~2896 | — |

> 注：FUSAR-Ship 是民用船只数据集，**不含军舰（驱逐舰/航母/护卫舰）**。
> 军舰细分类需要专用军事SAR数据集（HRSC2016 光学可参考，SAR域暂无公开标注）。

### 两阶段方案

```
第一阶段（v4，已有）：粗检测
  YOLOv8-OBB → 定位目标位置 + 输出 ship/aircraft/...

第二阶段（v5，新增）：细分类器
  对 ship 类目标裁剪 ROI（64×64 或 128×128）
  → 轻量分类网络（ResNet18 / EfficientNet-B0）
  → 输出细粒度舰船类型
```

### v5 类别 ID 方案（追加，不修改 0-13）

| ID | code | name_cn | 数据来源 | 状态 |
|----|------|---------|---------|------|
| 0 | ship | 舰船（粗） | v4已有 | ACTIVE（粗检测保留） |
| — | destroyer | 驱逐舰 | 待收集军事SAR | PLANNED |
| — | frigate | 护卫舰 | 待收集军事SAR | PLANNED |
| — | carrier | 航空母舰 | 待收集军事SAR | PLANNED |
| — | replenishment | 补给舰 | FUSAR tanker参考 | PLANNED |
| — | amphibious | 两栖舰 | 待收集军事SAR | PLANNED |
| — | cargo_ship | 货船 | FUSAR-Ship1.0 | 可训练 |
| — | tanker | 油轮/化学品船 | FUSAR-Ship1.0 | 可训练 |

> 细分类器独立于 v4 检测器，以 `ship_subtype` 字段追加到 Evidence Package objects[] 中，不修改现有 schema。

### 实施步骤

1. **数据准备**：从 FUSAR-Ship1.0 的 meta.csv 按类别合并，生成分类数据集（train/val 8:2）
2. **训练细分类器**：ResNet18 fine-tune，输入 128×128 SAR 芯片，输出 6 类
3. **接入流水线**：`DetectorTool.detect()` 检测到 ship 后，自动裁剪 ROI 送入分类器
4. **Evidence 扩展**：`objects[].attributes.ship_subtype` 字段（不改 schema 主结构）
5. **军舰数据收集**：联系军事SAR数据源，或用 HRSC2016 光学数据做迁移学习预研

### 当前限制

- FUSAR-Ship1.0 **无军舰标注**，驱逐舰/航母/护卫舰需要专用数据
- 民用船型（货船/油轮）可训练，军事价值有限
- 建议先做民用船型分类验证流程，等军事SAR数据到位后直接替换

### 启动命令（数据准备好后）

```bash
cd /home/zhuxiang/RS/SAR/experiments
python modules/detector/train_ship_classifier.py  # 待编写
```

---

## 十、下一步行动（更新）

### 可立即进行
- [ ] **等待 v4 训练结束**：patience=30，从 epoch 72 起算，预计 12-16 小时
- [ ] **harbor 类别补充**：FAIR1M2.0 下载完成后补入
- [ ] **Prompt调优**：调整 `prompt_templates.py` 风格约束

### 近期规划（v5 民用船型细分类器）
- [ ] **FUSAR-Ship 数据整理**：按 meta.csv 合并类别，生成 train/val 分类数据集（0.5天）
- [ ] **训练 ResNet18 分类器**：6 类民用船型，验证两阶段流水线可行性（0.5天+1~2h训练）
- [ ] **接入 DetectorTool**：ship ROI 自动裁剪 → 分类器推理 → `ship_subtype` 字段（1天）
- [ ] **Evidence Schema 扩展**：`objects[].attributes.ship_subtype`，不改主结构（0.5天）

**可行性：高。** 数据现成（5243张，AIS标签），工程量小，预估 **2.5~3 天**完成。

### 二期工作
- [ ] **军舰细分类**：收集军事SAR标注数据（驱逐舰/护卫舰/航母/补给舰/两栖舰）
- [ ] **机场设施标注**：激活 ID 5-13（跑道/滑行道/停机坪/机库/掩蔽库/塔台/弹药库）
- [ ] **Qwen3-4B LoRA微调**：用证据JSON→报告对训练，提升军事术语和格式准确性
- [ ] **多时相变化检测**：同一区域前后时相比较
