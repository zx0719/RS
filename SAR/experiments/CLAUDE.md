# SAR图像军事目标情报通报自动生成系统

## 项目定位

输入单景SAR卫星图像及其元数据，系统自动完成目标检测、坐标地理化、结构化证据生成、情报通报正文生成、Word文档组装。

**一期原则：专用感知工具 + 结构化证据 + LLM文本生成 + Word组装**，不做端到端VLM直接生成。

原因：SAR视觉域与自然图像差异大；军事情报场景对坐标/数量/类别/一致性要求高，不能接受幻觉。

## 总体架构（Tool Use主线）

```
输入：SAR图像（GeoTIFF/JPEG）+ 元数据 + 目标区域名称
  ↓
[M1] 预处理与元数据解析
  ↓
[M2] 感知工具层
  ├── YOLOv8-OBB 检测（ship/aircraft/tank/bridge/harbor，v4 5类）
  ├── [v5 新增] 船型细分类器（ResNet18，6类民用船型，解耦独立模块）
  │     FUSAR-Ship1.0 → cargo_ship/tanker/fishing/tug_service/passenger/other_vessel
  │     接入方式：ship ROI裁剪 → 分类推理 → objects[].attributes.ship_subtype
  ├── GDAL坐标地理化
  └── 统计汇总
  ↓
[M3] 证据融合层（检测结果 + 坐标 + 元数据 → 标准Evidence JSON）
  ↓
[M4] 通报生成层（纯文本LLM读取证据JSON生成通报正文；VLM仅辅助图像整体描述）
  ↓
[M5] 文档组装层（python-docx生成.docx，插入检测图/统计表/正文/元数据页脚）
  ↓
输出：标准通报.docx + 证据JSON + 标注图
```

## v5 船型细分类器（规划中）

- **数据**：FUSAR-Ship1.0，5243张 512×512 SAR芯片，AIS标签
- **架构**：ResNet18 fine-tune，6类，输入128×128
- **模块路径**：`modules/classifier/ship_classifier.py`（待创建）
- **接入**：DetectorTool检测到ship后自动触发，失败时fallback到`ship`
- **预估工期**：2.5~3天（v4训练结束后启动）
- **限制**：仅民用船型，军舰细分类待三期军事SAR数据

## 核心设计原则

- **事实由工具给出，文字由模型组织** — 数量/类别/坐标等事实必须来自工具输出，LLM只负责语言组织
- **结构化证据防幻觉** — 所有关键事实以JSON证据形式传递，不允许LLM自行推断事实
- **模块解耦** — 更换检测器/LLM/Word模板时不影响整体接口

## 一期明确不做

- 端到端自由生成式目标检测
- 无证据支撑的态势研判结论
- 敌我属性自动判断
- 跨时相复杂变化推断（二期）

---

## 统一证据JSON Schema（v1.0）

文档：`/home/zhuxiang/RS/SAR/文档/SAR统一证据JSON与模块接口定义_v1.md`

### 顶层Evidence Package结构

```json
{
  "schema_version": "1.0.0",
  "package_id": "sar-YYYYMMDD-XXXXXX",
  "task_type": "intel_brief",
  "status": "<状态枚举>",
  "input": {},       // 图像URI、元数据、任务信息
  "scene": {},       // 场景类型、地理边界、环境特征
  "objects": [],     // 目标实例列表（核心）
  "statistics": {},  // 按类统计、总计
  "attachments": {}, // 可视化图、裁剪图、日志
  "report": {},      // 通报正文、docx URI
  "quality": {},     // 一致性检查、幻觉检查
  "errors": []
}
```

### 状态枚举

```
RECEIVED → PREPROCESSED → DETECTED → GEOLOCATED → FUSED
→ READY_FOR_NLG → REPORT_DRAFTED → DOCX_RENDERED → REVIEW_PENDING → COMPLETED
```

### objects[] 单个目标结构

```json
{
  "object_id": "obj-000001",
  "class": {
    "code": "destroyer",
    "name_cn": "驱逐舰",
    "super_class": "ship",
    "priority": "HIGH"
  },
  "score": { "confidence": 0.92, "calibrated_confidence": 0.89 },
  "geometry": {
    "pixel": { "center_x": 1032.4, "center_y": 812.7, "width": 86.3, "height": 18.9, "angle_deg": -27.4 },
    "geo":   { "center_lon": 120.265000, "center_lat": 22.782000 }
  },
  "attributes": { "heading_deg": 332.6, "estimated_length_m": 86.3, "is_near_pier": true },
  "audit": { "review_status": "UNREVIEWED" }
}
```

### 目标类别编码

| 大类 | 编码 |
|------|------|
| 舰船 | carrier, destroyer, frigate, replenishment, amphibious, other_vessel |
| 飞机 | fighter, bomber, transport, aew, helicopter, other_aircraft |

### scene_type 枚举

`harbor, airport, anchorage, shipyard, airbase, coastal_area, unknown`

### 一期最小闭环五字段链路

1. `input.metadata`
2. `objects[].class + geometry.pixel`
3. `objects[].geometry.geo`
4. `statistics.by_class + totals`
5. `report.body + report.docx.uri`

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 目标检测 | YOLOv8-OBB（旋转框） |
| 坐标地理化 | GDAL（像素→WGS84） |
| 文本生成 | 纯文本LLM（读取Evidence JSON） |
| 文档组装 | python-docx |
| 输入格式 | GeoTIFF / JPEG |

## 数据集需求

- 证据JSON + 通报正文对：300~500份（训练LLM风格）
- 图像 + 证据 + 完整Word对：100~200份（系统联调）

## 一期待确认事项

1. 输入图像是否稳定提供GeoTIFF地理参考？
2. 一期是否只做舰船，飞机是否后置？
3. 舰船类别是否接受先粗后细的两阶段策略？
4. 通报是否必须完全对标既有示例版式？
5. 是否需要"审核版/正式版"双输出？
6. 是否需要保存证据JSON作为正式归档附件？

---

## Agent Team 配置

本项目使用4个专职Agent协作开发，每个Agent严格限定在自己的模块边界内。

### 全局规则

- **禁止修改Evidence Schema**：`SAR统一证据JSON与模块接口定义_v1.md` 是各模块的契约，任何Agent不得修改其字段定义
- **实现前必须获得计划批准**：每个Agent在写代码前必须先用 `EnterPlanMode` 提交计划，等待用户批准后再实施
- **代码修改必须使用worktree**：所有涉及代码编写/修改的Agent必须在独立git worktree中工作，避免互相干扰
- **模块边界**：每个Agent只能操作自己负责的目录，不得跨界修改其他模块代码

### Agent 1: detector-trainer

**职责**：M2感知工具层 — YOLOv8-OBB模型训练、数据集准备、检测器封装

**模块边界**：`modules/detector/`

**工作内容**：
- YOLOv8-OBB训练脚本（舰船/飞机）
- 数据集格式转换与增强
- 检测器推理接口封装（输入图像路径 → 输出符合`objects[]`结构的检测结果）
- 置信度校准

**输出契约**：填充Evidence Package中的 `objects[].class`、`objects[].score`、`objects[].geometry.pixel`、`objects[].evidence.detector_version`

**worktree名称**：`worktree-detector`

---

### Agent 2: geo-pipeline-engineer

**职责**：M1预处理 + M2坐标地理化 + M3证据融合层

**模块边界**：`modules/geo/` 和 `modules/evidence/`

**工作内容**：
- GeoTIFF元数据解析（GDAL）
- 像素坐标 → WGS84经纬度转换
- Evidence Package构建器（将检测结果、坐标、元数据组装为标准JSON）
- 统计汇总（`statistics.by_class`、`statistics.totals`）

**输出契约**：填充Evidence Package中的 `input`、`scene`、`objects[].geometry.geo`、`statistics`，并将状态推进到 `READY_FOR_NLG`

**worktree名称**：`worktree-geo`

---

### Agent 3: report-generator

**职责**：M4通报生成 + M5文档组装层

**模块边界**：`modules/report/`

**工作内容**：
- LLM调用封装（读取Evidence JSON → 生成通报正文）
- Prompt模板管理（通报风格、格式约束）
- python-docx文档组装（插入检测图/统计表/正文/元数据页脚）
- 审核版/正式版双输出支持

**输入契约**：接收状态为 `READY_FOR_NLG` 的Evidence Package，只读 `objects`、`statistics`、`scene`、`input.metadata`

**输出契约**：填充 `report.body`、`report.docx.uri`，状态推进到 `DOCX_RENDERED`

**worktree名称**：`worktree-report`

---

### Agent 4: eval-qa

**职责**：质量评估、一致性检查、幻觉检测、集成测试

**模块边界**：`modules/eval/` 和 `tests/`（只读其他模块代码，不修改）

**工作内容**：
- Evidence Package一致性校验（数量/类别/坐标是否自洽）
- 通报正文幻觉检测（数字/类别是否与证据一致）
- 端到端集成测试（输入图像 → 验证最终docx）
- 填充 `quality` 字段（`consistency_checks`、`nlg_checks`、`review_gate`）

**规则**：eval-qa不使用worktree（只写测试和评估脚本，不修改其他模块）

---

### 最终汇报格式

每个Agent完成工作后，汇报必须包含：

```
## 变更文件
- <文件路径>：<变更说明>

## 阻塞项
- <问题描述>（如无则写"无"）

## 下一步
- <后续工作建议>
```

---

## 数据集路径

所有数据集位于 `/mnt/data/mm_data/`，详见 `/mnt/data/mm_data/README.md`。

### 一期使用的数据集

| 模块 | 数据集 | 路径 | 说明 |
|------|--------|------|------|
| M2 检测训练 | SSDD RBox | `/mnt/data/mm_data/SAR/dection/SSDD/Official-SSDD-OPEN/RBox_SSDD/voc_style/` | VOC XML OBB格式，928张训练图 |
| M2 检测训练 | SARDet_100K | `/mnt/data/mm_data/SAR/SARDet_100K/` | COCO格式，94K张，ship+aircraft类别 |
| M5 NLG训练 | FSAR-Cap | `/mnt/data/mm_data/SAR/FSAR-Cap/FSAR-Captrain.json` | 50K SAR图文对 |
| M2 类别参考 | FUSAR-Ship | `/mnt/data/mm_data/SAR/dection/高分辨率船只数据集FUSAR-Ship1.0/SAR/` | 细粒度船型分类 |

### 准备好的训练数据目录
- 检测: `/home/zhuxiang/RS/SAR/experiments/data/ssdd_yolo_obb/` (运行prepare_ssdd.py后生成)
- NLG: `/home/zhuxiang/RS/SAR/experiments/data/nlg_training/` (运行build_nlg_training_data.py后生成)
