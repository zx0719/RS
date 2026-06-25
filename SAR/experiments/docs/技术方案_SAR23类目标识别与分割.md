# SAR 图像 23 类军事目标识别与分割 — 完整技术方案

> 版本 v2.0 | 2026-06-17

---

## 一、项目目标

输入单景 SAR 卫星图像，系统自动完成：
1. **舰船/飞机实例检测**：旋转框（OBB）+ 细粒度分类（16 类舰船 + 5 类飞机）
2. **港口/机场区域分割**：分割 mask + 面积估算（km²）
3. **结构化证据融合**：输出标准 Evidence JSON
4. **情报通报生成**：LLM 生成通报正文 → DOCX 文档组装

---

## 二、总体架构：两分支 + 门控

```
                        输入: SAR 卫星图像
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
   ┌────────────────────┐          ┌────────────────────┐
   │  分支 A: 实例检测   │          │  分支 B: 场景门控   │
   │                    │          │                    │
   │  M1 YOLOv8-OBB     │          │  Gate ResNet-18    │
   │  2类粗检测          │          │  3类场景分类        │
   │  · ship            │          │  · harbor          │
   │  · aircraft        │          │  · airport         │
   │                    │          │  · none            │
   │  训练: 公开+私人    │          │  训练: 私人+公开辅助 │
   └────────┬───────────┘          └────────┬───────────┘
            │                               │
            │                          harbor / airport?
            │                               │
            │                        ┌──────┴──────┐
            │                       Yes            No
            │                        │              │
            │                        ▼              │
            │             ┌──────────────────┐      │
            │             │  M3 FastSAM      │      │
            │             │  区域分割 (零样本) │      │
            │             │                  │      │
            │             │  输入: 原图       │      │
            │             │  prompt: bbox    │      │
            │             │  输出: mask+面积  │      │
            │             └────────┬─────────┘      │
            │                      │                │
            ▼                      ▼                │
   ┌──────────────────────────────────────┐        │
   │          M2 细分类器                 │        │
   │                                      │        │
   │  M2a ResNet-18: ship ROI → 16类舰船  │        │
   │  M2b ResNet-18: aircraft ROI → 5类飞机│        │
   │                                      │        │
   │  训练: 仅私人 300 张                  │        │
   └──────────────┬───────────────────────┘        │
                  │                                │
                  └────────────┬───────────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │     Evidence Builder         │
                │                              │
                │  objects[]:                   │
                │    舰船(16类) OBB + geo       │
                │    飞机(5类) OBB + geo        │
                │    空间包含: ship in harbor?  │
                │              aircraft in      │
                │                airport?       │
                │                              │
                │  attachments:                 │
                │    harbor/airport mask        │
                │    area_km²                   │
                └──────────────┬──────────────┘
                               ▼
                ┌─────────────────────────────┐
                │   LLM 通报生成 → DOCX 组装   │
                └─────────────────────────────┘
```

---

## 三、23 类完整体系

```
面积目标 (2类) — 分支 B: FastSAM 分割 + 面积估算
═══════════════════════════════════════════════
 0  港口 (harbor)          22  机场 (airport)

舰船 (16类) — 分支 A: YOLO粗检 → M2a ResNet细分类
═══════════════════════════════════════════════
 1  军用辅助舰船            11  集装箱船
 2  作战舰船                12  工程船
 3  液货船                  13  渔船
 4  港务船                  14  拖船
 5  散货船                  15  客船
 6  舰船_其他               16  滚装船
 7  调查船                  19  帆船
10  两栖舰船                20  研究船

飞机 (5类) — 分支 A: YOLO粗检 → M2b ResNet细分类
═══════════════════════════════════════════════
 8  作战飞机
 9  运输机
17  作战支援飞机
18  直升机
21  飞机_其他
```

---

## 四、各模块详细设计

### 4.1 M1 — YOLOv8-OBB 实例检测器（分支 A）

| 项目 | 说明 |
|------|------|
| **架构** | YOLOv8m-OBB (~26M 参数) |
| **输出类别** | 2 类: `ship` / `aircraft` |
| **输出格式** | 旋转框 OBB: `(cx, cy, w, h, angle_deg, confidence)` |
| **不检测** | harbor / airport（由分支 B 独立处理） |
| **大图处理** | 自动 tile (640×640, overlap=128, threshold=1280) |
| **训练数据** | 公开 ~150K 张 (ship/aircraft) + 私人 300 张 (ship/aircraft) |
| **预训练权重** | `yolov8m-obb.pt` (Ultralytics 官方) |
| **训练时间** | ~1 天 (双卡 A800, 100 epochs) |

**为什么 YOLO 只做 2 类？**

公开数据中 harbor/airport 存在严重的"缺失标注"问题：
- SARDet_100K 中 61,287 张船图和 1,328 张港口图**完全不重叠**
- 船图里的港口背景可见但未标注 → 如果 YOLO 学 harbor，会被教成"港口特征=背景"
- 机场同理：SAR-Airport-1.0 的 624 张机场图中飞机全部未标注

**解法**：YOLO 只管 ship/aircraft（公开数据标注完整），harbor/airport 走独立分支（不受公开数据污染）。

### 4.2 M2a — 舰船细分类器（分支 A）

| 项目 | 说明 |
|------|------|
| **架构** | ResNet-18 (~11M 参数) |
| **输入** | M1 检测到的 ship OBB → crop + pad(20%) → resize 128×128 |
| **输出** | 16 类舰船 subtype + 置信度 |
| **训练数据** | 仅私人 300 张的 ship ROI (~2100 个实例) |
| **每类样本** | ~130 个（均衡分布） |
| **数据增强** | 旋转 ±180°, 散斑噪声, 缩放 ±10%, 翻转 |
| **Fallback** | 置信度 < 0.3 → 保留粗标签 "ship" |
| **训练时间** | ~1 小时 (单卡, 100 epochs) |

### 4.3 M2b — 飞机细分类器（分支 A）

| 项目 | 说明 |
|------|------|
| **架构** | ResNet-18 (~11M 参数) |
| **输入** | M1 检测到的 aircraft OBB → crop + pad → resize 128×128 |
| **输出** | 5 类飞机 subtype + 置信度 |
| **训练数据** | 仅私人 300 张的 aircraft ROI (~2100 个实例) |
| **每类样本** | ~420 个（均衡分布） |
| **Fallback** | 置信度 < 0.3 → 保留粗标签 "aircraft" |
| **训练时间** | ~30 分钟 (单卡, 100 epochs) |

### 4.4 Gate — 场景门控分类器（分支 B 入口）

| 项目 | 说明 |
|------|------|
| **架构** | ResNet-18 (~11M 参数) |
| **输入** | SAR 图像 resize 到 512×512，灰度 |
| **输出** | 3 类: `harbor` / `airport` / `none` |
| **作用** | 判断图像是否包含港口/机场，决定是否触发 M3 FastSAM |
| **推理速度** | ~5ms/张 |
| **门控策略** | 偏向高召回（宁可误触发，不能漏检），阈值默认 0.5 可调至 0.35 |
| **二次校验** | FastSAM 输出后检查分割面积 ≥ 0.01 km²，否则丢弃 |

**训练数据构建**：

| 类别 | 来源 | 预估数量 |
|------|------|---------|
| **harbor** | 私人 300 中有 harbor 标注的图 | ~100 张 |
| **airport** | 私人 300 中有 airport 标注的图 + SAR-Airport-1.0 | ~700 张 |
| **none** | 公开数据随机抽样（纯海面/陆地，排除含港口机场的图） | ~2000 张 |

**训练配置**：
- 损失函数: CrossEntropyLoss + 类别权重（harbor 权重调高以补偿样本少）
- 优化器: AdamW, lr=1e-4
- Epochs: 50, EarlyStopping patience=10
- 训练时间: ~1 小时（单卡）

### 4.5 M3 — FastSAM 区域分割（分支 B）

| 项目 | 说明 |
|------|------|
| **架构** | FastSAM (轻量 Segment Anything, ~40MB) |
| **触发条件** | Gate 输出 harbor 或 airport |
| **输入** | SAR 原图 + bbox prompt |
| **预处理** | CLAHE 对比度增强（提升 SAR 图像分割质量） |
| **输出** | 分割 mask → 最大连通域多边形 → 像素数 × GSD² → 面积(km²) |
| **训练** | 零样本，无需训练 |
| **验证** | 私人 300 有 harbor/airport 分割 mask，可计算 mIoU 评估 |
| **推理速度** | ~200ms/张 |

---

## 五、推理流程

```
Phase 1 (并行执行):
  ├─ M1 YOLO: 检测所有 ship + aircraft → OBB 列表
  └─ Gate: 场景分类 → harbor / airport / none

Phase 2 (并行执行，条件触发):
  ├─ M2a: ship ROI 批量裁剪 → ResNet-18 → 16 类舰船 subtype
  ├─ M2b: aircraft ROI 批量裁剪 → ResNet-18 → 5 类飞机 subtype
  └─ 如果 Gate ≠ none → M3 FastSAM: 分割 mask + 面积(km²)

Phase 3:
  └─ Evidence Builder:
       · 合并 objects[] (OBB + subtype + geo坐标)
       · 空间包含判断: ship.center in harbor_mask? → attributes.in_harbor
       · 面积: harbor/airport mask 像素数 × GSD² → km²
       · 输出标准 Evidence JSON

Phase 4:
  └─ LLM 通报生成 → DOCX 文档组装
```

**Gate = none 时**：M3 不触发。Evidence Builder 中 scene_type = "unknown"，无 harbor/airport mask。管线正常运行。

**总推理时间预估**：M1+Gate 并行 ~100ms + M2+M3 并行 ~200ms ≈ **300ms/张**（不含 tile 开销）

---

## 六、双机分工（涉密数据约束）

```
当前机器 (联网)                      涉密主机 (离线)
════════════════                     ════════════════

可访问:                              可访问:
 · 公开数据集 ~150K 张                 · 私人标注 300 张大图
 · SAR-Airport-1.0 624 张             · USB 拷入的公开辅助数据
 · GitHub / pip / conda               · CUDA + Python 环境

执行:                                执行:
 Phase 1: 代码开发                     Phase 3: 离线训练
  · modules/classifier/ 包              · Gate 场景分类器
  · modules/segmentation/ 包            · M2a 舰船细分类器
  · class_labels.py 更新               · M2b 飞机细分类器
  · evidence/builder.py 更新
  · 训练脚本                           Phase 4: 权重导出
                                        · USB 拷回当前机器
 Phase 2: 离线训练包准备
  · 代码打包                           Phase 5 (当前机器):
  · 公开辅助数据 (SAR-Airport-1.0        · 所有权重就位
    + 负样本 ~2000 张)                  · M1 YOLO 训练
  · 依赖列表 + 离线安装脚本              · 端到端集成测试
  → USB 拷入涉密主机
```

### 离线训练包内容

```
offline_training_pack/
  ├── code/
  │   ├── modules/
  │   │   ├── classifier/        # 所有 classifier 代码
  │   │   │   ├── __init__.py
  │   │   │   ├── roi_crop.py
  │   │   │   ├── scene_gate.py
  │   │   │   ├── ship_classifier.py
  │   │   │   ├── aircraft_classifier.py
  │   │   │   ├── train_gate.py
  │   │   │   ├── train_ship_cls.py
  │   │   │   └── train_aircraft_cls.py
  │   │   └── segmentation/
  │   │       ├── __init__.py
  │   │       └── fastsam_segmenter.py
  │   ├── class_labels.py
  │   └── requirements_offline.txt
  ├── data/
  │   ├── airport_scenes/        # SAR-Airport-1.0 (624 张)
  │   ├── negative_samples/      # 公开负样本 (~2000 张)
  │   └── README.md              # 数据说明 + 训练命令
  └── install.sh                 # 离线 pip install 依赖
```

---

## 七、训练计划总览

| 阶段 | 模型 | 机器 | 数据 | 时间 |
|------|------|------|------|------|
| S1 | M1 YOLO 2类粗检测 | 当前(联网) | 公开 ~150K (ship/aircraft) | ~1天 (双A800) |
| S2 | Gate 场景分类器 | 涉密(离线) | 私人300 + SAR-Airport-1.0 + 负样本 | ~1小时 |
| S3a | M2a 舰船细分类 16类 | 涉密(离线) | 私人300 ship ROI (~2100) | ~1小时 |
| S3b | M2b 飞机细分类 5类 | 涉密(离线) | 私人300 aircraft ROI (~2100) | ~30分钟 |
| S4 | M3 FastSAM | 当前(联网) | 零样本 | 0 |
| S5 | 端到端集成测试 | 当前(联网) | 测试图像 | ~半天 |

---

## 八、公开数据缺失标注问题 — 完整分析

### 8.1 问题根因

SARDet_100K 由 10 个独立标注的源数据集拼合而成，每个源只标单一类别：

| 源数据集 | 只标注 | 缺失 |
|---------|--------|------|
| SSDD, HRSID, ShipDataset, AIR_SARShip | ship | harbor（船图背景中港口可见但未标） |
| SADD, SAR-AIRcraft | aircraft | airport（机场图中跑道可见但未标） |
| OGSOD | bridge, harbor, tank | ship（港口图中的船未标） |
| SAR-Airport-1.0 | airport | aircraft（机场图中的飞机未标） |

**结果**：ship+harbor 共现率 0%，aircraft+airport 共现率 0%。模型被教会"港口特征=没有船"。

### 8.2 本方案解决方式

| 问题 | 解决 |
|------|------|
| 船图里港口可见但未标注 | YOLO 不管 harbor → 船图标注完整可用 |
| 机场图里飞机可见但未标注 | YOLO 不管 airport → 飞机正常检测 |
| 港口只有 tile 级小框 | Gate 用私人数据训练做场景级判断；FastSAM 在原图上做完整分割 |
| 细分类数据不足（公开只有粗标签） | ResNet 仅用私人数据训练，YOLO+ResNet 两阶段解耦 |
| 私人数据只有 300 张（少） | YOLO 粗检测从公开 150K 学 SAR 特征，ResNet 只做小样本细分类 |

---

## 九、模块目录结构

```
modules/
  detector/               # M1: YOLOv8-OBB
    class_map.py           #   修改: 2类 ACTIVE (ship/aircraft)
    
  classifier/             # 新增: M2 细分类 + Gate
    __init__.py
    roi_crop.py            #   OBB ROI 裁剪工具
    scene_gate.py          #   Gate: ResNet-18 3类场景分类
    ship_classifier.py     #   M2a: ResNet-18 16类舰船
    aircraft_classifier.py #   M2b: ResNet-18 5类飞机
    train_gate.py          #   Gate 离线训练脚本
    train_ship_cls.py      #   M2a 离线训练脚本
    train_aircraft_cls.py  #   M2b 离线训练脚本
    
  segmentation/           # 新增: M3 FastSAM
    __init__.py
    fastsam_segmenter.py   #   FastSAM 分割 + 面积计算
    
  evidence/               # 修改
    builder.py             #   增加 segmentation + gate_result 参数
                           #   增加空间包含逻辑
    
  class_labels.py          # 修改: 23类中文标签 + super_class 映射
  geo/                     # 不变
  report/                  # 不变
  eval/                    # 不变
```

---

## 十、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| YOLO 不管 harbor/airport | 分支 B 独立处理 | 公开数据缺失标注，混在一起训练会污染模型 |
| 细分类用 ResNet 而非 YOLO 直接输出 | 两阶段 | 细分类数据仅 300 张，ResNet 小样本泛化更好，且不拖累 YOLO 粗检测 |
| Gate 用 ResNet-18 而非 CLIP | ResNet | CLIP 在 SAR 域上的 zero-shot 准确率仅 40-55%，不可靠 |
| FastSAM 而非 SAM | FastSAM | 推理速度快 50 倍，SAR 分割精度够用 |
| 门控偏向高召回 | 宁可误触发 | 误触发 FastSAM 只浪费 ~200ms；漏检丢失关键情报 |
