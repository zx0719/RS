# NLG Training Data — SAR 情报通报 SFT 数据集

## 生成方式

由 `data/build_nlg_training_data.py` 自动生成，数据来源为 FSAR-Cap 数据集。

**生成命令：**
```bash
python data/build_nlg_training_data.py \
    --fsar-cap /mnt/data/mm_data/SAR/FSAR-Cap/FSAR-Captrain.json \
    --output data/nlg_training/train.jsonl \
    --max-samples 3000 \
    --val-ratio 0.1
```

**生成流程：**
1. 解析 FSAR-Cap 英文 caption，提取舰船/飞机类别及数量（正则匹配）
2. 构建合成 Evidence JSON 包（含卫星名、成像日期、目标统计、空间分布）
3. 使用规则模板生成标准中文军事情报通报正文（100～300 字）
4. 按 9:1 比例划分 train/val

## JSONL 格式

每行为一个 JSON 对象，格式如下：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一名军事情报分析助手，负责根据SAR卫星侦察数据生成标准军事情报通报正文。..."
    },
    {
      "role": "user",
      "content": "请根据以下 <EVIDENCE> 生成SAR卫星情报通报正文。\n\n<EVIDENCE>\n卫星：高分三号\n成像日期：2025年5月7日\n..."
    },
    {
      "role": "assistant",
      "content": "据高分三号卫星2025年5月7日对某港口（港口）实施侦察，共发现舰船3艘，目标情况如下。..."
    }
  ]
}
```

## 数据集统计（2026-04-14）

| 指标 | 值 |
|------|-----|
| 训练集 (train.jsonl) | 2700 条 |
| 验证集 (val.jsonl) | 300 条 |
| 平均通报正文长度 | 136 字 |
| 通报长度范围 | 124～153 字 |
| 数据来源 | FSAR-Cap (50000 原始条目) |

**类别分布（目标实例数）：**

| 类别 | 中文名 | 实例数 |
|------|--------|--------|
| other_vessel | 其他舰船 | 5204 |
| fighter | 战斗机 | 2508 |
| replenishment | 补给舰 | 597 |
| destroyer | 驱逐舰 | 755 |
| carrier | 航空母舰 | 157 |
| helicopter | 直升机 | 34 |

> 注：SARLANG-1M 数据集为城市/陆地场景图像，不含舰船/飞机目标，贡献 0 条有效样本。

## 用于 Qwen3-4B SFT 训练

### 数据加载

```python
from modules.report.training_utils import load_sft_dataset, format_for_qwen

train_items = load_sft_dataset("data/nlg_training/train.jsonl")
val_items = load_sft_dataset("data/nlg_training/val.jsonl")
```

### 格式化为 Qwen3 Chat Template

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/data/zhuxiang/Qwen/Qwen3-4B")
formatted = format_for_qwen(train_items[0], tokenizer, add_generation_prompt=False)
# formatted["input_ids"] — token IDs for SFT loss computation
# formatted["text"]      — raw formatted string for inspection
```

### 推荐训练配置

- 模型：Qwen3-4B（`/mnt/data/zhuxiang/Qwen/Qwen3-4B`）
- 训练方式：全参数 SFT 或 LoRA
- 学习率：`2e-5`（全参）/ `1e-4`（LoRA）
- Batch size：8（梯度累积 4 步）
- 最大序列长度：1024 tokens
- Epoch：3
- 损失：仅对 assistant 回复计算 cross-entropy loss（mask system + user tokens）
