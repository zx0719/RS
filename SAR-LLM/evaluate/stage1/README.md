# Stage1 评测脚本使用说明

## 脚本概览

| 脚本 | 功能 |
|------|------|
| `test.py` | Stage1 统一评测入口；默认只跑生成评测，输出 BLEU-1/2/3/4、ROUGE-L 与每条样本结果 |
| `test_pt.py` | 旧兼容入口；内部仍保留完整评测实现，但日常使用不再推荐 |

---

## 环境依赖

```bash
pip install nltk rouge_score
python -c "import nltk; nltk.download('punkt_tab')"
```

如需启用 vLLM 生成加速，再额外安装：

```bash
pip install vllm
```

---

## 使用前配置

编辑 `config_local.py`，确认以下路径正确：

```python
class LocalPaths:
    qwen_path    = "/mnt/data/zhuxiang/Qwen/Qwen3-VL-4B-Instruct"  # Qwen 模型路径
    sarlang_root = "/mnt/data/mm_data/SAR/SARLANG-1M"              # 数据集根目录
    sartext_root = "..."
    sarcap_root  = "..."
    fsarcap_root = "..."
```

哪些数据集参与评测，由 `mixed_weight_*` 控制（0 表示跳过）：

```python
class TrainConfig:
    mixed_weight_sarlang: float = 1.0   # > 0 则评测
    mixed_weight_sartext: float = 1.0
    mixed_weight_sarcap:  float = 0.0   # 0 则跳过
    mixed_weight_fsarcap: float = 0.0
```

**注意：设备（GPU）不在 config 中指定，通过命令行 `--device` 参数传入。**

---

## 命令行参数

### `test.py` 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--projector` | 必填 | projector 权重路径（`.pt` 纯权重或含 `projector` key 的完整 checkpoint） |
| `--device` | `cuda:0` | 推理设备，所有组件（projector + Qwen）均在此设备上运行 |
| `--batch_size` | `4` | 推理 batch size |
| `--max_new_tokens` | `50` | 生成时最大新 token 数 |
| `--max_gen_samples` | `500` | 每个数据集 BLEU/ROUGE 评测的最大样本数，`-1` 表示全量 |
| `--datasets` | 无（按 config） | 逗号分隔的数据集名，如 `sarlang,sartext` |
| `--num_workers` | `0` | DataLoader worker 数量 |
| `--progress_every` | `20` | 每隔多少个 batch 打印一次生成进度 |
| `--gen_backend` | `auto` | 生成后端：优先用 `vllm`，若环境无 `vllm` 则回退到 `hf` |
| `--vllm_tensor_parallel_size` | `1` | vLLM tensor parallel size |
| `--vllm_gpu_memory_utilization` | `0.9` | vLLM GPU 显存占用比例 |
| `--vllm_max_model_len` | 模型默认 | vLLM `max_model_len` |
| `--vllm_dtype` | `auto` | vLLM 推理 dtype |
| `--vllm_enforce_eager` | 关闭 | 打开 vLLM eager 模式，便于兼容性排查 |
| `--no_print_samples` | 关闭 | 不打印每一条样本结果，只保存到记录文件 |
| `--print_prompt_for_each_sample` | 关闭 | 每条样本结果中额外打印完整 prompt |
| `--with_loss` | 关闭 | 走兼容实现，追加 Test Loss 评测 |
| `--loss_only` | 关闭 | 走兼容实现，只计算 Test Loss |

---

## 使用示例

### 默认生成评测（推荐）

```bash
python test.py \
    --projector /mnt/data/checkpoints/sar_projector.pt \
    --device cuda:0

# 全量 + 大 batch（显存充足时提速）
python test.py \
    --projector /mnt/data/checkpoints/sar_projector.pt \
    --device cuda:0 \
    --max_gen_samples -1 \
    --batch_size 8

# 显式使用 vLLM 做生成评测
python test.py \
    --projector /mnt/data/checkpoints/sar_projector.pt \
    --device cuda:0 \
    --gen_backend vllm
```

### 需要 loss 时

```bash
# 完整评测（走兼容实现）
python test.py \
    --projector /mnt/data/checkpoints/sar_projector.pt \
    --device cuda:0 \
    --with_loss

# 只算 Test Loss
python test.py \
    --projector /mnt/data/checkpoints/sar_projector.pt \
    --device cuda:0 \
    --loss_only
```

---

## 输出说明

评测结束后，终端和日志文件（`logs/test_pt_gen_<时间戳>.log`）中会打印：

```
============================================================
  汇总结果
============================================================
数据集            BLEU-1   BLEU-2   BLEU-3   BLEU-4   ROUGE-L    样本数
-----------------------------------------------------------------------
sarlang           0.1234   0.1010   0.0876   0.0765   0.3456       500
```

每条样本结果会自动保存到 `logs/test_pt_gen_<时间戳>_records/` 下的 `jsonl/json` 文件。

---

## 使用已有 checkpoint 的完整目录结构

训练产出的完整 checkpoint（含 optimizer 状态）和纯 projector 权重均可直接传给 `--projector`，脚本会自动识别：

```
checkpoint_step_005000.pt   ← 完整 checkpoint（含 "projector" key）
sar_projector.pt            ← 纯 projector 权重
```
