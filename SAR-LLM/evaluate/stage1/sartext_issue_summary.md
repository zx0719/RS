# sartext 生成异常排查总结

更新时间：2026-04-07

## 1. 现象

当前生成评测中，`sartext` 数据集出现明显异常：

- BLEU-4 基本为 `0.0000`
- ROUGE-L 很低，为 `0.0181`
- 多条生成结果退化为重复的 `0.000000...`

典型现象来自日志：

- 日志文件：[test_pt_gen_20260407_214854.log](/home/zhuxiang/RS/SAR-LLM/evaluate/stage1/logs/test_pt_gen_20260407_214854.log)
- `sartext` 示例输出中，多条 `Hyp` 为长串 `0.000000...`

## 2. 已核实的事实

### 2.1 `sartext` 数据文件本身格式正常

已检查：

- `/mnt/data/mm_data/SAR/SARTEXT/SAR-TEXT-data/SARTEXTtest_pt.json`

首条样本结构正常，包含：

- `messages`
- `images`
- `pt_path`

其首条内容与 `sarlang` 一致，都是：

- user prompt: `<image>Write a terse but informative summary of the picture.`
- assistant target: 正常英文 caption

因此，当前现象不像是 `sartext` 的 JSON 结构错误或字段缺失导致。

### 2.2 `sartext` 的 `.pt` 特征缓存看起来正常

抽查 `sartext` 前 20 个 `.pt` 文件后，结果如下：

- shape: `(195, 768)`
- dtype: `torch.float32`
- zero ratio 平均值: `0.0`
- 数值分布与 `sarlang` 的 `.pt` 特征非常接近

结论：

- 不是“特征文件全零”
- 不是“shape/dtype 明显错误”
- 也不像是 `pt_path` 大面积失效

### 2.3 问题不只出现在 `sartext`

旧日志 [test_pt_500_fixedgen_20260407_142053.log](/home/zhuxiang/RS/SAR-LLM/evaluate/stage1/logs/test_pt_500_fixedgen_20260407_142053.log) 显示：

- `sarlang` 的生成结果也已经明显异常
- 示例输出出现无关中文描述、乱码、与图像无关的内容

这说明当前 projector 并不是“只在 `sartext` 上坏掉”，而是整体生成对齐能力就偏弱。

区别只在于：

- `sarlang` 当前表现为“胡乱生成”
- `sartext` 当前进一步塌缩成“重复数字串”

### 2.4 当前评测脚本默认并不优先使用 `~/RS` 里的训练代码

评测脚本：

- [test.py](/home/zhuxiang/RS/SAR-LLM/evaluate/stage1/test.py)
- [test_pt.py](/home/zhuxiang/RS/SAR-LLM/evaluate/stage1/test_pt.py)

都会将下面这个目录加入 `sys.path`：

- `/home/qianwentao/SARClip/SARCLIP+QwenVL-pt`

这意味着当前评测实际导入的可能是外部目录中的：

- `mixed_dataset_pt.py`
- `qwen3_sar_model.py`
- `train_pt.py`
- `sarclip_module.py`

而不一定是 `~/RS` 当前仓库里的实现。

这会带来一个重要风险：

- 你眼前修改的代码和实际运行的代码，可能不是同一份

### 2.5 当前权重命名本身也值得警惕

当前使用的权重是：

- `/mnt/data/qianwentao/checkpoints_sarqwen/sar_projector_stage1_sarcap.pt`

但当前配置里又打开了：

- `mixed_weight_sarlang = 1.0`
- `mixed_weight_sartext = 1.0`

从命名上看，`sar_projector_stage1_sarcap.pt` 更像是早期或单域命名遗留，不足以证明它真的在 `sarlang + sartext` 混训设置下得到。

目前只能确认：

- 这是一个 projector 权重文件
- 不能仅凭名字证明它一定适配 `sartext`

## 3. 当前最合理的判断

基于现有证据，当前更可能的原因排序如下：

1. 当前 projector 对 `sartext` 域没有学好，导致生成阶段塌缩到数字 token 模式。
2. 当前使用的评测代码来自外部目录，与本地仓库代码不一致，导致行为不可控。
3. 当前权重文件可能并不是预期的混训产物，或者命名和训练来源不一致。

相对不支持的原因：

- `sartext` JSON 格式错误
- `sartext` 的 `.pt` 文件全零或维度错误
- 数据集字段读取完全失败

## 4. 建议的下一步

### 4.1 先固定评测代码来源

强制使用本地仓库代码，而不是 `/home/qianwentao/...`：

```bash
export SARCLIP_PROJECT_DIR=/home/zhuxiang/RS/SAR-LLM/train/stage1
cd /home/zhuxiang/RS/SAR-LLM/evaluate/stage1
```

### 4.2 分别跑 `loss_only`

先比较 `sarlang` 和 `sartext` 的测试 loss：

```bash
python test_pt.py --projector /mnt/data/qianwentao/checkpoints_sarqwen/sar_projector_stage1_sarcap.pt --datasets sarlang --loss_only
python test_pt.py --projector /mnt/data/qianwentao/checkpoints_sarqwen/sar_projector_stage1_sarcap.pt --datasets sartext --loss_only
```

如果 `sartext` 的 loss 明显高很多，那么更支持“该域没有学好”这个判断。

### 4.3 打印首 token 的 top-k logits

如果需要进一步确认“为什么会塌成 0”，建议在生成第一步打印：

- top-k token id
- 对应 token 文本
- 对应 logits

如果 `sartext` 在第一步就强烈偏向数字 token，而 `sarlang` 没有同样现象，那么可以更直接地确认是域适配塌缩。

## 5. 简短结论

`sartext` 现在确实有问题，而且不是单纯指标低，而是生成已经明显退化。

目前最可信的结论是：

- 数据文件本身大体正常
- `.pt` 特征缓存大体正常
- 当前 projector 或当前评测链路对 `sartext` 域不工作
- 同时，当前评测还存在“实际导入的是外部代码而不是本地代码”的风险

在继续分析生成细节前，优先建议先统一代码来源，再分别比较 `sarlang` / `sartext` 的 test loss。
