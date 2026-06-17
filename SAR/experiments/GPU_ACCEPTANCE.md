# GPU 小模型验收说明

本项目的上星前报告验收必须证明三件事：

1. 报告正文由真实小模型生成，不是 `template_v1` 兜底。
2. 本地小模型运行在 GPU 环境中，不接受 CPU-only 产物。
3. 模型输出通过数量/单位后验证，例如 `舰船47艘`、`桥梁13处`。
4. VLM 服务真实参与，每个样本必须写入 `scene_description` 和 `scene_description_trace`。

一键验收命令：

```bash
cd /home/zhuxiang/RS/SAR/experiments
bash scripts/run_gpu_small_model_acceptance.sh
```

上星验收默认要求 VLM。先启动 VLM 服务并导出配置：

```bash
bash scripts/start_vllm_vlm.sh 0
export SAR_VLM_URL=http://127.0.0.1:8101/v1
bash scripts/run_gpu_small_model_acceptance.sh
```

成功后检查：

```bash
jq '.summary' output/large_scene_original_format/gpu_acceptance_preflight.json
jq '.summary' output/large_scene_original_format/collaborative_validation_gpu_strict.json
jq '.summary' output/large_scene_original_format/release_readiness_gpu_strict.json
```

合格标准：

- `gpu_acceptance_preflight.json` 中 `ok` 为 `true`，并记录 `inference_loaded=false`
- `total_cases` 为 5
- `model_participation_cases` 为 5
- `gpu_requirement_cases` 为 5
- `local_gpu_requirement_cases` 为 5
- `release_readiness_gpu_strict.json` 中 `ready_cases` 为 5
- `failures` 为空数组

还要检查：

```bash
jq '.summary' output/large_scene_original_format/vlm_validation_gpu_strict.json
```

- `scene_description_cases` 为 5
- `vlm_trace_cases` 为 5
- `failures` 为空数组

当前 Codex 沙箱里 `/dev/nvidia*` 不可见，`nvidia-smi` 会失败；这是环境可见性问题，不代表脚本允许 CPU 路径。脚本会在 GPU 不可见时直接退出，防止误用 CPU 产物。
