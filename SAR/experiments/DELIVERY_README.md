# SAR 情报通报系统交付包说明

交付包用于离线查看、部署复现和验收归档。

## 1. 当前交付批次

```text
accept-20260612T084252Z-512968
```

验收结果：

```text
total_cases = 5
strict_release_ready_cases = 5
ready_cases = 5
model_participation_cases = 5
fresh_model_output_cases = 5
local_gpu_cases = 5
vlm_trace_cases = 5
failures = []
```

## 2. 包内内容

```text
docs/
  USER_GUIDE.md                 完整使用手册
  GPU_ACCEPTANCE.md             GPU+VLM 验收说明
  README.md                     项目概览
  ENVIRONMENT.md                环境说明
  DEPLOYMENT.md                 部署说明

config/
  .env.collaborative.example    严格验收配置样例
  .env.collaborative.recommended 阈值推荐配置

source/
  run_pipeline.py               单图端到端入口
  run_v4_test_report.py         v4 检测报告入口
  modules/                      核心代码
  scripts/                      生成、验收、校验脚本
  pyproject.toml / uv.lock      uv 环境定义
  setup_uv.sh                   uv 环境初始化

reports/
  large_scene_original_format/  已验收 DOCX、Evidence、manifest、验证报告

validation/
  acceptance_summary.json       交付摘要
  gpu_acceptance_preflight.json
  collaborative_validation_gpu_strict.json
  vlm_validation_gpu_strict.json
  release_readiness_gpu_strict.json
  release_manifest.json
```

## 3. 不包含内容

交付包不包含以下内容：

- Qwen3-4B 本地小模型权重
- Qwen3-VL-4B-Instruct VLM 权重
- Python 虚拟环境或 conda 环境
- `.report_cache` 缓存
- vLLM 服务日志
- Git 元数据

默认模型路径：

```text
/mnt/data/zhuxiang/Qwen/Qwen3-4B
/mnt/data/zhuxiang/Qwen/Qwen3-VL-4B-Instruct
```

如部署路径不同，请修改 `config/.env.collaborative.example` 或运行命令中的参数。

## 4. 快速复查

解包后：

```bash
cd sar_intel_delivery

jq '.summary' reports/large_scene_original_format/release_manifest.json
jq '.summary' reports/large_scene_original_format/release_readiness_gpu_strict.json
jq '.summary' reports/large_scene_original_format/collaborative_validation_gpu_strict.json
jq '.summary' reports/large_scene_original_format/vlm_validation_gpu_strict.json
```

全部应显示 5/5 且 `failures` 为空。

查看报告：

```bash
find reports/large_scene_original_format -maxdepth 2 -name '*.docx' -print
```

## 5. 重新验收

在原项目目录或解包后的 `source/` 目录准备好模型和环境后：

```bash
cd source

export SAR_VLM_URL=http://127.0.0.1:8101/v1
export SAR_VLM_MODEL=qwen3-vl-4b
export SAR_REQUIRE_VLM=1
export SAR_REQUIRE_GPU=1
export SAR_REQUIRE_LOCAL_GPU=1
export SAR_LLM_DEVICE=cuda:0
export SAR_ALLOW_TEMPLATE_FALLBACK=0

bash scripts/run_gpu_small_model_acceptance.sh
```

如果只需要查看已经生成的报告，不需要重新运行模型。

## 6. 服务状态说明

当前机器上 VLM 服务由 tmux 会话提供：

```bash
tmux list-sessions | grep sar_vlm_8101
curl -fsS http://127.0.0.1:8101/v1/models
```

停止：

```bash
tmux kill-session -t sar_vlm_8101
```
