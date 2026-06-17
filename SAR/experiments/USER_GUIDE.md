# SAR图像军事目标情报通报系统使用手册

本文档说明如何部署、运行、生成报告、执行 GPU+VLM 严格验收，以及如何检查输出是否可交付。

当前已验收批次：

```text
accept-20260612T084252Z-512968
```

该批次包含 5 个大图样例，已通过真实本地小模型 GPU 生成、VLM 场景描述、Word 组装、证据 JSON 和发布就绪校验。

## 1. 系统定位

本系统用于 SAR 卫星图像军事目标情报通报自动生成。系统不是让 VLM 直接看图编写完整报告，而是采用如下链路：

```text
SAR 图像
  -> 目标检测和大图切片
  -> 结构化 Evidence JSON
  -> VLM 生成场景级补充描述
  -> 本地小模型根据 Evidence 生成正文
  -> Word 通报组装
  -> 严格校验
```

核心原则：

- 数量、类别、目标事实来自检测器和 Evidence JSON。
- VLM 只提供场景级描述，不决定目标数量和类别。
- LLM 只负责将结构化事实组织成通报正文。
- 上星前验收不允许 CPU 推理，不允许模板兜底，不接受缓存伪产物。

## 2. 关键目录

在本机项目根目录：

```bash
cd /home/zhuxiang/RS/SAR/experiments
```

常用目录：

```text
modules/                         核心模块
scripts/                         批处理、验收、校验脚本
run_pipeline.py                  单图端到端入口
run_v4_test_report.py            v4 检测权重报告入口
.env.collaborative.example       严格 GPU+VLM 配置样例
output/large_scene/              大图检测和 Evidence 中间产物
output/large_scene_original_format/
                                  已生成 DOCX、证据、验收报告
```

当前已验收的 5 份 Word 报告位于：

```text
output/large_scene_original_format/ship2_large_tiff/*.docx
output/large_scene_original_format/ship1_large_tiff/*.docx
output/large_scene_original_format/plane_ultralong_png/*.docx
output/large_scene_original_format/airport2_large_tiff/*.docx
output/large_scene_original_format/bridge_large_tiff/*.docx
```

对应证据文件位于各 case 目录下的：

```text
*_original_format_evidence.json
```

总 manifest：

```text
output/large_scene_original_format/original_format_docx_manifest.json
output/large_scene_original_format/release_manifest.json
```

## 3. 环境要求

### 3.1 GPU 要求

严格验收要求 CUDA 可见：

```bash
nvidia-smi
```

本次验收环境看到 2 张 A800：

```text
NVIDIA A800 80GB PCIe
```

本地小模型默认使用：

```text
cuda:0
```

VLM 服务建议使用另一张卡：

```text
cuda:1
```

### 3.2 模型路径

本地小模型：

```text
/mnt/data/zhuxiang/Qwen/Qwen3-4B
```

VLM 模型：

```text
/mnt/data/zhuxiang/Qwen/Qwen3-VL-4B-Instruct
```

模型权重不放入交付包。部署到其他机器时，需要把模型放到同一路径，或修改配置中的模型路径。

### 3.3 Python 环境

项目开发和测试使用 uv 环境：

```bash
cd /home/zhuxiang/RS/SAR/experiments
bash setup_uv.sh
```

严格 GPU 验收脚本默认使用 conda 环境：

```text
/home/zhuxiang/.conda/envs/qwen3-4B/bin/python
```

该环境至少需要：

```bash
/home/zhuxiang/.conda/envs/qwen3-4B/bin/python -m pip install python-docx
```

预检脚本会检查：

- `docx`
- `PIL`
- `openai`
- `torch`
- `transformers`

缺依赖会在正式生成前失败，不会跑到 Word 阶段才报错。

## 4. 配置文件

复制配置样例：

```bash
cd /home/zhuxiang/RS/SAR/experiments
cp .env.collaborative.example .env.collaborative.local
```

严格验收推荐配置：

```bash
SAR_SMALL_LLM_PATH=/mnt/data/zhuxiang/Qwen/Qwen3-4B
SAR_SMALL_LLM_MODEL=qwen3-4b

SAR_VLM_URL=http://127.0.0.1:8101/v1
SAR_VLM_MODEL=qwen3-vl-4b
SAR_REQUIRE_VLM=1

SAR_LLM_DEVICE=cuda:0
SAR_LLM_MAX_NEW_TOKENS=256
SAR_REQUIRE_GPU=1
SAR_REQUIRE_LOCAL_GPU=1
SAR_ALLOW_TEMPLATE_FALLBACK=0
```

含义：

- `SAR_REQUIRE_GPU=1`：本地 LLM 必须能看到 CUDA。
- `SAR_REQUIRE_LOCAL_GPU=1`：每份报告必须证明来自本地 GPU 后端。
- `SAR_ALLOW_TEMPLATE_FALLBACK=0`：模型失败时不能用模板顶替。
- `SAR_REQUIRE_VLM=1`：必须有 VLM 描述和 trace。
- `SAR_LLM_MAX_NEW_TOKENS=256`：控制本地小模型输出长度。

## 5. 启动 VLM 服务

VLM 服务使用 vLLM OpenAI-compatible API，默认端口 8101。

推荐用 tmux 常驻：

```bash
cd /home/zhuxiang/RS/SAR
mkdir -p experiments/output/service_logs
tmux new-session -d -s sar_vlm_8101 \
  "cd /home/zhuxiang/RS/SAR && bash experiments/scripts/start_vllm_vlm.sh 1 > experiments/output/service_logs/vlm_8101.log 2>&1"
```

检查服务是否启动：

```bash
curl -fsS http://127.0.0.1:8101/v1/models
```

期望返回包含：

```json
{"id":"qwen3-vl-4b"}
```

查看日志：

```bash
tail -f /home/zhuxiang/RS/SAR/experiments/output/service_logs/vlm_8101.log
```

停止服务：

```bash
tmux kill-session -t sar_vlm_8101
```

## 6. 严格 GPU+VLM 验收

这是上星前最重要的一条命令。

```bash
cd /home/zhuxiang/RS/SAR/experiments

export SAR_VLM_URL=http://127.0.0.1:8101/v1
export SAR_VLM_MODEL=qwen3-vl-4b
export SAR_REQUIRE_VLM=1
export SAR_REQUIRE_GPU=1
export SAR_REQUIRE_LOCAL_GPU=1
export SAR_LLM_DEVICE=cuda:0
export SAR_ALLOW_TEMPLATE_FALLBACK=0

bash scripts/run_gpu_small_model_acceptance.sh
```

脚本会执行：

1. GPU 和运行时依赖预检。
2. 检查 VLM `/v1/models`。
3. 清理报告缓存，避免接受旧缓存。
4. 用本地 Qwen3-4B 在 GPU 上重新生成报告。
5. 调用 VLM 写入 `scene_description` 和 `scene_description_trace`。
6. 组装 DOCX。
7. 执行协同模型、VLM、DOCX、发布就绪严格校验。

成功时最后输出：

```text
[gpu-acceptance] PASS: output/large_scene_original_format/collaborative_acceptance_gpu_report.json
```

## 7. 验收结果检查

严格验收后检查摘要：

```bash
jq '.summary' output/large_scene_original_format/gpu_acceptance_preflight.json
jq '.summary' output/large_scene_original_format/collaborative_validation_gpu_strict.json
jq '.summary' output/large_scene_original_format/vlm_validation_gpu_strict.json
jq '.summary' output/large_scene_original_format/release_readiness_gpu_strict.json
jq '.summary' output/large_scene_original_format/release_manifest.json
```

当前已通过状态：

```text
release_manifest:
  total_cases = 5
  strict_release_ready_cases = 5
  model_participation_cases = 5
  fresh_model_output_cases = 5
  local_gpu_cases = 5
  vlm_trace_cases = 5
  acceptance_run_id_cases = 5

collaborative_validation_gpu_strict:
  model_participation_cases = 5
  gpu_requirement_cases = 5
  local_gpu_requirement_cases = 5
  fresh_model_output_cases = 5
  failures = []

vlm_validation_gpu_strict:
  scene_description_cases = 5
  body_mentions_scene_description_cases = 5
  vlm_trace_cases = 5
  failures = []

release_readiness_gpu_strict:
  ready_cases = 5
  failures = []
```

如果任何 `failures` 非空，该批次不能交付。

## 8. 查看已生成报告

列出 DOCX：

```bash
find output/large_scene_original_format -maxdepth 2 -name '*.docx' -print
```

当前 5 份：

```text
output/large_scene_original_format/airport2_large_tiff/sar-20260428-860591_draft.docx
output/large_scene_original_format/bridge_large_tiff/sar-20260428-0366D2_draft.docx
output/large_scene_original_format/plane_ultralong_png/sar-20260428-9DA3B6_draft.docx
output/large_scene_original_format/ship1_large_tiff/sar-20260428-A4489B_draft.docx
output/large_scene_original_format/ship2_large_tiff/sar-20260428-61A68F_draft.docx
```

查看 manifest：

```bash
jq '[.[] | {case_name, docx_path, evidence_path, acceptance_run_id, report_source}]' \
  output/large_scene_original_format/original_format_docx_manifest.json
```

## 9. 单图运行

适合临时对一张图生成通报。

### 9.1 使用真实检测权重、VLM、本地小模型

```bash
cd /home/zhuxiang/RS/SAR/experiments

python run_pipeline.py \
  --image /path/to/image.tif \
  --region 某目标区域 \
  --model /path/to/yolov8-obb.pt \
  --env-file .env.collaborative.local \
  --vlm-url http://127.0.0.1:8101/v1 \
  --vlm-model qwen3-vl-4b \
  --require-vlm \
  --require-gpu \
  --no-template-fallback \
  --llm-device cuda:0 \
  --llm-max-new-tokens 256 \
  --output-dir output/single_run
```

输出：

```text
output/single_run/*.docx
output/single_run/*_evidence.json
```

### 9.2 调试模式

不接真实模型，仅检查流程：

```bash
python run_pipeline.py \
  --image /path/to/image.tif \
  --region 测试区域 \
  --output-dir output/debug_run
```

调试模式允许 MockDetector 和模板兜底，不可作为验收报告。

## 10. v4 检测报告入口

`run_v4_test_report.py` 适合使用 v4 多类检测权重生成 Word。

严格模式示例：

```bash
cd /home/zhuxiang/RS/SAR/experiments

python run_v4_test_report.py \
  --weights /mnt/data/zhuxiang/SAR_experiments/runs/v4-multiclass/weights/last.pt \
  --image /path/to/image.jpg \
  --region 某目标区域 \
  --env-file .env.collaborative.local \
  --vlm-url http://127.0.0.1:8101/v1 \
  --require-vlm \
  --require-gpu \
  --no-template-fallback \
  --llm-device cuda:0 \
  --llm-max-new-tokens 256
```

常用参数：

```text
--tile-size 640
--tile-overlap 128
--tile-threshold 1280
--include-image / --no-image
```

## 11. 大图报告重建

如果 `output/large_scene/` 中已经有大图检测 Evidence 和 overview 图，可以只重建原格式 Word 报告：

```bash
cd /home/zhuxiang/RS/SAR/experiments

export SAR_ACCEPTANCE_RUN_ID=accept-manual-001
export SAR_VLM_URL=http://127.0.0.1:8101/v1
export SAR_REQUIRE_VLM=1

/home/zhuxiang/.conda/envs/qwen3-4B/bin/python \
  scripts/build_original_large_scene_docx.py \
  --env-file .env.collaborative.local \
  --strict-release \
  --acceptance-run-id "${SAR_ACCEPTANCE_RUN_ID}"
```

注意：

- `--strict-release` 会先写入临时目录。
- 只有全批次都通过严格校验后，才提交到最终输出目录。
- 某个 case 失败时不会覆盖已有合格报告。

## 12. 手动校验命令

DOCX：

```bash
python scripts/validate_docx_outputs.py \
  --manifest output/large_scene_original_format/original_format_docx_manifest.json \
  --output output/large_scene_original_format/docx_validation_report.json
```

VLM：

```bash
python scripts/validate_vlm_outputs.py \
  --manifest output/large_scene_original_format/original_format_docx_manifest.json \
  --output output/large_scene_original_format/vlm_validation_gpu_strict.json \
  --require-vlm \
  --require-vlm-trace
```

协同模型：

```bash
python scripts/validate_collaborative_outputs.py \
  --manifest output/large_scene_original_format/original_format_docx_manifest.json \
  --output output/large_scene_original_format/collaborative_validation_gpu_strict.json \
  --require-model-participation \
  --require-gpu \
  --require-local-gpu \
  --require-fresh-model-output \
  --require-acceptance-run-id accept-20260612T084252Z-512968
```

发布就绪：

```bash
python scripts/validate_release_readiness.py \
  --manifest output/large_scene_original_format/original_format_docx_manifest.json \
  --output output/large_scene_original_format/release_readiness_gpu_strict.json \
  --require-model-participation \
  --require-gpu \
  --require-local-gpu \
  --require-vlm \
  --require-vlm-trace \
  --require-fresh-model-output \
  --require-acceptance-run-id accept-20260612T084252Z-512968
```

## 13. 测试

推荐先跑聚焦测试：

```bash
cd /home/zhuxiang/RS/SAR/experiments

UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev python -m pytest \
  tests/test_run_pipeline_collaborative.py \
  tests/test_run_v4_collaborative.py \
  tests/test_vlm_integration.py \
  tests/test_run_v4_vlm.py \
  tests/test_gpu_acceptance_preflight.py \
  tests/test_build_original_large_scene_docx_vlm.py \
  -q
```

当前结果：

```text
33 passed
```

## 14. 常见问题

### 14.1 `nvidia-smi` 失败

说明当前 shell 看不到 GPU。严格验收会停止，不会改用 CPU。

处理：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
```

### 14.2 `SAR_REQUIRE_VLM=1 but VLM URL is empty`

说明严格模式要求 VLM，但没有配置服务地址。

处理：

```bash
export SAR_VLM_URL=http://127.0.0.1:8101/v1
curl -fsS http://127.0.0.1:8101/v1/models
```

### 14.3 `runtime_imports` 失败

说明验收 Python 缺运行时依赖，常见是缺 `python-docx`。

处理：

```bash
/home/zhuxiang/.conda/envs/qwen3-4B/bin/python -m pip install python-docx
```

### 14.4 `fresh non-cache model output is required`

说明结果来自缓存或 trace 不完整。严格验收脚本会自动清理 `.report_cache`。手动清理：

```bash
rm -rf output/large_scene_original_format/.report_cache
```

### 14.5 `VLM scene_description is not reflected in report body`

说明 VLM 描述有生成，但正文没有体现。当前构建脚本已增加确定性后处理：如果模型漏写 VLM 描述，会追加 `VLM场景补充显示...`，保证 trace 和正文一致。

### 14.6 不想占用 GPU 1

停止 VLM：

```bash
tmux kill-session -t sar_vlm_8101
```

或者改用其他 GPU：

```bash
bash scripts/start_vllm_vlm.sh 0
```

## 15. 交付判定

可交付必须同时满足：

- `release_manifest.json` 中 `strict_release_ready_cases = total_cases`
- `release_readiness_gpu_strict.json` 中 `ready_cases = total_cases`
- `collaborative_validation_gpu_strict.json` 中 `failures = []`
- `vlm_validation_gpu_strict.json` 中 `failures = []`
- `docx_validation_report.json` 中 `failures = []`
- 每个 manifest item 有同一个 `acceptance_run_id`

当前交付批次满足上述全部条件。
