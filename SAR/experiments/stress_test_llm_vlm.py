#!/usr/bin/env python3
"""
stress_test_llm_vlm.py — LLM + VLM vLLM 服务压力测试

测试内容：
  1. 健康检查（服务是否可达）
  2. 单请求延迟（首 token 延迟 + 总延迟）
  3. 并发压测（N 个并发请求，统计 P50/P95/P99 延迟 + 吞吐量）
  4. 显存占用（通过 nvidia-smi 采样）

用法：
    # 先启动服务（两个终端）：
    #   bash scripts/start_vllm_llm.sh 1
    #   bash scripts/start_vllm_vlm.sh 1

    conda activate sar-intel
    python stress_test_llm_vlm.py
    python stress_test_llm_vlm.py --llm-url http://localhost:8100/v1 --vlm-url http://localhost:8101/v1
    python stress_test_llm_vlm.py --concurrency 8 --requests 50
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── 默认参数 ────────────────────────────────────────────────────────────────
DEFAULT_LLM_URL = "http://localhost:8100/v1"
DEFAULT_VLM_URL = "http://localhost:8101/v1"
LLM_MODEL       = "qwen3-4b"
VLM_MODEL       = "qwen3-vl-4b"

# 测试图像（用已有的 val 图）
TEST_IMAGES = [
    "/mnt/data/zhuxiang/SAR_experiments/datasets/v4-multiclass/val/images/ms_04866.jpg",
    "/mnt/data/zhuxiang/SAR_experiments/datasets/v4-multiclass/val/images/ms_014553.jpg",
    "/mnt/data/zhuxiang/SAR_experiments/datasets/v4-multiclass/val/images/ms_015315.jpg",
]

# 测试用 Evidence Package（最小结构，不跑检测）
_DUMMY_PACKAGE = {
    "schema_version": "1.0.0",
    "package_id": "stress-test-001",
    "task_type": "intel_brief",
    "status": "READY_FOR_NLG",
    "input": {
        "image_uri": "file:///tmp/test.jpg",
        "metadata": {
            "satellite": "SAR卫星", "sensor": "SAR",
            "acquisition_time": "2026-05-13", "gsd_m": 3.0,
            "polarization": "VV",
        },
        "task": {"region_name": "某军港", "region_type": "harbor"},
    },
    "scene": {"scene_type": "harbor", "geo_bounds": None, "environment": {}},
    "objects": [],
    "statistics": {
        "by_class": [
            {"code": "ship", "name_cn": "舰船", "super_class": "ship", "count": 12},
            {"code": "aircraft", "name_cn": "飞机", "super_class": "aircraft", "count": 3},
        ],
        "totals": {"objects": 15},
    },
    "attachments": {},
    "report": {}, "quality": {}, "errors": [],
}


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def gpu_mem_mb(gpu_id: int = 1) -> tuple[int, int]:
    """返回 (used_mb, total_mb)，失败返回 (0, 0)。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip().split("\n")[0]
        used, total = [int(x.strip()) for x in out.split(",")]
        return used, total
    except Exception:
        return 0, 0


def encode_image_b64(image_path: str, max_side: int = 512) -> str:
    from PIL import Image
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()


def health_check(url: str, model: str) -> bool:
    try:
        import requests
        r = requests.get(f"{url}/models", timeout=5)
        if r.status_code == 200:
            names = [m["id"] for m in r.json().get("data", [])]
            return any(model in n for n in names)
    except Exception:
        pass
    return False


# ── LLM 单次请求 ─────────────────────────────────────────────────────────────

def llm_request(url: str, package: dict) -> tuple[float, str]:
    """发送一次 LLM 请求，返回 (latency_s, generated_text)。"""
    sys.path.insert(0, str(Path(__file__).parent))
    from modules.report.prompt_templates import build_system_prompt, build_user_prompt

    import requests

    system_prompt = build_system_prompt()
    user_prompt   = build_user_prompt(package) + " /no_think"  # 关闭 Qwen3 CoT

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }

    t0 = time.perf_counter()
    resp = requests.post(
        f"{url}/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer EMPTY"},
        timeout=60,
    )
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return latency, text


# ── VLM 单次请求 ─────────────────────────────────────────────────────────────

_VLM_SYSTEM = (
    "你是SAR图像分析专家。用1~2句话描述图像的整体场景特征，"
    "不得给出具体数量或类别。"
)

def vlm_request(url: str, image_b64: str) -> tuple[float, str]:
    """发送一次 VLM 请求，返回 (latency_s, description)。"""
    import requests

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": _VLM_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": "请描述这张SAR图像的整体场景特征。"},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 128,
    }

    t0 = time.perf_counter()
    resp = requests.post(
        f"{url}/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer EMPTY"},
        timeout=60,
    )
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return latency, text


# ── 压测核心 ─────────────────────────────────────────────────────────────────

def run_stress(
    name: str,
    request_fn,
    n_requests: int,
    concurrency: int,
    gpu_id: int,
) -> dict:
    """并发压测，返回统计结果。"""
    latencies: list[float] = []
    errors: int = 0
    outputs: list[str] = []

    mem_before = gpu_mem_mb(gpu_id)[0]
    t_wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request_fn) for _ in range(n_requests)]
        for fut in as_completed(futures):
            try:
                lat, text = fut.result()
                latencies.append(lat)
                outputs.append(text)
            except Exception as exc:
                errors += 1
                print(f"  [ERR] {exc}")

    t_wall = time.perf_counter() - t_wall_start
    mem_after = gpu_mem_mb(gpu_id)[0]

    if not latencies:
        return {"name": name, "error": "all requests failed"}

    latencies.sort()
    n = len(latencies)
    result = {
        "name":        name,
        "n_requests":  n_requests,
        "concurrency": concurrency,
        "errors":      errors,
        "success":     n,
        "wall_time_s": round(t_wall, 2),
        "throughput_rps": round(n / t_wall, 2),
        "latency": {
            "min":  round(min(latencies), 3),
            "mean": round(statistics.mean(latencies), 3),
            "p50":  round(latencies[int(n * 0.50)], 3),
            "p95":  round(latencies[int(n * 0.95)], 3),
            "p99":  round(latencies[min(int(n * 0.99), n - 1)], 3),
            "max":  round(max(latencies), 3),
        },
        "gpu_mem_delta_mb": mem_after - mem_before,
        "gpu_mem_after_mb": mem_after,
        "sample_output": outputs[0][:120] if outputs else "",
    }
    return result


def print_result(r: dict) -> None:
    if "error" in r:
        print(f"  [{r['name']}] 失败: {r['error']}")
        return
    l = r["latency"]
    print(f"\n  ── {r['name']} ──")
    print(f"  请求数: {r['success']}/{r['n_requests']}  错误: {r['errors']}")
    print(f"  吞吐量: {r['throughput_rps']} req/s  (并发={r['concurrency']}, 总耗时={r['wall_time_s']}s)")
    print(f"  延迟:   min={l['min']}s  mean={l['mean']}s  P50={l['p50']}s  P95={l['p95']}s  P99={l['p99']}s  max={l['max']}s")
    print(f"  显存:   {r['gpu_mem_after_mb']} MB (Δ{r['gpu_mem_delta_mb']:+d} MB)")
    print(f"  示例输出: {r['sample_output']}")


# ── 主程序 ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm-url",    default=DEFAULT_LLM_URL)
    p.add_argument("--vlm-url",    default=DEFAULT_VLM_URL)
    p.add_argument("--concurrency", type=int, default=4,
                   help="并发请求数，默认 4")
    p.add_argument("--requests",    type=int, default=20,
                   help="总请求数，默认 20")
    p.add_argument("--gpu-id",      type=int, default=1,
                   help="监控的 GPU ID，默认 1")
    p.add_argument("--skip-llm",    action="store_true")
    p.add_argument("--skip-vlm",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    results = []

    print("=" * 70)
    print("  vLLM 压力测试  (LLM: Qwen3-4B  |  VLM: Qwen3-VL-4B)")
    print(f"  并发={args.concurrency}  总请求={args.requests}  GPU={args.gpu_id}")
    print("=" * 70)

    # ── LLM 测试 ────────────────────────────────────────────────────────
    if not args.skip_llm:
        print(f"\n[LLM] 健康检查 {args.llm_url} ...")
        if not health_check(args.llm_url, LLM_MODEL):
            print(f"  [WARN] LLM 服务不可达，跳过 LLM 测试")
            print(f"  启动方式: bash scripts/start_vllm_llm.sh {args.gpu_id}")
        else:
            print("  服务正常")

            # 单请求预热
            print("[LLM] 预热（1 次请求）...")
            try:
                lat, text = llm_request(args.llm_url, _DUMMY_PACKAGE)
                print(f"  预热延迟: {lat:.2f}s")
                print(f"  输出: {text[:100]}")
            except Exception as exc:
                print(f"  预热失败: {exc}")

            # 压测
            print(f"\n[LLM] 压测 {args.requests} 请求 / 并发 {args.concurrency} ...")
            fn = lambda: llm_request(args.llm_url, _DUMMY_PACKAGE)
            r = run_stress("LLM Qwen3-4B", fn, args.requests, args.concurrency, args.gpu_id)
            print_result(r)
            results.append(r)

    # ── VLM 测试 ────────────────────────────────────────────────────────
    if not args.skip_vlm:
        print(f"\n[VLM] 健康检查 {args.vlm_url} ...")
        if not health_check(args.vlm_url, VLM_MODEL):
            print(f"  [WARN] VLM 服务不可达，跳过 VLM 测试")
            print(f"  启动方式: bash scripts/start_vllm_vlm.sh {args.gpu_id}")
        else:
            print("  服务正常")

            # 预编码测试图像
            img_path = next((p for p in TEST_IMAGES if Path(p).exists()), None)
            if img_path is None:
                print("  [WARN] 找不到测试图像，跳过 VLM 测试")
            else:
                print(f"  测试图像: {Path(img_path).name}")
                img_b64 = encode_image_b64(img_path)

                # 单请求预热
                print("[VLM] 预热（1 次请求）...")
                try:
                    lat, text = vlm_request(args.vlm_url, img_b64)
                    print(f"  预热延迟: {lat:.2f}s")
                    print(f"  输出: {text[:100]}")
                except Exception as exc:
                    print(f"  预热失败: {exc}")

                # 压测
                print(f"\n[VLM] 压测 {args.requests} 请求 / 并发 {args.concurrency} ...")
                fn = lambda: vlm_request(args.vlm_url, img_b64)
                r = run_stress("VLM Qwen3-VL-4B", fn, args.requests, args.concurrency, args.gpu_id)
                print_result(r)
                results.append(r)

    # ── 保存结果 ─────────────────────────────────────────────────────────
    if results:
        out_path = Path(__file__).parent / "output" / "stress_test_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\n  结果已保存: {out_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
