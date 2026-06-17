#!/usr/bin/env python3
"""
run_v4_test_report.py — 用 v4 last.pt 对指定图像跑检测，生成 Word 情报通报

用法：
    cd /home/zhuxiang/RS/SAR/experiments
    conda activate sar-intel
    python run_v4_test_report.py

    # 指定图像
    python run_v4_test_report.py --image /path/to/image.jpg --region 某某军港

    # 指定权重
    python run_v4_test_report.py --weights /path/to/best.pt

    # 不在 Word 中插入图像（纯文字+表格版）
    python run_v4_test_report.py --no-image

    # 接入 LLM（Qwen3-4B vLLM 服务，端口 8100）
    python run_v4_test_report.py --llm-url http://localhost:8100/v1

    # 接入 VLM（Qwen3-VL-4B vLLM 服务，端口 8101）
    python run_v4_test_report.py --vlm-url http://localhost:8101/v1

    # 同时接入 LLM + VLM
    python run_v4_test_report.py --llm-url http://localhost:8100/v1 --vlm-url http://localhost:8101/v1
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from modules.report.collab_config import collaborative_generator_kwargs, read_env_file, resolve_collaborative_config, resolve_vlm_config

# ── 默认参数 ────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "/mnt/data/zhuxiang/SAR_experiments/runs/v4-multiclass/weights/last.pt"
DEFAULT_IMAGE   = "/mnt/data/zhuxiang/SAR_experiments/datasets/v4-multiclass/val/images/ms_04866.jpg"
DEFAULT_REGION  = "某目标区域"
OUTPUT_DIR      = Path(__file__).parent / "output"

# ── 类别信息 ────────────────────────────────────────────────────────────────
CLASS_MAP = {
    0: {"code": "ship",     "name_cn": "舰船",      "super_class": "ship",           "priority": "HIGH"},
    1: {"code": "aircraft", "name_cn": "飞机",      "super_class": "aircraft",       "priority": "HIGH"},
    2: {"code": "tank",     "name_cn": "坦克/装甲车","super_class": "ground",         "priority": "HIGH"},
    3: {"code": "bridge",   "name_cn": "桥梁",      "super_class": "infrastructure", "priority": "MEDIUM"},
    4: {"code": "harbor",   "name_cn": "港口",      "super_class": "infrastructure", "priority": "HIGH"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--image",   default=DEFAULT_IMAGE)
    p.add_argument("--region",  default=DEFAULT_REGION)
    p.add_argument("--score-thresh", type=float, default=0.25)
    p.add_argument("--device",  default="0")
    # 大图切片参数
    p.add_argument("--tile-size",      type=int, default=640,
                   help="切片边长（像素），默认 640")
    p.add_argument("--tile-overlap",   type=int, default=128,
                   help="相邻切片重叠像素数，默认 128（约 20%%）")
    p.add_argument("--tile-threshold", type=int, default=1280,
                   help="触发切片推理的最小边长（像素），默认 1280")
    # LLM / VLM 服务地址（不传则走模板 fallback）
    p.add_argument("--llm-url", default=None,
                   help="Qwen3-4B LLM vLLM 服务地址，例如 http://localhost:8100/v1")
    p.add_argument("--large-llm-url", default=None,
                   help="大模型服务地址（可选，用于协同生成/重写）")
    p.add_argument("--large-llm-model", default="Qwen2.5-72B-Instruct",
                   help="大模型名称（默认 Qwen2.5-72B-Instruct）")
    p.add_argument("--use-collaborative-llm", action="store_true",
                   help="启用 small/large 协同生成")
    p.add_argument("--env-file", default=None,
                   help="可选 .env 配置文件，用于 small/large 模型参数")
    p.add_argument("--llm-device", default=None,
                   help="本地小模型推理设备，例如 cuda:0；严格验收不要使用 cpu")
    p.add_argument("--llm-max-new-tokens", type=int, default=None,
                   help="本地小模型最大生成 token 数")
    p.add_argument("--require-gpu", action="store_true", default=None,
                   help="要求本地 LLM 使用 CUDA 可见 GPU；GPU 不可见时失败")
    fallback_group = p.add_mutually_exclusive_group()
    fallback_group.add_argument("--allow-template-fallback", dest="allow_template_fallback",
                                action="store_true", default=None,
                                help="允许 LLM 不可用时使用规则模板兜底")
    fallback_group.add_argument("--no-template-fallback", dest="allow_template_fallback",
                                action="store_false", default=None,
                                help="禁止规则模板兜底，要求真实 LLM 输出")
    p.add_argument("--vlm-url", default=None,
                   help="Qwen3-VL-4B VLM vLLM 服务地址，例如 http://localhost:8101/v1")
    p.add_argument("--vlm-model", default=None,
                   help="可选 VLM 模型名称，默认从 env-file 或共享配置读取")
    p.add_argument("--require-vlm", action="store_true", default=None,
                   help="要求 VLM 成功生成场景描述；未配置或空结果时失败")
    # Word 图像控制：默认包含检测标注图；--no-image 生成纯文字版
    img_grp = p.add_mutually_exclusive_group()
    img_grp.add_argument("--include-image", dest="include_image", action="store_true",
                         default=True, help="Word 中插入检测标注图（默认）")
    img_grp.add_argument("--no-image", dest="include_image", action="store_false",
                         help="Word 中不插入图像，仅文字+表格")
    return p.parse_args()


def run_detection(weights: str, image: str, score_thresh: float, device: str,
                  save_vis: bool = True,
                  tile_size: int = 640,
                  tile_overlap: int = 128,
                  tile_threshold: int = 1280) -> tuple[list[dict], str | None]:
    """调用 DetectorTool 跑 OBB 检测，返回 (objects[], vis_path)。"""
    sys.path.insert(0, str(Path(__file__).parent))
    from modules.detector import DetectorTool
    from modules.detector.class_map import CLASS_MAP as FULL_CLASS_MAP

    tool = DetectorTool(
        model_path=weights,
        class_map=FULL_CLASS_MAP,
        score_thresh=score_thresh,
        device=device,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_threshold=tile_threshold,
    )
    return tool.detect(image, save_vis=save_vis, vis_dir=str(OUTPUT_DIR / "v4_vis"))


def build_statistics(objects: list[dict]) -> dict:
    """按类统计目标数量，by_class 格式与 generator._template_fallback 兼容。"""
    counts: dict[str, int] = {}
    for obj in objects:
        code = obj["class"]["code"]
        counts[code] = counts.get(code, 0) + 1

    by_class = []
    for code, cnt in counts.items():
        desc = next((v for v in CLASS_MAP.values() if v["code"] == code), None)
        by_class.append({
            "code": code,
            "name_cn": desc["name_cn"] if desc else code,
            "super_class": desc["super_class"] if desc else "unknown",
            "count": cnt,
        })

    return {
        "by_class": by_class,
        "totals": {"objects": len(objects)},
    }


def build_evidence_package(image: str, region: str, objects: list[dict], stats: dict) -> dict:
    """构建最小 Evidence Package（无 GDAL 地理化，坐标留空）。"""
    now = datetime.now(timezone.utc).isoformat()
    pkg_id = f"sar-{datetime.now().strftime('%Y%m%d')}-{datetime.now().strftime('%H%M%S')}"

    return {
        "schema_version": "1.0.0",
        "package_id": pkg_id,
        "task_type": "intel_brief",
        "status": "READY_FOR_NLG",
        "input": {
            "image_uri": f"file://{Path(image).resolve()}",
            "metadata": {
                "satellite": "SAR卫星",
                "sensor": "SAR",
                "acquisition_time": now[:10],
                "gsd_m": None,
                "polarization": "未知",
            },
            "task": {"region_name": region, "region_type": "harbor"},
        },
        "scene": {
            "scene_type": "harbor",
            "geo_bounds": None,
            "environment": {},
        },
        "objects": objects,
        "statistics": stats,
        "attachments": {
            "annotated_image_uri": None,
        },
        "report": {},
        "quality": {},
        "errors": [],
    }


def generate_report_package(
    package: dict,
    llm_url: str | None = None,
    large_llm_url: str | None = None,
    use_collaborative_llm: bool = False,
    large_llm_model: str = "Qwen2.5-72B-Instruct",
    env_file: str | None = None,
    local_device: str | None = None,
    local_max_new_tokens: int | None = None,
    require_gpu: bool | None = None,
    allow_template_fallback: bool | None = None,
) -> dict:
    """生成通报正文包。"""
    sys.path.insert(0, str(Path(__file__).parent))
    config = resolve_collaborative_config(
        cache_dir=str(OUTPUT_DIR / ".report_cache"),
        env_file=env_file,
        small_url=llm_url,
        small_model="qwen3-4b",
        large_url=large_llm_url,
        large_model=large_llm_model,
        local_device=local_device,
        local_max_new_tokens=local_max_new_tokens,
        require_gpu=require_gpu,
        allow_template_fallback=allow_template_fallback,
    )
    if (
        use_collaborative_llm
        or large_llm_url
        or config.small_model_path
        or config.large_model_path
        or config.require_gpu
        or not config.allow_template_fallback
    ):
        from modules.report.collaborative import CollaborativeReportGenerator
        gen = CollaborativeReportGenerator(**collaborative_generator_kwargs(config))
        return gen.generate(package)

    if not config.allow_template_fallback:
        raise RuntimeError("LLM is required but no model backend is configured.")

    from modules.report.generator import ReportGenerator

    gen = ReportGenerator(
        base_url=llm_url,
        model_name="qwen3-4b",
        allow_template_fallback=config.allow_template_fallback,
    )
    return gen.generate(package)


def assemble_word(package: dict, output_dir: Path, include_image: bool = True) -> str:
    """调用 DocxAssembler 生成 Word 文件。"""
    sys.path.insert(0, str(Path(__file__).parent))
    from modules.report.docx_assembler import DocxAssembler

    assembler = DocxAssembler()
    docx_path = assembler.assemble(
        evidence_package=package,
        output_dir=str(output_dir),
        draft_mode=True,
        include_image=include_image,
    )
    return docx_path


def print_summary(objects: list[dict], stats: dict, docx_path: str, vis_path: str,
                  include_image: bool = True):
    print("\n" + "=" * 60)
    print(f"  检测结果汇总")
    print("=" * 60)
    print(f"  总目标数: {stats['totals']['objects']}")
    for item in stats["by_class"]:
        print(f"    {item['name_cn']}({item['code']}): {item['count']}")
    print(f"\n  Word 报告: {docx_path}")
    if include_image:
        print(f"  可视化图: {vis_path}")
    else:
        print(f"  可视化图: 已跳过（--no-image）")
    print("=" * 60)


def main():
    args = parse_args()

    weights = Path(args.weights)
    image   = Path(args.image)

    if not weights.exists():
        print(f"[ERROR] 权重不存在: {weights}")
        sys.exit(1)
    if not image.exists():
        print(f"[ERROR] 图像不存在: {image}")
        sys.exit(1)

    env_values = read_env_file(args.env_file)
    llm_url = args.llm_url or env_values.get("SAR_SMALL_LLM_URL")
    large_llm_url = args.large_llm_url or env_values.get("SAR_LARGE_LLM_URL")
    vlm_url = args.vlm_url or env_values.get("SAR_VLM_URL")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "v4_vis").mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 检测中: {image.name}  (weights={weights.name})")
    objects, vis_path = run_detection(str(weights), str(image), args.score_thresh, args.device,
                                      save_vis=args.include_image,
                                      tile_size=args.tile_size,
                                      tile_overlap=args.tile_overlap,
                                      tile_threshold=args.tile_threshold)
    print(f"      检测到 {len(objects)} 个目标")

    print("[2/5] 构建 Evidence Package ...")
    stats   = build_statistics(objects)
    package = build_evidence_package(str(image), args.region, objects, stats)

    # 将标注图路径写入 attachments（直接使用 detect() 返回的路径）
    if args.include_image and vis_path:
        package["attachments"]["annotated_image"] = {"uri": f"file://{vis_path}"}

    # VLM 图像整体描述（可选，vlm_url 为 None 时跳过）
    vlm_config = resolve_vlm_config(
        env_file=args.env_file,
        vlm_url=vlm_url,
        vlm_model=args.vlm_model,
        require_vlm=args.require_vlm,
    )
    if vlm_config.vlm_base_url:
        print(f"[3/5] VLM 图像描述中 ({vlm_config.vlm_base_url}) ...")
        sys.path.insert(0, str(Path(__file__).parent))
        from modules.report.vlm_describer import VLMDescriber
        from modules.report.vlm_trace import attach_vlm_scene_description
        describer = VLMDescriber(
            base_url=vlm_config.vlm_base_url,
            model_name=vlm_config.vlm_model_name or "qwen3-vl-4b",
            api_key=vlm_config.api_key or "EMPTY",
            timeout=vlm_config.timeout,
        )
        scene_desc = describer.describe(str(image))
        if scene_desc:
            attach_vlm_scene_description(
                package,
                description=scene_desc,
                describer=describer,
                image_path=str(image),
            )
            print(f"      VLM: {scene_desc}")
        elif vlm_config.require_vlm:
            raise RuntimeError("SAR_REQUIRE_VLM=1 but VLM returned an empty scene description.")
        else:
            print("      VLM: 描述失败，已跳过")
    elif vlm_config.require_vlm:
        raise RuntimeError("SAR_REQUIRE_VLM=1 but SAR_VLM_URL/--vlm-url is empty.")
    else:
        print("[3/5] VLM 图像描述: 未配置（--vlm-url），跳过")

    mode_label = "(协同LLM)" if (args.use_collaborative_llm or large_llm_url) else ("(LLM)" if llm_url else "(模板)")
    print(f"[4/5] 生成通报正文 {mode_label}...")
    package = generate_report_package(
        package,
        llm_url=llm_url,
        large_llm_url=large_llm_url,
        use_collaborative_llm=args.use_collaborative_llm,
        large_llm_model=args.large_llm_model,
        env_file=args.env_file,
        local_device=args.llm_device,
        local_max_new_tokens=args.llm_max_new_tokens,
        require_gpu=args.require_gpu,
        allow_template_fallback=args.allow_template_fallback,
    )

    print("[5/5] 组装 Word 文档 ...")
    docx_path = assemble_word(package, OUTPUT_DIR, include_image=args.include_image)
    package["report"]["docx_uri"] = f"file://{docx_path}"
    package["status"] = "DOCX_RENDERED"

    # 保存 evidence JSON
    pkg_id = package["package_id"]
    json_path = OUTPUT_DIR / f"{pkg_id}_evidence.json"
    json_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(objects, stats, docx_path, vis_path or "", include_image=args.include_image)


def _stats_display(stats: dict) -> dict[str, int]:
    """将 by_class 列表转为 {name_cn: count} 供打印用。"""
    return {item["name_cn"]: item["count"] for item in stats.get("by_class", [])}


if __name__ == "__main__":
    main()
