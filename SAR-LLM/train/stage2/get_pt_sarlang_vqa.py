"""
get_pt_sarlang_vqa.py
=====================
为 SARLANG-1M VQA 数据集提取 SARCLIP patch token 特征（.pt 文件），
并在原 json 基础上新增 "pt_path" 字段，写成 *_pt.json。

策略（高效版）：
  1. 扫描 SARDET_TRAIN_DIR 建 stem→path 索引（O(N_imgs)）
  2. 读所有 split json，收集唯一图片 stem
  3. 对尚未缓存的唯一图片批量提取 pt（GPU 利用率高）
  4. 回写各 split 的 *_pt.json（纯 dict 查找，O(N_samples)）

前提：
  - SARDet_100K train 图片已解压到
    /mnt/data/mm_data/SAR/SARDet_100K/data/Images/train/
  - SARCLIP ViT-B-16 模型已就位：
    /home/qianwentao/SARClip/Model/ViT-B-16/

处理的数据集：
  - SARVQA1_train.json / SARVQA1_test.json
  - SARVQA2_train.json / SARVQA2_test.json

输出：
  - pt 文件  → /mnt/data/mm_data/SAR/SARDet_100K/data/Images/pt_cache/
  - pt json  → /mnt/data/mm_data/SAR/SARLANG-1M/Text/VQA/*/SARVQA*_pt.json

用法：
  python get_pt_sarlang_vqa.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ─────────────────────────────────────────────────────────────
# 0. sys.path：SARCLIP 库 + 项目 stage1 工具
# ─────────────────────────────────────────────────────────────
SARCLIP_QWEN_DIR = Path("/home/qianwentao/SARClip/SARCLIP+QwenVL")
SARCLIP_REPO_DIR = Path("/home/qianwentao/SARClip/SARCLIP-main")
STAGE1_DIR       = Path(__file__).resolve().parent.parent / "stage1"

# SARCLIP dirs must come before STAGE1_DIR to avoid shadowing sarclip_module.py
for _p in [str(SARCLIP_QWEN_DIR), str(SARCLIP_REPO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# STAGE1_DIR provides image_io; append so it doesn't shadow SARCLIP's sarclip_module
if str(STAGE1_DIR) not in sys.path:
    sys.path.append(str(STAGE1_DIR))

from sarclip_module import SarClipFrozenEncoder   # noqa: E402
from image_io import read_sar_as_rgb, BadSarImageError  # noqa: E402

# ─────────────────────────────────────────────────────────────
# 1. 配置
# ─────────────────────────────────────────────────────────────
SARCLIP_MODEL_NAME = "ViT-B-16"
SARCLIP_CACHE_DIR  = "/home/qianwentao/SARClip/Model/ViT-B-16"
DEVICE             = "cuda:0"
PRECISION          = "fp32"
BATCH_SIZE         = 128   # 更大 batch 提升 GPU 利用率

# 图片根目录（SARDet-100K train 解压后）
SARDET_TRAIN_DIR = Path("/mnt/data/mm_data/SAR/SARDet_100K/data/Images/train")

# pt 输出目录
PT_CACHE_DIR = Path("/mnt/data/mm_data/SAR/SARDet_100K/data/Images/pt_cache")

# VQA json 根目录
VQA_ROOT = Path("/mnt/data/mm_data/SAR/SARLANG-1M/Text/VQA")

# 要处理的 split
SPLITS = [
    {
        "name":        "SARVQA1_train",
        "input_json":  VQA_ROOT / "train" / "SARVQA1_train.json",
        "output_json": VQA_ROOT / "train" / "SARVQA1_train_pt.json",
    },
    {
        "name":        "SARVQA1_test",
        "input_json":  VQA_ROOT / "test"  / "SARVQA1_test.json",
        "output_json": VQA_ROOT / "test"  / "SARVQA1_test_pt.json",
    },
    {
        "name":        "SARVQA2_train",
        "input_json":  VQA_ROOT / "train" / "SARVQA2_train.json",
        "output_json": VQA_ROOT / "train" / "SARVQA2_train_pt.json",
    },
    {
        "name":        "SARVQA2_test",
        "input_json":  VQA_ROOT / "test"  / "SARVQA2_test.json",
        "output_json": VQA_ROOT / "test"  / "SARVQA2_test_pt.json",
    },
]

ALLOWED_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


# ─────────────────────────────────────────────────────────────
# 2. 图片索引（一次性扫描，O(N_imgs)）
# ─────────────────────────────────────────────────────────────
def build_image_index(train_dir: Path) -> Dict[str, Path]:
    """stem → full path，支持多后缀。"""
    idx: Dict[str, Path] = {}
    for p in train_dir.iterdir():
        if p.suffix.lower() in ALLOWED_EXTS:
            idx[p.stem] = p
    return idx


# ─────────────────────────────────────────────────────────────
# 3. 从 VQA json 提取图片 stem
# ─────────────────────────────────────────────────────────────
def collect_stems_from_split(input_json: Path) -> List[str]:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    stems = []
    for sample in data:
        images = sample.get("images", [])
        if images:
            stems.append(Path(images[0]).stem)
    return stems


# ─────────────────────────────────────────────────────────────
# 4. 批量提取唯一图片的 pt 特征
# ─────────────────────────────────────────────────────────────
def extract_unique_images(
    unique_stems: List[str],
    img_index: Dict[str, Path],
    encoder: SarClipFrozenEncoder,
) -> Dict[str, Optional[Path]]:
    """
    返回 stem → pt_path (None 表示提取失败)。
    只处理 pt_cache 中尚未存在的 stem。
    """
    PT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    preprocess = encoder.preprocess

    # 分 already-done 和 need-extract
    stem_to_pt: Dict[str, Optional[Path]] = {}
    to_extract: List[str] = []

    for stem in unique_stems:
        pt_path = PT_CACHE_DIR / f"{stem}.pt"
        if pt_path.exists():
            stem_to_pt[stem] = pt_path
        elif stem in img_index:
            to_extract.append(stem)
        else:
            stem_to_pt[stem] = None   # 图片不存在

    n_skip    = len(stem_to_pt)
    n_total   = len(to_extract)
    n_success = 0
    n_fail    = 0
    t0        = time.time()

    print(f"  唯一图片: {len(unique_stems)}  |  已缓存={n_skip}  |  待提取={n_total}")

    for batch_start in range(0, len(to_extract), BATCH_SIZE):
        batch_stems = to_extract[batch_start : batch_start + BATCH_SIZE]
        imgs_tensor = []
        valid_stems = []

        for stem in batch_stems:
            img_path = img_index[stem]
            try:
                tensor = preprocess(read_sar_as_rgb(str(img_path)))
                imgs_tensor.append(tensor)
                valid_stems.append(stem)
            except BadSarImageError as e:
                print(f"  [WARN] {img_path.name}: {e}")
                n_fail += 1
                stem_to_pt[stem] = None
            except Exception as e:
                print(f"  [WARN] {img_path.name}: {e}")
                n_fail += 1
                stem_to_pt[stem] = None

        if not imgs_tensor:
            continue

        imgs_batch = torch.stack(imgs_tensor).to(DEVICE)
        with torch.no_grad():
            tokens = encoder.encode_image_tokens(
                imgs_batch, normalize=False, drop_cls=True
            ).float().cpu()   # (B, 195, 768)

        for i, stem in enumerate(valid_stems):
            pt_path = PT_CACHE_DIR / f"{stem}.pt"
            try:
                torch.save(tokens[i], pt_path)
                stem_to_pt[stem] = pt_path
                n_success += 1
            except Exception as e:
                print(f"  [WARN] 保存失败 {stem}.pt: {e}")
                n_fail += 1
                stem_to_pt[stem] = None

        done    = batch_start + len(batch_stems)
        elapsed = time.time() - t0
        speed   = n_success / max(elapsed, 1e-6)
        eta     = (n_total - done) / max((n_success + n_fail) / max(elapsed, 1e-6), 1e-6)
        print(f"  提取进度 {done}/{n_total}  "
              f"成功={n_success} 失败={n_fail}  "
              f"{speed:.1f}新图/s  ETA={eta:.0f}s")

    elapsed_total = time.time() - t0
    print(f"  提取完成  成功={n_success} 已跳过={n_skip} 失败={n_fail}  耗时={elapsed_total:.1f}s")
    return stem_to_pt


# ─────────────────────────────────────────────────────────────
# 5. 回写单个 split 的 *_pt.json
# ─────────────────────────────────────────────────────────────
def write_split_json(
    split: dict,
    stem_to_pt: Dict[str, Optional[Path]],
) -> dict:
    name        = split["name"]
    input_json  = split["input_json"]
    output_json = split["output_json"]

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(input_json, "r", encoding="utf-8") as f:
        raw_data: list = json.load(f)

    output_data = []
    n_ok = n_skip = n_fail = 0
    for sample in raw_data:
        images = sample.get("images", [])
        if not images:
            n_fail += 1
            continue
        stem = Path(images[0]).stem
        pt_path = stem_to_pt.get(stem)
        if pt_path is None:
            n_fail += 1
            continue
        new_sample = dict(sample)
        new_sample["pt_path"] = str(pt_path)
        output_data.append(new_sample)
        n_ok += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False)

    print(f"  [{name}] 写出 {len(output_data)} 条  "
          f"(ok={n_ok} fail={n_fail})  → {output_json}")

    return dict(name=name, n_ok=n_ok, n_fail=n_fail, n_out=len(output_data))


# ─────────────────────────────────────────────────────────────
# 6. main
# ─────────────────────────────────────────────────────────────
def main():
    # 检查图片目录
    if not SARDET_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"SARDet-100K train 目录不存在: {SARDET_TRAIN_DIR}\n"
            "请先解压 train_full.zip：\n"
            "  7z x .../train_full.zip -o.../Images/ -y"
        )

    # Step 1: 扫描图片目录
    print(f"[1/4] 扫描图片目录: {SARDET_TRAIN_DIR}")
    t0 = time.time()
    img_index = build_image_index(SARDET_TRAIN_DIR)
    print(f"  找到 {len(img_index)} 张图片  ({time.time()-t0:.1f}s)")

    # Step 2: 收集所有 split 中的唯一图片 stem
    print(f"\n[2/4] 扫描各 split json，收集唯一图片 stem ...")
    all_stems: set = set()
    for sp in SPLITS:
        stems = collect_stems_from_split(sp["input_json"])
        all_stems.update(stems)
        print(f"  {sp['name']}: {len(stems)} 样本")
    unique_stems = sorted(all_stems)
    print(f"  共 {len(unique_stems)} 个唯一图片 stem")

    # Step 3: 初始化 encoder + 批量提取
    print(f"\n[3/4] 初始化 SARCLIP encoder ({SARCLIP_MODEL_NAME})")
    encoder = SarClipFrozenEncoder(
        model_name = SARCLIP_MODEL_NAME,
        pretrained = None,
        cache_dir  = SARCLIP_CACHE_DIR,
        device     = DEVICE,
        precision  = PRECISION,
    )
    encoder.eval()

    with torch.no_grad():
        _dummy = torch.zeros(1, 3, 224, 224, device=DEVICE)
        _tok   = encoder.encode_image_tokens(_dummy, normalize=False, drop_cls=True)
    token_count, token_dim = _tok.shape[1], _tok.shape[2]
    del _dummy, _tok
    print(f"  token shape: ({token_count}, {token_dim})")

    stem_to_pt = extract_unique_images(unique_stems, img_index, encoder)

    # Step 4: 回写各 split json
    print(f"\n[4/4] 回写各 split *_pt.json ...")
    t_total = time.time()
    all_stats = [write_split_json(sp, stem_to_pt) for sp in SPLITS]

    print("\n" + "="*60)
    print("  SARLANG-1M VQA 提取完成 —— 汇总")
    print("="*60)
    print(f"  {'split':<22} {'ok':>10} {'fail':>8} {'输出':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*8}")
    for s in all_stats:
        print(f"  {s['name']:<22} {s['n_ok']:>10} {s['n_fail']:>8} {s['n_out']:>8}")
    print(f"\n  token 格式: ({token_count}, {token_dim}), dtype=torch.float32")
    print(f"  写 json 耗时: {time.time()-t_total:.1f}s")
    print("="*60)

    # 提示后续步骤
    print("\n[下一步] 在 train/stage2/config_local.py 中修改：")
    print("  mixed_weight_sarlang_vqa = 1.0")


if __name__ == "__main__":
    main()
