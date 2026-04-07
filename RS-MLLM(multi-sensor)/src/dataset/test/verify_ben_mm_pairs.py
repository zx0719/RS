from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def scan_s2(root: Path):
    s2_root = root / 'BigEarthNet-S2'
    mapping = {}
    for tile_dir in s2_root.iterdir():
        if not tile_dir.is_dir():
            continue
        for patch_dir in tile_dir.iterdir():
            if not patch_dir.is_dir():
                continue
            pid = patch_dir.name
            if (patch_dir / f'{pid}_B04.tif').exists():
                mapping[pid] = patch_dir
    return mapping


def scan_s1(root: Path):
    s1_root = root / 'BigEarthNet-S1'
    mapping = {}
    for scene_dir in s1_root.iterdir():
        if not scene_dir.is_dir():
            continue
        for patch_dir in scene_dir.iterdir():
            if not patch_dir.is_dir():
                continue
            pid = patch_dir.name
            if (patch_dir / f'{pid}_VV.tif').exists() and (patch_dir / f'{pid}_VH.tif').exists():
                mapping[pid] = patch_dir
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='BEN-MM 根目录，里面应包含 BigEarthNet-S2 / BigEarthNet-S1 / metadata.parquet')
    parser.add_argument('--metadata-file', default='metadata.parquet')
    parser.add_argument('--split', default=None, choices=[None, 'train', 'val', 'validation', 'test'], nargs='?')
    args = parser.parse_args()

    root = Path(args.root)
    metadata_path = root / args.metadata_file
    if not metadata_path.exists():
        raise FileNotFoundError(f'未找到 metadata 文件: {metadata_path}')

    df = pd.read_parquet(metadata_path)
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    patch_col = pick('patch_id', 'patch_name', 'name')
    s1_col = pick('s1_name', 'patch_name_s1', 's1_patch_name')
    split_col = pick('split')

    if patch_col is None or s1_col is None:
        raise RuntimeError('metadata.parquet 缺少 patch_id 或 s1_name 列，无法校验配对')

    if args.split is not None and split_col is not None:
        split_value = 'val' if args.split == 'validation' else args.split
        df = df[df[split_col].astype(str).str.lower() == split_value]

    s2_map = scan_s2(root)
    s1_map = scan_s1(root)

    total = len(df)
    missing_s2 = []
    missing_s1 = []
    ok = 0

    for _, row in df.iterrows():
        patch_id = str(row[patch_col])
        s1_name = str(row[s1_col]) if row[s1_col] is not None else None
        has_s2 = patch_id in s2_map
        has_s1 = s1_name in s1_map if s1_name else False

        if not has_s2:
            missing_s2.append(patch_id)
        if not has_s1:
            missing_s1.append((patch_id, s1_name))
        if has_s2 and has_s1:
            ok += 1

    print(f'total metadata rows      : {total}')
    print(f'paired rows existing     : {ok}')
    print(f'missing S2 patch count   : {len(missing_s2)}')
    print(f'missing S1 patch count   : {len(missing_s1)}')

    if missing_s2:
        print('\nExamples missing S2:')
        for x in missing_s2[:10]:
            print('  ', x)

    if missing_s1:
        print('\nExamples missing S1:')
        for patch_id, s1_name in missing_s1[:10]:
            print('  ', patch_id, '->', s1_name)

    if ok == total and total > 0:
        print('\n[OK] metadata 里的 S2 patch 和 S1 patch 在本地目录中都能对应上。')
    else:
        print('\n[WARN] 存在配对缺失。先确认 S1 是否已完整解压，再检查 metadata 与目录版本是否一致。')


if __name__ == '__main__':
    main()
