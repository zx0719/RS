import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import DataLoader

sys.path.insert(0, '/mnt/data')
import ben_mm_dataset as ben


def _write_tif(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=h,
        width=w,
        count=1,
        dtype=arr.dtype,
        crs='EPSG:4326',
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(arr, 1)


def _make_s2_patch(root: Path, tile: str, patch_id: str, bands=None):
    bands = bands or ben.S2_ALL_BANDS
    patch_dir = root / 'BigEarthNet-S2' / tile / patch_id
    for i, band in enumerate(bands):
        if band in {'B01', 'B09'}:
            size = 20
        elif band in {'B05', 'B06', 'B07', 'B8A', 'B11', 'B12'}:
            size = 60
        else:
            size = 120
        arr = np.full((size, size), fill_value=(i + 1) * 1000, dtype=np.uint16)
        _write_tif(patch_dir / f'{patch_id}_{band}.tif', arr)
    return patch_dir


def _make_s1_patch(root: Path, scene: str, s1_name: str):
    patch_dir = root / 'BigEarthNet-S1' / scene / s1_name
    vv = np.full((120, 120), -10, dtype=np.float32)
    vh = np.full((120, 120), -15, dtype=np.float32)
    _write_tif(patch_dir / f'{s1_name}_VV.tif', vv)
    _write_tif(patch_dir / f'{s1_name}_VH.tif', vh)
    return patch_dir


def test_s2_only_without_metadata(tmp_path: Path):
    patch_id = 'S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR_63_74'
    _make_s2_patch(tmp_path, 'S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR', patch_id)

    ds = ben.BENMMRGBMSSARDataset(
        root=str(tmp_path),
        split=None,
        enable_rgb=True,
        enable_ms=True,
        enable_sar=False,
        use_metadata=False,
    )

    assert len(ds) == 1
    sample = ds[0]
    assert sample['patch_id'] == patch_id
    assert tuple(sample['rgb'].shape) == (3, 120, 120)
    assert tuple(sample['ms'].shape) == (10, 120, 120)
    assert tuple(sample['labels'].shape) == (19,)
    assert torch.all(sample['labels'] == 0)
    assert sample['rgb'].dtype == torch.float32
    assert sample['ms'].dtype == torch.float32
    assert float(sample['rgb'].max()) <= 1.5



def test_rgb_ms_sar_with_metadata_and_split(monkeypatch, tmp_path: Path):
    patch_id = 'S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR_63_74'
    s1_name = 'S1A_IW_GRDH_1SDV_20170801T045000_20170801T045025_017553_01D6EC_63_74'
    _make_s2_patch(tmp_path, 'S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR', patch_id)
    _make_s1_patch(tmp_path, 'S1A_IW_GRDH_1SDV_20170801T045000_20170801T045025_017553_01D6EC', s1_name)

    metadata_path = tmp_path / 'metadata.parquet'
    metadata_path.touch()

    df = pd.DataFrame([
        {
            'patch_id': patch_id,
            'split': 'train',
            's1_name': s1_name,
            'labels': ['Urban fabric', 'Inland waters'],
        },
        {
            'patch_id': 'unused_patch',
            'split': 'val',
            's1_name': 'unused_s1',
            'labels': ['Marine waters'],
        },
    ])

    monkeypatch.setattr(ben.pd, 'read_parquet', lambda path: df)

    ds = ben.BENMMRGBMSSARDataset(
        root=str(tmp_path),
        split='train',
        enable_rgb=True,
        enable_ms=True,
        enable_sar=True,
        use_metadata=True,
    )

    assert len(ds) == 1
    sample = ds[0]
    assert tuple(sample['rgb'].shape) == (3, 120, 120)
    assert tuple(sample['ms'].shape) == (10, 120, 120)
    assert tuple(sample['sar'].shape) == (2, 120, 120)
    assert sample['s1_name'] == s1_name
    assert sample['label_names'] == ['Urban fabric', 'Inland waters']
    assert sample['labels'][ben.BEN19_LABELS.index('Urban fabric')] == 1
    assert sample['labels'][ben.BEN19_LABELS.index('Inland waters')] == 1
    assert float(sample['sar'].min()) >= 0.0
    assert float(sample['sar'].max()) <= 1.0



def test_metadata_43_labels_are_mapped_to_ben19(monkeypatch, tmp_path: Path):
    patch_id = 'S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR_63_75'
    s1_name = 'S1A_IW_GRDH_1SDV_20170801T045000_20170801T045025_017553_01D6EC_63_75'
    _make_s2_patch(tmp_path, 'S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR', patch_id)
    _make_s1_patch(tmp_path, 'S1A_IW_GRDH_1SDV_20170801T045000_20170801T045025_017553_01D6EC', s1_name)
    (tmp_path / 'metadata.parquet').touch()

    df = pd.DataFrame([
        {
            'patch_id': patch_id,
            'split': 'train',
            's1_name': s1_name,
            'original_labels': ['Continuous urban fabric', 'Water bodies'],
        }
    ])
    monkeypatch.setattr(ben.pd, 'read_parquet', lambda path: df)

    ds = ben.BENMMRGBMSSARDataset(
        root=str(tmp_path),
        split='train',
        enable_rgb=False,
        enable_ms=True,
        enable_sar=True,
        use_metadata=True,
    )
    sample = ds[0]
    assert 'rgb' not in sample
    assert tuple(sample['ms'].shape) == (10, 120, 120)
    assert tuple(sample['sar'].shape) == (2, 120, 120)
    assert 'Urban fabric' in sample['label_names']
    assert 'Inland waters' in sample['label_names']



def test_dataloader_batching(tmp_path: Path):
    for idx in range(2):
        patch_id = f'S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR_63_{74+idx}'
        _make_s2_patch(tmp_path, 'S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR', patch_id)

    ds = ben.BENMMRGBMSSARDataset(
        root=str(tmp_path),
        split=None,
        enable_rgb=True,
        enable_ms=False,
        enable_sar=False,
        use_metadata=False,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert tuple(batch['rgb'].shape) == (2, 3, 120, 120)
    assert tuple(batch['labels'].shape) == (2, 19)
    assert len(batch['patch_id']) == 2



def test_invalid_all_modalities_disabled(tmp_path: Path):
    with pytest.raises(ValueError):
        ben.BENMMRGBMSSARDataset(
            root=str(tmp_path),
            enable_rgb=False,
            enable_ms=False,
            enable_sar=False,
            use_metadata=False,
        )
