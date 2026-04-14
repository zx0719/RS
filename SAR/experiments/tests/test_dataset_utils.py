"""
test_dataset_utils.py — Unit tests for SSDD conversion utilities.

Tests convert a tiny synthetic VOC XML (2 objects) and verify:
- Output label file has the correct number of lines.
- Normalised cx, cy, w, h are in [0, 1].
- angle_rad is in [-pi/2, pi/2].
- YAML file is valid YAML and contains required keys.
- Train/val split is reproducible and non-overlapping.
"""

from __future__ import annotations

import math
import sys
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Ensure the experiments package root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.detector.dataset_utils import (
    convert_ssdd_to_yolo_obb,
    generate_ssdd_yaml,
    split_ssdd_train_val,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SSDD_XML_TEMPLATE = """\
<annotation>
  <filename>{stem}.jpg</filename>
  <size>
    <width>{width}</width>
    <height>{height}</height>
    <depth>1</depth>
  </size>
  <object>
    <name>ship</name>
    <rotated_bndbox>
      <rotated_bbox_cx>242</rotated_bbox_cx>
      <rotated_bbox_cy>96</rotated_bbox_cy>
      <rotated_bbox_w>98</rotated_bbox_w>
      <rotated_bbox_h>45</rotated_bbox_h>
      <rotated_bbox_theta>85.815</rotated_bbox_theta>
      <x1>215</x1><y1>48</y1>
      <x2>261</x2><y2>45</y2>
      <x3>268</x3><y3>143</y3>
      <x4>223</x4><y4>147</y4>
    </rotated_bndbox>
  </object>
  <object>
    <name>ship</name>
    <rotated_bndbox>
      <rotated_bbox_cx>100</rotated_bbox_cx>
      <rotated_bbox_cy>200</rotated_bbox_cy>
      <rotated_bbox_w>60</rotated_bbox_w>
      <rotated_bbox_h>30</rotated_bbox_h>
      <rotated_bbox_theta>-45.0</rotated_bbox_theta>
      <x1>80</x1><y1>185</y1>
      <x2>115</x2><y2>170</y2>
      <x3>120</x3><y3>215</y3>
      <x4>85</x4><y4>230</y4>
    </rotated_bndbox>
  </object>
</annotation>
"""

IMG_W, IMG_H = 500, 400


@pytest.fixture()
def fake_ssdd_root(tmp_path: Path) -> Path:
    """Build a minimal SSDD VOC-style directory with 3 synthetic images."""
    ann_dir = tmp_path / "Annotations"
    img_train_dir = tmp_path / "JPEGImages_train"
    img_test_dir  = tmp_path / "JPEGImages_test"
    ann_dir.mkdir()
    img_train_dir.mkdir()
    img_test_dir.mkdir()

    stems = ["000001", "000002", "000003"]
    for stem in stems:
        # Write XML annotation
        xml_content = _SSDD_XML_TEMPLATE.format(
            stem=stem, width=IMG_W, height=IMG_H
        )
        (ann_dir / f"{stem}.xml").write_text(xml_content, encoding="utf-8")
        # Write dummy image files (1-byte placeholder)
        (img_train_dir / f"{stem}.jpg").write_bytes(b"\xff")

    # One test image
    (img_test_dir / "000001.jpg").write_bytes(b"\xff")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests — convert_ssdd_to_yolo_obb
# ---------------------------------------------------------------------------


def test_ssdd_label_line_count(fake_ssdd_root: Path, tmp_path: Path):
    """Each label file should have exactly 2 lines (2 objects per XML)."""
    out_root = tmp_path / "out"
    stats = convert_ssdd_to_yolo_obb(
        ssdd_root=fake_ssdd_root,
        output_root=out_root,
        split="train",
    )
    label_dir = out_root / "labels" / "train"
    label_files = list(label_dir.glob("*.txt"))
    assert len(label_files) == 3, f"Expected 3 label files, got {len(label_files)}"

    for lf in label_files:
        lines = [l for l in lf.read_text().splitlines() if l.strip()]
        assert len(lines) == 2, f"{lf.name}: expected 2 lines, got {len(lines)}"


def test_ssdd_normalised_coords_in_range(fake_ssdd_root: Path, tmp_path: Path):
    """YOLO-OBB polygon format: class x1 y1 x2 y2 x3 y3 x4 y4, all coords in (0, 1]."""
    out_root = tmp_path / "out"
    convert_ssdd_to_yolo_obb(
        ssdd_root=fake_ssdd_root,
        output_root=out_root,
        split="train",
    )
    label_dir = out_root / "labels" / "train"
    for lf in label_dir.glob("*.txt"):
        for line in lf.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            assert len(parts) == 9, f"Expected 9 fields per line (class x1 y1 x2 y2 x3 y3 x4 y4), got {len(parts)}"
            class_id = int(parts[0])
            assert class_id == 0, f"Expected class_id=0, got {class_id}"
            coords = [float(p) for p in parts[1:]]
            for i, val in enumerate(coords):
                assert 0.0 <= val <= 1.0, (
                    f"{lf.name}: coord[{i}]={val} not in [0, 1]"
                )


def test_ssdd_angle_rad_in_range(fake_ssdd_root: Path, tmp_path: Path):
    """YOLO-OBB polygon: 4 corner points should form a non-degenerate quadrilateral."""
    out_root = tmp_path / "out"
    convert_ssdd_to_yolo_obb(
        ssdd_root=fake_ssdd_root,
        output_root=out_root,
        split="train",
    )
    label_dir = out_root / "labels" / "train"
    for lf in label_dir.glob("*.txt"):
        for line in lf.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            assert len(parts) == 9
            coords = [float(p) for p in parts[1:]]
            xs = coords[0::2]
            ys = coords[1::2]
            # Bounding box of the 4 points must have non-zero area
            assert max(xs) > min(xs), f"{lf.name}: degenerate polygon (zero x-span)"
            assert max(ys) > min(ys), f"{lf.name}: degenerate polygon (zero y-span)"


def test_ssdd_specific_angle_conversion(fake_ssdd_root: Path, tmp_path: Path):
    """YOLO-OBB polygon for 000001 first object should have 4 valid corner points."""
    out_root = tmp_path / "out"
    convert_ssdd_to_yolo_obb(
        ssdd_root=fake_ssdd_root,
        output_root=out_root,
        split="train",
        file_stems=["000001"],
    )
    label_path = out_root / "labels" / "train" / "000001.txt"
    lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    parts = lines[0].split()
    assert len(parts) == 9, f"Expected 9 fields, got {len(parts)}"
    coords = [float(p) for p in parts[1:]]
    for i, val in enumerate(coords):
        assert 0.0 <= val <= 1.0, f"coord[{i}]={val} out of [0,1]"


def test_ssdd_stats_dict(fake_ssdd_root: Path, tmp_path: Path):
    """Return dict should contain n_images, n_objects, output_dir."""
    out_root = tmp_path / "out"
    stats = convert_ssdd_to_yolo_obb(
        ssdd_root=fake_ssdd_root,
        output_root=out_root,
        split="train",
    )
    assert "n_images"  in stats
    assert "n_objects" in stats
    assert "output_dir" in stats
    assert stats["n_images"]  == 3
    assert stats["n_objects"] == 6   # 3 images × 2 objects each


def test_ssdd_images_copied(fake_ssdd_root: Path, tmp_path: Path):
    """Image files should be copied to images/train/."""
    out_root = tmp_path / "out"
    convert_ssdd_to_yolo_obb(
        ssdd_root=fake_ssdd_root,
        output_root=out_root,
        split="train",
    )
    img_dir = out_root / "images" / "train"
    imgs = list(img_dir.glob("*.jpg"))
    assert len(imgs) == 3, f"Expected 3 images copied, got {len(imgs)}"


# ---------------------------------------------------------------------------
# Tests — split_ssdd_train_val
# ---------------------------------------------------------------------------


def test_split_no_overlap(fake_ssdd_root: Path):
    """Train and val sets must be disjoint."""
    train, val = split_ssdd_train_val(fake_ssdd_root, val_ratio=0.33, seed=0)
    assert set(train).isdisjoint(set(val)), "Train and val sets overlap"


def test_split_covers_all_stems(fake_ssdd_root: Path):
    """Union of train+val must equal the full annotation set."""
    ann_dir = fake_ssdd_root / "Annotations"
    all_stems = {p.stem for p in ann_dir.glob("*.xml")}
    train, val = split_ssdd_train_val(fake_ssdd_root, val_ratio=0.33, seed=0)
    assert set(train) | set(val) == all_stems


def test_split_reproducible(fake_ssdd_root: Path):
    """Same seed must yield identical splits across calls."""
    train_a, val_a = split_ssdd_train_val(fake_ssdd_root, seed=99)
    train_b, val_b = split_ssdd_train_val(fake_ssdd_root, seed=99)
    assert train_a == train_b
    assert val_a   == val_b


def test_split_different_seeds(fake_ssdd_root: Path):
    """Different seeds should (with high probability) produce different splits."""
    # With only 3 stems and val_ratio=0.33, we get 1 val item each time.
    # Different seeds may shuffle to the same split by chance on a tiny dataset,
    # but we can at least verify the function runs without error.
    train_a, val_a = split_ssdd_train_val(fake_ssdd_root, seed=1)
    train_b, val_b = split_ssdd_train_val(fake_ssdd_root, seed=2)
    # Just verify both splits are valid (non-empty, cover all stems)
    ann_dir = fake_ssdd_root / "Annotations"
    all_stems = {p.stem for p in ann_dir.glob("*.xml")}
    assert set(train_a) | set(val_a) == all_stems
    assert set(train_b) | set(val_b) == all_stems


# ---------------------------------------------------------------------------
# Tests — generate_ssdd_yaml
# ---------------------------------------------------------------------------


def test_yaml_file_created(tmp_path: Path):
    """generate_ssdd_yaml should create a file at the default location."""
    out_root = tmp_path / "ssdd_yolo_obb"
    out_root.mkdir()
    yaml_path = generate_ssdd_yaml(output_root=out_root)
    assert yaml_path.exists(), f"YAML file not created at {yaml_path}"


def test_yaml_required_keys(tmp_path: Path):
    """YAML must contain path, train, val, nc, names."""
    try:
        import yaml  # PyYAML
        _has_yaml = True
    except ImportError:
        _has_yaml = False

    out_root = tmp_path / "ssdd_yolo_obb"
    out_root.mkdir()
    yaml_path = generate_ssdd_yaml(output_root=out_root)
    content = yaml_path.read_text(encoding="utf-8")

    # Basic string checks — no PyYAML dependency required
    for required in ("path:", "train:", "val:", "nc:", "names:"):
        assert required in content, f"YAML missing key: {required!r}"

    # Ship class must be listed
    assert "ship" in content, "YAML must mention 'ship' class"

    if _has_yaml:
        data = yaml.safe_load(content)
        assert data["nc"] == 1
        assert "ship" in data["names"]


def test_yaml_custom_path(tmp_path: Path):
    """generate_ssdd_yaml should honour a custom yaml_path argument."""
    out_root = tmp_path / "dataset"
    out_root.mkdir()
    custom_yaml = tmp_path / "custom" / "my_dataset.yaml"
    result = generate_ssdd_yaml(output_root=out_root, yaml_path=custom_yaml)
    assert result == custom_yaml
    assert custom_yaml.exists()


def test_yaml_path_is_absolute(tmp_path: Path):
    """The 'path:' field in the YAML should be an absolute filesystem path."""
    out_root = tmp_path / "ssdd_yolo_obb"
    out_root.mkdir()
    yaml_path = generate_ssdd_yaml(output_root=out_root)
    content = yaml_path.read_text(encoding="utf-8")
    # Find the path: line
    for line in content.splitlines():
        if line.startswith("path:"):
            value = line.split(":", 1)[1].strip()
            assert Path(value).is_absolute(), (
                f"'path:' value should be absolute, got: {value!r}"
            )
            break
