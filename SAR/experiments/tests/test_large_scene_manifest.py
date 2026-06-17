from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_large_scene_tests.py"
SPEC = importlib.util.spec_from_file_location("run_large_scene_tests", SCRIPT_PATH)
large_scene = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = large_scene
SPEC.loader.exec_module(large_scene)


def test_manifest_selects_smoke_and_full_cases() -> None:
    manifest = large_scene.load_manifest(
        Path(__file__).resolve().parent / "fixtures" / "large_scene_manifest.json"
    )

    smoke = large_scene.select_cases(manifest, suite="smoke")
    full = large_scene.select_cases(manifest, suite="full")

    assert [case.name for case in smoke] == ["ship2_large_tiff", "plane_ultralong_png"]
    assert [case.name for case in full] == [
        "ship2_large_tiff",
        "ship1_large_tiff",
        "plane_ultralong_png",
        "airport2_large_tiff",
        "bridge_large_tiff",
    ]
    assert all(not Path(raw["relative_path"]).is_absolute() for raw in manifest["cases"])


def test_tile_grid_covers_ultralong_scene_edges() -> None:
    windows = large_scene.tile_grid(width=9431, height=16409, tile_size=1536, overlap=256)

    assert windows[0] == (0, 0, 1536, 1536)
    assert max(x + w for x, _y, w, _h in windows) == 9431
    assert max(y + h for _x, y, _w, h in windows) == 16409
    assert all(w <= 1536 and h <= 1536 for _x, _y, w, h in windows)


def test_shift_clamp_and_bounds_validation() -> None:
    obj = {
        "object_id": "obj-test",
        "class": {"code": "ship"},
        "score": {"confidence": 0.9},
        "geometry": {
            "pixel": {
                "center_x": 100,
                "center_y": 100,
                "polygon": [[-5, -5], [20, -5], [20, 20], [-5, 20]],
                "bbox_axis_aligned": [-5, -5, 20, 20],
            }
        },
    }

    shifted = large_scene.shift_object_to_global(
        obj, offset_x=920, offset_y=920, image_width=1000, image_height=1000
    )

    pixel = shifted["geometry"]["pixel"]
    assert pixel["center_x"] == 999
    assert pixel["center_y"] == 999
    assert pixel["bbox_axis_aligned"] == [915.0, 915.0, 940.0, 940.0]
    large_scene.validate_object_bounds([shifted], image_width=1000, image_height=1000)


def test_nms_suppresses_same_class_duplicate_only() -> None:
    def obj(object_id: str, code: str, confidence: float, box: list[float]) -> dict:
        return {
            "object_id": object_id,
            "class": {"code": code},
            "score": {"confidence": confidence},
            "geometry": {"pixel": {"bbox_axis_aligned": box}},
        }

    objects = [
        obj("a", "ship", 0.9, [0, 0, 100, 100]),
        obj("b", "ship", 0.8, [5, 5, 105, 105]),
        obj("c", "aircraft", 0.7, [5, 5, 105, 105]),
    ]

    kept = large_scene.nms_objects(objects, iou_thresh=0.5)

    assert [item["object_id"] for item in kept] == ["a", "c"]
