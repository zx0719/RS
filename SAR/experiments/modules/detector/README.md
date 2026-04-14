# modules/detector — M2 SAR Target Detection (YOLOv8-OBB)

This module implements the **M2 Detection** stage of the SAR intelligence reporting pipeline.

It wraps a trained YOLOv8-OBB model and produces `objects[]` entries conforming to the
**Evidence Package schema v1.0** (`SAR统一证据JSON与模块接口定义_v1.md`).

---

## File overview

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports (`DetectorTool`, `CLASS_MAP`, `get_class_descriptor`) |
| `detector.py` | `DetectorTool` inference class |
| `class_map.py` | YOLO class index → Evidence Package class descriptor |
| `train.py` | YOLOv8-OBB training script |
| `dataset_utils.py` | DOTA-format → YOLO-OBB annotation converter |
| `README.md` | This file |

---

## Quick start

### 1. Install dependencies

```bash
pip install ultralytics numpy
```

### 2. Run inference

```python
from modules.detector import DetectorTool

tool = DetectorTool(
    model_path="runs/detect/train/weights/best.pt",
    score_thresh=0.25,
    nms_thresh=0.5,
    device="cuda:0",   # or "cpu"
)

objects = tool.detect("scene.tif")
# objects is a list[dict] — each dict is one Evidence Package objects[] entry

# Insert into an existing Evidence Package
evidence_package["objects"] = objects
```

### 3. Train a model

```bash
# Convert DOTA annotations first (if needed)
python modules/detector/dataset_utils.py \
    --dota_dir /data/DOTA/train/labelTxt \
    --output_dir /data/yolo_obb/labels/train \
    --img_width 1024 --img_height 1024 \
    --gen_yaml --train_images /data/yolo_obb/images/train \
               --val_images   /data/yolo_obb/images/val

# Then train
python modules/detector/train.py \
    --data /data/yolo_obb/dataset.yaml \
    --epochs 100 \
    --imgsz 1024 \
    --model yolov8m-obb.pt \
    --device cuda:0
```

Best weights are saved to `runs/detect/train/weights/best.pt`.

---

## Class map

The default class ordering (index 0–11) is:

| Index | Code | Chinese | Super class | Priority |
|-------|------|---------|-------------|----------|
| 0 | `carrier` | 航空母舰 | ship | CRITICAL |
| 1 | `destroyer` | 驱逐舰 | ship | HIGH |
| 2 | `frigate` | 护卫舰 | ship | HIGH |
| 3 | `replenishment` | 补给舰 | ship | MEDIUM |
| 4 | `amphibious` | 两栖舰 | ship | HIGH |
| 5 | `other_vessel` | 其他舰船 | ship | LOW |
| 6 | `fighter` | 战斗机 | aircraft | HIGH |
| 7 | `bomber` | 轰炸机 | aircraft | CRITICAL |
| 8 | `transport` | 运输机 | aircraft | MEDIUM |
| 9 | `aew` | 预警机 | aircraft | HIGH |
| 10 | `helicopter` | 直升机 | aircraft | MEDIUM |
| 11 | `other_aircraft` | 其他飞机 | aircraft | LOW |

**Important**: the ordering in `class_map.py` must match the `names:` list in your `dataset.yaml`.
Edit `class_map.py` to reorder or add classes as needed.

---

## Output schema

Each entry returned by `DetectorTool.detect()` looks like:

```json
{
  "object_id": "obj-A3F21C",
  "source_module": "detector_yolov8_obb_v1",
  "status": "VALID",
  "class": {
    "code": "destroyer",
    "name_cn": "驱逐舰",
    "super_class": "ship",
    "priority": "HIGH"
  },
  "score": {
    "confidence": 0.923400,
    "calibrated_confidence": 0.895698
  },
  "geometry": {
    "pixel": {
      "center_x": 1032.4,
      "center_y": 812.7,
      "width": 86.3,
      "height": 18.9,
      "angle_deg": -27.4,
      "polygon": [[991.1, 806.2], [1068.4, 766.0], [1073.7, 819.4], [996.3, 859.1]],
      "bbox_axis_aligned": [991.1, 766.0, 1073.7, 859.1]
    }
  },
  "evidence": {
    "crop_uri": null,
    "detector_version": "yolov8-obb-shipair-v1.3"
  },
  "audit": {
    "review_status": "UNREVIEWED",
    "review_comment": "",
    "reviewer": ""
  }
}
```

`geometry.geo` is **not** filled by this module — that is the responsibility of the
`geo-pipeline-engineer` (M3 Geolocate stage).

---

## Notes

- The `calibrated_confidence` value is currently a placeholder (`confidence × 0.97`).
  Replace with a proper isotonic regression or temperature-scaling calibration once you
  have a held-out calibration set.
- For GeoTIFF inputs, only pixel data is read by YOLOv8. Geo-referencing metadata is
  handled downstream.
- To adjust confidence thresholds post-training, simply change `score_thresh` in
  `DetectorTool.__init__()` without retraining.
