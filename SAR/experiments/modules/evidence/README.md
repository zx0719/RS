# modules/evidence — M4 Evidence Fusion

Assembles a fully-validated **Evidence Package** (schema v1.0) from the
outputs of M1 (preprocessing) and M3 (geolocalisation) together with the
M2 detector results.

---

## Quick Start

```python
from modules.evidence import EvidenceBuilder

mission = {
    "region_name": "某某军港",
    "region_type": "harbor",
    "priority": "HIGH",
    "user_prompt": "生成标准军事情报通报",
}

# objects[] come from the detector (geometry.pixel already filled)
objects = [
    {
        "object_id": "obj-000001",
        "class": {"code": "destroyer", "name_cn": "驱逐舰",
                  "super_class": "ship", "priority": "HIGH"},
        "score": {"confidence": 0.92},
        "geometry": {
            "pixel": {
                "center_x": 1032.4, "center_y": 812.7,
                "width": 86.3, "height": 18.9, "angle_deg": -27.4,
            }
        },
        "audit": {"review_status": "UNREVIEWED"},
    }
]

builder = EvidenceBuilder()
package = builder.build(
    image_uri="file:///data/sample.tif",
    mission=mission,
    objects=objects,
    scene=None,   # auto-derived from GeoTIFF transform
)

print(package["status"])          # READY_FOR_NLG
print(package["statistics"])      # computed from objects[], no LLM
```

---

## Module Contents

| File | Description |
|---|---|
| `builder.py` | `EvidenceBuilder.build()` — main entry point |
| `schema_validator.py` | `validate_evidence_package()` — structural checks |
| `__init__.py` | Exports `EvidenceBuilder` |

---

## Design Decisions

- **Statistics are always computed programmatically** from `objects[]`.
  No LLM or hardcoded values are ever used for counts, class names, or
  confidence summaries.
- **Geo-reference fallback**: if the GeoTIFF has no CRS, each object
  receives `status = "PARTIAL_SUCCESS"` and `geometry.geo` is omitted.
  The overall package `status` is still set to `READY_FOR_NLG` to allow
  downstream modules to proceed with reduced fidelity.
- **Validation**: `validate_evidence_package()` raises
  `SchemaValidationError` (a subclass of `ValueError`) with a full list
  of field-level errors.

---

## Dependencies

| Library | Role | Required? |
|---|---|---|
| `rasterio` | GeoTIFF read, affine transform | Strongly recommended |
| `osgeo.gdal` | Fallback geo-reference | Optional |
| stdlib only | sha256, statistics math | Always available |

Install the recommended library:

```bash
pip install rasterio
```
