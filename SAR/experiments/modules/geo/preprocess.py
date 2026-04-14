"""
preprocess.py — M1 Image Preprocessing and Metadata Parsing

Reads a GeoTIFF (or plain image file) and returns the Evidence Package
`input` block conforming to schema v1.0.

Dependency priority:
  1. rasterio  (preferred — pure-Python wheel friendly)
  2. GDAL / osgeo.gdal  (fallback)
  3. Graceful degradation when neither is available
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — import lazily so the rest of the codebase
# can still be imported even if geospatial libraries are absent.
# ---------------------------------------------------------------------------

_RASTERIO_AVAILABLE = False
_GDAL_AVAILABLE = False

try:
    import rasterio  # type: ignore
    from rasterio.crs import CRS  # type: ignore
    _RASTERIO_AVAILABLE = True
except ImportError:
    logger.warning(
        "rasterio not found. GeoTIFF geo-reference parsing will be limited. "
        "Install with: pip install rasterio"
    )

if not _RASTERIO_AVAILABLE:
    try:
        from osgeo import gdal  # type: ignore
        _GDAL_AVAILABLE = True
    except ImportError:
        logger.warning(
            "Neither rasterio nor GDAL (osgeo) is available. "
            "Geo-reference metadata will not be extracted. "
            "Install rasterio: pip install rasterio"
        )

_PIL_AVAILABLE = False
try:
    from PIL import Image as _PILImage  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_FORMAT = "GeoTIFF"
_FALLBACK_SATELLITE = "UNKNOWN"
_FALLBACK_SENSOR = "SAR"


class ImagePreprocessor:
    """M1 — Parse image file + mission dict → Evidence Package `input` block.

    Usage
    -----
    >>> ip = ImagePreprocessor()
    >>> input_block = ip.parse("file:///data/sample.tif", mission)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, image_uri: str, mission: dict[str, Any]) -> dict[str, Any]:
        """Return the Evidence Package ``input`` block.

        Parameters
        ----------
        image_uri:
            URI of the source image (``file://`` or plain path).
        mission:
            Dict containing at minimum ``region_name``, ``region_type``,
            ``priority``, and optionally ``user_prompt``.

        Returns
        -------
        dict conforming to the ``input`` block of Evidence Package schema v1.0.
        """
        local_path = self._uri_to_path(image_uri)
        file_name = Path(local_path).name

        sha256 = self._compute_sha256(local_path)

        image_meta: dict[str, Any] = {
            "width": None,
            "height": None,
            "bands": None,
            "bit_depth": None,
            "crs": None,
        }

        if _RASTERIO_AVAILABLE:
            image_meta = self._parse_with_rasterio(local_path)
        elif _GDAL_AVAILABLE:
            image_meta = self._parse_with_gdal(local_path)
        else:
            logger.warning(
                "No geospatial library available; image metadata fields will be "
                "partially filled using PIL (size only). "
                "Install rasterio to enable full metadata extraction."
            )
            image_meta = self._parse_with_pil(local_path)

        now_iso = datetime.now(timezone.utc).isoformat()

        input_block: dict[str, Any] = {
            "input_id": f"input-{uuid.uuid4().hex[:8]}",
            "image": {
                "uri": image_uri,
                "file_name": file_name,
                "format": _SCHEMA_FORMAT,
                "width": image_meta.get("width"),
                "height": image_meta.get("height"),
                "bands": image_meta.get("bands"),
                "bit_depth": image_meta.get("bit_depth"),
                "sha256": sha256,
            },
            "metadata": {
                "satellite": image_meta.get("satellite", _FALLBACK_SATELLITE),
                "sensor": image_meta.get("sensor", _FALLBACK_SENSOR),
                "acquisition_time": image_meta.get("acquisition_time", now_iso),
                "resolution_m": image_meta.get("resolution_m"),
                "polarization": image_meta.get("polarization", "UNKNOWN"),
                "orbit_direction": image_meta.get("orbit_direction", "UNKNOWN"),
                "incidence_angle_deg": image_meta.get("incidence_angle_deg"),
                "crs": image_meta.get("crs"),
            },
            "mission": {
                "region_name": mission.get("region_name", ""),
                "region_type": mission.get("region_type", "unknown"),
                "priority": mission.get("priority", "MEDIUM"),
                "user_prompt": mission.get("user_prompt", ""),
            },
        }

        return input_block

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _uri_to_path(image_uri: str) -> str:
        """Convert a file URI or plain path to a local filesystem path."""
        parsed = urlparse(image_uri)
        if parsed.scheme in ("file", ""):
            return parsed.path if parsed.path else image_uri
        # For non-file URIs return as-is (e.g. object storage paths may be
        # handled by a VFS layer in the calling environment).
        return image_uri

    @staticmethod
    def _compute_sha256(path: str) -> Optional[str]:
        """Return the hex SHA-256 digest of the file at *path*."""
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError as exc:
            logger.warning("sha256 computation failed for %s: %s", path, exc)
            return None

    @staticmethod
    def _parse_with_rasterio(path: str) -> dict[str, Any]:
        """Extract metadata using rasterio."""
        meta: dict[str, Any] = {}
        try:
            with rasterio.open(path) as ds:
                meta["width"] = ds.width
                meta["height"] = ds.height
                meta["bands"] = ds.count

                # Infer bit depth from the first band's dtype
                dtype = ds.dtypes[0] if ds.dtypes else "unknown"
                meta["bit_depth"] = _dtype_to_bits(dtype)

                # CRS
                if ds.crs is not None:
                    try:
                        meta["crs"] = ds.crs.to_epsg()
                        if meta["crs"] is not None:
                            meta["crs"] = f"EPSG:{meta['crs']}"
                        else:
                            meta["crs"] = ds.crs.to_string()
                    except Exception:
                        meta["crs"] = str(ds.crs)
                else:
                    logger.warning(
                        "Image %s has no CRS. Geolocalisation will be skipped.", path
                    )
                    meta["crs"] = None

                # GSD: derive from affine transform pixel size (average x/y)
                transform = ds.transform
                if transform is not None:
                    pixel_x_m = abs(transform.a)
                    pixel_y_m = abs(transform.e)
                    meta["resolution_m"] = round((pixel_x_m + pixel_y_m) / 2, 4)

                # GDAL tags (if embedded, e.g. GF-3 standard products)
                tags = ds.tags()
                meta.update(_extract_tags(tags))

        except Exception as exc:
            logger.error("rasterio failed to open %s: %s", path, exc)

        return meta

    @staticmethod
    def _parse_with_pil(path: str) -> dict[str, Any]:
        """Extract basic image dimensions using PIL/Pillow.

        Used when neither rasterio nor GDAL is installed.  Geo-reference fields
        (``crs``, ``resolution_m``, etc.) will be ``None``; the pipeline
        continues without raising an exception.

        Parameters
        ----------
        path:
            Local filesystem path to the image file (JPEG, PNG, TIFF, …).

        Returns
        -------
        dict with ``width``, ``height``, and ``bands`` populated when PIL can
        open the file; all other geo fields remain ``None``.
        """
        meta: dict[str, Any] = {
            "width": None,
            "height": None,
            "bands": None,
            "bit_depth": None,
            "crs": None,
        }
        if not _PIL_AVAILABLE:
            logger.warning(
                "PIL (Pillow) is not installed; image size cannot be determined "
                "without rasterio/GDAL. Install with: pip install Pillow"
            )
            return meta

        try:
            with _PILImage.open(path) as img:
                meta["width"] = img.width
                meta["height"] = img.height
                # Map PIL mode to approximate band count
                _mode_bands: dict[str, int] = {
                    "1": 1, "L": 1, "P": 1, "RGB": 3,
                    "RGBA": 4, "CMYK": 4, "YCbCr": 3,
                    "LAB": 3, "HSV": 3, "I": 1, "F": 1,
                    "LA": 2, "PA": 2, "RGBa": 4, "La": 2,
                    "I;16": 1,
                }
                meta["bands"] = _mode_bands.get(img.mode, 1)
                logger.debug(
                    "PIL parsed %s: %dx%d mode=%s",
                    path, img.width, img.height, img.mode,
                )
        except Exception as exc:
            logger.warning("PIL failed to open %s: %s", path, exc)

        return meta

    @staticmethod
    def _parse_with_gdal(path: str) -> dict[str, Any]:
        """Extract metadata using GDAL (fallback)."""
        meta: dict[str, Any] = {}
        try:
            from osgeo import gdal, osr  # type: ignore

            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is None:
                logger.error("GDAL could not open %s", path)
                return meta

            meta["width"] = ds.RasterXSize
            meta["height"] = ds.RasterYSize
            meta["bands"] = ds.RasterCount

            if ds.RasterCount > 0:
                band = ds.GetRasterBand(1)
                dtype_name = gdal.GetDataTypeName(band.DataType)
                meta["bit_depth"] = _dtype_to_bits(dtype_name)

            srs = osr.SpatialReference()
            wkt = ds.GetProjection()
            if wkt:
                srs.ImportFromWkt(wkt)
                epsg = srs.GetAttrValue("AUTHORITY", 1)
                meta["crs"] = f"EPSG:{epsg}" if epsg else wkt
            else:
                logger.warning(
                    "Image %s has no CRS. Geolocalisation will be skipped.", path
                )
                meta["crs"] = None

            gt = ds.GetGeoTransform()
            if gt and gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                pixel_x_m = abs(gt[1])
                pixel_y_m = abs(gt[5])
                meta["resolution_m"] = round((pixel_x_m + pixel_y_m) / 2, 4)

            tags = {k: v for k, v in ds.GetMetadata().items()}
            meta.update(_extract_tags(tags))

            ds = None  # close

        except Exception as exc:
            logger.error("GDAL failed to parse %s: %s", path, exc)

        return meta


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _dtype_to_bits(dtype_str: str) -> Optional[int]:
    """Map numpy / GDAL dtype strings to bit-depth integers."""
    mapping = {
        # numpy dtypes
        "uint8": 8, "int8": 8,
        "uint16": 16, "int16": 16,
        "uint32": 32, "int32": 32,
        "float32": 32, "float64": 64,
        # GDAL type names
        "Byte": 8,
        "UInt16": 16, "Int16": 16,
        "UInt32": 32, "Int32": 32,
        "Float32": 32, "Float64": 64,
        "CFloat32": 32, "CFloat64": 64,
    }
    return mapping.get(dtype_str)


def _extract_tags(tags: dict[str, str]) -> dict[str, Any]:
    """Extract known metadata fields from GDAL/rasterio tag dicts.

    Supports common GF-3 / Sentinel-1 / generic tag conventions.
    Returns only non-None values.
    """
    result: dict[str, Any] = {}

    _tag_map: dict[str, list[str]] = {
        "satellite":         ["SATELLITE_ID", "SPACECRAFT_ID", "MISSION", "satellite"],
        "sensor":            ["SENSOR_ID", "INSTRUMENT", "sensor"],
        "acquisition_time":  ["ACQUISITION_DATE_TIME", "START_TIME", "SCENE_CENTER_TIME",
                              "acquisition_time"],
        "polarization":      ["POLARIZATION", "polarization"],
        "orbit_direction":   ["ORBIT_DIRECTION", "PASS", "orbit_direction"],
        "incidence_angle_deg": ["INCIDENCE_ANGLE", "NEAR_INCIDENCE_ANGLE",
                                "incidence_angle_deg"],
    }

    for field, candidates in _tag_map.items():
        for key in candidates:
            val = tags.get(key)
            if val is not None:
                if field == "incidence_angle_deg":
                    try:
                        result[field] = float(val)
                    except ValueError:
                        pass
                else:
                    result[field] = val
                break

    return result
