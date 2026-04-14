"""
geolocalize.py — M3 Pixel-to-WGS84 Geolocalisation

Reads the affine transform from a GeoTIFF and converts per-object pixel
coordinates to WGS-84 (lon, lat).  Also computes estimated physical dimensions
from pixel size × GSD (ground sample distance).

Dependency priority:
  1. rasterio  (preferred)
  2. GDAL / osgeo.gdal  (fallback)
  3. Graceful degradation — sets object status to PARTIAL_SUCCESS when
     neither library is available or when the image lacks a CRS.

Plain JPEG/PNG fallback:
  When rasterio/GDAL are absent *or* the file is a plain image with no
  embedded CRS, ``GeoLocalizer`` sets ``has_geo = False``, logs a warning,
  and allows the pipeline to continue.  ``localize_objects()`` marks each
  object with ``status = "PARTIAL_SUCCESS"`` and leaves ``geometry.geo``
  empty rather than raising an exception.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_RASTERIO_AVAILABLE = False
_GDAL_AVAILABLE = False
_PIL_AVAILABLE = False

try:
    import rasterio  # type: ignore
    from rasterio.transform import AffineTransformer  # type: ignore
    _RASTERIO_AVAILABLE = True
except ImportError:
    logger.warning(
        "rasterio not found.  Pixel→geo transform will be unavailable. "
        "Install with: pip install rasterio"
    )

if not _RASTERIO_AVAILABLE:
    try:
        from osgeo import gdal  # type: ignore
        _GDAL_AVAILABLE = True
    except ImportError:
        logger.warning(
            "Neither rasterio nor GDAL is available. "
            "Geolocalisation cannot be performed."
        )

try:
    from PIL import Image as _PILImage  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:
    pass

# Low-confidence threshold — objects below this confidence are flagged for review.
_LOW_CONFIDENCE_THRESHOLD = 0.5


class GeoLocalizer:
    """M3 — Convert pixel coordinates to WGS-84 and fill geometry.geo.

    Parameters
    ----------
    image_uri:
        URI (``file://...``) or plain path of the source GeoTIFF.

    Attributes
    ----------
    has_geo : bool
        True when a valid affine transform + CRS were loaded successfully.
    gsd_x_m : float | None
        Ground sample distance in the X direction (metres per pixel).
    gsd_y_m : float | None
        Ground sample distance in the Y direction (metres per pixel).
    """

    def __init__(self, image_uri: str) -> None:
        self.image_uri = image_uri
        self._local_path = self._uri_to_path(image_uri)

        # Affine transform coefficients (GDAL convention: gt[0..5])
        # gt[0] = top-left X, gt[1] = pixel width, gt[2] = row rotation
        # gt[3] = top-left Y, gt[4] = col rotation, gt[5] = pixel height (neg)
        self._gt: Optional[tuple[float, ...]] = None
        self.has_geo: bool = False
        self.gsd_x_m: Optional[float] = None
        self.gsd_y_m: Optional[float] = None
        self.crs: Optional[str] = None

        self._load_transform()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pixel_to_lonlat(self, px: float, py: float) -> tuple[float, float]:
        """Convert pixel centre (col, row) to (lon, lat) in WGS-84.

        Parameters
        ----------
        px : column index (x), 0-based
        py : row index (y), 0-based

        Returns
        -------
        (lon, lat) in decimal degrees.

        Raises
        ------
        RuntimeError
            If no valid geo-reference was loaded.
        """
        if not self.has_geo or self._gt is None:
            raise RuntimeError(
                "No geo-reference available for this image. "
                "Ensure the GeoTIFF contains a valid CRS and affine transform."
            )
        gt = self._gt
        lon = gt[0] + px * gt[1] + py * gt[2]
        lat = gt[3] + px * gt[4] + py * gt[5]
        return round(lon, 7), round(lat, 7)

    def localize_objects(self, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fill ``geometry.geo`` for every object in *objects*.

        Also populates:
        - ``attributes.estimated_length_m``
        - ``attributes.estimated_width_m``

        Objects are **not** mutated in place; a deep copy is returned.

        If geo-reference is unavailable, each object receives
        ``status = "PARTIAL_SUCCESS"`` and ``geometry.geo`` is left absent.

        Parameters
        ----------
        objects:
            List of object dicts from the detector (geometry.pixel filled).

        Returns
        -------
        List of updated object dicts.
        """
        result = copy.deepcopy(objects)

        if not self.has_geo:
            logger.warning(
                "No geo-reference loaded.  Setting status=PARTIAL_SUCCESS "
                "on all objects and skipping geometry.geo fill."
            )
            for obj in result:
                obj.setdefault("geometry", {})["geo"] = {}
                obj["status"] = "PARTIAL_SUCCESS"
            return result

        for obj in result:
            pixel = obj.get("geometry", {}).get("pixel", {})
            if not pixel:
                logger.warning(
                    "Object %s has no geometry.pixel block; skipping.",
                    obj.get("object_id", "?"),
                )
                obj.setdefault("status", "PARTIAL_SUCCESS")
                continue

            try:
                cx = float(pixel["center_x"])
                cy = float(pixel["center_y"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Object %s: invalid pixel centre — %s",
                    obj.get("object_id", "?"), exc,
                )
                obj.setdefault("status", "PARTIAL_SUCCESS")
                continue

            # Centre coordinate
            lon, lat = self.pixel_to_lonlat(cx, cy)

            # Oriented bounding box polygon (4 corners)
            polygon = self._obb_to_polygon_wgs84(pixel)

            obj.setdefault("geometry", {})["geo"] = {
                "center_lon": lon,
                "center_lat": lat,
                "polygon_wgs84": polygon,
            }

            # Physical dimensions
            attrs = obj.setdefault("attributes", {})
            if self.gsd_x_m is not None and self.gsd_y_m is not None:
                w_px = float(pixel.get("width", 0) or 0)
                h_px = float(pixel.get("height", 0) or 0)
                gsd = (self.gsd_x_m + self.gsd_y_m) / 2
                attrs["estimated_length_m"] = round(max(w_px, h_px) * gsd, 1)
                attrs["estimated_width_m"] = round(min(w_px, h_px) * gsd, 1)

            obj.setdefault("status", "VALID")

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_transform(self) -> None:
        """Load the affine transform from the GeoTIFF into ``self._gt``.

        Falls back gracefully when rasterio/GDAL are absent or the file is a
        plain JPEG/PNG without embedded geo-reference information.  In that
        case ``self.has_geo`` remains ``False`` and no exception is raised so
        the pipeline can continue in a degraded (PARTIAL_SUCCESS) mode.
        """
        if _RASTERIO_AVAILABLE:
            self._load_with_rasterio()
        elif _GDAL_AVAILABLE:
            self._load_with_gdal()
        else:
            # Neither geospatial library is present.  Attempt a lightweight
            # PIL/Pillow read just to validate the file is readable; geo
            # coordinates cannot be derived.
            self._load_plain_image_fallback()

        if not self.has_geo:
            logger.warning(
                "rasterio not available or no geo reference found, "
                "geo coordinates will not be filled"
            )

    def _load_plain_image_fallback(self) -> None:
        """Attempt a PIL read to confirm the file is accessible.

        Used when neither rasterio nor GDAL is installed.  No geo-reference
        can be extracted from a plain JPEG/PNG, so ``self.has_geo`` stays
        ``False``.  This method never raises — it only logs.
        """
        if _PIL_AVAILABLE:
            try:
                with _PILImage.open(self._local_path) as img:
                    logger.debug(
                        "PIL opened %s (%s, %dx%d) — no geo-reference available.",
                        self._local_path,
                        img.format,
                        img.width,
                        img.height,
                    )
            except Exception as exc:
                logger.warning(
                    "PIL could not open %s: %s", self._local_path, exc
                )
        else:
            logger.debug(
                "PIL not available; skipping plain-image fallback for %s.",
                self._local_path,
            )
        # has_geo intentionally stays False — no affine transform available.

    def _load_with_rasterio(self) -> None:
        try:
            with rasterio.open(self._local_path) as ds:
                if ds.crs is None:
                    logger.warning(
                        "Image %s has no CRS; geo-reference disabled.",
                        self._local_path,
                    )
                    return

                t = ds.transform
                # Convert rasterio Affine to GDAL 6-tuple convention
                # Affine: (a=col_scale, b=col_shear, c=x_origin,
                #          d=row_shear, e=row_scale, f=y_origin)
                self._gt = (t.c, t.a, t.b, t.f, t.d, t.e)
                self.gsd_x_m = abs(t.a)
                self.gsd_y_m = abs(t.e)

                try:
                    epsg = ds.crs.to_epsg()
                    self.crs = f"EPSG:{epsg}" if epsg else ds.crs.to_string()
                except Exception:
                    self.crs = str(ds.crs)

                self.has_geo = True
                logger.debug(
                    "Loaded transform from %s via rasterio. CRS=%s GSD=(%.4f, %.4f)",
                    self._local_path, self.crs, self.gsd_x_m, self.gsd_y_m,
                )
        except Exception as exc:
            logger.error("rasterio failed for %s: %s", self._local_path, exc)

    def _load_with_gdal(self) -> None:
        try:
            from osgeo import gdal, osr  # type: ignore

            ds = gdal.Open(self._local_path, gdal.GA_ReadOnly)
            if ds is None:
                logger.error("GDAL could not open %s", self._local_path)
                return

            wkt = ds.GetProjection()
            if not wkt:
                logger.warning(
                    "Image %s has no projection; geo-reference disabled.",
                    self._local_path,
                )
                ds = None
                return

            gt = ds.GetGeoTransform()
            if gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                logger.warning(
                    "Image %s has default/identity geo-transform; treating as no CRS.",
                    self._local_path,
                )
                ds = None
                return

            self._gt = tuple(gt)
            self.gsd_x_m = abs(gt[1])
            self.gsd_y_m = abs(gt[5])

            srs = osr.SpatialReference()
            srs.ImportFromWkt(wkt)
            epsg = srs.GetAttrValue("AUTHORITY", 1)
            self.crs = f"EPSG:{epsg}" if epsg else wkt

            self.has_geo = True
            ds = None
            logger.debug(
                "Loaded transform from %s via GDAL. CRS=%s GSD=(%.4f, %.4f)",
                self._local_path, self.crs, self.gsd_x_m, self.gsd_y_m,
            )
        except Exception as exc:
            logger.error("GDAL failed for %s: %s", self._local_path, exc)

    def _obb_to_polygon_wgs84(
        self, pixel: dict[str, Any]
    ) -> list[list[float]]:
        """Compute the 4-corner WGS-84 polygon for an oriented bounding box.

        The OBB is defined by (center_x, center_y, width, height, angle_deg).
        If any required field is missing the polygon is returned as an empty list.
        """
        try:
            cx = float(pixel["center_x"])
            cy = float(pixel["center_y"])
            w = float(pixel.get("width", 0) or 0)
            h = float(pixel.get("height", 0) or 0)
            angle_deg = float(pixel.get("angle_deg", 0) or 0)
        except (KeyError, TypeError, ValueError):
            return []

        if w == 0 or h == 0:
            return []

        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        hw, hh = w / 2.0, h / 2.0

        # Four corners in local frame (before rotation)
        corners_local = [
            (-hw, -hh),
            ( hw, -hh),
            ( hw,  hh),
            (-hw,  hh),
        ]

        polygon: list[list[float]] = []
        for dx, dy in corners_local:
            px = cx + dx * cos_a - dy * sin_a
            py = cy + dx * sin_a + dy * cos_a
            try:
                lon, lat = self.pixel_to_lonlat(px, py)
                polygon.append([lon, lat])
            except RuntimeError:
                return []

        return polygon

    @staticmethod
    def _uri_to_path(image_uri: str) -> str:
        parsed = urlparse(image_uri)
        if parsed.scheme in ("file", ""):
            return parsed.path if parsed.path else image_uri
        return image_uri
