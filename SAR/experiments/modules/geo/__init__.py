"""
modules/geo — M1 preprocessing and M3 geolocalisation.

Public API
----------
ImagePreprocessor  : parse GeoTIFF metadata → Evidence Package `input` block
GeoLocalizer       : pixel coordinates → WGS-84 lon/lat, fills objects[].geometry.geo
"""

from .preprocess import ImagePreprocessor
from .geolocalize import GeoLocalizer

__all__ = ["ImagePreprocessor", "GeoLocalizer"]
