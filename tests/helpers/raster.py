# tests/helpers/raster.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


__all__ = [
    "write_geotiff",
    "read_geotiff",
    "transform_from_bounds",
    "transform_from_origin",
    "tiny_geotiff",
]


def _require_rasterio():
    """
    Local/lazy import so that importing tests.helpers.* never hard-fails
    when optional raster deps are missing. Tests should skip via optional.py.
    """
    try:
        import rasterio  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "tests.helpers.raster requires optional dependency 'rasterio'. "
            "In tests, use pytest.importorskip('rasterio') or tests.helpers.optional.requires_rasterio()."
        ) from e
    return rasterio


def transform_from_bounds(bounds: tuple[float, float, float, float], *, width: int, height: int):
    """
    Convenience wrapper around rasterio.transform.from_bounds.
    bounds = (west, south, east, north)
    """
    rasterio = _require_rasterio()
    west, south, east, north = bounds
    return rasterio.transform.from_bounds(west, south, east, north, width=width, height=height)


def transform_from_origin(*, west: float, north: float, xsize: float, ysize: float):
    """
    Convenience wrapper around rasterio.transform.from_origin.
    """
    rasterio = _require_rasterio()
    return rasterio.transform.from_origin(west, north, xsize, ysize)


def write_geotiff(
    path: Path,
    data: np.ndarray,
    *,
    crs: str,
    transform,
    nodata: Any = None,
) -> None:
    """
    Write a single-band GeoTIFF to disk.

    Drop-in compatible with the previous helper (same name + signature),
    but imports rasterio lazily to avoid import-time failures.
    """
    rasterio = _require_rasterio()

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"write_geotiff expects 2D array (height,width). Got shape {data.shape}.")

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": int(data.shape[0]),
        "width": int(data.shape[1]),
        "count": 1,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def read_geotiff(path: Path) -> tuple[np.ndarray, Mapping[str, Any]]:
    """
    Read a single-band GeoTIFF from disk; returns (data, profile).
    """
    rasterio = _require_rasterio()

    with rasterio.open(path, "r") as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def tiny_geotiff(
    tmp_path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:4326",
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    nodata: Any = None,
    filename: str = "test.tif",
) -> Path:
    """
    One-liner GeoTIFF creator for tests.

    - Computes a transform from bounds matching the provided data shape.
    - Writes a single-band GeoTIFF.
    - Returns the file path.
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"tiny_geotiff expects 2D array (height,width). Got shape {arr.shape}.")
    height, width = int(arr.shape[0]), int(arr.shape[1])

    transform = transform_from_bounds(bounds, width=width, height=height)
    path = Path(tmp_path) / filename
    write_geotiff(path, arr, crs=crs, transform=transform, nodata=nodata)
    return path
