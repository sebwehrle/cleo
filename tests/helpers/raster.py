from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def write_geotiff(
    path: Path,
    data: np.ndarray,
    *,
    crs: str,
    transform,
    nodata=None,
) -> None:
    """Write a single-band GeoTIFF to disk."""
    data = np.asarray(data)
    profile = {
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

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
