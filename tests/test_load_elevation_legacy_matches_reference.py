"""Oracle test: legacy elevation must be reproject-matched to reference grid."""

import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr
import rioxarray as rxr

import pytest

from cleo.loaders import load_elevation


def test_load_elevation_legacy_matches_reference_grid(tmp_path, monkeypatch):
    iso3 = "AUT"
    raw_dir = tmp_path / "data" / "raw" / iso3
    raw_dir.mkdir(parents=True, exist_ok=True)

    legacy = raw_dir / f"{iso3}_elevation_w_bathymetry.tif"

    # Legacy raster: 2x2, pixel size 1
    transform = from_origin(0, 2, 1, 1)
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": None,
    }
    arr = np.ones((1, 2, 2), dtype="float32")
    with rasterio.open(legacy, "w", **profile) as dst:
        dst.write(arr)

    # Reference grid: 4x4, pixel size 0.5, same CRS
    ref = xr.DataArray(
        np.ones((4, 4), dtype="float32"),
        dims=("y", "x"),
        coords={"x": [0.25, 0.75, 1.25, 1.75], "y": [1.75, 1.25, 0.75, 0.25]},
    ).rio.write_crs("EPSG:4326")

    # If CopDEM branch is called, fail the test
    def boom(*args, **kwargs):
        raise AssertionError("CopDEM download should not be called when legacy exists")

    monkeypatch.setattr("cleo.copdem.download_copdem_tiles_for_bbox", boom)

    out = load_elevation(tmp_path, iso3, ref)

    assert out.shape == ref.shape
    assert np.array_equal(out["x"].values, ref["x"].values)
    assert np.array_equal(out["y"].values, ref["y"].values)
    assert out.rio.crs == ref.rio.crs
