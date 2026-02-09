"""spatial: test_geotiff.
Merged test file (imports preserved per chunk).
"""

from tests.helpers.optional import requires_rioxarray, requires_rasterio

requires_rioxarray()
requires_rasterio()

import pytest
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
import rasterio
from pathlib import Path
from rasterio.transform import Affine
from cleo.spatial import save_to_geotiff

# --- merged from tests/_staging/test_save_to_geotiff_contract.py ---

def test_save_to_geotiff_3d_raises_typeerror(tmp_path: Path):
    """3D array must be rejected with clear TypeError."""
    da = xr.DataArray(
        np.ones((2, 3, 3)),
        dims=("band", "y", "x"),
        coords={"band": [0, 1], "x": [0, 1, 2], "y": [0, 1, 2]},
    )
    with pytest.raises(TypeError, match="expects exactly dims"):
        save_to_geotiff(da, "EPSG:4326", tmp_path, "test.tif")


def test_save_to_geotiff_2d_no_nans_succeeds(tmp_path: Path):
    """2D array without NaNs should succeed."""
    da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    da = da.rio.write_crs("EPSG:4326")
    save_to_geotiff(da, "EPSG:4326", tmp_path, "no_nans.tif")
    assert (tmp_path / "no_nans.tif").is_file()


def test_save_to_geotiff_writes_crs(tmp_path: Path):
    """Saved GeoTIFF must have CRS metadata (not None)."""
    da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=("y", "x"),
        coords={"x": [100.0, 200.0], "y": [50.0, 60.0]},
    )
    da = da.rio.write_crs("EPSG:3035")
    save_to_geotiff(da, "EPSG:3035", tmp_path, "with_crs.tif")

    with rasterio.open(tmp_path / "with_crs.tif") as src:
        assert src.crs is not None, "CRS should be written to GeoTIFF"
        assert src.crs.to_epsg() == 3035, "CRS should match EPSG:3035"


def test_save_to_geotiff_writes_transform(tmp_path: Path):
    """Saved GeoTIFF must have a valid transform (not identity or None)."""
    # Create data with explicit coordinates that produce non-identity transform
    x_coords = [100.0, 200.0, 300.0]
    y_coords = [500.0, 400.0]  # y typically decreasing (north-up)

    da = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        dims=("y", "x"),
        coords={"x": x_coords, "y": y_coords},
    )
    da = da.rio.write_crs("EPSG:3035")
    save_to_geotiff(da, "EPSG:3035", tmp_path, "with_transform.tif")

    with rasterio.open(tmp_path / "with_transform.tif") as src:
        transform = src.transform
        assert transform is not None, "Transform should not be None"

        # Identity transform has scale=1 and no offset
        identity = Affine.identity()
        assert transform != identity, "Transform should not be identity"

        # Verify transform reflects our coordinate spacing
        # x resolution should be 100 (200-100)
        # y resolution should be -100 (400-500, negative for north-up)
        assert transform.a == 100.0, f"Expected x scale 100, got {transform.a}"
        assert transform.e == -100.0, f"Expected y scale -100, got {transform.e}"


def test_save_to_geotiff_with_nans_writes_nodata(tmp_path: Path):
    """Array with NaNs should write nodata value."""
    da = xr.DataArray(
        np.array([[1.0, np.nan], [3.0, 4.0]]),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    da = da.rio.write_crs("EPSG:4326")
    save_to_geotiff(da, "EPSG:4326", tmp_path, "with_nans.tif", nodata_value=-9999)

    with rasterio.open(tmp_path / "with_nans.tif") as src:
        assert src.nodata == -9999, "Nodata value should be written"
        data = src.read(1)
        assert -9999 in data, "Nodata value should appear in raster"


# --- merged from tests/_staging/test_save_to_geotiff_dim_check.py ---

def test_save_to_geotiff_rejects_non_2d_xy(tmp_path: Path):
    da = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=("band", "y", "x"),
        coords={"band": [0], "x": [0, 1], "y": [0, 1]},
    )
    with pytest.raises(TypeError, match="expects exactly dims"):
        save_to_geotiff(da, "EPSG:4326", tmp_path, "tmp.tif")


# --- merged from tests/_staging/test_save_to_geotiff_no_nans.py ---

def test_save_to_geotiff_all_finite_succeeds(tmp_path: Path):
    da = xr.DataArray(np.ones((2, 2)), dims=("y", "x"), coords={"x": [0, 1], "y": [0, 1]})
    da = da.rio.write_crs("EPSG:4326")
    save_to_geotiff(da, "EPSG:4326", tmp_path, "tmp.tif")
    assert (tmp_path / "tmp.tif").is_file()
