"""Oracle-driven IO contract tests for cleo/class_helpers.py build_netcdf.

These tests assert the INTENDED behavior:
1. build_netcdf writes to the correct path (data/processed/...)
2. open_dataset leaves self.data usable (not backed by closed file handle)
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path


class MockParent:
    """Minimal mock of the Atlas parent object."""

    def __init__(self, path, country, region, crs):
        self.path = path
        self.country = country
        self.region = region
        self.crs = crs


class MockSelf:
    """Minimal mock of the WindAtlas/LandscapeAtlas self object."""

    def __init__(self, parent):
        self.parent = parent
        self.data = None


def create_minimal_geotiff(path, width=4, height=4):
    """Create a minimal valid GeoTIFF with EPSG:4326 CRS."""
    path.parent.mkdir(parents=True, exist_ok=True)

    transform = from_bounds(9.5, 46.5, 17.0, 49.0, width, height)
    data = np.ones((height, width), dtype=np.float32)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_build_netcdf_writes_to_correct_path(tmp_path, monkeypatch):
    """
    Test A: build_netcdf must write to data/processed/WindAtlas_AUT.nc,
    NOT to a nested path under data/raw/AUT/.

    Oracle:
    - Path("data/processed/WindAtlas_AUT.nc").is_file() is True
    - Path("data/raw/AUT/data/processed/WindAtlas_AUT.nc").exists() is False
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Create the minimal raw input raster
        raw_path = Path("data/raw/AUT")
        tif_path = raw_path / "AUT_combined-Weibull-A_100.tif"
        create_minimal_geotiff(tif_path)

        # Create ONLY the correct processed directory
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        # Monkeypatch ensure_crs_from_gwa to avoid network calls
        import cleo.loaders
        monkeypatch.setattr(cleo.loaders, "ensure_crs_from_gwa", lambda ds, iso3: ds)

        from rasterio.crs import CRS
        parent = MockParent(
            path=Path("."),
            country="AUT",
            region=None,
            crs=CRS.from_epsg(4326),
        )

        dummy = MockSelf(parent)

        from cleo.class_helpers import build_netcdf
        build_netcdf(dummy, "WindAtlas")

        # Oracle assertions
        correct_path = Path("data/processed/WindAtlas_AUT.nc")
        wrong_path = Path("data/raw/AUT/data/processed/WindAtlas_AUT.nc")

        assert correct_path.is_file(), (
            f"NetCDF not written to correct path: {correct_path}"
        )
        assert not wrong_path.exists(), (
            f"NetCDF incorrectly written to nested path: {wrong_path}"
        )

    finally:
        os.chdir(original_cwd)


def test_open_dataset_data_accessible_after_build(tmp_path, monkeypatch):
    """
    Test B: After build_netcdf opens an existing dataset, self.data must be
    usable - not backed by a closed file handle.

    Oracle: dummy.data[var].values must succeed without closed-file errors.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Create raw directory with input tif (needed for cleo imports)
        raw_path = Path("data/raw/AUT")
        tif_path = raw_path / "AUT_combined-Weibull-A_100.tif"
        create_minimal_geotiff(tif_path)

        # Create processed directory
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        # Monkeypatch ensure_crs_from_gwa to avoid network calls
        import cleo.loaders
        monkeypatch.setattr(cleo.loaders, "ensure_crs_from_gwa", lambda ds, iso3: ds)

        from rasterio.crs import CRS
        import xarray as xr

        # Create a minimal NetCDF at the correct location
        correct_nc_path = Path("data/processed/WindAtlas_AUT.nc")

        y_coords = np.linspace(46.5, 49.0, 4)
        x_coords = np.linspace(9.5, 17.0, 4)
        ds = xr.Dataset(
            {"test_var": (["y", "x"], np.ones((4, 4), dtype=np.float32))},
            coords={"y": y_coords, "x": x_coords},
        )
        ds = ds.rio.write_crs("EPSG:4326")
        ds.to_netcdf(correct_nc_path)
        ds.close()

        parent = MockParent(
            path=Path("."),
            country="AUT",
            region=None,
            crs=CRS.from_epsg(4326),
        )

        dummy = MockSelf(parent)

        from cleo.class_helpers import build_netcdf
        build_netcdf(dummy, "WindAtlas")

        # Oracle: data must be accessible
        assert dummy.data is not None, "self.data should be set"

        data_vars = list(dummy.data.data_vars)
        assert len(data_vars) > 0, "Dataset should have data variables"

        var = data_vars[0]
        # Force load - must not raise closed-file error
        _ = dummy.data[var].values

    finally:
        os.chdir(original_cwd)
