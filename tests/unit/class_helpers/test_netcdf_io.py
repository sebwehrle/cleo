"""class_helpers: test_netcdf_io.

Contracts:
- build_netcdf writes to data/processed/{AtlasName}_{ISO3}.nc (no nested raw path).
- If an existing NetCDF CRS conflicts with parent.crs, build_netcdf must reproject
  (not merely overwrite CRS metadata).
- After build_netcdf opens/loads a dataset, self.data must be usable (no closed-handle issues).
"""

from __future__ import annotations

from tests.helpers.optional import requires_rioxarray, requires_rasterio

requires_rioxarray()
requires_rasterio()

import os
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pytest
import xarray as xr
import rasterio
import rioxarray  # noqa: F401  (ensures .rio accessor is registered)
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from cleo.class_helpers import build_netcdf


@dataclass
class Parent:
    path: Path
    country: str
    region: str | None
    crs: object  # accepts "EPSG:3035", int, or rasterio CRS


@dataclass
class AtlasSelf:
    parent: Parent
    data: xr.Dataset | None = None

    def _set_var(self, name, da):
        """Simple mock for _set_var - just assigns directly."""
        if self.data is None:
            raise RuntimeError("self.data is None")
        self.data[name] = da


def _write_minimal_geotiff_epsg4326(path: Path, *, width: int = 4, height: int = 4) -> None:
    """Create a minimal valid GeoTIFF with EPSG:4326 CRS and sane transform."""
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

@contextmanager
def _chdir(path: Path):
    """Tiny context manager: keep tests isolated from CWD side-effects."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def test_build_netcdf_reprojects_when_existing_crs_conflicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Existing dataset is EPSG:4326 with degree-like coords, but parent.crs is EPSG:3035.
    Contract: build_netcdf must reproject (coords become meter-like), not just overwrite CRS.
    """
    parent = Parent(path=tmp_path, country="AUT", region=None, crs="EPSG:3035")
    atlas = AtlasSelf(parent)

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    fname = processed / "WindAtlas_AUT.nc"

    # Existing dataset in EPSG:4326 with degree-like coordinates
    ds = xr.Dataset(
        {"a": (("y", "x"), np.ones((2, 2), dtype=np.float32))},
        coords={"x": [10.0, 11.0], "y": [45.0, 46.0]},
    ).rio.write_crs("EPSG:4326")
    ds.to_netcdf(fname)
    ds.close()

    # Avoid any external calls in build_netcdf
    import cleo.loaders

    monkeypatch.setattr(cleo.loaders, "ensure_crs_from_gwa", lambda dset, iso3: dset)

    build_netcdf(atlas, "WindAtlas")
    assert atlas.data is not None

    assert str(atlas.data.rio.crs) == str(CRS.from_user_input(parent.crs))

    # If CRS was merely overwritten without reprojection, x would still be ~10..11 (degrees).
    # After reprojection to EPSG:3035, x should be O(1e6) meters.
    assert float(np.max(np.abs(atlas.data["x"].values))) > 1_000.0


def test_build_netcdf_writes_to_correct_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    build_netcdf must write to data/processed/WindAtlas_AUT.nc, NOT to a nested path
    under data/raw/AUT/data/processed/.
    """
    import cleo.loaders

    monkeypatch.setattr(cleo.loaders, "ensure_crs_from_gwa", lambda dset, iso3: dset)

    with _chdir(tmp_path):
        # minimal raw input raster (what build_netcdf expects to exist)
        tif_path = Path("data/raw/AUT/AUT_combined-Weibull-A_100.tif")
        _write_minimal_geotiff_epsg4326(tif_path)

        Path("data/processed").mkdir(parents=True, exist_ok=True)

        parent = Parent(path=Path("."), country="AUT", region=None, crs=CRS.from_epsg(4326))
        atlas = AtlasSelf(parent)

        build_netcdf(atlas, "WindAtlas")

        correct_path = Path("data/processed/WindAtlas_AUT.nc")
        wrong_path = Path("data/raw/AUT/data/processed/WindAtlas_AUT.nc")

        assert correct_path.is_file(), f"NetCDF not written to correct path: {correct_path}"
        assert not wrong_path.exists(), f"NetCDF incorrectly written to nested path: {wrong_path}"


def test_open_dataset_data_accessible_after_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If build_netcdf opens an existing dataset, self.data must remain usable.
    Oracle: accessing dummy.data[var].values must not raise closed-file errors.
    """
    import cleo.loaders

    monkeypatch.setattr(cleo.loaders, "ensure_crs_from_gwa", lambda dset, iso3: dset)

    with _chdir(tmp_path):
        # raw tif needed for build_netcdf's raw-file existence checks
        tif_path = Path("data/raw/AUT/AUT_combined-Weibull-A_100.tif")
        _write_minimal_geotiff_epsg4326(tif_path)

        Path("data/processed").mkdir(parents=True, exist_ok=True)

        # Create a minimal NetCDF at the expected location
        correct_nc_path = Path("data/processed/WindAtlas_AUT.nc")
        y_coords = np.linspace(46.5, 49.0, 4)
        x_coords = np.linspace(9.5, 17.0, 4)
        ds = xr.Dataset(
            {"test_var": (("y", "x"), np.ones((4, 4), dtype=np.float32))},
            coords={"y": y_coords, "x": x_coords},
        ).rio.write_crs("EPSG:4326")
        ds.to_netcdf(correct_nc_path)
        ds.close()

        parent = Parent(path=Path("."), country="AUT", region=None, crs=CRS.from_epsg(4326))
        atlas = AtlasSelf(parent)

        build_netcdf(atlas, "WindAtlas")

        assert atlas.data is not None
        assert atlas.data.data_vars, "Dataset should have at least one data variable"

        var = next(iter(atlas.data.data_vars))
        _ = atlas.data[var].values  # must not raise


def _write_minimal_geotiff_with_band(path: Path, *, width: int = 4, height: int = 4) -> None:
    """Create a minimal valid GeoTIFF with EPSG:4326 CRS that will have a 'band' coord when loaded."""
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


def test_build_netcdf_drops_band_coord(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    build_netcdf must drop any 'band' coord that leaks from rasterio single-band rasters.
    Contract: After build_netcdf completes, 'band' must not be in coords or dims; template must be 2D (y,x).
    """
    import cleo.loaders

    monkeypatch.setattr(cleo.loaders, "ensure_crs_from_gwa", lambda dset, iso3: dset)

    with _chdir(tmp_path):
        # Create a minimal raw input raster (rasterio single-band will produce 'band' coord)
        tif_path = Path("data/raw/AUT/AUT_combined-Weibull-A_100.tif")
        _write_minimal_geotiff_with_band(tif_path)

        Path("data/processed").mkdir(parents=True, exist_ok=True)

        parent = Parent(path=Path("."), country="AUT", region=None, crs=CRS.from_epsg(4326))
        atlas = AtlasSelf(parent)

        build_netcdf(atlas, "WindAtlas")

        assert atlas.data is not None, "build_netcdf did not create a dataset"
        assert "band" not in atlas.data.coords, "'band' coord leaked into Dataset coords"
        assert "band" not in atlas.data.dims, "'band' dim leaked into Dataset dims"
        assert "template" in atlas.data.data_vars, "template missing from dataset"
        assert tuple(atlas.data["template"].dims) == ("y", "x"), f"template dims wrong: {atlas.data['template'].dims}"
