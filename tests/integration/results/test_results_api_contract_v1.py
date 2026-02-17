"""Integration tests for Results API v1.

Tests are offline-only: no network calls, uses local test fixtures.

Tests verify:
- new_run_id prefix validation
- persist creates store and open_result reads it
- persist raises FileExistsError on collision
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS

from cleo.classes import Atlas
from cleo.unify import GWA_HEIGHTS


def _create_gwa_raster(
    path: Path,
    shape: tuple[int, int] = (20, 20),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (4000000, 2600000, 4100000, 2700000),
    fill_value: float | None = None,
    add_nodata_region: bool = True,
) -> None:
    """Create a minimal GWA-style GeoTIFF with CRS."""
    if fill_value is not None:
        data = np.full(shape, fill_value, dtype=np.float32)
    else:
        data = np.random.rand(*shape).astype(np.float32) * 10 + 1

    if add_nodata_region:
        data[:3, :3] = np.nan

    transform = rasterio.transform.from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3], shape[1], shape[0]
    )

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "crs": CRS.from_epsg(crs_epsg),
        "transform": transform,
        "nodata": np.nan,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_elevation_raster(path: Path, **kwargs) -> None:
    """Create elevation raster file."""
    _create_gwa_raster(path, fill_value=500.0, add_nodata_region=False, **kwargs)


def _create_all_gwa_rasters(atlas_path: Path, country: str = "AUT") -> None:
    """Create all required GWA rasters for wind materialization."""
    raw_dir = atlas_path / "data" / "raw" / country

    for h in GWA_HEIGHTS:
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-A_{h}.tif",
            fill_value=8.0 + h * 0.01,
        )
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-k_{h}.tif",
            fill_value=2.0 + h * 0.001,
        )
        _create_gwa_raster(
            raw_dir / f"{country}_air-density_{h}.tif",
            fill_value=1.2 - h * 0.0001,
        )


@pytest.fixture
def atlas(tmp_path: Path) -> Atlas:
    """Create and materialize an Atlas for testing."""
    _create_all_gwa_rasters(tmp_path)
    elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
    _create_elevation_raster(elev_path)

    atlas = Atlas(tmp_path, "AUT", "epsg:3035")
    atlas.materialize_canonical()
    return atlas


class TestNewRunIdPrefixValidation:
    """Tests for new_run_id prefix validation."""

    def test_new_run_id_valid_prefix(self, atlas: Atlas) -> None:
        """Valid prefix characters are accepted."""
        run_id = atlas.new_run_id("abc_01-XY")
        assert run_id.startswith("abc_01-XY-")
        # Should have format: prefix-YYYYMMDDTHHMMSSz-8chars
        parts = run_id.split("-")
        assert len(parts) >= 3
        assert parts[0] == "abc_01"
        assert parts[1] == "XY"

    def test_new_run_id_no_prefix(self, atlas: Atlas) -> None:
        """No prefix generates valid run_id."""
        run_id = atlas.new_run_id()
        # Should have format: YYYYMMDDTHHMMSSz-8chars
        assert "T" in run_id
        assert "Z" in run_id
        assert len(run_id) >= 20  # Timestamp + uuid

    def test_new_run_id_invalid_prefix_slash(self, atlas: Atlas) -> None:
        """Prefix with slash raises ValueError."""
        with pytest.raises(ValueError, match="must match"):
            atlas.new_run_id("a/b")

    def test_new_run_id_invalid_prefix_colon(self, atlas: Atlas) -> None:
        """Prefix with colon raises ValueError."""
        with pytest.raises(ValueError, match="must match"):
            atlas.new_run_id("a:b")

    def test_new_run_id_invalid_prefix_space(self, atlas: Atlas) -> None:
        """Prefix with space raises ValueError."""
        with pytest.raises(ValueError, match="must match"):
            atlas.new_run_id("a b")


class TestPersistAndOpenResult:
    """Tests for persist and open_result methods."""

    def test_persist_and_open_result(self, atlas: Atlas) -> None:
        """persist creates store and open_result reads it."""
        run = atlas.new_run_id("t")
        ds_in = xr.Dataset({"a": ("x", [1, 2, 3])})

        # Persist
        store_path = atlas.persist("m", ds_in, run_id=run, params={"p": 1})

        assert store_path.exists()
        assert store_path.name == "m.zarr"

        # Open result
        ds = atlas.open_result(run, "m")
        assert "a" in ds
        np.testing.assert_array_equal(ds["a"].values, [1, 2, 3])

        # Verify attrs
        assert ds.attrs["store_state"] == "complete"
        assert ds.attrs["run_id"] == run
        assert ds.attrs["metric_name"] == "m"
        assert "created_at" in ds.attrs
        assert ds.attrs["params_json"] == '{"p":1}'

    def test_persist_dataarray(self, atlas: Atlas) -> None:
        """persist accepts DataArray and converts to Dataset."""
        run = atlas.new_run_id("da")
        da_in = xr.DataArray([4, 5, 6], dims=["x"], name="vals")

        store_path = atlas.persist("metric", da_in, run_id=run)

        ds = atlas.open_result(run, "metric")
        assert "vals" in ds
        np.testing.assert_array_equal(ds["vals"].values, [4, 5, 6])

    def test_persist_dataarray_unnamed_gets_metric_name(self, atlas: Atlas) -> None:
        """persist assigns metric_name to unnamed DataArray."""
        run = atlas.new_run_id("unnamed")
        da_in = xr.DataArray([7, 8, 9], dims=["x"])  # No name

        atlas.persist("my_metric", da_in, run_id=run)

        ds = atlas.open_result(run, "my_metric")
        assert "my_metric" in ds

    def test_persist_auto_generates_run_id(self, atlas: Atlas) -> None:
        """persist generates run_id if not provided."""
        ds_in = xr.Dataset({"b": ("y", [10, 20])})

        store_path = atlas.persist("auto", ds_in)

        # Extract run_id from path
        run_id = store_path.parent.name
        assert len(run_id) >= 20

        # Should be readable
        ds = atlas.open_result(run_id, "auto")
        assert "b" in ds

    def test_open_result_works(self, atlas: Atlas) -> None:
        """open_result reads result store without consolidated metadata."""
        run = atlas.new_run_id("cons")
        ds_in = xr.Dataset({"c": ("z", [1, 2])})

        atlas.persist("check", ds_in, run_id=run)

        # Should open without error (uses consolidated=False internally)
        ds = atlas.open_result(run, "check")
        assert "c" in ds

    def test_open_result_missing_raises(self, atlas: Atlas) -> None:
        """open_result raises FileNotFoundError for missing store."""
        with pytest.raises(FileNotFoundError, match="not found"):
            atlas.open_result("nonexistent-run", "nonexistent-metric")


class TestPersistCollision:
    """Tests for persist collision detection."""

    def test_persist_raises_on_collision(self, atlas: Atlas) -> None:
        """persist raises FileExistsError on collision."""
        run = atlas.new_run_id("collision")
        ds1 = xr.Dataset({"d": ("w", [1])})
        ds2 = xr.Dataset({"e": ("w", [2])})

        # First persist succeeds
        atlas.persist("metric", ds1, run_id=run)

        # Second persist with same run/metric raises
        with pytest.raises(FileExistsError, match="already exists"):
            atlas.persist("metric", ds2, run_id=run)

    def test_persist_different_metric_same_run_ok(self, atlas: Atlas) -> None:
        """persist allows different metrics in same run."""
        run = atlas.new_run_id("multi")
        ds1 = xr.Dataset({"f": ("v", [1])})
        ds2 = xr.Dataset({"g": ("v", [2])})

        atlas.persist("metric1", ds1, run_id=run)
        atlas.persist("metric2", ds2, run_id=run)

        # Both readable
        assert "f" in atlas.open_result(run, "metric1")
        assert "g" in atlas.open_result(run, "metric2")


class TestExportResultNetcdfStringCoords:
    """Regression tests for NetCDF export with string/object dtype coords."""

    def test_export_result_netcdf_handles_string_coords_roundtrip(
        self, tmp_path: Path
    ) -> None:
        """NetCDF export handles string turbine coordinates correctly.

        Regression test: export_result_netcdf must encode object/string dtype
        coordinates as fixed-length byte strings for NetCDF4 compatibility.
        """
        # Setup: minimal Atlas (no materialize needed for persist/export)
        atlas = Atlas(tmp_path, "AUT", "epsg:3035")

        # Build dataset with string turbine coordinate
        ds = xr.Dataset(
            {
                "capacity_factors": xr.DataArray(
                    np.ones((2, 2, 2), dtype=float),
                    dims=("turbine", "y", "x"),
                    coords={"turbine": ["turbA", "turbB"]},
                )
            }
        )

        # Persist
        run_id = atlas.new_run_id("strcoord")
        atlas.persist("capacity_factors", ds, run_id=run_id)

        # Act: export to NetCDF
        out = atlas.export_result_netcdf(run_id, "capacity_factors", tmp_path / "cf.nc")

        # Assert: file exists and is non-empty
        assert out.exists()
        assert out.stat().st_size > 0

        # Assert: roundtrip succeeds
        ds_nc = xr.open_dataset(out)
        assert "capacity_factors" in ds_nc

        # Assert: turbine coord values match (handle bytes decoding)
        vals = ds_nc["turbine"].values
        if vals.dtype.kind == "S":
            vals = [v.decode("utf-8") for v in vals]
        assert list(vals) == ["turbA", "turbB"]

        ds_nc.close()
