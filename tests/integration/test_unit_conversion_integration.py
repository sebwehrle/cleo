"""Integration tests for unit conversion end-to-end behavior.

Verifies:
- Unit metadata survives compute -> materialize -> persist -> export round-trip
- Domain convert_units() works end-to-end
- Conversion is dask-friendly (no eager loading regression)
"""

import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS

import cleo
from cleo import Atlas
from cleo.units import (
    UNIT_ATTR_KEY,
    convert_dataarray,
    conversion_factor,
)
from cleo.unification.gwa_io import GWA_HEIGHTS


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

    transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], shape[1], shape[0])

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "crs": CRS.from_epsg(crs_epsg),
        "transform": transform,
        "nodata": -9999.0,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_all_required_gwa_files(atlas_path: Path, country: str = "AUT") -> None:
    """Create all required GWA files for testing with correct naming convention."""
    raw_dir = atlas_path / "data" / "raw" / country

    for h in GWA_HEIGHTS:
        for layer, prefix in [
            ("weibull_A", f"{country}_combined-Weibull-A_{h}.tif"),
            ("weibull_k", f"{country}_combined-Weibull-k_{h}.tif"),
            ("rho", f"{country}_air-density_{h}.tif"),
        ]:
            p = raw_dir / prefix
            if layer == "weibull_A":
                _create_gwa_raster(p, fill_value=8.0)
            elif layer == "weibull_k":
                _create_gwa_raster(p, fill_value=2.0)
            else:  # rho
                _create_gwa_raster(p, fill_value=1.225)

    # Elevation
    elev_path = raw_dir / f"{country}_elevation_w_bathymetry.tif"
    _create_gwa_raster(elev_path, fill_value=500.0, add_nodata_region=False)


def _copy_turbine_yaml(atlas_path: Path, turbine_name: str = "Enercon.E40.500") -> None:
    """Copy a turbine YAML from package resources."""
    resources_src = Path(cleo.__file__).resolve().parent / "resources"
    resources_dest = atlas_path / "resources"
    resources_dest.mkdir(parents=True, exist_ok=True)
    src = resources_src / f"{turbine_name}.yml"
    if src.exists():
        shutil.copy(src, resources_dest / f"{turbine_name}.yml")


class TestUnitMetadataSurvivesRoundTrip:
    """Tests for unit metadata preservation through compute/materialize/persist/export."""

    @pytest.fixture
    def materialized_atlas(self, tmp_path: Path) -> Atlas:
        """Create a fully materialized atlas for testing."""
        _create_all_required_gwa_files(tmp_path)
        _copy_turbine_yaml(tmp_path, "Enercon.E40.500")

        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="EPSG:3035",
        )
        atlas.configure_turbines(["Enercon.E40.500"])
        atlas.build()
        atlas.wind.select(turbines=["Enercon.E40.500"])
        return atlas

    def test_capacity_factors_has_units_after_materialize(self, materialized_atlas: Atlas) -> None:
        """Capacity factors have units attr after materialization."""
        atlas = materialized_atlas

        result = atlas.wind.compute("capacity_factors", method="hub_height_weibull")
        result.materialize()

        # Reload from store
        cf = atlas.wind.data["capacity_factors"]
        assert "units" in cf.attrs
        assert cf.attrs["units"] == "1"

    def test_rotor_equivalent_wind_speed_has_units_after_materialize(self, materialized_atlas: Atlas) -> None:
        """REWS has units attr after materialization."""
        atlas = materialized_atlas

        result = atlas.wind.compute("wind_speed", method="rotor_equivalent")
        result.materialize()

        rews = atlas.wind.data["rotor_equivalent_wind_speed"]
        assert "units" in rews.attrs
        assert rews.attrs["units"] == "m/s"

    def test_mean_wind_speed_has_units_after_compute(self, materialized_atlas: Atlas) -> None:
        """Mean wind speed has units attr after compute."""
        atlas = materialized_atlas

        result = atlas.wind.compute("wind_speed", height=100)

        assert "units" in result.data.attrs
        assert result.data.attrs["units"] == "m/s"

    def test_units_survive_persist_and_open(self, materialized_atlas: Atlas) -> None:
        """Unit metadata survives persist and open_result round-trip."""
        atlas = materialized_atlas

        result = atlas.wind.compute("capacity_factors", method="hub_height_weibull")
        store_path = result.persist(run_id="test_units")

        # Open persisted result (returns Dataset, access the variable)
        opened_ds = atlas.open_result(store_path.parent.name, "capacity_factors")
        opened = opened_ds["capacity_factors"]

        assert "units" in opened.attrs
        assert opened.attrs["units"] == "1"

    def test_units_survive_netcdf_export(self, materialized_atlas: Atlas) -> None:
        """Unit metadata survives NetCDF export."""
        atlas = materialized_atlas

        result = atlas.wind.compute("capacity_factors", method="hub_height_weibull")
        store_path = result.persist(run_id="test_export")

        # Export to NetCDF
        nc_path = atlas.path / "test_cf.nc"
        atlas.export_result_netcdf(store_path.parent.name, "capacity_factors", nc_path)

        # Open NetCDF and check units
        with xr.open_dataset(nc_path) as ds:
            assert "units" in ds["capacity_factors"].attrs
            assert ds["capacity_factors"].attrs["units"] == "1"


class TestDomainConvertUnitsEndToEnd:
    """End-to-end tests for domain convert_units() method."""

    @pytest.fixture
    def materialized_atlas(self, tmp_path: Path) -> Atlas:
        """Create a fully materialized atlas for testing."""
        _create_all_required_gwa_files(tmp_path)
        _copy_turbine_yaml(tmp_path, "Enercon.E40.500")

        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="EPSG:3035",
        )
        atlas.configure_turbines(["Enercon.E40.500"])
        atlas.build()
        atlas.wind.select(turbines=["Enercon.E40.500"])
        return atlas

    def test_wind_convert_units_inplace_end_to_end(self, materialized_atlas: Atlas) -> None:
        """Wind domain convert_units with inplace=True works end-to-end."""
        atlas = materialized_atlas

        # Compute mean wind speed
        atlas.wind.compute("wind_speed", method="height_weibull_mean", height=100)

        # Convert in place
        atlas.wind.convert_units("mean_wind_speed", "km/h", inplace=True)

        # Verify conversion is visible
        mws = atlas.wind.data["mean_wind_speed"]
        assert mws.attrs["units"] == "km/h"

    def test_wind_convert_units_returns_converted(self, materialized_atlas: Atlas) -> None:
        """Wind domain convert_units returns converted DataArray."""
        atlas = materialized_atlas

        # Compute mean wind speed
        atlas.wind.compute("wind_speed", method="height_weibull_mean", height=100)

        # Convert without inplace
        result = atlas.wind.convert_units("mean_wind_speed", "km/h")

        assert result is not None
        assert result.attrs["units"] == "km/h"

        # Original should be unchanged
        original = atlas.wind.data["mean_wind_speed"]
        assert original.attrs["units"] == "m/s"


class TestConversionDaskFriendly:
    """Tests to verify conversion doesn't trigger eager loading."""

    def test_convert_dataarray_preserves_dask_chunks(self) -> None:
        """convert_dataarray preserves dask chunks."""
        pytest.importorskip("dask")
        import dask.array as dask_array

        data = dask_array.from_array(np.arange(100.0).reshape(10, 10), chunks=5)
        da = xr.DataArray(data, dims=["y", "x"])
        da.attrs["units"] = "m"

        result = convert_dataarray(da, "km")

        # Result should still be dask-backed
        assert hasattr(result.data, "dask")
        # Chunks should be preserved
        assert result.data.chunks == data.chunks

    def test_conversion_factor_is_scalar(self) -> None:
        """conversion_factor returns a scalar for multiplication."""
        factor = conversion_factor("m", "km")

        assert isinstance(factor, float)
        assert factor == 0.001

    def test_conversion_no_compute_until_needed(self) -> None:
        """Conversion doesn't trigger compute on dask arrays."""
        pytest.importorskip("dask")
        import dask.array as dask_array

        # Track if compute is called
        compute_called = []

        class TrackedArray(dask_array.Array):
            def compute(self, **kwargs):
                compute_called.append(True)
                return super().compute(**kwargs)

        data = dask_array.from_array(np.arange(100.0).reshape(10, 10), chunks=5)
        da = xr.DataArray(data, dims=["y", "x"])
        da.attrs["units"] = "m"

        # Conversion should not trigger compute
        result = convert_dataarray(da, "km")

        # Verify no compute was triggered by checking the result is still lazy
        assert hasattr(result.data, "dask")


class TestConversionOverheadBenchmark:
    """Micro-benchmarks for conversion overhead."""

    def test_conversion_factor_overhead_minimal(self) -> None:
        """conversion_factor has minimal overhead per call."""
        # Warm up
        for _ in range(10):
            conversion_factor("m", "km")

        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            conversion_factor("m", "km")
        elapsed = time.perf_counter() - start

        avg_us = (elapsed / iterations) * 1_000_000
        # Should be well under 1ms per call (typically ~10-50 microseconds)
        assert avg_us < 1000, f"conversion_factor too slow: {avg_us:.1f} µs/call"

    def test_convert_dataarray_overhead_minimal(self) -> None:
        """convert_dataarray overhead is minimal for small arrays."""
        da = xr.DataArray(np.ones((10, 10)), dims=["y", "x"])
        da.attrs["units"] = "m"

        # Warm up
        for _ in range(10):
            convert_dataarray(da, "km")

        # Benchmark
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            convert_dataarray(da, "km")
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / iterations) * 1000
        # Should be well under 10ms per call for small arrays
        assert avg_ms < 10, f"convert_dataarray too slow: {avg_ms:.1f} ms/call"


class TestCanonicalUnitsEnforced:
    """Tests for canonical unit enforcement at boundaries."""

    @pytest.fixture
    def materialized_atlas(self, tmp_path: Path) -> Atlas:
        """Create a fully materialized atlas for testing."""
        _create_all_required_gwa_files(tmp_path)
        _copy_turbine_yaml(tmp_path, "Enercon.E40.500")

        atlas = Atlas(
            tmp_path,
            country="AUT",
            crs="EPSG:3035",
        )
        atlas.configure_turbines(["Enercon.E40.500"])
        atlas.build()
        atlas.wind.select(turbines=["Enercon.E40.500"])
        return atlas

    def test_turbine_metadata_has_canonical_units(self, materialized_atlas: Atlas) -> None:
        """Turbine metadata has canonical units attr."""
        atlas = materialized_atlas
        ds = atlas.wind.data

        # Check turbine metadata
        assert ds["turbine_capacity"].attrs.get("units") == "kW"
        assert ds["turbine_hub_height"].attrs.get("units") == "m"
        assert ds["turbine_rotor_diameter"].attrs.get("units") == "m"
        assert ds["power_curve"].attrs.get("units") == "1"

    def test_computed_metrics_use_canonical_key(self, materialized_atlas: Atlas) -> None:
        """Computed metrics use canonical 'units' key, not legacy 'unit'."""
        atlas = materialized_atlas

        result = atlas.wind.compute("capacity_factors", method="hub_height_weibull")

        # Should have canonical key
        assert UNIT_ATTR_KEY in result.data.attrs
        # Should NOT have legacy key
        assert "unit" not in result.data.attrs
