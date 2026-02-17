"""Integration tests for WindDomain.compute() API.

Tests are offline-only: uses local test fixtures with no network calls.

Tests verify:
- Unknown metric raises ValueError with supported list
- mean_wind_speed works and is not all-NaN
- capacity_factors requires turbines (enforcement)
- capacity_factors works and is not all-NaN
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS

import cleo
from cleo.classes import WindDomain
from cleo.unify import Unifier, GWA_HEIGHTS


def _copy_default_turbine(atlas_path: Path) -> None:
    """Copy a default turbine YAML to resources for testing."""
    resources_src = Path(cleo.__file__).resolve().parent / "resources"
    resources_dest = atlas_path / "resources"
    resources_dest.mkdir(parents=True, exist_ok=True)

    turbine_name = "Enercon.E40.500"
    src = resources_src / f"{turbine_name}.yml"
    if src.exists():
        shutil.copy(src, resources_dest / f"{turbine_name}.yml")


class MockAtlas:
    """Minimal Atlas-like object for testing.

    Provides the minimal interface expected by Unifier and WindDomain.
    """

    def __init__(
        self,
        path: Path,
        country: str = "AUT",
        crs: str = "epsg:3035",
        region: str | None = None,
        turbines: list[str] | None = None,
    ):
        self.path = path
        self.country = country
        self.crs = crs
        self._region = region  # Region selection state
        self._turbines_configured = tuple(turbines) if turbines else None
        self._wind_selected_turbines: tuple[str, ...] | None = None

        self.chunk_policy = {"y": 64, "x": 64}
        self.fingerprint_method = "path_mtime_size"

        # Store paths for WindDomain
        self.wind_store_path = path / "wind.zarr"
        self.landscape_store_path = path / "landscape.zarr"

        # Domain object caches
        self._wind_domain = None
        self._landscape_domain = None

        # Create required directories
        (path / "data" / "raw" / country).mkdir(parents=True, exist_ok=True)
        (path / "resources").mkdir(parents=True, exist_ok=True)
        (path / "intermediates" / "crs_cache").mkdir(parents=True, exist_ok=True)

        # Copy default turbine for wind materialization (when not explicitly configured)
        _copy_default_turbine(path)

    @property
    def turbines_configured(self):
        return self._turbines_configured

    @property
    def region(self):
        return self._region

    def _active_wind_store_path(self) -> Path:
        """Return active wind store path (no region support in mock)."""
        return self.wind_store_path

    def _active_landscape_store_path(self) -> Path:
        """Return active landscape store path (no region support in mock)."""
        return self.landscape_store_path

    def get_nuts_region(self, region: str):
        """Mock - would return GeoDataFrame."""
        return None

    @property
    def wind(self) -> WindDomain:
        """Access WindDomain."""
        if self._wind_domain is None:
            self._wind_domain = WindDomain(self)
        return self._wind_domain

    @property
    def wind_data(self) -> xr.Dataset:
        """Direct access to wind dataset."""
        return self.wind.data

    @property
    def landscape_data(self) -> xr.Dataset:
        """Direct access to landscape dataset."""
        ds = xr.open_zarr(self.landscape_store_path, consolidated=False)
        state = ds.attrs.get("store_state", "")
        if state != "complete":
            raise RuntimeError(f"Landscape store incomplete (store_state={state!r})")
        return ds


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

    # Add nodata region in upper-left corner to test mask
    if add_nodata_region:
        data[:3, :3] = np.nan

    transform = rasterio.transform.from_bounds(
        *bounds, shape[1], shape[0]
    )

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "transform": transform,
        "crs": CRS.from_epsg(crs_epsg),
        "nodata": np.nan,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_all_required_gwa_files(atlas: MockAtlas, **kwargs) -> list[Path]:
    """Create all 15 required GWA files for testing."""
    raw_dir = atlas.path / "data" / "raw" / atlas.country
    paths = []

    for h in GWA_HEIGHTS:
        for layer, prefix in [
            ("weibull_A", f"{atlas.country}_combined-Weibull-A_{h}.tif"),
            ("weibull_k", f"{atlas.country}_combined-Weibull-k_{h}.tif"),
            ("rho", f"{atlas.country}_air-density_{h}.tif"),
        ]:
            p = raw_dir / prefix
            _create_gwa_raster(p, **kwargs)
            paths.append(p)

    return paths


def _copy_turbine_yamls(atlas: MockAtlas, turbine_names: list[str]) -> None:
    """Copy turbine YAML files from package resources to test resources."""
    resources_src = Path(cleo.__file__).resolve().parent / "resources"
    resources_dest = atlas.path / "resources"

    for name in turbine_names:
        src = resources_src / f"{name}.yml"
        if src.exists():
            shutil.copy(src, resources_dest / f"{name}.yml")


def _create_elevation_raster(atlas: MockAtlas, **kwargs) -> Path:
    """Create elevation raster file."""
    raw_dir = atlas.path / "data" / "raw" / atlas.country
    p = raw_dir / f"{atlas.country}_elevation_w_bathymetry.tif"
    _create_gwa_raster(p, fill_value=500.0, add_nodata_region=False, **kwargs)
    return p


class TestWindDomainCompute:
    """Tests for WindDomain.compute() method."""

    def test_compute_unknown_metric_lists_supported(self, tmp_path: Path) -> None:
        """Unknown metric raises ValueError with list of supported metrics."""
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)
        _create_elevation_raster(atlas)

        # Materialize stores
        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Try unknown metric
        with pytest.raises(ValueError, match="Unknown metric") as exc_info:
            atlas.wind.compute("___nope___")

        # Should list supported metrics
        error_msg = str(exc_info.value)
        assert "mean_wind_speed" in error_msg
        assert "Supported" in error_msg

    def test_mean_wind_speed_works_and_not_all_nan(self, tmp_path: Path) -> None:
        """mean_wind_speed metric computes valid values."""
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)
        _create_elevation_raster(atlas)

        # Materialize stores
        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Deterministic validity guard
        wind = atlas.wind_data
        land = atlas.landscape_data
        var_A = "weibull_A" if "weibull_A" in wind else "weibull_a"
        ok = wind[var_A].sel(height=100).notnull() & land["valid_mask"]
        assert bool(ok.any().compute()) is True, "Fixture must have valid cells"

        # Compute mean wind speed (compute() returns MetricResult, use .data for DataArray)
        da = atlas.wind.compute("mean_wind_speed", height=100).data

        # Assert not all NaN
        assert bool(da.notnull().any().compute()) is True, (
            "mean_wind_speed should have at least one valid (non-NaN) value"
        )


class TestCapacityFactorsEnforcement:
    """Tests for capacity_factors turbine enforcement and computation."""

    def test_capacity_factors_requires_turbines(self, tmp_path: Path) -> None:
        """capacity_factors without turbines raises ValueError with hint."""
        turbine_names = ["Enercon.E40.500"]
        atlas = MockAtlas(tmp_path, turbines=turbine_names)
        _create_all_required_gwa_files(atlas)
        _copy_turbine_yamls(atlas, turbine_names)
        _create_elevation_raster(atlas)

        # Materialize stores
        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Try without turbines - should fail
        with pytest.raises(ValueError) as exc_info:
            atlas.wind.compute("capacity_factors", height=100)

        error_msg = str(exc_info.value)
        assert "turbines" in error_msg.lower()
        # Should hint at atlas.wind.turbines or atlas.wind.select
        assert "atlas.wind.turbines" in error_msg or "atlas.wind.select" in error_msg

    def test_capacity_factors_not_all_nan(self, tmp_path: Path) -> None:
        """capacity_factors computes valid values."""
        turbine_names = ["Enercon.E40.500"]
        atlas = MockAtlas(tmp_path, turbines=turbine_names)
        _create_all_required_gwa_files(atlas)
        _copy_turbine_yamls(atlas, turbine_names)
        _create_elevation_raster(atlas)

        # Materialize stores
        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Mandatory fixture guard
        wind = atlas.wind_data
        land = atlas.landscape_data
        var_A = "weibull_A" if "weibull_A" in wind else "weibull_a"
        ok = wind[var_A].sel(height=100).notnull() & land["valid_mask"]
        assert bool(ok.any().compute()) is True, "Fixture must have valid cells"

        # Get turbine IDs
        tids = atlas.wind.turbines[:1]

        # Compute capacity factors (compute() returns MetricResult, use .data for DataArray)
        da = atlas.wind.compute("capacity_factors", turbines=tids, height=100).data

        # Assert not all NaN
        assert bool(da.notnull().any().compute()) is True, (
            "capacity_factors should have at least one valid (non-NaN) value"
        )

    def test_capacity_factors_with_air_density(self, tmp_path: Path) -> None:
        """capacity_factors with air_density=True computes valid values."""
        turbine_names = ["Enercon.E40.500"]
        atlas = MockAtlas(tmp_path, turbines=turbine_names)
        _create_all_required_gwa_files(atlas)
        _copy_turbine_yamls(atlas, turbine_names)
        _create_elevation_raster(atlas)

        # Materialize stores
        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Get turbine IDs
        tids = atlas.wind.turbines[:1]

        # Compute capacity factors with air density correction
        da = atlas.wind.compute("capacity_factors", turbines=tids, height=100, air_density=True).data

        # Assert not all NaN
        assert bool(da.notnull().any().compute()) is True, (
            "capacity_factors with air_density should have at least one valid value"
        )

    def test_capacity_factors_uses_selected_turbines(self, tmp_path: Path) -> None:
        """capacity_factors uses selected_turbines when turbines kwarg not provided."""
        turbine_names = ["Enercon.E40.500", "Enercon.E82.3000"]
        atlas = MockAtlas(tmp_path, turbines=turbine_names)
        _create_all_required_gwa_files(atlas)
        _copy_turbine_yamls(atlas, turbine_names)
        _create_elevation_raster(atlas)

        # Materialize stores
        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Select first turbine (persistent selection on Atlas)
        atlas.wind.select(turbines=[turbine_names[0]])

        # Compute without passing turbines - should use selected (use .data for DataArray)
        da = atlas.wind.compute("capacity_factors", height=100).data

        # Should have only one turbine in result
        assert "turbine" in da.dims
        assert da.sizes["turbine"] == 1
        assert str(da.coords["turbine"].values[0]) == turbine_names[0]
