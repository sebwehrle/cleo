"""Integration tests for WindDomain API.

Tests are offline-only: uses local test fixtures with no network calls.

Tests verify:
- WindDomain.turbines property returns turbine IDs from wind store
- WindDomain.select() mutates persistent Atlas selection state and validates turbine IDs
- Mean wind speed computation from canonical store is not all-NaN
"""

from __future__ import annotations

import shutil
from pathlib import Path
from scipy.special import gamma

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS

import cleo
from cleo.domains import WindDomain
from cleo.unification.gwa_io import GWA_HEIGHTS
from cleo.unification.unifier import Unifier


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

    transform = rasterio.transform.from_bounds(*bounds, shape[1], shape[0])

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


class TestWindDomainTurbineSelection:
    """Tests for WindDomain.turbines, selected_turbines, and select()."""

    def test_wind_select_is_persistent_and_validates(self, tmp_path: Path) -> None:
        """WindDomain.select() is persistent on Atlas and validates turbine IDs."""
        turbine_names = ["Enercon.E40.500", "Enercon.E82.3000"]

        atlas = MockAtlas(tmp_path, turbines=turbine_names)
        _create_all_required_gwa_files(atlas)
        _copy_turbine_yamls(atlas, turbine_names)
        _create_elevation_raster(atlas)

        # Materialize wind store
        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)

        # Get WindDomain
        w0 = WindDomain(atlas)

        # Verify turbines property returns both turbines
        tids = w0.turbines
        assert len(tids) == 2
        assert turbine_names[0] in tids
        assert turbine_names[1] in tids

        # Test persistent select: selection persists on Atlas
        result = w0.select(turbines=[tids[0]])
        assert result is None, "select() should return None (in-place command)"
        assert w0.selected_turbines == (tids[0],), "Selection must persist"

        # Test selecting multiple turbines
        w0.select(turbines=list(tids))
        assert w0.selected_turbines == tuple(tids)

        # Test selecting by positional turbine indices
        w0.select(turbine_indices=[1, 0])
        assert w0.selected_turbines == (tids[1], tids[0])

        # Test clearing selection via clear_selection()
        w0.clear_selection()
        assert w0.selected_turbines is None

        # Test empty list raises
        with pytest.raises(ValueError, match="non-empty"):
            w0.select(turbines=[])

        # Test unknown turbine raises with hint
        with pytest.raises(ValueError, match="atlas.wind.turbines"):
            w0.select(turbines=["Unknown.Turbine.999"])

        # Verify error message contains unknown turbine name
        with pytest.raises(ValueError, match="Unknown.Turbine.999"):
            w0.select(turbines=["Unknown.Turbine.999"])

        # Reject single-string iterable footgun
        with pytest.raises(ValueError, match="single string/bytes"):
            w0.select(turbines=tids[0])

        # Must provide exactly one selection mode
        with pytest.raises(ValueError, match="exactly one"):
            w0.select(turbines=[tids[0]], turbine_indices=[0])
        with pytest.raises(ValueError, match="exactly one"):
            w0.select()

        # Index-based validation
        with pytest.raises(ValueError, match="out of range"):
            w0.select(turbine_indices=[99])
        with pytest.raises(ValueError, match="Duplicate turbine index"):
            w0.select(turbine_indices=[0, 0])
        with pytest.raises(ValueError, match="must be an integer"):
            w0.select(turbine_indices=[0, "1"])  # type: ignore[list-item]

        # compute() rejects unknown kwargs like height for capacity_factors
        with pytest.raises(ValueError, match="Unknown parameter"):
            w0.compute("capacity_factors", height=100)


class TestMeanWindSpeedComputation:
    """Tests for mean wind speed computation from canonical store."""

    def test_mean_wind_speed_not_all_nan(self, tmp_path: Path) -> None:
        """Mean wind speed computed from canonical store is not all-NaN.

        Uses pure compute: mean_wind_speed = weibull_A * gamma(1 / weibull_k + 1)
        This is canonical-only (no raw I/O).
        """
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)
        _create_elevation_raster(atlas)

        # Materialize both stores
        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Get canonical wind data
        wind = WindDomain(atlas).data

        # Verify fixture has valid cells in weibull_A at height=100
        var_A = "weibull_A"
        assert var_A in wind.data_vars, f"wind store must have {var_A}"
        weibull_A_100 = wind[var_A].sel(height=100)
        assert (
            bool(weibull_A_100.notnull().any().compute()) is True
        ), "Fixture must have at least one valid weibull_A cell at height=100"

        # Verify weibull_k also has valid cells
        var_k = "weibull_k"
        assert var_k in wind.data_vars, f"wind store must have {var_k}"
        weibull_k_100 = wind[var_k].sel(height=100)
        assert (
            bool(weibull_k_100.notnull().any().compute()) is True
        ), "Fixture must have at least one valid weibull_k cell at height=100"

        # Verify landscape has valid_mask with some valid cells
        landscape = xr.open_zarr(atlas.landscape_store_path, consolidated=False)
        assert "valid_mask" in landscape.data_vars
        assert (
            bool(landscape["valid_mask"].any().compute()) is True
        ), "Fixture must have at least one valid cell in landscape valid_mask"

        # Compute mean wind speed using pure formula (canonical-only, no raw I/O)
        # Formula: mean_wind_speed = weibull_A * gamma(1 / weibull_k + 1)
        mean_wind_speed = weibull_A_100 * gamma(1 / weibull_k_100 + 1)

        # Assert NOT all NaN (some values should be valid)
        assert (
            bool(mean_wind_speed.notnull().any().compute()) is True
        ), "Mean wind speed should have at least one valid (non-NaN) value"

        # Sanity check: valid values should be positive (wind speed > 0)
        valid_values = mean_wind_speed.values[~np.isnan(mean_wind_speed.values)]
        assert len(valid_values) > 0, "Should have at least one valid value"
        assert np.all(valid_values > 0), "Valid wind speeds should be positive"
