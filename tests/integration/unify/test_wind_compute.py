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
import textwrap

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS

import cleo
from cleo.dask_utils import compute as dask_compute
from cleo.domains import WindDomain
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


def _write_custom_turbine_yaml(
    atlas: MockAtlas,
    *,
    turbine_id: str,
    hub_height: float,
    rotor_diameter: float,
) -> None:
    """Write a deterministic custom turbine resource for integration tests."""
    resources_dest = atlas.path / "resources"
    resources_dest.mkdir(parents=True, exist_ok=True)
    yaml_text = textwrap.dedent(
        f"""\
        manufacturer: TestCo
        model: TallRotor
        capacity: 3000
        rotor_diameter: {float(rotor_diameter)}
        hub_height: {float(hub_height)}
        commissioning_year: 2018
        V: [0.0, 3.0, 12.0, 25.0, 30.0]
        cf: [0.0, 0.0, 1.0, 1.0, 0.0]
        """
    )
    (resources_dest / f"{turbine_id}.yml").write_text(yaml_text, encoding="utf-8")


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

        # Compute mean wind speed (compute() returns DomainResult, use .data for DataArray)
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

        # Compute capacity factors (compute() returns DomainResult, use .data for DataArray)
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


class TestDomainResultCacheOverwrite:
    """Tests for DomainResult.cache() overwrite behavior."""

    def test_cache_overwrite_true_replaces_variable(self, tmp_path: Path) -> None:
        """cache(overwrite=True) replaces existing variable completely."""
        turbine_names = ["Enercon.E40.500"]
        atlas = MockAtlas(tmp_path, turbines=turbine_names)
        _create_all_required_gwa_files(atlas)
        _copy_turbine_yamls(atlas, turbine_names)
        _create_elevation_raster(atlas)

        # Materialize stores
        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Select turbine
        atlas.wind.select(turbines=turbine_names)

        # Cache capacity_factors first time (air_density=False -> cleo:air_density=0)
        atlas.wind.compute("capacity_factors", height=100, air_density=False).cache()

        # Check first cache
        assert "capacity_factors" in atlas.wind.data
        first_air_density = atlas.wind.data["capacity_factors"].attrs.get("cleo:air_density")
        assert first_air_density == 0, "First cache should have air_density=0"

        # Cache again with air_density=True - should overwrite
        atlas.wind.compute(
            "capacity_factors", height=100, air_density=True
        ).cache(overwrite=True)

        # Verify overwrite - attrs should be updated
        assert "capacity_factors" in atlas.wind.data
        second_air_density = atlas.wind.data["capacity_factors"].attrs.get("cleo:air_density")
        assert second_air_density == 1, "cache(overwrite=True) must update attrs (air_density=1)"

    def test_cache_overwrite_true_with_mode_change(self, tmp_path: Path) -> None:
        """cache(overwrite=True, allow_mode_change=True) allows changing cf mode."""
        turbine_names = ["Enercon.E40.500"]
        atlas = MockAtlas(tmp_path, turbines=turbine_names)
        _create_all_required_gwa_files(atlas)
        _copy_turbine_yamls(atlas, turbine_names)
        _create_elevation_raster(atlas)

        # Materialize stores
        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Select turbine
        atlas.wind.select(turbines=turbine_names)

        # Cache capacity_factors with mode="hub"
        atlas.wind.compute("capacity_factors", height=100, mode="hub").cache()

        # Verify first mode
        assert atlas.wind.data["capacity_factors"].attrs.get("cleo:cf_mode") == "hub"

        # Cache with mode="rews" should fail without allow_mode_change
        with pytest.raises(ValueError, match="allow_mode_change"):
            atlas.wind.compute("capacity_factors", height=100, mode="rews").cache()

        # Cache with mode="rews" and allow_mode_change=True should succeed
        atlas.wind.compute(
            "capacity_factors", height=100, mode="rews"
        ).cache(overwrite=True, allow_mode_change=True)

        # Verify mode changed
        assert atlas.wind.data["capacity_factors"].attrs.get("cleo:cf_mode") == "rews"

    def test_cache_overwrite_false_raises_if_exists(self, tmp_path: Path) -> None:
        """cache(overwrite=False) raises if variable already exists."""
        turbine_names = ["Enercon.E40.500"]
        atlas = MockAtlas(tmp_path, turbines=turbine_names)
        _create_all_required_gwa_files(atlas)
        _copy_turbine_yamls(atlas, turbine_names)
        _create_elevation_raster(atlas)

        # Materialize stores
        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Select turbine
        atlas.wind.select(turbines=turbine_names)

        # Cache first time
        atlas.wind.compute("capacity_factors", height=100).cache()

        # Second cache with overwrite=False should raise
        with pytest.raises(ValueError, match="already exists"):
            atlas.wind.compute("capacity_factors", height=100).cache(overwrite=False)


class TestComputeBackendParity:
    """Numerical parity across local compute backends."""

    def test_capacity_factors_parity_serial_threads_processes(self, tmp_path: Path) -> None:
        np.random.seed(42)
        turbine_names = ["Enercon.E40.500", "Vestas.V112.3075"]
        atlas = MockAtlas(tmp_path, turbines=turbine_names)
        _create_all_required_gwa_files(atlas)
        _copy_turbine_yamls(atlas, turbine_names)
        _create_elevation_raster(atlas)

        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        atlas.wind.select(turbines=turbine_names)
        da = atlas.wind.compute("capacity_factors", height=100, mode="hub").data

        serial = dask_compute(da, backend="serial").values
        threads = dask_compute(da, backend="threads").values
        assert np.allclose(serial, threads, equal_nan=True)

        try:
            processes = dask_compute(da, backend="processes").values
        except Exception as exc:
            msg = str(exc)
            if "Operation not permitted" in msg or "PermissionError" in msg:
                pytest.skip("Process backend unavailable in this execution environment.")
            raise
        assert np.allclose(serial, processes, equal_nan=True)


class TestVerticalRewsIntegration:
    """Integration gates for PR4 closure: tall-rotor and within-200 envelope."""

    def test_tall_rotor_direct_cf_and_rews_mps_are_finite(self, tmp_path: Path) -> None:
        """Tall rotor (z_top > 200 m) computes in direct mode without structural failure."""
        turbine_id = "Test.TallRotor3000"
        atlas = MockAtlas(tmp_path, turbines=[turbine_id])
        _create_all_required_gwa_files(atlas)
        _create_elevation_raster(atlas)
        _write_custom_turbine_yaml(
            atlas,
            turbine_id=turbine_id,
            hub_height=140.0,
            rotor_diameter=240.0,  # z_top = 260 m (> 200 m)
        )

        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        atlas.wind.select(turbines=[turbine_id])

        cf = atlas.wind.compute("capacity_factors", mode="direct_cf_quadrature", rews_n=12).data
        rews = atlas.wind.compute("rews_mps", rews_n=12).data

        assert cf.attrs.get("cleo:cf_mode") == "direct_cf_quadrature"
        assert bool(cf.notnull().any().compute()) is True
        assert bool(rews.notnull().any().compute()) is True

    def test_within_200_direct_vs_legacy_rews_acceptance_envelope(self, tmp_path: Path) -> None:
        """Within-200 envelope check between direct_cf_quadrature and legacy rews mode."""
        turbine_names = ["Enercon.E40.500", "Enercon.E82.3000"]
        atlas = MockAtlas(tmp_path, turbines=turbine_names)
        _create_all_required_gwa_files(atlas)
        _copy_turbine_yamls(atlas, turbine_names)
        _create_elevation_raster(atlas)

        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        atlas.wind.select(turbines=turbine_names)
        cf_direct = atlas.wind.compute(
            "capacity_factors",
            mode="direct_cf_quadrature",
            rews_n=12,
        ).data
        cf_legacy = atlas.wind.compute(
            "capacity_factors",
            mode="rews",
            rews_n=12,
        ).data

        delta = np.abs((cf_legacy - cf_direct).where(cf_direct.notnull() & cf_legacy.notnull()))
        delta_vals = delta.values[np.isfinite(delta.values)]
        assert delta_vals.size > 0, "Expected at least one valid comparison pixel."

        med = float(np.median(delta_vals))
        p95 = float(np.quantile(delta_vals, 0.95))
        # Envelope-only check for legacy comparison under new_everywhere policy.
        assert med <= 0.05, f"median(|ΔCF|) too high: {med}"
        assert p95 <= 0.15, f"P95(|ΔCF|) too high: {p95}"
