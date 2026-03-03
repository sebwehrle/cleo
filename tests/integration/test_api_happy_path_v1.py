"""End-to-end integration test for v1 public API workflow.

Tests the full happy path:
Atlas.build → wind/landscape data access → turbine selection →
compute → persist → open_result → export_result_netcdf.

Offline-only: no network calls, uses local synthetic fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import zarr
from rasterio.crs import CRS

from cleo import Atlas

# GWA heights constant
GWA_HEIGHTS = [10, 50, 100, 150, 200]


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
        "nodata": np.nan,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


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


def _create_elevation_raster(path: Path, **kwargs) -> None:
    """Create elevation raster file."""
    _create_gwa_raster(path, fill_value=500.0, add_nodata_region=False, **kwargs)


def _get_expected_turbine_ids(resources_dir: Path) -> tuple[str, ...]:
    """Get expected turbine IDs from resources directory.

    Returns sorted tuple of turbine stems from all *.yml files,
    excluding non-turbine config files (clc_codes.yml, cost_assumptions.yml).
    """
    non_turbine_files = {"clc_codes.yml", "cost_assumptions.yml"}
    return tuple(sorted(p.stem for p in resources_dir.glob("*.yml") if p.name not in non_turbine_files))


@pytest.fixture
def offline_atlas(tmp_path: Path) -> Atlas:
    """Create a fully offline Atlas with synthetic data and all packaged turbines.

    This fixture sets up everything needed for the happy path test:
    - All GWA rasters (Weibull A/k, air density at all heights)
    - Elevation raster (to avoid CopDEM network calls)
    - All turbine YAMLs from package resources (default discovery per v1.3 contract)

    Contract v1.3: NO configure_turbines() call - exercises default discovery path.
    All turbine YAMLs in resources/ should be auto-discovered during materialize().

    Returns:
        Atlas ready for materialize().
    """
    country = "AUT"

    # Create all required GWA rasters
    _create_all_gwa_rasters(tmp_path, country=country)

    # Create elevation raster (prevents CopDEM network call)
    elev_path = tmp_path / "data" / "raw" / country / f"{country}_elevation_w_bathymetry.tif"
    _create_elevation_raster(elev_path)

    # Create Atlas - this will deploy all packaged resources including turbine YAMLs
    # NO configure_turbines() call - testing default discovery path (contract v1.3)
    # All turbine YAMLs in resources/ will be auto-discovered during materialize()
    atlas = Atlas(tmp_path, country, "epsg:3035")

    return atlas


def _assert_no_string_dtype_in_zarr(store_path: Path) -> None:
    """Assert no arrays in zarr store have dtype.kind in ("U","S","O").

    This enforces Zarr v3 compatibility by preventing string/object arrays
    which cause FixedLengthUTF32 or NullTerminatedBytes warnings.
    """
    forbidden_kinds = ("U", "S", "O")
    violations = []

    root = zarr.open_group(store_path, mode="r")

    # Use members() with max_depth=None to recursively get all arrays
    for name, item in root.members(max_depth=None):
        if isinstance(item, zarr.Array):
            kind = item.dtype.kind
            if kind in forbidden_kinds:
                violations.append(f"{name}: dtype={item.dtype} (kind={kind})")

    if violations:
        pytest.fail(
            f"Zarr store {store_path.name} contains string/object dtype arrays:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


def test_api_happy_path_v1(offline_atlas: Atlas, tmp_path: Path) -> None:
    """End-to-end test of the v1 public API workflow.

    Verifies:
    - materialize() creates canonical stores
    - wind_data/landscape_data aliases work
    - valid_mask exists and has valid pixels
    - weibull variables exist (A and k)
    - turbine selection works
    - capacity_factors compute produces valid data
    - persist/open_result roundtrip works
    - export_result_netcdf creates a file
    """
    atlas = offline_atlas

    # === Public API boundary checks (Atlas/domain contract) ===
    for method in ("build", "build_canonical", "select", "new_run_id", "open_result", "export_result_netcdf"):
        assert hasattr(atlas, method), f"Atlas missing public method: {method}"
    for method in ("compute", "select", "clear_selection", "clear_computed"):
        assert hasattr(atlas.wind, method), f"WindDomain missing public method: {method}"
    for method in ("add", "rasterize", "clear_staged"):
        assert hasattr(atlas.landscape, method), f"LandscapeDomain missing public method: {method}"

    # === Materialize (canonical v1 API, offline-safe) ===
    atlas.build_canonical()

    # === Zarr v3 dtype invariant guard: no string/object dtypes ===
    wind_store = Path(atlas.path) / "wind.zarr"
    land_store = Path(atlas.path) / "landscape.zarr"
    _assert_no_string_dtype_in_zarr(wind_store)
    _assert_no_string_dtype_in_zarr(land_store)

    # === Lock the dataset aliases ===
    wind_ds = atlas.wind_data
    land_ds = atlas.landscape_data

    # Landscape must have valid_mask
    assert "valid_mask" in land_ds

    # Wind var naming: allow common aliases
    assert any(v in wind_ds.data_vars for v in ("weibull_A", "weibull_a"))
    assert any(v in wind_ds.data_vars for v in ("weibull_k", "weibull_K", "weibull_kappa"))

    # === Deterministic fixture guards ===
    # Guard 1: valid_mask must have at least one True
    assert bool(land_ds["valid_mask"].any().compute()) is True, (
        "Fixture sanity: valid_mask must have at least one True pixel"
    )

    # Guard 2: weibull_A at height=100 must have at least one non-NaN on valid_mask
    weibull_A_var = "weibull_A" if "weibull_A" in wind_ds.data_vars else "weibull_a"
    weibull_A_100 = wind_ds[weibull_A_var].sel(height=100)
    valid_weibull = land_ds["valid_mask"] & weibull_A_100.notnull()
    assert bool(valid_weibull.any().compute()) is True, (
        f"Fixture sanity: {weibull_A_var} at height=100 must have valid data on valid_mask"
    )

    # === Guard 3: Default turbine discovery (contract v1.3) ===
    # Verify that without configure_turbines(), default discovery finds ALL turbines in resources/
    # Per contract A3: "If not configured, all available turbines are materialized."
    assert atlas.turbines_configured is None, "Fixture should NOT call configure_turbines() - testing default discovery"
    assert "turbine" in wind_ds.dims, "wind.zarr must have turbine dimension after default discovery"
    # Build expected turbine IDs from the resources dir used by this atlas instance
    resources_dir = atlas.path / "resources"
    expected_turbines = _get_expected_turbine_ids(resources_dir)
    actual_turbines = tuple(sorted(atlas.wind.turbines))
    assert actual_turbines == expected_turbines, (
        f"Default discovery mismatch:\n  actual:   {actual_turbines}\n  expected: {expected_turbines}"
    )
    assert len(actual_turbines) > 0, "At least one turbine YAML must exist in resources/"

    # === Turbine selection (persistent, contract v1) ===
    tids = atlas.wind.turbines
    assert len(tids) >= 1, "Fixture must have at least one turbine"
    tid = tids[0]

    # Persistent selection: select() mutates selection state on Atlas
    result = atlas.wind.select(turbines=[tid])
    assert result is None, "select() should return None (in-place command)"
    assert atlas.wind.selected_turbines == (tid,), "Selection must persist on Atlas, not ephemeral WindDomain instance"

    # === Compute capacity factors (result wrapper API) ===
    # compute() returns DomainResult with .data and .materialize()
    metric_result = atlas.wind.compute("capacity_factors", air_density=False)
    cf = metric_result.data  # Access the DataArray

    # Ensure stable name before persist
    cf = cf.rename("capacity_factors")

    # Must not be all-NaN on valid cells
    ok = land_ds["valid_mask"] & cf.notnull().all(dim="turbine")
    assert bool(ok.any().compute()) is True, (
        "capacity_factors should have at least one valid (non-NaN) value on valid_mask"
    )

    # === Contract v1: compute().materialize() pattern ===
    # Test the canonical examples: select + compute + materialize writes to wind.zarr
    atlas.wind.clear_selection()
    assert atlas.wind.selected_turbines is None, "clear_selection() must clear"

    atlas.wind.select(turbines=[tid])
    materialized_cf = atlas.wind.compute(
        metric="capacity_factors",
        air_density=False,
        loss_factor=1.0,
    ).materialize()

    # Verify metric appears in atlas.wind.data after materialize
    assert "capacity_factors" in atlas.wind.data, (
        "materialize() must write metric into wind.zarr and surface in atlas.wind.data"
    )
    assert atlas.wind.selected_turbines == (tid,), "Selection must remain persistent after compute/materialize"

    # === Persist ===
    run_id = atlas.new_run_id(prefix="api")
    assert run_id.startswith("api-")

    store_path = metric_result.persist(
        run_id=run_id,
        params={"height": 100, "turbine": tid},
    )
    assert store_path.exists()

    # === Zarr v3 dtype invariant guard for result store ===
    _assert_no_string_dtype_in_zarr(store_path)

    # === Open result ===
    ds_result = atlas.open_result(run_id, "capacity_factors")
    assert "capacity_factors" in ds_result
    assert ds_result.attrs.get("store_state") == "complete"
    assert ds_result.attrs.get("run_id") == run_id

    # === Export to NetCDF ===
    out_path = tmp_path / "capacity_factors.nc"
    exported = atlas.export_result_netcdf(run_id, "capacity_factors", out_path)

    # Verify file exists and is non-empty (no xr.open_dataset per constraints)
    assert exported.exists()
    assert exported.stat().st_size > 0
