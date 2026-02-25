"""Integration tests for region selection and region stores.

Tests verify:
- atlas.select(region=...) persists region selection
- Region stores are created with smaller y/x than country stores
- DomainResult.materialize() writes to region store (not country store)
- atlas.wind.data surfaces data from region store when region selected

Offline-only: uses local synthetic fixtures with no network calls.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import box

from cleo import Atlas

# GWA heights constant
GWA_HEIGHTS = [10, 50, 100, 150, 200]


def _create_gwa_raster(
    path: Path,
    shape: tuple[int, int] = (40, 40),  # Larger for region subsetting
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (4000000, 2600000, 4200000, 2800000),
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


def _create_all_gwa_rasters(atlas_path: Path, country: str = "AUT", **kwargs) -> None:
    """Create all required GWA rasters for wind materialization."""
    raw_dir = atlas_path / "data" / "raw" / country

    for h in GWA_HEIGHTS:
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-A_{h}.tif",
            fill_value=8.0 + h * 0.01,
            **kwargs,
        )
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-k_{h}.tif",
            fill_value=2.0 + h * 0.001,
            **kwargs,
        )
        _create_gwa_raster(
            raw_dir / f"{country}_air-density_{h}.tif",
            fill_value=1.2 - h * 0.0001,
            **kwargs,
        )


def _create_elevation_raster(path: Path, **kwargs) -> None:
    """Create elevation raster file."""
    _create_gwa_raster(path, fill_value=500.0, add_nodata_region=False, **kwargs)


def _create_nuts_shapefile(
    atlas_path: Path,
    country: str = "AUT",
    bounds: tuple[float, float, float, float] = (4000000, 2600000, 4200000, 2800000),
) -> None:
    """Create a synthetic NUTS shapefile for testing.

    Creates a small region in the center of the data extent.
    Must include all columns expected by Atlas.get_nuts_region():
    - CNTR_CODE: country code (AT)
    - NAME_LATN: region name in Latin script (used for lookup)
    - LEVL_CODE: NUTS level (for get_nuts_country)
    """
    nuts_dir = atlas_path / "data" / "nuts"
    nuts_dir.mkdir(parents=True, exist_ok=True)

    # Create a smaller region in the center (approximately 1/4 of the extent)
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    width = (bounds[2] - bounds[0]) / 4
    height = (bounds[3] - bounds[1]) / 4

    region_bounds = (
        center_x - width,
        center_y - height,
        center_x + width,
        center_y + height,
    )

    # Create additional nested/adjacent regions for level-aware discovery tests.
    level1_bounds = (
        bounds[0],
        bounds[1],
        center_x,
        bounds[3],
    )
    level3_bounds = (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )

    # Deliberate cross-level name ambiguity (no level-2 entry with this name):
    # "SharedName" appears at NUTS-1 and NUTS-3 to test ambiguity handling.
    shared_l1_bounds = (
        center_x,
        bounds[1],
        bounds[2],
        center_y,
    )
    shared_l3_bounds = (
        center_x,
        center_y,
        bounds[2],
        bounds[3],
    )

    # Add an additional level-2 region ("Wien") for realistic small-region workflow tests.
    wien_bounds = (
        center_x,
        center_y - height / 2,
        center_x + width,
        center_y + height / 2,
    )

    # Create GeoDataFrame with test regions
    # Include all columns expected by Atlas.get_nuts_region()
    gdf = gpd.GeoDataFrame(
        {
            "NUTS_ID": [
                "AT",
                "Niederösterreich",
                "AT13",
                "AT1",
                "AT123",
                "ATX1",
                "ATX3",
            ],
            "NUTS_NAME": [
                "Österreich",
                "Niederösterreich",
                "Wien",
                "Westregion",
                "Sankt Pölten",
                "SharedName",
                "SharedName",
            ],
            "NAME_LATN": [
                "Österreich",
                "Niederösterreich",
                "Wien",
                "Westregion",
                "Sankt Pölten",
                "SharedName",
                "SharedName",
            ],  # Required for lookup
            "CNTR_CODE": ["AT", "AT", "AT", "AT", "AT", "AT", "AT"],
            "LEVL_CODE": [0, 2, 2, 1, 3, 1, 3],  # country + levels 1/2/3
            "geometry": [
                box(*bounds),
                box(*region_bounds),
                box(*wien_bounds),
                box(*level1_bounds),
                box(*level3_bounds),
                box(*shared_l1_bounds),
                box(*shared_l3_bounds),
            ],
        },
        crs="EPSG:3035",
    )

    # Write shapefile
    shp_path = nuts_dir / "NUTS_RG_01M_2021_3035.shp"
    gdf.to_file(shp_path)


@pytest.fixture
def region_atlas(tmp_path: Path) -> Atlas:
    """Create an Atlas with synthetic data for region testing.

    Creates larger rasters so region subsetting produces meaningful subsets.
    Calls materialize() to build region name index for select() to work.
    """
    country = "AUT"
    shape = (40, 40)
    bounds = (4000000, 2600000, 4200000, 2800000)

    # Create GWA rasters
    _create_all_gwa_rasters(tmp_path, country=country, shape=shape, bounds=bounds)

    # Create elevation raster
    elev_path = tmp_path / "data" / "raw" / country / f"{country}_elevation_w_bathymetry.tif"
    _create_elevation_raster(elev_path, shape=shape, bounds=bounds)

    # Create NUTS shapefile for region lookup
    _create_nuts_shapefile(tmp_path, country=country, bounds=bounds)

    # Create Atlas (no region selected initially)
    atlas = Atlas(tmp_path, country, "epsg:3035")

    # Materialize to build region name index (required for select() to work)
    atlas.build()

    return atlas


class TestRegionSelection:
    """Tests for atlas.select(region=...) API."""

    def test_select_region_persists(self, region_atlas: Atlas) -> None:
        """atlas.select(region=...) persists region selection."""
        atlas = region_atlas

        # Initially no region selected
        assert atlas.region is None

        # Select region
        atlas.select(region="Niederösterreich", inplace=True)

        # Region persists
        assert atlas.region == "Niederösterreich"

    def test_select_region_can_be_cleared(self, region_atlas: Atlas) -> None:
        """atlas.select(region=None) clears region selection."""
        atlas = region_atlas

        atlas.select(region="Niederösterreich", inplace=True)
        assert atlas.region == "Niederösterreich"

        atlas.select(region=None, inplace=True)
        assert atlas.region is None

    def test_select_region_validates_empty(self, region_atlas: Atlas) -> None:
        """atlas.select(region='') raises ValueError."""
        atlas = region_atlas

        with pytest.raises(ValueError, match="empty"):
            atlas.select(region="", inplace=True)

        with pytest.raises(ValueError, match="empty"):
            atlas.select(region="   ", inplace=True)


class TestNutsRegionDiscovery:
    """Tests for NUTS region discovery and level-aware selection."""

    def test_nuts_regions_defaults_to_level_2(self, region_atlas: Atlas) -> None:
        """atlas.nuts_regions returns level-2 names by default."""
        atlas = region_atlas

        regions = atlas.nuts_regions
        assert isinstance(regions, tuple)
        assert regions, "Expected at least one level-2 region"
        assert all(getattr(r, "level", None) == 2 for r in regions)
        assert "Niederösterreich" in [str(r) for r in regions]
        assert "Sankt Pölten" not in [str(r) for r in regions]

    def test_nuts_regions_level_returns_tagged_names(self, region_atlas: Atlas) -> None:
        """atlas.nuts_regions_level(level) returns names tagged with that level."""
        atlas = region_atlas

        level3 = atlas.nuts_regions_level(3)
        assert isinstance(level3, tuple)
        assert level3, "Expected at least one level-3 region"
        assert all(getattr(r, "level", None) == 3 for r in level3)
        assert "Sankt Pölten" in [str(r) for r in level3]
        assert "Niederösterreich" not in [str(r) for r in level3]

    def test_select_accepts_level_tagged_region_without_region_level(self, region_atlas: Atlas) -> None:
        """Selecting a region object from nuts_regions_level infers its level."""
        atlas = region_atlas

        target = [r for r in atlas.nuts_regions_level(3) if str(r) == "Sankt Pölten"][0]
        atlas.select(region=target, inplace=True)

        assert atlas.region == "Sankt Pölten"
        assert atlas._region_id == "AT123"

    def test_select_ambiguous_plain_name_requires_level(self, region_atlas: Atlas) -> None:
        """Ambiguous plain names across levels raise unless region_level is provided."""
        atlas = region_atlas

        with pytest.raises(ValueError, match="ambiguous"):
            atlas.select(region="SharedName", inplace=True)

        atlas.select(region="SharedName", region_level=1, inplace=True)
        assert atlas.region == "SharedName"
        assert atlas._region_id == "ATX1"


class TestRegionStores:
    """Tests for region store creation and structure."""

    def test_region_stores_created_on_materialize(self, region_atlas: Atlas) -> None:
        """Selecting region then materializing creates region stores."""
        atlas = region_atlas

        # First materialize base stores
        atlas.build()

        # Get base store dims
        base_wind = xr.open_zarr(atlas.wind_store_path, consolidated=False)
        base_y_size = base_wind.sizes["y"]
        base_x_size = base_wind.sizes["x"]
        base_wind.close()

        # Select region and materialize again
        atlas.select(region="Niederösterreich", inplace=True)
        atlas.build()

        # Region stores should exist
        region_root = atlas.path / "regions" / atlas._region_id
        region_wind_path = region_root / "wind.zarr"
        region_land_path = region_root / "landscape.zarr"

        assert region_wind_path.exists(), "Region wind store should exist"
        assert region_land_path.exists(), "Region landscape store should exist"

        # Region stores should have smaller dims than base
        region_wind = xr.open_zarr(region_wind_path, consolidated=False)
        region_y_size = region_wind.sizes["y"]
        region_x_size = region_wind.sizes["x"]
        region_wind.close()

        assert region_y_size < base_y_size, (
            f"Region y ({region_y_size}) should be smaller than base ({base_y_size})"
        )
        assert region_x_size < base_x_size, (
            f"Region x ({region_x_size}) should be smaller than base ({base_x_size})"
        )

    def test_region_data_routes_to_region_store(self, region_atlas: Atlas) -> None:
        """atlas.wind.data routes to region store when region selected."""
        atlas = region_atlas

        # Materialize base stores
        atlas.build()

        # Get base dims
        base_data = atlas.wind.data
        base_y_size = base_data.sizes["y"]

        # Clear cached data
        atlas._wind_domain._data = None

        # Select region and materialize
        atlas.select(region="Niederösterreich", inplace=True)
        atlas.build()

        # Data should now come from region store (smaller dims)
        region_data = atlas.wind.data
        region_y_size = region_data.sizes["y"]

        assert region_y_size < base_y_size, (
            f"Region data y ({region_y_size}) should be smaller than base ({base_y_size})"
        )

    def test_region_stores_rebuild_when_base_inputs_id_changes(self, region_atlas: Atlas) -> None:
        """Region store freshness tracks base inputs; stale region stores must rebuild."""
        atlas = region_atlas

        # Build initial region stores from default base turbine set.
        atlas.select(region="Niederösterreich", inplace=True)
        atlas.build()
        region_id = atlas._region_id
        region_wind_path = atlas.path / "regions" / region_id / "wind.zarr"

        region_before = xr.open_zarr(region_wind_path, consolidated=False)
        n_before = int(region_before.sizes["turbine"])
        base_link_before = region_before.attrs.get("base_wind_inputs_id", "")
        region_before.close()

        # Rebuild base with a smaller configured turbine set.
        atlas.select(region=None, inplace=True)
        subset = list(atlas.wind.turbines[:2])
        assert len(subset) == 2
        atlas.configure_turbines(subset)
        atlas.build_canonical()

        base_after = xr.open_zarr(atlas.wind_store_path, consolidated=False)
        base_inputs_after = base_after.attrs.get("inputs_id", "")
        assert int(base_after.sizes["turbine"]) == len(subset)
        base_after.close()

        # Re-materialize region. Region store must rebuild to track new base inputs.
        atlas.select(region="Niederösterreich", inplace=True)
        atlas.build()

        region_after = xr.open_zarr(region_wind_path, consolidated=False)
        assert int(region_after.sizes["turbine"]) == len(subset)
        assert region_after.attrs.get("base_wind_inputs_id") == base_inputs_after
        assert region_after.attrs.get("base_wind_inputs_id") != base_link_before
        # Ensure this test meaningfully exercised a changed base-set path.
        assert n_before >= len(subset)
        region_after.close()


class TestRegionMaterialization:
    """Tests for DomainResult.materialize() with region selection."""

    def test_materialize_writes_to_region_store(self, region_atlas: Atlas) -> None:
        """DomainResult.materialize() writes to region store, not base store."""
        atlas = region_atlas

        # Materialize base stores first
        atlas.build()

        # Select region and materialize region stores
        atlas.select(region="Niederösterreich", inplace=True)
        atlas.build()

        # Select turbine and compute
        tid = atlas.wind.turbines[0]
        atlas.wind.select(turbines=[tid])

        # Cache capacity factors
        atlas.wind.compute(
            metric="capacity_factors",
            air_density=False,
            loss_factor=1.0,
        ).materialize()

        # Verify metric appears in atlas.wind.data (region store)
        assert "capacity_factors" in atlas.wind.data, (
            "capacity_factors should appear in atlas.wind.data after materialize()"
        )

        # Verify it was written to region store, NOT base store
        region_wind_path = atlas.path / "regions" / atlas._region_id / "wind.zarr"
        region_wind = xr.open_zarr(region_wind_path, consolidated=False)
        assert "capacity_factors" in region_wind, (
            "capacity_factors should be in region wind store"
        )
        region_wind.close()

        # Base store should NOT have capacity_factors
        base_wind = xr.open_zarr(atlas.wind_store_path, consolidated=False)
        assert "capacity_factors" not in base_wind, (
            "capacity_factors should NOT be in base wind store"
        )
        base_wind.close()

    def test_selection_persists_after_materialize(self, region_atlas: Atlas) -> None:
        """Region and turbine selection persist after materialize()."""
        atlas = region_atlas

        # Materialize
        atlas.build()
        atlas.select(region="Niederösterreich", inplace=True)
        atlas.build()

        # Select turbine
        tid = atlas.wind.turbines[0]
        atlas.wind.select(turbines=[tid])

        # Cache
        atlas.wind.compute(
            metric="capacity_factors",
            air_density=False,
        ).materialize()

        # Selections should persist
        assert atlas.region == "Niederösterreich"
        assert atlas.wind.selected_turbines == (tid,)

    def test_wien_capacity_factors_exist_and_not_all_nan(self, region_atlas: Atlas) -> None:
        """Small-region workflow: Wien + limited turbines yields non-empty, non-all-NaN CF."""
        atlas = region_atlas

        # Build base, select small region, and build region stores.
        atlas.build()
        atlas.select(region="Wien", inplace=True)
        atlas.build()

        # Limit turbine set for compute workload.
        turbines = list(atlas.wind.turbines[:2])
        assert len(turbines) == 2
        atlas.wind.select(turbines=turbines)

        # Compute + materialize and verify surfaced output.
        cf = atlas.wind.compute("capacity_factors", mode="direct_cf_quadrature", air_density=False).materialize()
        assert "capacity_factors" in atlas.wind.data
        assert cf.size > 0
        assert bool(cf.notnull().any().compute()) is True
