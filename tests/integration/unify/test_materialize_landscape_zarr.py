"""Integration tests for Unifier.materialize_landscape.

Tests are offline-only: no network calls, uses local test fixtures.

These tests verify that landscape.zarr is correctly created with:
- valid_mask derived from wind data
- elevation data from legacy GeoTIFF
- Proper alignment to wind grid
- Correct manifest entries
"""

from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from rasterio.crs import CRS
from shapely.geometry import box

import cleo
import json

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

    Provides the minimal interface expected by Unifier methods.
    """

    def __init__(
        self,
        path: Path,
        country: str = "AUT",
        crs: str = "epsg:3035",
        area: str | None = None,
    ):
        self.path = path
        self.country = country
        self.crs = crs
        self.area = area
        self.turbines_configured = None
        self.chunk_policy = {"y": 1024, "x": 1024}
        self.fingerprint_method = "path_mtime_size"

        # Create required directories
        (path / "data" / "raw" / country).mkdir(parents=True, exist_ok=True)
        (path / "resources").mkdir(parents=True, exist_ok=True)
        (path / "intermediates" / "crs_cache").mkdir(parents=True, exist_ok=True)

        # Copy default turbine for wind materialization
        _copy_default_turbine(path)

    def get_nuts_area(self, area: str):
        """Return a GeoDataFrame for the specified NUTS area."""
        nuts_dir = self.path / "data" / "nuts"
        if not nuts_dir.exists():
            raise FileNotFoundError(f"NUTS shapefile not found under {nuts_dir}")

        shp_files = list(nuts_dir.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"NUTS shapefile not found under {nuts_dir}")

        gdf = gpd.read_file(shp_files[0])
        # Filter by NAME_LATN matching area
        region_gdf = gdf[gdf["NAME_LATN"] == area]
        if region_gdf.empty:
            raise ValueError(f"Region {area} not found in NUTS shapefile")
        return region_gdf


def _create_gwa_raster(
    path: Path,
    shape: tuple[int, int] = (20, 20),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (4000000, 2600000, 4100000, 2700000),
    fill_value: float | None = None,
    add_nodata_region: bool = True,
) -> None:
    """Create a minimal GWA-style GeoTIFF with CRS.

    Args:
        path: Path to write the raster.
        shape: (height, width) of the raster.
        crs_epsg: EPSG code for CRS.
        bounds: (minx, miny, maxx, maxy) bounds.
        fill_value: If set, fill with this value; otherwise random.
        add_nodata_region: If True, add some nodata cells in corners.
    """
    if fill_value is not None:
        data = np.full(shape, fill_value, dtype=np.float32)
    else:
        data = np.random.rand(*shape).astype(np.float32) * 10 + 1

    # Add nodata area in upper-left corner to test mask
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

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_elevation_raster(
    path: Path,
    shape: tuple[int, int] = (25, 25),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (3990000, 2590000, 4110000, 2710000),
) -> None:
    """Create a legacy elevation GeoTIFF.

    Creates a raster slightly larger than the GWA bounds to test clipping.
    """
    # Elevation values between 0 and 3000m
    data = np.random.rand(*shape).astype(np.float32) * 3000

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

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_all_gwa_rasters(atlas_path: Path, country: str = "AUT") -> None:
    """Create all required GWA rasters for wind materialization."""
    raw_dir = atlas_path / "data" / "raw" / country

    for h in GWA_HEIGHTS:
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-A_{h}.tif",
            fill_value=8.0 + h * 0.01,  # Different values per height
        )
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-k_{h}.tif",
            fill_value=2.0 + h * 0.001,
        )
        _create_gwa_raster(
            raw_dir / f"{country}_air-density_{h}.tif",
            fill_value=1.2 - h * 0.0001,
        )


def _create_nuts_shapefile(
    nuts_dir: Path,
    region_name: str,
    bounds: tuple[float, float, float, float],
    crs_epsg: int = 3035,
) -> None:
    """Create a minimal NUTS shapefile with one area.

    Args:
        nuts_dir: Directory to write the shapefile.
        region_name: Latin name for the area.
        bounds: (minx, miny, maxx, maxy) for the area polygon.
        crs_epsg: CRS EPSG code.
    """
    nuts_dir.mkdir(parents=True, exist_ok=True)

    # Create a polygon covering part of the wind grid
    polygon = box(*bounds)

    gdf = gpd.GeoDataFrame(
        {
            "NAME_LATN": [region_name],
            "CNTR_CODE": ["AT"],  # Austria
            "LEVL_CODE": [2],
        },
        geometry=[polygon],
        crs=f"EPSG:{crs_epsg}",
    )

    gdf.to_file(nuts_dir / "test_nuts.shp")


class TestMaterializeLandscapeZarr:
    """Integration tests for Unifier.materialize_landscape."""

    def test_creates_landscape_zarr_complete(self, tmp_path: Path) -> None:
        """materialize_landscape creates landscape.zarr with store_state=complete."""
        atlas = MockAtlas(tmp_path)

        # Create GWA rasters
        _create_all_gwa_rasters(tmp_path)

        # Create elevation raster
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        # Materialize wind first
        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)

        # Then materialize landscape
        unifier.materialize_landscape(atlas)

        # Check landscape.zarr exists and is complete
        landscape_path = tmp_path / "landscape.zarr"
        assert landscape_path.exists()

        root = zarr.open_group(landscape_path, mode="r")
        assert root.attrs["store_state"] == "complete"

    def test_landscape_grid_aligned_to_wind(self, tmp_path: Path) -> None:
        """landscape.zarr y/x coords match wind.zarr exactly."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Open both stores
        wind = xr.open_zarr(tmp_path / "wind.zarr", consolidated=False)
        landscape = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)

        # Check grid_id matches
        wind_root = zarr.open_group(tmp_path / "wind.zarr", mode="r")
        landscape_root = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        assert landscape_root.attrs["grid_id"] == wind_root.attrs["grid_id"]

        # Check y/x coords are identical
        np.testing.assert_array_equal(landscape["y"].values, wind["y"].values)
        np.testing.assert_array_equal(landscape["x"].values, wind["x"].values)

    def test_landscape_has_valid_mask_and_elevation(self, tmp_path: Path) -> None:
        """landscape.zarr contains valid_mask (bool) and elevation variables."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        ds = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)

        # Check variables exist
        assert "valid_mask" in ds.data_vars
        assert "elevation" in ds.data_vars

        # Check valid_mask is boolean
        assert ds["valid_mask"].dtype == np.dtype("bool")

        # Check dimensions
        assert set(ds["valid_mask"].dims) == {"y", "x"}
        assert set(ds["elevation"].dims) == {"y", "x"}

    def test_elevation_nan_where_valid_mask_false(self, tmp_path: Path) -> None:
        """Elevation values are NaN where valid_mask is False."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        ds = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)

        # Load data for inspection
        valid_mask = ds["valid_mask"].values
        elevation = ds["elevation"].values

        # Where valid_mask is False, elevation should be NaN
        invalid_mask = ~valid_mask
        assert np.all(np.isnan(elevation[invalid_mask])), "Elevation should be NaN where valid_mask is False"

    def test_manifest_contains_sources_and_variables(self, tmp_path: Path) -> None:
        """Manifest sources and variables exist with correct entries."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        landscape_path = tmp_path / "landscape.zarr"

        # Check sources
        root = zarr.open_group(landscape_path, mode="r")
        sources = json.loads(root.attrs.get("cleo_manifest_sources_json", "[]"))
        source_ids = [s["source_id"] for s in sources]
        assert "mask:derived_from_wind" in source_ids
        assert "elevation:local" in source_ids

        # Check variables
        variables = json.loads(root.attrs.get("cleo_manifest_variables_json", "[]"))
        var_names = [v["variable_name"] for v in variables]
        assert "valid_mask" in var_names
        assert "elevation" in var_names

    def test_idempotent_materialize(self, tmp_path: Path) -> None:
        """Calling materialize_landscape twice doesn't recreate store if unchanged."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Get modification time
        landscape_path = tmp_path / "landscape.zarr"
        mtime1 = landscape_path.stat().st_mtime

        # Call again
        unifier.materialize_landscape(atlas)

        # Should not be recreated (same mtime)
        mtime2 = landscape_path.stat().st_mtime
        assert mtime1 == mtime2

    def test_fails_if_wind_not_complete(self, tmp_path: Path) -> None:
        """materialize_landscape raises if wind.zarr is not complete."""
        atlas = MockAtlas(tmp_path)

        # Create skeleton wind store (not complete)
        wind_path = tmp_path / "wind.zarr"
        root = zarr.open_group(wind_path, mode="w")
        root.attrs["store_state"] = "skeleton"

        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})

        with pytest.raises(RuntimeError, match="wind.zarr is not complete"):
            unifier.materialize_landscape(atlas)

    def test_area_selection_does_not_change_base_stores(self, tmp_path: Path) -> None:
        """Base stores are country-wide: a selected area neither clips them nor changes their identity."""
        # Area first, on a clean workdir, so a clipped first build cannot hide behind an earlier country build.
        atlases = (MockAtlas(tmp_path, area="TestRegion"), MockAtlas(tmp_path))
        _create_nuts_shapefile(
            tmp_path / "data" / "nuts",
            region_name="TestRegion",
            bounds=(4020000, 2620000, 4080000, 2680000),
        )
        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        ids = []
        for atlas in atlases:
            unifier.materialize_wind(atlas)
            unifier.materialize_landscape(atlas)
            if atlas.area is not None:
                land = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
                x, y = land["x"], land["y"]
                in_area_box = (x >= 4020000) & (x <= 4080000) & (y >= 2620000) & (y <= 2680000)
                outside = land["valid_mask"] & ~in_area_box
                assert bool(outside.any())
                assert bool(land["elevation"].where(outside).notnull().sum() == outside.sum())
            ids.append(
                tuple(
                    zarr.open_group(tmp_path / f"{n}.zarr", mode="r").attrs["inputs_id"] for n in ("wind", "landscape")
                )
            )

        assert ids[0] == ids[1]

    def test_crs_and_transform_from_wind(self, tmp_path: Path) -> None:
        """landscape.zarr has y/x coords matching wind reference."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Open datasets
        wind = xr.open_zarr(tmp_path / "wind.zarr", consolidated=False)
        landscape = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)

        # Verify y/x coords are identical (grid alignment)
        np.testing.assert_array_equal(landscape["y"].values, wind["y"].values)
        np.testing.assert_array_equal(landscape["x"].values, wind["x"].values)

        # grid_id should match between stores (indicates same spatial reference)
        wind_root = zarr.open_group(tmp_path / "wind.zarr", mode="r")
        landscape_root = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        assert landscape_root.attrs["grid_id"] == wind_root.attrs["grid_id"]


class TestLandscapeWarmBuild:
    """A complete, up-to-date landscape store is reused without reading elevation values again."""

    @staticmethod
    def _inputs_id(path: Path) -> str:
        return zarr.open_group(path / "landscape.zarr", mode="r").attrs["inputs_id"]

    def test_local_elevation_warm_build_and_workdir_spelling(self, tmp_path: Path, monkeypatch) -> None:
        from cleo.unification.materializers import _landscape_core

        workdir = tmp_path / "ws"
        atlas = MockAtlas(workdir)
        _create_all_gwa_rasters(workdir)
        _create_elevation_raster(workdir / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif")
        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)
        built_id = self._inputs_id(workdir)

        def _no_read(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            raise AssertionError("elevation values read although the store is up to date")

        monkeypatch.setattr(_landscape_core, "_open_local_elevation", _no_read)
        unifier.materialize_landscape(atlas)
        # The same workdir under another spelling (symlink) has the same identity.
        (tmp_path / "link").symlink_to(workdir, target_is_directory=True)
        unifier.materialize_landscape(MockAtlas(tmp_path / "link"))
        assert self._inputs_id(workdir) == built_id

    def test_copdem_warm_build_skips_mosaic(self, tmp_path: Path, monkeypatch) -> None:
        from cleo.unification.materializers import _landscape_core

        atlas = MockAtlas(tmp_path)
        _create_all_gwa_rasters(tmp_path)
        unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
        unifier.materialize_wind(atlas)

        meta = {"provider": "copernicus", "version": "GLO-30", "bbox_4326": [9.0, 46.0, 10.0, 47.0], "clip": "none"}
        tiles = [tmp_path / "N46_00_E009_00.tif"]
        monkeypatch.setattr(
            _landscape_core, "_resolve_copdem_tiles", lambda *a: (tiles, {**meta, "tile_ids": [t.stem for t in tiles]})
        )
        mosaics: list[int] = []

        def _mosaic(reference, tile_paths):  # noqa: ANN001, ANN202
            mosaics.append(len(tile_paths))
            return xr.full_like(reference, 500.0)

        monkeypatch.setattr(_landscape_core, "build_copdem_elevation_like", _mosaic)
        unifier.materialize_landscape(atlas)
        built_id = self._inputs_id(tmp_path)
        unifier.materialize_landscape(atlas)
        assert mosaics == [1] and self._inputs_id(tmp_path) == built_id

        # A change in the available tiles is a real input change: rebuild.
        tiles.append(tmp_path / "N46_00_E010_00.tif")
        unifier.materialize_landscape(atlas)
        assert mosaics == [1, 2] and self._inputs_id(tmp_path) != built_id


class TestBuildCanonical:
    """Tests for Atlas.build_canonical method via integration."""

    def test_build_canonical_creates_both_stores(self, tmp_path: Path) -> None:
        """build_canonical creates both wind.zarr and landscape.zarr."""
        # Use a minimal atlas-like object that has the required interface
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        # Call build_canonical logic directly via Unifier
        unifier = Unifier(
            chunk_policy=atlas.chunk_policy,
            fingerprint_method=atlas.fingerprint_method,
        )
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Both stores should be complete
        wind_root = zarr.open_group(tmp_path / "wind.zarr", mode="r")
        landscape_root = zarr.open_group(tmp_path / "landscape.zarr", mode="r")

        assert wind_root.attrs["store_state"] == "complete"
        assert landscape_root.attrs["store_state"] == "complete"


class TestZarrV3Compatibility:
    """Tests for Zarr v3 compatibility (no string arrays, no consolidated metadata)."""

    def test_landscape_store_no_string_arrays(self, tmp_path: Path) -> None:
        """landscape.zarr must not contain string/unicode/object dtype arrays.

        Regression test for Zarr v3 compatibility. String arrays trigger
        UnstableSpecificationWarning because they have no stable Zarr v3 spec.
        """
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        ds = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)

        # Check all data variables
        for name, var in ds.data_vars.items():
            dtype_kind = getattr(var.dtype, "kind", None)
            assert dtype_kind not in (
                "U",
                "S",
                "O",
            ), f"Data variable {name!r} has string/object dtype {var.dtype}, which is not Zarr v3 compatible."

        # Check all coordinates (except spatial_ref which is special)
        for name, coord in ds.coords.items():
            if name == "spatial_ref":
                continue
            dtype_kind = getattr(coord.dtype, "kind", None)
            assert dtype_kind not in (
                "U",
                "S",
                "O",
            ), f"Coordinate {name!r} has string/object dtype {coord.dtype}, which is not Zarr v3 compatible."

    def test_landscape_store_no_consolidated_metadata(self, tmp_path: Path) -> None:
        """landscape.zarr must be readable with consolidated=False."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)
        unifier.materialize_landscape(atlas)

        # Should open successfully without consolidated metadata
        ds = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        assert "valid_mask" in ds.data_vars
        assert "elevation" in ds.data_vars
