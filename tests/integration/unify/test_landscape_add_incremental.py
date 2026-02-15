"""Integration tests for incremental landscape additions.

Tests are offline-only: no network calls, uses local test fixtures.

These tests verify that atlas.landscape.add() correctly:
- Registers sources in __manifest__
- Materializes aligned raster variables
- Enforces valid_mask semantics
- Does NOT recompute existing variables
- Updates inputs_id deterministically
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from rasterio.crs import CRS

import cleo
import json

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
    """Minimal Atlas-like object for testing."""

    def __init__(
        self,
        path: Path,
        country: str = "AUT",
        crs: str = "epsg:3035",
        region: str | None = None,
    ):
        self.path = path
        self.country = country
        self.crs = crs
        self.region = region
        self.turbines_configured = None
        self.chunk_policy = {"y": 1024, "x": 1024}
        self.fingerprint_method = "path_mtime_size"
        self._canonical_ready = False

        # Create required directories
        (path / "data" / "raw" / country).mkdir(parents=True, exist_ok=True)
        (path / "resources").mkdir(parents=True, exist_ok=True)
        (path / "intermediates" / "crs_cache").mkdir(parents=True, exist_ok=True)

        # Copy default turbine for wind materialization
        _copy_default_turbine(path)

    def get_nuts_region(self, region: str):
        """Return None - no NUTS region for basic tests."""
        return None

    def materialize_canonical(self) -> None:
        """Materialize both wind.zarr and landscape.zarr."""
        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        u.materialize_wind(self)
        u.materialize_landscape(self)
        self._canonical_ready = True

    def landscape_add(
        self,
        name: str,
        source_path: str | Path,
        kind: str = "raster",
        params: dict | None = None,
        materialize: bool = True,
    ) -> None:
        """Add a new variable to the landscape store."""
        if not self._canonical_ready:
            self.materialize_canonical()

        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        u.register_landscape_source(
            self,
            name=name,
            source_path=Path(source_path),
            kind=kind,
            params=params or {},
        )
        if materialize:
            u.materialize_landscape_variable(self, variable_name=name)


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

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_elevation_raster(
    path: Path,
    shape: tuple[int, int] = (25, 25),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (3990000, 2590000, 4110000, 2710000),
) -> None:
    """Create a legacy elevation GeoTIFF."""
    data = np.random.rand(*shape).astype(np.float32) * 3000
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

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_extra_layer_raster(
    path: Path,
    shape: tuple[int, int] = (30, 30),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (3980000, 2580000, 4120000, 2720000),
    fill_value: float = 42.0,
) -> None:
    """Create an extra layer GeoTIFF with constant values.

    Extent is larger than AOI to test clipping.
    """
    data = np.full(shape, fill_value, dtype=np.float32)
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


class TestLandscapeAddIncremental:
    """Integration tests for incremental landscape additions."""

    def test_add_creates_new_variable(self, tmp_path: Path) -> None:
        """landscape.add() creates a new variable in landscape.zarr."""
        atlas = MockAtlas(tmp_path)

        # Create required rasters
        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)

        # Create and materialize canonical stores
        atlas.materialize_canonical()

        # Create extra layer raster
        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        # Add the extra layer
        atlas.landscape_add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        # Re-open landscape with consolidated=False to verify .zmetadata updated
        landscape = xr.open_zarr(
            tmp_path / "landscape.zarr",
            consolidated=False,
            chunks={"y": 1024, "x": 1024},
        )

        assert "extra_layer" in landscape.data_vars

    def test_extra_layer_nan_where_valid_mask_false(self, tmp_path: Path) -> None:
        """Extra layer values are NaN where valid_mask is False."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape_add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        landscape = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)

        valid_mask = landscape["valid_mask"].values
        extra_layer = landscape["extra_layer"].values

        # Where valid_mask is False, extra_layer should be NaN
        invalid_mask = ~valid_mask
        assert np.all(np.isnan(extra_layer[invalid_mask])), (
            "extra_layer should be NaN where valid_mask is False"
        )

    def test_existing_vars_unchanged(self, tmp_path: Path) -> None:
        """Adding a new layer does NOT modify existing variables."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        # Capture existing data before adding new layer
        landscape_before = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        valid_mask_before = landscape_before["valid_mask"].values.copy()
        elevation_before = landscape_before["elevation"].values.copy()

        # Add extra layer
        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape_add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        # Re-open and compare
        landscape_after = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        valid_mask_after = landscape_after["valid_mask"].values
        elevation_after = landscape_after["elevation"].values

        np.testing.assert_array_equal(
            valid_mask_before, valid_mask_after,
            err_msg="valid_mask should not change after adding new layer"
        )
        np.testing.assert_array_equal(
            elevation_before, elevation_after,
            err_msg="elevation should not change after adding new layer"
        )

    def test_manifest_sources_updated(self, tmp_path: Path) -> None:
        """Manifest sources contains new source after landscape.add()."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape_add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        root = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        sources = json.loads(root.attrs.get("cleo_manifest_sources_json", "[]"))
        source_ids = [s["source_id"] for s in sources]

        assert "land:raster:extra_layer" in source_ids

    def test_manifest_variables_updated(self, tmp_path: Path) -> None:
        """Manifest variables maps extra_layer to its source."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape_add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        root = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        variables = json.loads(root.attrs.get("cleo_manifest_variables_json", "[]"))
        var_names = [v["variable_name"] for v in variables]

        assert "extra_layer" in var_names

        var_entry = next(v for v in variables if v["variable_name"] == "extra_layer")
        assert var_entry["source_id"] == "land:raster:extra_layer"

    def test_inputs_id_changes_after_materialize(self, tmp_path: Path) -> None:
        """inputs_id changes deterministically after materialization."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        # Get inputs_id before
        root_before = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        inputs_id_before = root_before.attrs["inputs_id"]

        # Add extra layer
        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape_add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        # Get inputs_id after
        root_after = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        inputs_id_after = root_after.attrs["inputs_id"]

        assert inputs_id_before != inputs_id_after, (
            "inputs_id should change after materialization"
        )

    def test_grid_id_unchanged(self, tmp_path: Path) -> None:
        """grid_id remains unchanged after adding new layer."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        root_before = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        grid_id_before = root_before.attrs["grid_id"]

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape_add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        root_after = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        grid_id_after = root_after.attrs["grid_id"]

        assert grid_id_before == grid_id_after

    def test_register_only_without_materialize(self, tmp_path: Path) -> None:
        """materialize=False only registers source without creating variable."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        # Register only, don't materialize
        atlas.landscape_add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=False,
        )

        # Source should be registered
        root = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        sources = json.loads(root.attrs.get("cleo_manifest_sources_json", "[]"))
        source_ids = [s["source_id"] for s in sources]
        assert "land:raster:extra_layer" in source_ids

        # But variable should NOT exist yet
        landscape = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        assert "extra_layer" not in landscape.data_vars

    def test_invalid_kind_raises(self, tmp_path: Path) -> None:
        """kind != 'raster' raises ValueError in v1."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        with pytest.raises(ValueError, match="Only kind='raster' supported"):
            atlas.landscape_add(
                "extra_layer",
                extra_path,
                kind="vector",  # Invalid in v1
                params={},
                materialize=False,
            )

    def test_categorical_uses_nearest_resampling(self, tmp_path: Path) -> None:
        """categorical=True uses nearest neighbor resampling."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_categorical.tif"
        _create_extra_layer_raster(extra_path, fill_value=5.0)

        atlas.landscape_add(
            "categorical_layer",
            extra_path,
            kind="raster",
            params={"categorical": True},
            materialize=True,
        )

        # Check manifest records nearest resampling
        root = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        variables = json.loads(root.attrs.get("cleo_manifest_variables_json", "[]"))
        var_entry = next(v for v in variables if v["variable_name"] == "categorical_layer")
        resampling = var_entry.get("resampling_method", "")

        assert resampling == "nearest"
