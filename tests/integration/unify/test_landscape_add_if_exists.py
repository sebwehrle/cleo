"""Integration tests for LandscapeDomain.add() with if_exists semantics.

Tests are offline-only: no network calls, uses local test fixtures.

These tests verify:
- if_exists="error" raises ValueError when variable exists
- if_exists="noop" silently skips when variable exists
- if_exists="replace" atomically replaces existing variable data
- consolidated=False reads see new/replaced vars
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
from cleo.domains import LandscapeDomain
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

        # Store paths
        self.wind_store_path = path / "wind.zarr"
        self.landscape_store_path = path / "landscape.zarr"

        # Domain object caches
        self._landscape_domain = None

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

    @property
    def landscape(self) -> LandscapeDomain:
        """Access LandscapeDomain."""
        if self._landscape_domain is None:
            self._landscape_domain = LandscapeDomain(self)
        return self._landscape_domain


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
    """Create an extra layer GeoTIFF with constant values."""
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


class TestIfExistsError:
    """Tests for if_exists='error' (default behavior)."""

    def test_add_new_variable_succeeds(self, tmp_path: Path) -> None:
        """Adding a new variable with if_exists='error' succeeds."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        # Add new variable with default if_exists='error'
        atlas.landscape.add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        # Verify variable exists
        landscape = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        assert "extra_layer" in landscape.data_vars

    def test_add_existing_variable_raises(self, tmp_path: Path) -> None:
        """Adding an existing variable with if_exists='error' raises ValueError."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        # Add variable first time
        atlas.landscape.add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        # Try to add again with if_exists='error' - should raise
        with pytest.raises(ValueError, match="already exists"):
            atlas.landscape.add(
                "extra_layer",
                extra_path,
                kind="raster",
                params={"categorical": False},
                materialize=True,
                if_exists="error",
            )


class TestIfExistsNoop:
    """Tests for if_exists='noop' behavior."""

    def test_add_new_variable_succeeds(self, tmp_path: Path) -> None:
        """Adding a new variable with if_exists='noop' succeeds."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape.add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
            if_exists="noop",
        )

        landscape = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        assert "extra_layer" in landscape.data_vars

    def test_add_existing_variable_skips_silently_with_exact_match(self, tmp_path: Path) -> None:
        """Adding an existing variable with exact same config and if_exists='noop' skips."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        # Add variable first time
        atlas.landscape.add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        # Capture original values and inputs_id
        landscape_before = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        original_values = landscape_before["extra_layer"].values.copy()
        original_inputs_id = landscape_before.attrs["inputs_id"]

        # Add again with if_exists='noop' and SAME source (exact match)
        atlas.landscape.add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
            if_exists="noop",
        )

        # Values and inputs_id should be unchanged
        landscape_after = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        np.testing.assert_array_equal(
            landscape_after["extra_layer"].values,
            original_values,
            err_msg="Variable should not be modified with if_exists='noop'",
        )
        assert landscape_after.attrs["inputs_id"] == original_inputs_id, (
            "inputs_id should not change with if_exists='noop'"
        )


class TestIfExistsReplace:
    """Tests for if_exists='replace' behavior."""

    def test_add_new_variable_succeeds(self, tmp_path: Path) -> None:
        """Adding a new variable with if_exists='replace' succeeds."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape.add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
            if_exists="replace",
        )

        landscape = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        assert "extra_layer" in landscape.data_vars

    def test_replace_existing_variable_updates_data(self, tmp_path: Path) -> None:
        """Replacing an existing variable with if_exists='replace' updates data."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        # Add variable with fill_value=42.0
        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape.add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        # Verify original value
        landscape_before = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        valid_mask = landscape_before["valid_mask"].values
        original_valid = landscape_before["extra_layer"].values[valid_mask]
        assert np.allclose(original_valid, 42.0, rtol=0.1), "Original should be ~42"

        # Create new raster with fill_value=99.0
        extra_path_v2 = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer_v2.tif"
        _create_extra_layer_raster(extra_path_v2, fill_value=99.0)

        # Replace with if_exists='replace'
        atlas.landscape.add(
            "extra_layer",
            extra_path_v2,
            kind="raster",
            params={"categorical": False},
            materialize=True,
            if_exists="replace",
        )

        # Verify new value - re-open with consolidated=False
        landscape_after = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        valid_mask_after = landscape_after["valid_mask"].values
        new_valid = landscape_after["extra_layer"].values[valid_mask_after]
        assert np.allclose(new_valid, 99.0, rtol=0.1), "Replaced should be ~99"

    def test_replace_updates_manifest(self, tmp_path: Path) -> None:
        """Replacing updates manifest sources with new fingerprint."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        # Add variable first time
        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape.add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        # Get original fingerprint
        root_before = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        sources_before = json.loads(root_before.attrs.get("cleo_manifest_sources_json", "[]"))
        source_entry = next(s for s in sources_before if s["source_id"] == "land:raster:extra_layer")
        fingerprint_before = source_entry["fingerprint"]

        # Create new raster (different file, different fingerprint)
        extra_path_v2 = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer_v2.tif"
        _create_extra_layer_raster(extra_path_v2, fill_value=99.0)

        # Replace
        atlas.landscape.add(
            "extra_layer",
            extra_path_v2,
            kind="raster",
            params={"categorical": False},
            materialize=True,
            if_exists="replace",
        )

        # Get new fingerprint
        root_after = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
        sources_after = json.loads(root_after.attrs.get("cleo_manifest_sources_json", "[]"))
        source_entry_after = next(s for s in sources_after if s["source_id"] == "land:raster:extra_layer")
        fingerprint_after = source_entry_after["fingerprint"]

        assert fingerprint_before != fingerprint_after, (
            "Fingerprint should change after replace"
        )


class TestConsolidatedReads:
    """Tests verifying consolidated=False reads see updated variables."""

    def test_consolidated_sees_new_variable(self, tmp_path: Path) -> None:
        """consolidated=False read sees newly added variable."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape.add(
            "extra_layer",
            extra_path,
            kind="raster",
            params={"categorical": False},
            materialize=True,
        )

        # Open with consolidated=False
        landscape = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        assert "extra_layer" in landscape.data_vars

    def test_consolidated_sees_replaced_variable(self, tmp_path: Path) -> None:
        """consolidated=False read sees replaced variable data."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        # Add with fill_value=42
        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        atlas.landscape.add("extra_layer", extra_path, materialize=True)

        # Replace with fill_value=99
        extra_path_v2 = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer_v2.tif"
        _create_extra_layer_raster(extra_path_v2, fill_value=99.0)

        atlas.landscape.add(
            "extra_layer",
            extra_path_v2,
            materialize=True,
            if_exists="replace",
        )

        # Verify with consolidated=False
        landscape = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        valid_mask = landscape["valid_mask"].values
        valid_values = landscape["extra_layer"].values[valid_mask]
        assert np.allclose(valid_values, 99.0, rtol=0.1)


class TestInvalidIfExists:
    """Tests for invalid if_exists values."""

    def test_invalid_if_exists_raises(self, tmp_path: Path) -> None:
        """Invalid if_exists value raises ValueError."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        extra_path = tmp_path / "data" / "raw" / "AUT" / "AUT_extra_layer.tif"
        _create_extra_layer_raster(extra_path, fill_value=42.0)

        with pytest.raises(ValueError, match="if_exists must be one of"):
            atlas.landscape.add(
                "extra_layer",
                extra_path,
                materialize=True,
                if_exists="invalid_option",
            )


class TestExistingVarsUnchanged:
    """Tests verifying existing variables are not modified."""

    def test_replace_does_not_modify_other_vars(self, tmp_path: Path) -> None:
        """Replacing one variable does not modify other variables."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        # Add two variables
        layer1_path = tmp_path / "data" / "raw" / "AUT" / "AUT_layer1.tif"
        _create_extra_layer_raster(layer1_path, fill_value=11.0)

        layer2_path = tmp_path / "data" / "raw" / "AUT" / "AUT_layer2.tif"
        _create_extra_layer_raster(layer2_path, fill_value=22.0)

        atlas.landscape.add("layer1", layer1_path, materialize=True)
        atlas.landscape.add("layer2", layer2_path, materialize=True)

        # Capture layer1 values
        landscape_before = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        layer1_before = landscape_before["layer1"].values.copy()
        elevation_before = landscape_before["elevation"].values.copy()

        # Replace layer2
        layer2_v2_path = tmp_path / "data" / "raw" / "AUT" / "AUT_layer2_v2.tif"
        _create_extra_layer_raster(layer2_v2_path, fill_value=99.0)

        atlas.landscape.add(
            "layer2",
            layer2_v2_path,
            materialize=True,
            if_exists="replace",
        )

        # Verify layer1 and elevation unchanged
        landscape_after = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
        np.testing.assert_array_equal(
            landscape_after["layer1"].values,
            layer1_before,
            err_msg="layer1 should not change when layer2 is replaced",
        )
        np.testing.assert_array_equal(
            landscape_after["elevation"].values,
            elevation_before,
            err_msg="elevation should not change when layer2 is replaced",
        )


class TestNoopExactMatchSemantics:
    """Tests for tightened noop semantics requiring exact match."""

    def test_noop_raises_on_registration_mismatch(self, tmp_path: Path) -> None:
        """noop raises ValueError when registering with different path."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        # Create two different source files
        path_a = tmp_path / "data" / "raw" / "AUT" / "AUT_foo_a.tif"
        _create_extra_layer_raster(path_a, fill_value=42.0)

        path_b = tmp_path / "data" / "raw" / "AUT" / "AUT_foo_b.tif"
        _create_extra_layer_raster(path_b, fill_value=99.0)

        # Register with pathA (materialize=False)
        atlas.landscape.add("foo", path_a, materialize=False)

        # Try to register with pathB using noop - should raise
        with pytest.raises(ValueError, match="if_exists='replace'"):
            atlas.landscape.add("foo", path_b, materialize=False, if_exists="noop")

    def test_noop_repeat_ok_after_replace_and_rejects_old(self, tmp_path: Path) -> None:
        """noop accepts exact match after replace but rejects old config."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.materialize_canonical()

        # Create two different source files
        path_a = tmp_path / "data" / "raw" / "AUT" / "AUT_foo_a.tif"
        _create_extra_layer_raster(path_a, fill_value=42.0)

        path_b = tmp_path / "data" / "raw" / "AUT" / "AUT_foo_b.tif"
        _create_extra_layer_raster(path_b, fill_value=99.0)

        # Add with pathA
        atlas.landscape.add("foo", path_a, materialize=True)
        old_inputs = xr.open_zarr(
            tmp_path / "landscape.zarr", consolidated=False
        ).attrs["inputs_id"]

        # Replace with pathB
        atlas.landscape.add("foo", path_b, materialize=True, if_exists="replace")
        mid_inputs = xr.open_zarr(
            tmp_path / "landscape.zarr", consolidated=False
        ).attrs["inputs_id"]
        assert mid_inputs != old_inputs, "inputs_id should change after replace"

        # noop with pathB should succeed (exact match)
        atlas.landscape.add("foo", path_b, materialize=True, if_exists="noop")
        after_inputs = xr.open_zarr(
            tmp_path / "landscape.zarr", consolidated=False
        ).attrs["inputs_id"]
        assert after_inputs == mid_inputs, "inputs_id should not change after noop"

        # noop with pathA should raise (config mismatch)
        with pytest.raises(ValueError, match="if_exists='replace'"):
            atlas.landscape.add("foo", path_a, materialize=True, if_exists="noop")
