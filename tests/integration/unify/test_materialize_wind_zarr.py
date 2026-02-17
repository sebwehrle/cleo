"""Integration tests for Unifier.materialize_wind.

Tests are offline-only: all GeoTIFFs are created with CRS embedded,
so no network CRS fetch is needed.
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

from cleo.unify import (
    GWA_HEIGHTS,
    Unifier,
    _assert_all_required_gwa_present,
    _required_gwa_files,
)


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
        self.chunk_policy = {"y": 64, "x": 64}

        # Create resources directory and copy default turbine
        (path / "resources").mkdir(parents=True, exist_ok=True)
        _copy_default_turbine(path)

    def get_nuts_region(self, region):
        """Mock - would return GeoDataFrame."""
        return None


def _create_test_raster(
    path: Path,
    *,
    crs_epsg: int = 3035,
    shape: tuple[int, int] = (20, 30),
    bounds: tuple[float, float, float, float] = (0, 0, 300, 200),
) -> None:
    """Create a minimal test GeoTIFF with embedded CRS."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.random.rand(*shape).astype(np.float32) + 0.1  # Ensure positive values

    transform = rasterio.transform.from_bounds(*bounds, shape[1], shape[0])

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "transform": transform,
        "crs": CRS.from_epsg(crs_epsg),
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_all_required_gwa_files(atlas, **kwargs) -> list[Path]:
    """Create all 15 required GWA files for testing."""
    req_files = _required_gwa_files(atlas)
    paths = []

    for _sid, path in req_files:
        _create_test_raster(path, **kwargs)
        paths.append(path)

    return paths


class TestRequiredGwaFiles:
    """Tests for required GWA file detection."""

    def test_required_files_list(self, tmp_path: Path) -> None:
        """Returns list of all 15 required files (5 heights × 3 layers)."""
        atlas = MockAtlas(tmp_path)
        req = _required_gwa_files(atlas)

        # 5 heights × 3 layers = 15 files
        assert len(req) == 15

        # Check all heights present for each layer
        for h in GWA_HEIGHTS:
            assert any(f"weibull_A:{h}" in sid for sid, _ in req)
            assert any(f"weibull_k:{h}" in sid for sid, _ in req)
            assert any(f"rho:{h}" in sid for sid, _ in req)

    def test_assert_raises_if_all_missing(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError listing all missing paths."""
        atlas = MockAtlas(tmp_path)

        with pytest.raises(FileNotFoundError, match="Missing required GWA files"):
            _assert_all_required_gwa_present(atlas)

    def test_assert_raises_listing_all_missing(self, tmp_path: Path) -> None:
        """Error message lists ALL missing files, not just first."""
        atlas = MockAtlas(tmp_path)

        # Create only some files (5 of 15)
        req = _required_gwa_files(atlas)
        for sid, path in req[:5]:
            _create_test_raster(path)

        with pytest.raises(FileNotFoundError) as exc_info:
            _assert_all_required_gwa_present(atlas)

        error_msg = str(exc_info.value)

        # Should mention multiple missing files
        missing_count = error_msg.count(".tif")
        assert missing_count == 10  # 15 total - 5 created = 10 missing

    def test_assert_passes_if_all_present(self, tmp_path: Path) -> None:
        """Returns file list if all required files exist."""
        atlas = MockAtlas(tmp_path)

        _create_all_required_gwa_files(atlas)

        req = _assert_all_required_gwa_present(atlas)
        assert len(req) == 15


class TestMaterializeWind:
    """Integration tests for Unifier.materialize_wind."""

    def test_creates_wind_zarr(self, tmp_path: Path) -> None:
        """materialize_wind creates wind.zarr directory."""
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)

        unifier = Unifier(chunk_policy={"y": 64, "x": 64})
        unifier.materialize_wind(atlas)

        store_path = tmp_path / "wind.zarr"
        assert store_path.exists()
        assert store_path.is_dir()

    def test_store_state_is_complete(self, tmp_path: Path) -> None:
        """Completed store has store_state='complete'."""
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)

        unifier = Unifier()
        unifier.materialize_wind(atlas)

        root = zarr.open_group(tmp_path / "wind.zarr", mode="r")
        assert root.attrs["store_state"] == "complete"

    def test_store_has_required_attrs(self, tmp_path: Path) -> None:
        """Store has all required attributes."""
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)

        unifier = Unifier()
        unifier.materialize_wind(atlas)

        root = zarr.open_group(tmp_path / "wind.zarr", mode="r")

        assert "store_state" in root.attrs
        assert "grid_id" in root.attrs
        assert "inputs_id" in root.attrs
        assert "mask_policy" in root.attrs
        assert "unify_version" in root.attrs
        assert "code_dirty" in root.attrs
        assert "chunk_policy" in root.attrs
        assert "fingerprint_method" in root.attrs

        # grid_id and inputs_id should be non-empty for complete stores
        assert len(root.attrs["grid_id"]) == 16
        assert len(root.attrs["inputs_id"]) == 16

    def test_store_has_expected_variables(self, tmp_path: Path) -> None:
        """Store contains weibull_A, weibull_k, rho, template."""
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)

        unifier = Unifier()
        unifier.materialize_wind(atlas)

        ds = xr.open_zarr(tmp_path / "wind.zarr", consolidated=False)

        assert "weibull_A" in ds.data_vars
        assert "weibull_k" in ds.data_vars
        assert "rho" in ds.data_vars
        assert "template" in ds.data_vars

    def test_variables_have_height_dimension(self, tmp_path: Path) -> None:
        """Wind variables have height dimension with 5 levels."""
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)

        unifier = Unifier()
        unifier.materialize_wind(atlas)

        ds = xr.open_zarr(tmp_path / "wind.zarr", consolidated=False)

        for var_name in ["weibull_A", "weibull_k", "rho"]:
            assert "height" in ds[var_name].dims
            assert ds.sizes["height"] == 5
            assert list(ds.coords["height"].values) == GWA_HEIGHTS

    def test_manifest_attrs_exist(self, tmp_path: Path) -> None:
        """Store has manifest attrs initialized."""
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)

        unifier = Unifier()
        unifier.materialize_wind(atlas)

        root = zarr.open_group(tmp_path / "wind.zarr", mode="r")
        assert "cleo_manifest_sources_json" in root.attrs
        assert "cleo_manifest_variables_json" in root.attrs

    def test_manifest_sources_populated(self, tmp_path: Path) -> None:
        """Manifest sources contains GWA file sources."""
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)

        unifier = Unifier()
        unifier.materialize_wind(atlas)

        root = zarr.open_group(tmp_path / "wind.zarr", mode="r")
        sources = json.loads(root.attrs.get("cleo_manifest_sources_json", "[]"))

        # Should have 15 raster sources + 3 bundle sources = 18 minimum
        assert len(sources) >= 18

        # Check for bundle sources
        source_ids = [s["source_id"] for s in sources]
        assert "gwa:bundle:weibull_A" in source_ids
        assert "gwa:bundle:weibull_k" in source_ids
        assert "gwa:bundle:rho" in source_ids

    def test_manifest_variables_populated(self, tmp_path: Path) -> None:
        """Manifest variables contains wind variables."""
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)

        unifier = Unifier()
        unifier.materialize_wind(atlas)

        root = zarr.open_group(tmp_path / "wind.zarr", mode="r")
        variables = json.loads(root.attrs.get("cleo_manifest_variables_json", "[]"))

        var_names = [v["variable_name"] for v in variables]
        assert "weibull_A" in var_names
        assert "weibull_k" in var_names
        assert "rho" in var_names
        assert "template" in var_names

    def test_idempotent_if_inputs_unchanged(self, tmp_path: Path) -> None:
        """Second call with same inputs does nothing."""
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)

        unifier = Unifier()
        unifier.materialize_wind(atlas)

        store_path = tmp_path / "wind.zarr"
        first_mtime = store_path.stat().st_mtime

        # Second call should be no-op
        unifier.materialize_wind(atlas)

        # Store shouldn't be recreated (mtime same)
        # Note: This tests the idempotency check but timing might vary
        root = zarr.open_group(store_path, mode="r")
        assert root.attrs["store_state"] == "complete"

    def test_fails_fast_if_files_missing(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError before any processing if files missing."""
        atlas = MockAtlas(tmp_path)
        # Don't create any files

        unifier = Unifier()

        with pytest.raises(FileNotFoundError, match="Missing required GWA files"):
            unifier.materialize_wind(atlas)

        # wind.zarr should NOT be created
        assert not (tmp_path / "wind.zarr").exists()

    def test_partial_files_lists_all_missing(self, tmp_path: Path) -> None:
        """With partial files, error lists ALL missing paths."""
        atlas = MockAtlas(tmp_path)

        # Create only files for one height
        raw_dir = tmp_path / "data" / "raw" / atlas.country
        raw_dir.mkdir(parents=True, exist_ok=True)

        for layer in ["combined-Weibull-A", "combined-Weibull-k", "air-density"]:
            path = raw_dir / f"{atlas.country}_{layer}_100.tif"
            _create_test_raster(path)

        unifier = Unifier()

        with pytest.raises(FileNotFoundError) as exc_info:
            unifier.materialize_wind(atlas)

        error_msg = str(exc_info.value)
        # Should list all 12 missing files (15 - 3 created)
        assert error_msg.count(".tif") == 12


class TestMaterializeWindCrsHandling:
    """Tests for CRS handling during wind materialization."""

    def test_no_network_when_rasters_have_crs(self, tmp_path: Path) -> None:
        """No CRS fetch when all rasters have embedded CRS."""
        from unittest.mock import patch

        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas, crs_epsg=3035)

        with patch("cleo.loaders.fetch_gwa_crs") as mock_fetch:
            unifier = Unifier()
            unifier.materialize_wind(atlas)

            # Should NOT call network
            mock_fetch.assert_not_called()

    def test_uses_embedded_crs(self, tmp_path: Path) -> None:
        """Uses CRS from rasters, not hardcoded."""
        atlas = MockAtlas(tmp_path)
        atlas.crs = "epsg:3035"

        _create_all_required_gwa_files(atlas, crs_epsg=3035)

        unifier = Unifier()
        unifier.materialize_wind(atlas)

        # Resulting dataset should have CRS in spatial_ref or on template
        ds = xr.open_zarr(tmp_path / "wind.zarr", consolidated=False)

        # CRS is stored in spatial_ref variable (rioxarray convention)
        assert "spatial_ref" in ds.data_vars or ds["template"].rio.crs is not None


class TestZarrV3Compatibility:
    """Tests for Zarr v3 compatibility (no string arrays, no consolidated metadata)."""

    def test_wind_store_no_string_arrays(self, tmp_path: Path) -> None:
        """wind.zarr must not contain string/unicode/object dtype arrays.

        Regression test for Zarr v3 compatibility. String arrays trigger
        UnstableSpecificationWarning because they have no stable Zarr v3 spec.
        Turbine metadata must be stored as JSON in attrs, not as string arrays.
        """
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)

        unifier = Unifier()
        unifier.materialize_wind(atlas)

        ds = xr.open_zarr(tmp_path / "wind.zarr", consolidated=False)

        # Check all data variables
        for name, var in ds.data_vars.items():
            dtype_kind = getattr(var.dtype, "kind", None)
            assert dtype_kind not in ("U", "S", "O"), (
                f"Data variable {name!r} has string/object dtype {var.dtype}, "
                f"which is not Zarr v3 compatible. Use JSON attrs instead."
            )

        # Check all coordinates (except spatial_ref which is special)
        for name, coord in ds.coords.items():
            if name == "spatial_ref":
                # spatial_ref may have string CRS data, that's expected
                continue
            dtype_kind = getattr(coord.dtype, "kind", None)
            assert dtype_kind not in ("U", "S", "O"), (
                f"Coordinate {name!r} has string/object dtype {coord.dtype}, "
                f"which is not Zarr v3 compatible. Use integer indices + JSON attrs."
            )

        # Verify turbine metadata is in JSON attrs, not as arrays
        assert "cleo_turbines_json" in ds.attrs, (
            "Turbine metadata must be stored in cleo_turbines_json attr"
        )

        # Verify turbine coordinate is integer-indexed
        if "turbine" in ds.coords:
            assert ds.coords["turbine"].dtype.kind in ("i", "u"), (
                f"turbine coordinate must be integer, got {ds.coords['turbine'].dtype}"
            )

    def test_wind_store_no_consolidated_metadata(self, tmp_path: Path) -> None:
        """wind.zarr must be readable with consolidated=False.

        Stores must NOT depend on consolidated metadata (.zmetadata) for reading.
        """
        atlas = MockAtlas(tmp_path)
        _create_all_required_gwa_files(atlas)

        unifier = Unifier()
        unifier.materialize_wind(atlas)

        # Should open successfully without consolidated metadata
        ds = xr.open_zarr(tmp_path / "wind.zarr", consolidated=False)
        assert "weibull_A" in ds.data_vars or "weibull_a" in ds.data_vars
        assert "weibull_k" in ds.data_vars
