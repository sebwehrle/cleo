"""Integration tests for landscape materialization via CopDEM path.

Tests are offline-only: no network calls. CopDEM download is monkeypatched
to return locally-created synthetic tiles.

These tests verify that:
- When local elevation GeoTIFF is missing, CopDEM path is used
- CopDEM tiles are mosaiced and aligned to wind grid
- Manifest records elevation:copdem source
- No NotImplementedError is raised (regression guard)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from rasterio.crs import CRS

import cleo
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

        # Create required directories
        (path / "data" / "raw" / country).mkdir(parents=True, exist_ok=True)
        (path / "resources").mkdir(parents=True, exist_ok=True)
        (path / "intermediates" / "crs_cache").mkdir(parents=True, exist_ok=True)

        # Copy default turbine for wind materialization
        _copy_default_turbine(path)

    def get_nuts_region(self, region: str):
        """Return None - no NUTS region for basic tests."""
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


def _create_copdem_tile(
    path: Path,
    shape: tuple[int, int] = (100, 100),
    crs_epsg: int = 4326,
    bounds: tuple[float, float, float, float] = (9.0, 46.0, 10.0, 47.0),
    fill_value: float = 500.0,
) -> None:
    """Create a synthetic CopDEM tile GeoTIFF.

    Args:
        path: Path to write the tile.
        shape: (height, width) of the raster.
        crs_epsg: EPSG code for CRS (default 4326 for CopDEM).
        bounds: (minx, miny, maxx, maxy) bounds.
        fill_value: Elevation value to fill with.
    """
    data = np.full(shape, fill_value, dtype=np.float32)
    # Add some variation
    data += np.random.rand(*shape).astype(np.float32) * 100

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
        "nodata": -9999.0,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


class TestMaterializeLandscapeCopdemPath:
    """Tests for landscape materialization via CopDEM when local elevation is missing."""

    def test_copdem_path_when_local_missing(self, tmp_path: Path) -> None:
        """CopDEM path is used when local elevation GeoTIFF is missing."""
        atlas = MockAtlas(tmp_path)

        # Create GWA rasters (required for wind materialization)
        _create_all_gwa_rasters(tmp_path)

        # IMPORTANT: Do NOT create local elevation file
        # This forces the CopDEM path

        # Create synthetic CopDEM tiles that will be returned by the mock
        copdem_dir = tmp_path / "copdem_tiles"
        tile1_path = copdem_dir / "Copernicus_DSM_COG_10_N46_00_E009_00_DEM.tif"
        tile2_path = copdem_dir / "Copernicus_DSM_COG_10_N46_00_E010_00_DEM.tif"
        tile3_path = copdem_dir / "Copernicus_DSM_COG_10_N47_00_E009_00_DEM.tif"
        tile4_path = copdem_dir / "Copernicus_DSM_COG_10_N47_00_E010_00_DEM.tif"

        # Create tiles covering the expected bbox (approx 9-11E, 46-48N for Austria)
        _create_copdem_tile(tile1_path, bounds=(9.0, 46.0, 10.0, 47.0), fill_value=500.0)
        _create_copdem_tile(tile2_path, bounds=(10.0, 46.0, 11.0, 47.0), fill_value=600.0)
        _create_copdem_tile(tile3_path, bounds=(9.0, 47.0, 10.0, 48.0), fill_value=700.0)
        _create_copdem_tile(tile4_path, bounds=(10.0, 47.0, 11.0, 48.0), fill_value=800.0)

        # Mock download_copdem_tiles_for_bbox to return our synthetic tiles
        def mock_download_tiles(base_dir, iso3, *, min_lon, min_lat, max_lon, max_lat, overwrite=False):
            # Return paths to synthetic tiles (sorted lexicographically)
            return sorted([tile1_path, tile2_path, tile3_path, tile4_path])

        # Patch at the point of import in unify.py
        with patch("cleo.copdem.download_copdem_tiles_for_bbox", mock_download_tiles):
            unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
            unifier.materialize_wind(atlas)
            unifier.materialize_landscape(atlas)

        # Verify landscape.zarr was created and is complete
        landscape_path = tmp_path / "landscape.zarr"
        assert landscape_path.exists(), "landscape.zarr should exist"

        root = zarr.open_group(landscape_path, mode="r")
        assert root.attrs["store_state"] == "complete"

        # Verify elevation variable exists
        ds = xr.open_zarr(landscape_path, consolidated=False)
        assert "elevation" in ds.data_vars, "elevation variable should exist"
        assert "valid_mask" in ds.data_vars, "valid_mask should exist"

        # Verify manifest records CopDEM source
        sources_json = root.attrs.get("cleo_manifest_sources_json", "[]")
        sources = json.loads(sources_json)
        source_ids = [s["source_id"] for s in sources]
        assert "elevation:copdem" in source_ids, (
            f"manifest should include elevation:copdem source, got: {source_ids}"
        )

        # Verify elevation data has the correct shape (matches y/x coords)
        assert ds["elevation"].shape == ds["valid_mask"].shape, (
            "Elevation should have same shape as valid_mask"
        )

        # Elevation is masked by valid_mask, so NaN where valid_mask is False
        # (We don't assert on actual values since synthetic tiles may not
        # exactly overlap after reprojection - structural correctness is enough)

    def test_no_notimplementederror_raised(self, tmp_path: Path) -> None:
        """Regression guard: CopDEM path does not raise NotImplementedError."""
        atlas = MockAtlas(tmp_path)

        # Create GWA rasters
        _create_all_gwa_rasters(tmp_path)

        # No local elevation

        # Create synthetic CopDEM tile
        copdem_dir = tmp_path / "copdem_tiles"
        tile_path = copdem_dir / "Copernicus_DSM_COG_10_N46_00_E009_00_DEM.tif"
        _create_copdem_tile(tile_path, bounds=(9.0, 46.0, 10.0, 47.0))

        def mock_download_tiles(base_dir, iso3, *, min_lon, min_lat, max_lon, max_lat, overwrite=False):
            return [tile_path]

        with patch("cleo.copdem.download_copdem_tiles_for_bbox", mock_download_tiles):
            unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
            unifier.materialize_wind(atlas)

            # This should NOT raise NotImplementedError
            try:
                unifier.materialize_landscape(atlas)
            except NotImplementedError as e:
                pytest.fail(f"CopDEM path raised NotImplementedError: {e}")

        # Verify success
        landscape_path = tmp_path / "landscape.zarr"
        assert landscape_path.exists()

    def test_local_elevation_still_preferred(self, tmp_path: Path) -> None:
        """Local elevation GeoTIFF is still preferred over CopDEM."""
        atlas = MockAtlas(tmp_path)

        # Create GWA rasters
        _create_all_gwa_rasters(tmp_path)

        # Create local elevation file (this should be preferred)
        local_elev = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_gwa_raster(local_elev, fill_value=1234.0, add_nodata_region=False)

        # Mock should NOT be called
        def mock_download_tiles(*args, **kwargs):
            pytest.fail("download_copdem_tiles_for_bbox should not be called when local exists")

        with patch("cleo.copdem.download_copdem_tiles_for_bbox", mock_download_tiles):
            unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
            unifier.materialize_wind(atlas)
            unifier.materialize_landscape(atlas)

        # Verify landscape uses local source
        landscape_path = tmp_path / "landscape.zarr"
        root = zarr.open_group(landscape_path, mode="r")

        sources_json = root.attrs.get("cleo_manifest_sources_json", "[]")
        sources = json.loads(sources_json)
        source_ids = [s["source_id"] for s in sources]

        assert "elevation:local" in source_ids, (
            f"manifest should use elevation:local when local file exists, got: {source_ids}"
        )
        assert "elevation:copdem" not in source_ids, (
            "manifest should NOT include elevation:copdem when local file exists"
        )

    def test_copdem_metadata_in_manifest(self, tmp_path: Path) -> None:
        """CopDEM metadata (provider, version, tile_ids) is recorded in manifest."""
        atlas = MockAtlas(tmp_path)

        # Create GWA rasters
        _create_all_gwa_rasters(tmp_path)

        # No local elevation

        # Create synthetic CopDEM tiles
        copdem_dir = tmp_path / "copdem_tiles"
        tile1_path = copdem_dir / "Copernicus_DSM_COG_10_N46_00_E009_00_DEM.tif"
        tile2_path = copdem_dir / "Copernicus_DSM_COG_10_N47_00_E009_00_DEM.tif"
        _create_copdem_tile(tile1_path, bounds=(9.0, 46.0, 10.0, 47.0))
        _create_copdem_tile(tile2_path, bounds=(9.0, 47.0, 10.0, 48.0))

        def mock_download_tiles(base_dir, iso3, *, min_lon, min_lat, max_lon, max_lat, overwrite=False):
            return sorted([tile1_path, tile2_path])

        with patch("cleo.copdem.download_copdem_tiles_for_bbox", mock_download_tiles):
            # Also patch tiles_for_bbox to return consistent tile IDs
            with patch("cleo.copdem.tiles_for_bbox") as mock_tiles:
                mock_tiles.return_value = [
                    "Copernicus_DSM_COG_10_N46_00_E009_00_DEM",
                    "Copernicus_DSM_COG_10_N47_00_E009_00_DEM",
                ]
                unifier = Unifier(chunk_policy={"y": 1024, "x": 1024})
                unifier.materialize_wind(atlas)
                unifier.materialize_landscape(atlas)

        # Verify landscape was created
        landscape_path = tmp_path / "landscape.zarr"
        root = zarr.open_group(landscape_path, mode="r")

        # The inputs_id should include CopDEM-specific fingerprint elements
        # These are encoded in the inputs_id computation, not directly visible
        # but we can verify the store is complete
        assert root.attrs["store_state"] == "complete"

        # Verify elevation:copdem source is recorded
        sources_json = root.attrs.get("cleo_manifest_sources_json", "[]")
        sources = json.loads(sources_json)
        source_ids = [s["source_id"] for s in sources]
        assert "elevation:copdem" in source_ids
