"""Test elevation source selection with legacy-file preference."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import rioxarray as rxr
import xarray as xr
from pathlib import Path

from cleo.loaders import load_elevation


def test_load_elevation_uses_legacy_when_present(monkeypatch, tmp_path):
    """
    Test that load_elevation uses legacy file when it exists.

    CopDEM functions should NOT be called when legacy file is present.
    """
    iso3 = "AUT"

    # Create directory structure
    raw_dir = tmp_path / "data" / "raw" / iso3
    raw_dir.mkdir(parents=True)

    # Create legacy elevation file
    legacy_path = raw_dir / f"{iso3}_elevation_w_bathymetry.tif"
    legacy_data = np.full((4, 4), 500.0, dtype=np.float32)  # 500m elevation
    legacy_transform = from_bounds(9, 46, 17, 49, 4, 4)

    with rasterio.open(
        legacy_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=legacy_data.dtype,
        crs="EPSG:4326",
        transform=legacy_transform,
    ) as dst:
        dst.write(legacy_data, 1)

    # Create reference raster (A_100-like)
    ref_path = raw_dir / f"{iso3}_combined-Weibull-A_100.tif"
    ref_data = np.ones((4, 4), dtype=np.float32)
    ref_transform = from_bounds(9, 46, 17, 49, 4, 4)

    with rasterio.open(
        ref_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=ref_data.dtype,
        crs="EPSG:4326",
        transform=ref_transform,
    ) as dst:
        dst.write(ref_data, 1)

    reference_da = rxr.open_rasterio(ref_path).squeeze()

    # Track if CopDEM functions are called
    copdem_called = {"download": False, "build": False}

    def mock_download(*args, **kwargs):
        copdem_called["download"] = True
        raise AssertionError("download_copdem_tiles_for_bbox should not be called")

    def mock_build(*args, **kwargs):
        copdem_called["build"] = True
        raise AssertionError("build_copdem_elevation_like should not be called")

    monkeypatch.setattr("cleo.copdem.download_copdem_tiles_for_bbox", mock_download)
    monkeypatch.setattr("cleo.copdem.build_copdem_elevation_like", mock_build)

    # Also mock ensure_crs_from_gwa to avoid network call
    def mock_ensure_crs(ds, iso3):
        return ds

    monkeypatch.setattr("cleo.loaders.ensure_crs_from_gwa", mock_ensure_crs)

    # Call load_elevation
    elevation = load_elevation(tmp_path, iso3, reference_da)

    # Assert CopDEM functions were NOT called
    assert not copdem_called["download"], "download should not be called when legacy exists"
    assert not copdem_called["build"], "build should not be called when legacy exists"

    # Assert elevation was loaded from legacy file (value 500)
    assert elevation.name == "elevation"
    assert np.allclose(elevation.values, 500.0)


def test_load_elevation_uses_copdem_when_legacy_absent(monkeypatch, tmp_path):
    """
    Test that load_elevation uses CopDEM when legacy file is absent.

    CopDEM functions should be called with correct parameters.
    """
    iso3 = "AUT"

    # Create directory structure (but NO legacy file)
    raw_dir = tmp_path / "data" / "raw" / iso3
    raw_dir.mkdir(parents=True)

    # Create reference raster (A_100-like)
    ref_path = raw_dir / f"{iso3}_combined-Weibull-A_100.tif"
    ref_data = np.ones((4, 4), dtype=np.float32)
    ref_transform = from_bounds(9.0, 46.0, 17.0, 49.0, 4, 4)

    with rasterio.open(
        ref_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=ref_data.dtype,
        crs="EPSG:4326",
        transform=ref_transform,
    ) as dst:
        dst.write(ref_data, 1)

    reference_da = rxr.open_rasterio(ref_path).squeeze()

    # Track calls to CopDEM functions
    calls = {"download": None, "build": None}

    def mock_download(base_dir, iso3, *, min_lon, min_lat, max_lon, max_lat, overwrite=False):
        calls["download"] = {
            "base_dir": base_dir,
            "iso3": iso3,
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        }
        # Return fake tile paths
        return [Path("/fake/tile1.tif"), Path("/fake/tile2.tif")]

    def mock_build(reference_da, tile_paths):
        calls["build"] = {
            "tile_paths": tile_paths,
        }
        # Return a DataArray with known elevation values
        result = xr.DataArray(
            np.full((4, 4), 1000.0, dtype=np.float32),
            dims=["y", "x"],
            name="elevation",
        )
        result = result.rio.write_crs("EPSG:4326")
        return result

    monkeypatch.setattr("cleo.copdem.download_copdem_tiles_for_bbox", mock_download)
    monkeypatch.setattr("cleo.copdem.build_copdem_elevation_like", mock_build)

    # Call load_elevation
    elevation = load_elevation(tmp_path, iso3, reference_da)

    # Assert CopDEM download was called with correct bbox from reference
    assert calls["download"] is not None, "download should be called when legacy absent"
    assert calls["download"]["iso3"] == iso3
    # Bounds from reference: (9.0, 46.0, 17.0, 49.0)
    assert calls["download"]["min_lon"] == 9.0
    assert calls["download"]["min_lat"] == 46.0
    assert calls["download"]["max_lon"] == 17.0
    assert calls["download"]["max_lat"] == 49.0

    # Assert build was called with the tile paths
    assert calls["build"] is not None, "build should be called when legacy absent"
    assert len(calls["build"]["tile_paths"]) == 2

    # Assert returned elevation matches mocked DataArray (value 1000)
    assert elevation.name == "elevation"
    assert np.allclose(elevation.values, 1000.0)
