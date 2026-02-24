"""copdem: test_elevation_and_mosaic.
Merged test file (imports preserved per chunk).
"""

from tests.helpers.optional import requires_rioxarray, requires_rasterio

requires_rioxarray()
requires_rasterio()

import pytest
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
from pathlib import Path
from rasterio.transform import from_bounds, from_origin
from cleo.unification.raster_io import build_copdem_elevation_like
from cleo.loaders import load_elevation


def test_build_copdem_elevation_like(tmp_path):
    """
    Test mosaicking two tiles and reprojecting to match a reference raster.

    Create two synthetic 2x2 tiles:
        Tile1: lon [0,2], lat [0,2], fill value 10
        Tile2: lon [2,4], lat [0,2], fill value 20

    Reference raster: lon [0,4], lat [0,2], shape 2x4

    Expected output: shape (2,4), left half=10, right half=20
    """
    # Create tile 1: lon [0,2], lat [0,2], 2x2 pixels, value=10
    tile1_path = tmp_path / "tile1.tif"
    tile1_data = np.full((2, 2), 10, dtype=np.float32)
    tile1_transform = from_bounds(0, 0, 2, 2, 2, 2)  # (west, south, east, north, width, height)

    with rasterio.open(
        tile1_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=tile1_data.dtype,
        crs="EPSG:4326",
        transform=tile1_transform,
    ) as dst:
        dst.write(tile1_data, 1)

    # Create tile 2: lon [2,4], lat [0,2], 2x2 pixels, value=20
    tile2_path = tmp_path / "tile2.tif"
    tile2_data = np.full((2, 2), 20, dtype=np.float32)
    tile2_transform = from_bounds(2, 0, 4, 2, 2, 2)

    with rasterio.open(
        tile2_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=tile2_data.dtype,
        crs="EPSG:4326",
        transform=tile2_transform,
    ) as dst:
        dst.write(tile2_data, 1)

    # Create reference raster: lon [0,4], lat [0,2], 2x4 pixels
    ref_path = tmp_path / "reference.tif"
    ref_data = np.zeros((2, 4), dtype=np.float32)
    ref_transform = from_bounds(0, 0, 4, 2, 4, 2)

    with rasterio.open(
        ref_path,
        "w",
        driver="GTiff",
        height=2,
        width=4,
        count=1,
        dtype=ref_data.dtype,
        crs="EPSG:4326",
        transform=ref_transform,
    ) as dst:
        dst.write(ref_data, 1)

    # Open reference as DataArray
    reference_da = rxr.open_rasterio(ref_path).squeeze()

    # Call build_copdem_elevation_like
    result = build_copdem_elevation_like(reference_da, [tile1_path, tile2_path])

    # Assert output shape equals reference shape (2, 4)
    assert result.shape == (2, 4), f"Expected shape (2, 4), got {result.shape}"

    # Assert left half (columns 0,1) are 10, right half (columns 2,3) are 20
    result_values = result.values
    assert np.all(result_values[:, :2] == 10), f"Left half should be 10, got {result_values[:, :2]}"
    assert np.all(result_values[:, 2:] == 20), f"Right half should be 20, got {result_values[:, 2:]}"

    # Assert CRS is set and matches reference
    assert result.rio.crs is not None, "Result should have CRS set"
    assert result.rio.crs == reference_da.rio.crs, "Result CRS should match reference CRS"

    # Assert name is "elevation"
    assert result.name == "elevation", f"Expected name 'elevation', got {result.name}"


def test_build_copdem_elevation_like_empty_tiles():
    """Test that empty tile_paths raises ValueError."""
    import pytest
    import xarray as xr

    # Create a dummy reference DataArray
    reference_da = xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])
    reference_da = reference_da.rio.write_crs("EPSG:4326")

    with pytest.raises(ValueError, match="empty"):
        build_copdem_elevation_like(reference_da, [])



def test_build_copdem_elevation_like_masks_nodata(tmp_path):
    tile_path = tmp_path / "tile.tif"
    arr = np.array([[-9999, 10], [20, 30]], dtype="int16")
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 2, 1, 1),
        "nodata": -9999,
    }
    with rasterio.open(tile_path, "w", **profile) as dst:
        dst.write(arr, 1)

    ref_path = tmp_path / "ref.tif"
    ref_data = np.zeros((2, 2), dtype=np.float32)
    with rasterio.open(
        ref_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=ref_data.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
    ) as dst:
        dst.write(ref_data, 1)

    reference_da = rxr.open_rasterio(ref_path).squeeze()

    out = build_copdem_elevation_like(reference_da, [tile_path])
    assert np.isnan(out.values).any()



def test_load_elevation_uses_air_density_when_reference_lacks_crs(monkeypatch, tmp_path):
    """
    Test that load_elevation uses air-density file as reference when
    the provided reference_da lacks CRS.
    """
    iso3 = "AUT"

    # Create directory structure
    raw_dir = tmp_path / "data" / "raw" / iso3
    raw_dir.mkdir(parents=True)

    # Create air-density file with CRS
    air_density_path = raw_dir / f"{iso3}_air-density_100.tif"
    air_density_data = np.ones((4, 8), dtype=np.float32)
    air_density_transform = from_bounds(9.0, 46.0, 17.0, 49.0, 8, 4)

    with rasterio.open(
        air_density_path,
        "w",
        driver="GTiff",
        height=4,
        width=8,
        count=1,
        dtype=air_density_data.dtype,
        crs="EPSG:4326",
        transform=air_density_transform,
    ) as dst:
        dst.write(air_density_data, 1)

    # Create a reference DataArray WITHOUT CRS
    reference_da_no_crs = xr.DataArray(
        np.zeros((4, 8), dtype=np.float32),
        dims=["y", "x"],
    )

    # Sentinel value to verify correct DataArray is returned
    sentinel_elevation = xr.DataArray(
        np.full((4, 8), 999.0, dtype=np.float32),
        dims=["y", "x"],
        name="elevation",
    )
    sentinel_elevation = sentinel_elevation.rio.write_crs("EPSG:4326")

    # Track calls
    build_calls = []

    def mock_download(base_dir, iso3, *, min_lon, min_lat, max_lon, max_lat, overwrite=False):
        return [Path("/fake/tile1.tif"), Path("/fake/tile2.tif")]

    def mock_build(ref_da, tile_paths):
        # Assert that the reference_da passed to build has CRS
        assert ref_da.rio.crs is not None, "reference_da passed to build should have CRS"
        build_calls.append({"ref_crs": str(ref_da.rio.crs)})
        return sentinel_elevation

    monkeypatch.setattr("cleo.unification.raster_io.download_copdem_tiles_for_bbox", mock_download)
    monkeypatch.setattr("cleo.unification.raster_io.build_copdem_elevation_like", mock_build)

    # Call load_elevation with a reference that has no CRS
    result = load_elevation(tmp_path, iso3, reference_da_no_crs)

    # Assert build was called
    assert len(build_calls) == 1, "build_copdem_elevation_like should be called once"

    # Assert the reference passed to build had CRS
    assert "4326" in build_calls[0]["ref_crs"]

    # Assert result is the sentinel
    assert np.allclose(result.values, 999.0)
    assert result.name == "elevation"


def test_load_elevation_raises_when_no_crs_and_no_air_density(monkeypatch, tmp_path):
    """
    Test that load_elevation raises ValueError when reference_da lacks CRS
    and air-density file does not exist.
    """
    import pytest

    iso3 = "AUT"

    # Create directory structure but NO air-density file
    raw_dir = tmp_path / "data" / "raw" / iso3
    raw_dir.mkdir(parents=True)

    # Create a reference DataArray WITHOUT CRS
    reference_da_no_crs = xr.DataArray(
        np.zeros((4, 8), dtype=np.float32),
        dims=["y", "x"],
    )

    with pytest.raises(ValueError) as exc_info:
        load_elevation(tmp_path, iso3, reference_da_no_crs)

    assert "no CRS" in str(exc_info.value)
    assert "air-density" in str(exc_info.value)



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

    monkeypatch.setattr("cleo.unification.raster_io.download_copdem_tiles_for_bbox", mock_download)
    monkeypatch.setattr("cleo.unification.raster_io.build_copdem_elevation_like", mock_build)

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

    monkeypatch.setattr("cleo.unification.raster_io.download_copdem_tiles_for_bbox", mock_download)
    monkeypatch.setattr("cleo.unification.raster_io.build_copdem_elevation_like", mock_build)

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



def test_load_elevation_legacy_matches_reference_grid(tmp_path, monkeypatch):
    iso3 = "AUT"
    raw_dir = tmp_path / "data" / "raw" / iso3
    raw_dir.mkdir(parents=True, exist_ok=True)

    legacy = raw_dir / f"{iso3}_elevation_w_bathymetry.tif"

    # Legacy raster: 2x2, pixel size 1
    transform = from_origin(0, 2, 1, 1)
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": None,
    }
    arr = np.ones((1, 2, 2), dtype="float32")
    with rasterio.open(legacy, "w", **profile) as dst:
        dst.write(arr)

    # Reference grid: 4x4, pixel size 0.5, same CRS
    ref = xr.DataArray(
        np.ones((4, 4), dtype="float32"),
        dims=("y", "x"),
        coords={"x": [0.25, 0.75, 1.25, 1.75], "y": [1.75, 1.25, 0.75, 0.25]},
    ).rio.write_crs("EPSG:4326")

    # If CopDEM branch is called, fail the test
    def boom(*args, **kwargs):
        raise AssertionError("CopDEM download should not be called when legacy exists")

    monkeypatch.setattr("cleo.unification.raster_io.download_copdem_tiles_for_bbox", boom)

    out = load_elevation(tmp_path, iso3, ref)

    assert out.shape == ref.shape
    assert np.array_equal(out["x"].values, ref["x"].values)
    assert np.array_equal(out["y"].values, ref["y"].values)
    assert out.rio.crs == ref.rio.crs
