"""Test build_copdem_elevation_like mosaic and reproject helper."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import rioxarray as rxr

from cleo.copdem import build_copdem_elevation_like


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
