"""Test that LandscapeAtlas.rasterize accepts string/Path to shapefile."""

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from unittest.mock import MagicMock, patch


def _create_mock_landscape_data():
    """Create minimal xarray Dataset with template for rasterize."""
    x = np.arange(5, dtype=float)
    y = np.arange(5, dtype=float)
    ds = xr.Dataset(coords={"x": x, "y": y})
    ds["template"] = xr.DataArray(
        np.zeros((5, 5)),
        dims=["y", "x"],
        coords={"y": y, "x": x},
    )
    ds = ds.rio.write_crs("EPSG:4326")
    ds["template"] = ds["template"].rio.write_crs("EPSG:4326")
    ds.attrs["country"] = "AUT"
    return ds


def test_rasterize_accepts_path_string(tmp_path):
    """rasterize() should accept a path string and load the shapefile."""
    # Create a simple GeoJSON file
    gdf = gpd.GeoDataFrame(
        {"value": [42]},
        geometry=[box(1, 1, 3, 3)],
        crs="EPSG:4326",
    )
    shape_path = tmp_path / "test_shape.geojson"
    gdf.to_file(shape_path, driver="GeoJSON")

    # Create mock LandscapeAtlas
    with patch("cleo.classes._LandscapeAtlas.__init__", return_value=None):
        from cleo.classes import _LandscapeAtlas

        landscape = _LandscapeAtlas.__new__(_LandscapeAtlas)
        landscape.data = _create_mock_landscape_data()

        # Call rasterize with path string
        landscape.rasterize(str(shape_path), column="value", name="r_test")

        # Assert no exception and r_test exists in data
        assert "r_test" in landscape.data.data_vars, "r_test should be added to landscape.data"


def test_rasterize_accepts_path_object(tmp_path):
    """rasterize() should accept a Path object and load the shapefile."""
    # Create a simple GeoJSON file
    gdf = gpd.GeoDataFrame(
        {"value": [100]},
        geometry=[box(0, 0, 2, 2)],
        crs="EPSG:4326",
    )
    shape_path = tmp_path / "test_shape.geojson"
    gdf.to_file(shape_path, driver="GeoJSON")

    # Create mock LandscapeAtlas
    with patch("cleo.classes._LandscapeAtlas.__init__", return_value=None):
        from cleo.classes import _LandscapeAtlas

        landscape = _LandscapeAtlas.__new__(_LandscapeAtlas)
        landscape.data = _create_mock_landscape_data()

        # Call rasterize with Path object (not string)
        landscape.rasterize(shape_path, column="value", name="path_test")

        assert "path_test" in landscape.data.data_vars


def test_rasterize_error_message_for_missing_column(tmp_path):
    """rasterize() should give clear error when column is missing."""
    import pytest

    gdf = gpd.GeoDataFrame(
        {"existing_col": [1]},
        geometry=[box(1, 1, 2, 2)],
        crs="EPSG:4326",
    )
    shape_path = tmp_path / "test_shape.geojson"
    gdf.to_file(shape_path, driver="GeoJSON")

    with patch("cleo.classes._LandscapeAtlas.__init__", return_value=None):
        from cleo.classes import _LandscapeAtlas

        landscape = _LandscapeAtlas.__new__(_LandscapeAtlas)
        landscape.data = _create_mock_landscape_data()

        with pytest.raises(ValueError) as exc_info:
            landscape.rasterize(str(shape_path), column="nonexistent", name="test")

        # Error message should mention the missing column and available columns
        assert "nonexistent" in str(exc_info.value)
        assert "existing_col" in str(exc_info.value)
