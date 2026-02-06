"""classes: test_rasterize_features.
Merged test file (imports preserved per chunk).
"""

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box, Point
from unittest.mock import patch
from cleo.classes import _LandscapeAtlas

# --- merged from tests/_staging/test_landscape_rasterize_accepts_path.py ---

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


# --- merged from tests/_staging/test_rasterize_features_oracle.py ---

def _make_template():
    # 3x3 grid in EPSG:4326 with pixel boxes:
    # x centers: 0.5,1.5,2.5 ; y centers: 2.5,1.5,0.5 (descending y)
    y = np.array([2.5, 1.5, 0.5])
    x = np.array([0.5, 1.5, 2.5])
    arr = np.full((3, 3), np.nan, dtype=np.float32)
    da = xr.DataArray(arr, dims=("y", "x"), coords={"y": y, "x": x}, name="template")
    da = da.rio.write_crs("EPSG:4326")
    return da


def _oracle_center_mask(template, polys):
    # all_touched=False oracle: pixel center inside polygon => burned
    out = np.zeros(template.shape, dtype=bool)
    ys = template["y"].values
    xs = template["x"].values
    for iy, yy in enumerate(ys):
        for ix, xx in enumerate(xs):
            pt = Point(float(xx), float(yy))
            for poly in polys:
                if poly.contains(pt):
                    out[iy, ix] = True
                    break
    return out


def test_rasterize_column_none_center_oracle():
    atlas = _LandscapeAtlas.__new__(_LandscapeAtlas)
    atlas.parent = type("P", (), {"crs": "EPSG:4326"})()
    template = _make_template()

    # atlas.data must have CRS
    atlas.data = xr.Dataset({"template": template}).rio.write_crs("EPSG:4326")

    poly = box(0.0, 1.0, 2.0, 3.0)  # covers centers at x=0.5,1.5 and y=2.5,1.5
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")

    out = atlas.rasterize(template, gdf, column=None, all_touched=False)
    mask = _oracle_center_mask(template, [poly])

    got = np.isfinite(out.values) & (out.values == 1.0)
    assert np.array_equal(got, mask)


def test_rasterize_column_overlap_last_wins():
    atlas = _LandscapeAtlas.__new__(_LandscapeAtlas)
    atlas.parent = type("P", (), {"crs": "EPSG:4326"})()
    template = _make_template()
    atlas.data = xr.Dataset({"template": template}).rio.write_crs("EPSG:4326")

    # Two overlapping polys; second should win where overlap occurs
    p1 = box(0.0, 0.0, 2.0, 3.0)  # covers x=0.5,1.5 all y
    p2 = box(1.0, 0.0, 3.0, 3.0)  # covers x=1.5,2.5 all y (overlap at x=1.5)

    gdf = gpd.GeoDataFrame({"geometry": [p1, p2], "v": [1.0, 2.0]}, crs="EPSG:4326")

    out = atlas.rasterize(template, gdf, column="v", all_touched=False).values

    ys = template["y"].values
    xs = template["x"].values
    expected = np.full((3, 3), np.nan, dtype=np.float32)

    # oracle: last-wins by center containment
    for iy, yy in enumerate(ys):
        for ix, xx in enumerate(xs):
            pt = Point(float(xx), float(yy))
            val = None
            if p1.contains(pt):
                val = 1.0
            if p2.contains(pt):
                val = 2.0  # overwrite if both
            if val is not None:
                expected[iy, ix] = val

    assert np.allclose(np.nan_to_num(out, nan=-9999), np.nan_to_num(expected, nan=-9999))
