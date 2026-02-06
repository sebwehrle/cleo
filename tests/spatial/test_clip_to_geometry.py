"""spatial: test_clip_to_geometry.
Merged test file (imports preserved per chunk).
"""

import pytest
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
import geopandas as gpd
from types import SimpleNamespace
from shapely.geometry import Polygon
from cleo.spatial import clip_to_geometry

# --- merged from tests/_staging/test_clip_to_geometry_bad_path.py ---

def test_clip_to_geometry_missing_path_raises():
    ds = xr.Dataset({"a": (("y", "x"), np.ones((2, 2)))}, coords={"x": [0, 1], "y": [0, 1]})
    ds = ds.rio.write_crs("EPSG:4326")
    dummy = SimpleNamespace(data=ds, parent=SimpleNamespace(crs="EPSG:4326"))

    with pytest.raises(ValueError, match="Cannot read clipping geometry"):
        clip_to_geometry(dummy, "nonexistent.shp")


# --- merged from tests/_staging/test_clip_to_geometry_invalid_geom_repairs.py ---

def test_clip_to_geometry_repairs_invalid_geometry():
    # self-intersecting polygon (invalid)
    poly = Polygon([(0.2, 0.2), (0.8, 0.8), (0.8, 0.2), (0.2, 0.8), (0.2, 0.2)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    ds = xr.Dataset({"a": (("y", "x"), np.ones((2, 2)))}, coords={"x": [0, 1], "y": [0, 1]})
    ds = ds.rio.write_crs("EPSG:4326")
    dummy = SimpleNamespace(data=ds, parent=SimpleNamespace(crs="EPSG:4326"))

    clipped, used = clip_to_geometry(dummy, gdf)

    # Oracle: the function must not proceed with invalid geometry
    assert used.is_valid.all()
    assert "a" in clipped.data_vars


# --- merged from tests/_staging/test_clip_to_geometry_invalid_geom_repairs_tiny_polygon.py ---

def test_clip_to_geometry_repairs_invalid_geom_tiny_polygon_does_not_raise():
    # Self-intersecting polygon that, after repair, is tiny and may not contain pixel centers
    poly = Polygon([(0.2, 0.2), (0.8, 0.8), (0.8, 0.2), (0.2, 0.8), (0.2, 0.2)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    ds = xr.Dataset(
        {"a": (("y", "x"), np.ones((2, 2)))},
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    ).rio.write_crs("EPSG:4326")

    dummy = SimpleNamespace(data=ds, parent=SimpleNamespace(crs="EPSG:4326"))

    clipped, used = clip_to_geometry(dummy, gdf)

    assert isinstance(clipped, xr.Dataset)
    assert "a" in clipped.data_vars
    assert isinstance(used, gpd.GeoDataFrame)
    # At least one pixel should survive (or be present with masking, depending on drop behaviour)
    assert clipped["a"].sizes["x"] >= 1
    assert clipped["a"].sizes["y"] >= 1
