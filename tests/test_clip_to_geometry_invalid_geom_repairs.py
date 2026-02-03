import pytest
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
from types import SimpleNamespace
import rioxarray  # noqa: F401
from cleo.spatial import clip_to_geometry

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
