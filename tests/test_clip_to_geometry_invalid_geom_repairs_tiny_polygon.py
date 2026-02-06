import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
from types import SimpleNamespace

from cleo.spatial import clip_to_geometry


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
