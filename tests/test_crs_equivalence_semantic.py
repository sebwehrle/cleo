"""
Test that CRS comparisons are semantic (rasterio CRS objects),
not string/text based. Case-mismatched but equivalent CRS should pass.
"""
import numpy as np
import types
import xarray as xr
import rioxarray  # noqa: F401
import geopandas as gpd
from shapely.geometry import box

from cleo.class_helpers import set_attributes
from cleo.spatial import clip_to_geometry


def test_set_attributes_accepts_case_mismatched_crs():
    """
    set_attributes should accept case-mismatched but equivalent CRS.
    E.g., "EPSG:4326" (data) vs "epsg:4326" (parent) are semantically equal.
    """
    # Create minimal dataset with CRS in uppercase
    ds = xr.Dataset({
        "a": xr.DataArray(np.zeros((2, 2)), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]})
    }).rio.write_crs("EPSG:4326")

    # Dummy object with lowercase CRS
    dummy = types.SimpleNamespace()
    dummy.data = ds
    dummy.parent = types.SimpleNamespace(country="AUT", region=None, crs="epsg:4326")

    # Should NOT raise - CRS are semantically equivalent
    set_attributes(dummy)

    # Verify attributes were set
    assert dummy.data.attrs["country"] == "AUT"


def test_clip_to_geometry_accepts_case_mismatched_crs():
    """
    clip_to_geometry should accept case-mismatched but equivalent CRS.
    E.g., "EPSG:4326" (data) vs "epsg:4326" (parent) are semantically equal.
    """
    # Create minimal dataset with CRS in uppercase
    ds = xr.Dataset({
        "template": xr.DataArray(
            np.ones((3, 3)),
            dims=("y", "x"),
            coords={"y": [0.0, 0.5, 1.0], "x": [0.0, 0.5, 1.0]}
        )
    }).rio.write_crs("EPSG:4326")

    # Dummy object with lowercase CRS
    dummy = types.SimpleNamespace()
    dummy.data = ds
    dummy.parent = types.SimpleNamespace(crs="epsg:4326")

    # Create a tiny valid GeoDataFrame in EPSG:4326
    gdf = gpd.GeoDataFrame(geometry=[box(0.1, 0.1, 0.9, 0.9)], crs="EPSG:4326")

    # Should NOT raise - CRS are semantically equivalent
    result, _ = clip_to_geometry(dummy, gdf)

    # Verify result is an xr.Dataset
    assert isinstance(result, xr.Dataset)
