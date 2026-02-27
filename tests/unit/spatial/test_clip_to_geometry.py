"""spatial: test_clip_to_geometry.
Merged test file (imports preserved per chunk).
"""

from tests.helpers.optional import requires_rioxarray, requires_geopandas

requires_rioxarray()
requires_geopandas()

import pytest
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
import geopandas as gpd
from types import SimpleNamespace
from shapely.geometry import Polygon
from cleo.spatial import clip_to_geometry, _rio_clip_robust
from tests.helpers.factories import wind_speed_axis


def test_clip_to_geometry_rejects_path_string():
    """spatial.py is primitives-only: paths must be handled by caller (classes.py)."""
    ds = xr.Dataset({"a": (("y", "x"), np.ones((2, 2)))}, coords={"x": [0, 1], "y": [0, 1]})
    ds = ds.rio.write_crs("EPSG:4326")
    dummy = SimpleNamespace(data=ds, parent=SimpleNamespace(crs="EPSG:4326"))

    with pytest.raises(TypeError, match="clip_shape must be a geopandas.GeoDataFrame"):
        clip_to_geometry(dummy, "nonexistent.shp")


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


def test_clip_to_geometry_preserves_non_spatial_coords():
    """
    clip_to_geometry must preserve non-spatial coords like wind_speed
    which are not in data_vars and have no x/y dims.
    """
    # Build a Dataset with non-spatial coord wind_speed
    wind_speed = wind_speed_axis()
    ds = xr.Dataset(
        {"a": (("y", "x"), np.ones((2, 2)))},
        coords={
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "wind_speed": wind_speed,
        },
    )
    ds = ds.rio.write_crs("EPSG:4326")
    ds.attrs["test_attr"] = "preserved"

    dummy = SimpleNamespace(data=ds, parent=SimpleNamespace(crs="EPSG:4326"))

    # Clip geometry that overlaps the raster
    poly = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    clipped, _ = clip_to_geometry(dummy, gdf)

    # Assertions
    assert "wind_speed" in clipped.coords, "wind_speed coord should be preserved"
    np.testing.assert_array_equal(
        clipped.coords["wind_speed"].values,
        ds.coords["wind_speed"].values,
        err_msg="wind_speed values should match source",
    )
    assert "a" in clipped.data_vars, "data_var 'a' should be preserved"
    assert clipped.attrs.get("test_attr") == "preserved", "attrs should be copied"


def test_rio_clip_robust_fallback_path(monkeypatch):
    """
    _rio_clip_robust must retry with all_touched=True when NoDataInBounds is raised.
    Verify:
    - clip is called twice on NoDataInBounds
    - second call uses all_touched=True with same drop value
    - result is a DataArray
    """
    from rioxarray.exceptions import NoDataInBounds

    # Create a simple DataArray with rio accessor
    da = xr.DataArray(
        np.ones((2, 2)),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    ).rio.write_crs("EPSG:4326")

    geoms = [Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)])]

    # Track calls to rio.clip
    call_args_list = []
    call_count = [0]

    original_clip = da.rio.clip

    def mock_clip(geoms, all_touched=False, drop=True):
        call_args_list.append({"all_touched": all_touched, "drop": drop})
        call_count[0] += 1
        if call_count[0] == 1:
            # First call raises NoDataInBounds
            raise NoDataInBounds("simulated")
        # Second call succeeds
        return original_clip(geoms, all_touched=all_touched, drop=drop)

    # Monkeypatch the rio.clip method on the DataArray's accessor
    monkeypatch.setattr(da.rio, "clip", mock_clip)

    # Call with drop=False to test that drop is preserved
    result = _rio_clip_robust(da, geoms, drop=False, all_touched_primary=False)

    # Assert clip was called twice
    assert len(call_args_list) == 2, f"Expected 2 calls, got {len(call_args_list)}"

    # First call: all_touched=False (primary), drop=False
    assert call_args_list[0]["all_touched"] is False, "First call should use all_touched=False"
    assert call_args_list[0]["drop"] is False, "First call should use drop=False"

    # Second call: all_touched=True (fallback), drop=False (same as requested)
    assert call_args_list[1]["all_touched"] is True, "Second call should use all_touched=True"
    assert call_args_list[1]["drop"] is False, "Second call should preserve drop=False"

    # Result is a DataArray
    assert isinstance(result, xr.DataArray), "Result should be a DataArray"
