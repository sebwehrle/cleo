"""Phase 5 spatial guardrail/error-path tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tests.helpers.optional import requires_geopandas, requires_rioxarray

requires_geopandas()
requires_rioxarray()

import geopandas as gpd
from shapely.geometry import Polygon, box

from cleo.spatial import bbox, clip_to_geometry, reproject


def _dummy_with_data(ds: xr.Dataset, parent_crs: str = "EPSG:4326"):
    return SimpleNamespace(data=ds, parent=SimpleNamespace(crs=parent_crs))


def test_clip_to_geometry_raises_when_data_is_none() -> None:
    dummy = SimpleNamespace(data=None, parent=SimpleNamespace(crs="EPSG:4326"))
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="There is no data"):
        clip_to_geometry(dummy, gdf)


def test_clip_to_geometry_raises_when_parent_crs_missing() -> None:
    ds = xr.Dataset({"a": (("y", "x"), np.ones((2, 2), dtype=np.float64))}, coords={"x": [0, 1], "y": [0, 1]}).rio.write_crs("EPSG:4326")
    dummy = SimpleNamespace(data=ds, parent=SimpleNamespace(crs=None))
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="parent CRS is missing"):
        clip_to_geometry(dummy, gdf)


def test_clip_to_geometry_raises_on_crs_inconsistency() -> None:
    ds = xr.Dataset({"a": (("y", "x"), np.ones((2, 2), dtype=np.float64))}, coords={"x": [0, 1], "y": [0, 1]}).rio.write_crs("EPSG:4326")
    dummy = _dummy_with_data(ds, parent_crs="EPSG:3035")
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="CRS inconsistency"):
        clip_to_geometry(dummy, gdf)


def test_clip_to_geometry_raises_on_empty_or_crsless_geometry() -> None:
    ds = xr.Dataset({"a": (("y", "x"), np.ones((2, 2), dtype=np.float64))}, coords={"x": [0, 1], "y": [0, 1]}).rio.write_crs("EPSG:4326")
    dummy = _dummy_with_data(ds)
    gdf_empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    with pytest.raises(ValueError, match="geometry is empty"):
        clip_to_geometry(dummy, gdf_empty)

    gdf_no_crs = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])
    with pytest.raises(ValueError, match="CRS is missing"):
        clip_to_geometry(dummy, gdf_no_crs)


def test_clip_to_geometry_raises_when_invalid_geometry_not_repaired(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = xr.Dataset({"a": (("y", "x"), np.ones((2, 2), dtype=np.float64))}, coords={"x": [0, 1], "y": [0, 1]}).rio.write_crs("EPSG:4326")
    dummy = _dummy_with_data(ds)
    # Self-intersecting bow-tie polygon
    poly = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    monkeypatch.setattr("cleo.spatial.to_crs_if_needed", lambda g, _crs: g)
    # Force clip op to fail consistently so post-repair invalid path is exercised.
    monkeypatch.setattr("cleo.spatial._rio_clip_robust", lambda *a, **k: (_ for _ in ()).throw(ValueError("clip failed")))
    with pytest.raises(ValueError, match="Error clipping data variable"):
        clip_to_geometry(dummy, gdf)


def test_bbox_unsupported_object_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported object type for bbox"):
        bbox(SimpleNamespace())


def test_reproject_noop_when_crs_equal_keeps_parent_and_data() -> None:
    ds = xr.Dataset({"a": (("y", "x"), np.ones((2, 2), dtype=np.float64))}, coords={"x": [0, 1], "y": [0, 1]}).rio.write_crs("EPSG:4326")
    dummy = SimpleNamespace(data=ds, parent=SimpleNamespace(crs="epsg:4326"), crs="epsg:4326")
    before = dummy.data
    reproject(dummy, "EPSG:4326")
    assert dummy.data is before
    assert dummy.parent.crs == "epsg:4326"

