import numpy as np
import xarray as xr
import geopandas as gpd
import pytest

from shapely.geometry import box, Point

from cleo.classes import _LandscapeAtlas


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
