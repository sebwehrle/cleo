"""Additional spatial error-path coverage tests."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import rioxarray  # noqa: F401
import geopandas as gpd
import pyproj
from shapely.geometry import box
from types import SimpleNamespace

from cleo.spatial import canonical_crs_str, clip_to_geometry, reproject


def test_canonical_crs_str_invalid_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Cannot parse CRS"):
        canonical_crs_str("not-a-valid-crs")


def test_clip_to_geometry_reprojection_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = xr.Dataset(
        {"a": (("y", "x"), np.ones((2, 2), dtype=np.float64))},
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    ).rio.write_crs("EPSG:4326")
    gdf = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:4326")
    dummy = SimpleNamespace(data=ds, parent=SimpleNamespace(crs="EPSG:4326"))

    def _raise_crs(*args, **kwargs):  # noqa: ANN002, ANN003
        raise pyproj.exceptions.CRSError("bad reprojection")

    monkeypatch.setattr("cleo.spatial.to_crs_if_needed", _raise_crs)
    with pytest.raises(ValueError, match="Error reprojecting clipping geometry"):
        clip_to_geometry(dummy, gdf)


def test_reproject_error_path_is_non_raising_and_keeps_data() -> None:
    class _FakeRio:
        crs = "EPSG:4326"

    class _FakeData:
        rio = _FakeRio()

        def map(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("map failed")

    dummy = SimpleNamespace(
        data=_FakeData(),
        parent=SimpleNamespace(crs="epsg:4326"),
        crs="epsg:4326",
    )

    # Should not raise; function logs and returns.
    reproject(dummy, "epsg:3857")
    assert isinstance(dummy.data, _FakeData)

