"""Tests for backend-aware evaluation in export_result_netcdf."""

from __future__ import annotations

import xarray as xr

from cleo.atlas import Atlas


def test_export_result_netcdf_uses_evaluate_for_io(tmp_path) -> None:
    atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
    ds = xr.Dataset({"m": (("y", "x"), [[1.0, 2.0], [3.0, 4.0]])}, coords={"x": [0.0, 1.0], "y": [1.0, 0.0]})
    atlas.persist("m", ds, run_id="run1")

    calls = []

    def _spy(obj):  # noqa: ANN001
        calls.append(obj)
        return obj

    atlas._evaluate_for_io = _spy  # type: ignore[method-assign]
    out = atlas.export_result_netcdf("run1", "m", tmp_path / "m.nc")

    assert out.exists()
    assert len(calls) == 1
