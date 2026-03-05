from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from cleo.unification.unifier import Unifier


class _AtlasStub:
    def __init__(self, root: Path) -> None:
        self.path = root
        self.landscape_store_path = root / "landscape.zarr"
        self._area_name = "R1"
        self._area_id = "R1"
        self.chunk_policy = {"y": 32, "x": 32}
        self.fingerprint_method = "path_mtime_size"

    def _active_landscape_store_path(self) -> Path:
        return self.path / "areas" / self._area_id / "landscape.zarr"


def _make_landscape_store(path: Path) -> None:
    y = [0.0, 1.0]
    x = [0.0, 1.0]
    ds = xr.Dataset(
        data_vars={
            "valid_mask": xr.DataArray(
                np.ones((2, 2), dtype=bool),
                dims=("y", "x"),
                coords={"y": y, "x": x},
            )
        },
        coords={"y": y, "x": x},
    )
    ds.to_zarr(path, mode="w", consolidated=False)
    root = zarr.open_group(path, mode="a")
    root.attrs["store_state"] = "complete"


def test_materialize_distance_variables_writes_to_active_region_store(tmp_path: Path) -> None:
    atlas = _AtlasStub(tmp_path)
    base_store = atlas.landscape_store_path
    region_store = atlas._active_landscape_store_path()
    base_store.parent.mkdir(parents=True, exist_ok=True)
    region_store.parent.mkdir(parents=True, exist_ok=True)
    _make_landscape_store(base_store)
    _make_landscape_store(region_store)

    dist = xr.DataArray(
        np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        name="distance_roads",
    )

    u = Unifier(chunk_policy=atlas.chunk_policy, fingerprint_method=atlas.fingerprint_method)
    summary = u.materialize_landscape_computed_variables(
        atlas,
        variables={"distance_roads": dist},
        if_exists="error",
    )

    assert summary == {"written": ["distance_roads"], "skipped": []}

    ds_region = xr.open_zarr(region_store, consolidated=False)
    ds_base = xr.open_zarr(base_store, consolidated=False)
    assert "distance_roads" in ds_region.data_vars
    assert "distance_roads" not in ds_base.data_vars


def test_materialize_distance_variables_reports_partial_failure(tmp_path: Path) -> None:
    atlas = _AtlasStub(tmp_path)
    region_store = atlas._active_landscape_store_path()
    region_store.parent.mkdir(parents=True, exist_ok=True)
    _make_landscape_store(region_store)

    ok = xr.DataArray(
        np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        name="distance_roads",
    )
    bad = xr.DataArray(
        np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 2.0]},  # mismatched x grid
        name="distance_water",
    )

    u = Unifier(chunk_policy=atlas.chunk_policy, fingerprint_method=atlas.fingerprint_method)
    with pytest.raises(RuntimeError, match="written=.*skipped=.*failed=") as exc_info:
        u.materialize_landscape_computed_variables(
            atlas,
            variables={
                "distance_roads": ok,
                "distance_water": bad,
            },
            if_exists="error",
        )

    err = exc_info.value
    assert getattr(err, "written", ()) == ("distance_roads",)
    assert getattr(err, "skipped", ()) == ()
    assert getattr(err, "failed", ()) == ("distance_water",)

    ds = xr.open_zarr(region_store, consolidated=False)
    assert "distance_roads" in ds.data_vars
    assert "distance_water" not in ds.data_vars
