"""Tests for export execution semantics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr
import zarr

from cleo.exports import export_analysis_dataset_zarr


def _atlas_double(tmp_path: Path) -> SimpleNamespace:
    wind = xr.Dataset(
        {"capacity_factors": (("y", "x"), np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))},
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
        attrs={"grid_id": "wind-grid", "inputs_id": "wind-inputs", "store_state": "complete"},
    )
    landscape = xr.Dataset(
        {"valid_mask": (("y", "x"), np.array([[True, True], [True, False]], dtype=bool))},
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
        attrs={"grid_id": "land-grid", "inputs_id": "land-inputs", "store_state": "complete"},
    )
    return SimpleNamespace(
        _canonical_ready=True,
        compute_backend="threads",
        compute_workers=2,
        path=tmp_path,
        wind_data=wind,
        landscape_data=landscape,
    )


def test_export_compute_true_uses_explicit_precompute(monkeypatch, tmp_path: Path) -> None:
    atlas = _atlas_double(tmp_path)
    export_path = tmp_path / "export_true.zarr"
    compute_calls: list[tuple[str, int | None]] = []
    write_calls: list[dict] = []
    original_to_zarr = xr.Dataset.to_zarr

    def _fake_dask_compute(obj, *, backend, num_workers=None):
        compute_calls.append((backend, num_workers))
        return obj

    def _wrapped_to_zarr(self, store, *args, **kwargs):
        write_calls.append(dict(kwargs))
        return original_to_zarr(self, store, *args, **kwargs)

    monkeypatch.setattr("cleo.exports.dask_compute", _fake_dask_compute)
    monkeypatch.setattr(xr.Dataset, "to_zarr", _wrapped_to_zarr)

    export_analysis_dataset_zarr(atlas, export_path, domain="both", compute=True)

    assert compute_calls == [("threads", 2)]
    assert len(write_calls) == 1
    assert write_calls[0].get("compute", True) is not False
    group = zarr.open_group(export_path, mode="r")
    assert group.attrs["store_state"] == "complete"


def test_export_compute_false_skips_precompute_but_still_writes_synchronously(monkeypatch, tmp_path: Path) -> None:
    atlas = _atlas_double(tmp_path)
    export_path = tmp_path / "export_false.zarr"
    compute_calls: list[tuple[str, int | None]] = []
    write_calls: list[dict] = []
    original_to_zarr = xr.Dataset.to_zarr

    def _fake_dask_compute(obj, *, backend, num_workers=None):
        compute_calls.append((backend, num_workers))
        return obj

    def _wrapped_to_zarr(self, store, *args, **kwargs):
        write_calls.append(dict(kwargs))
        return original_to_zarr(self, store, *args, **kwargs)

    monkeypatch.setattr("cleo.exports.dask_compute", _fake_dask_compute)
    monkeypatch.setattr(xr.Dataset, "to_zarr", _wrapped_to_zarr)

    export_analysis_dataset_zarr(atlas, export_path, domain="both", compute=False)

    assert compute_calls == []
    assert len(write_calls) == 1
    assert write_calls[0].get("compute", True) is not False
    group = zarr.open_group(export_path, mode="r")
    assert group.attrs["store_state"] == "complete"
