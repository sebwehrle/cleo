"""Tests for DomainResult cache materialization backend dispatch."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import xarray as xr

from cleo.results import DomainResult


def _seed_wind_store(path) -> None:
    ds = xr.Dataset(
        {"template": (("y", "x"), np.ones((2, 2), dtype=np.float32))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
        attrs={
            "cleo_turbines_json": json.dumps(
                [{"id": "Enercon.E40.500"}],
                sort_keys=True,
                separators=(",", ":"),
            )
        },
    )
    ds.to_zarr(path, mode="w", consolidated=False)


def test_domain_result_cache_uses_atlas_compute_backend(monkeypatch, tmp_path) -> None:
    store_path = tmp_path / "wind.zarr"
    _seed_wind_store(store_path)

    calls = []

    def _fake_dask_compute(obj, *, backend, num_workers=None):
        calls.append((backend, num_workers))
        return obj

    monkeypatch.setattr("cleo.results.dask_compute", _fake_dask_compute)

    atlas = SimpleNamespace(
        compute_backend="processes",
        compute_workers=3,
        _active_wind_store_path=lambda: store_path,
    )
    domain = SimpleNamespace(_atlas=atlas, _data=None)
    da = xr.DataArray(
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
        name="mean_wind_speed",
    )
    result = DomainResult(domain, "mean_wind_speed", da, {})

    result.cache()

    assert calls == [("processes", 3)]


def test_domain_result_cache_defaults_to_serial_backend(monkeypatch, tmp_path) -> None:
    store_path = tmp_path / "wind.zarr"
    _seed_wind_store(store_path)

    calls = []

    def _fake_dask_compute(obj, *, backend, num_workers=None):
        calls.append((backend, num_workers))
        return obj

    monkeypatch.setattr("cleo.results.dask_compute", _fake_dask_compute)

    atlas = SimpleNamespace(
        _active_wind_store_path=lambda: store_path,
    )
    domain = SimpleNamespace(_atlas=atlas, _data=None)
    da = xr.DataArray(
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
        name="mean_wind_speed",
    )
    result = DomainResult(domain, "mean_wind_speed", da, {})

    result.cache()

    assert calls == [("serial", None)]


def test_domain_result_cache_prefers_atlas_evaluate_for_io(monkeypatch, tmp_path) -> None:
    store_path = tmp_path / "wind.zarr"
    _seed_wind_store(store_path)

    eval_calls = []
    compute_calls = []

    def _fake_dask_compute(obj, *, backend, num_workers=None):
        compute_calls.append((backend, num_workers))
        return obj

    def _evaluate_for_io(obj):
        eval_calls.append(obj)
        return obj

    monkeypatch.setattr("cleo.results.dask_compute", _fake_dask_compute)

    atlas = SimpleNamespace(
        compute_backend="processes",
        _active_wind_store_path=lambda: store_path,
        _evaluate_for_io=_evaluate_for_io,
    )
    domain = SimpleNamespace(_atlas=atlas, _data=None)
    da = xr.DataArray(
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
        name="mean_wind_speed",
    )
    result = DomainResult(domain, "mean_wind_speed", da, {})

    result.cache()

    assert len(eval_calls) == 1
    assert compute_calls == []
