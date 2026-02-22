"""Tests for DomainResult.persist() fluent persistence behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import xarray as xr
import zarr

from cleo.results import DomainResult


def test_domain_result_persist_writes_store_without_atlas_persist(tmp_path) -> None:
    def _forbid_persist(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("DomainResult.persist() must not call atlas.persist()")

    atlas = SimpleNamespace(
        results_root=tmp_path / "results",
        new_run_id=lambda: "run_001",
        _canonical_ready=False,
        persist=_forbid_persist,
    )
    domain = SimpleNamespace(_atlas=atlas)
    da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
        name="capacity_factors",
    )
    result = DomainResult(domain, "capacity_factors", da, {"height": 100})

    store_path = result.persist()

    assert store_path == tmp_path / "results" / "run_001" / "capacity_factors.zarr"
    group = zarr.open_group(store_path, mode="r")
    assert group.attrs["store_state"] == "complete"
    assert group.attrs["run_id"] == "run_001"
    assert group.attrs["metric_name"] == "capacity_factors"
    assert json.loads(group.attrs["params_json"]) == {"height": 100}


def test_domain_result_persist_allows_metric_name_and_params_override(tmp_path) -> None:
    atlas = SimpleNamespace(
        results_root=tmp_path / "results",
        new_run_id=lambda: "run_002",
        _canonical_ready=False,
    )
    domain = SimpleNamespace(_atlas=atlas)
    da = xr.DataArray(
        np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
        name="land_use",
    )
    result = DomainResult(domain, "land_use", da, {"source": "default"})

    store_path = result.persist(metric_name="landscape_land_use", params={"source": "custom"})

    assert store_path == tmp_path / "results" / "run_002" / "landscape_land_use.zarr"
    group = zarr.open_group(store_path, mode="r")
    assert group.attrs["metric_name"] == "landscape_land_use"
    assert json.loads(group.attrs["params_json"]) == {"source": "custom"}


def test_domain_result_persist_uses_atlas_evaluate_for_io_when_available(tmp_path) -> None:
    calls = []

    def _evaluate_for_io(obj):
        calls.append(obj)
        return obj

    atlas = SimpleNamespace(
        results_root=tmp_path / "results",
        new_run_id=lambda: "run_003",
        _canonical_ready=False,
        _evaluate_for_io=_evaluate_for_io,
    )
    domain = SimpleNamespace(_atlas=atlas)
    da = xr.DataArray(
        np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
        name="var",
    )
    result = DomainResult(domain, "var", da, {})

    result.persist()

    assert len(calls) == 1
