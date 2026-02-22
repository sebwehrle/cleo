"""Phase 5 dask_utils fallback/error path tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from cleo import dask_utils as D


def test_normalize_chunks_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        D.normalize_chunks({"x": 0})


def test_is_dask_backed_handles_none_and_module_prefix() -> None:
    obj = SimpleNamespace(data=SimpleNamespace(__module__="dask.array.core"))
    assert D.is_dask_backed(obj) is True
    assert D.is_dask_backed(SimpleNamespace(data=None)) is False


def test_maybe_chunk_auto_wraps_chunk_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    da = xr.DataArray(np.ones((4, 4), dtype=np.float64), dims=("y", "x"))
    monkeypatch.setattr("cleo.dask_utils.ensure_dask_available", lambda feature: None)
    monkeypatch.setattr(xr.DataArray, "chunk", lambda self, *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ValueError, match="Failed to chunk object with chunks='auto'"):
        D.maybe_chunk(da, chunks="auto", enabled=True)


def test_maybe_chunk_dict_wraps_chunk_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    da = xr.DataArray(np.ones((4, 4), dtype=np.float64), dims=("y", "x"))
    monkeypatch.setattr("cleo.dask_utils.ensure_dask_available", lambda feature: None)
    monkeypatch.setattr(xr.DataArray, "chunk", lambda self, *_a, **_k: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ValueError, match="Failed to chunk object with chunks="):
        D.maybe_chunk(da, chunks={"y": 2, "x": 2}, enabled=True)


def test_get_distributed_client_and_dashboard_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cleo.dask_utils.ensure_dask_available", lambda feature: None)

    real_import = __import__

    def _import_fail(name, *args, **kwargs):  # noqa: ANN002, ANN003
        if name == "dask.distributed":
            raise ImportError("missing distributed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _import_fail)
    with pytest.raises(RuntimeError, match="requires 'dask\\[distributed\\]'"):
        D.get_distributed_client_and_dashboard()


def test_scheduler_context_distributed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}
    monkeypatch.setattr("cleo.dask_utils.get_distributed_client_and_dashboard", lambda: called.__setitem__("n", called["n"] + 1) or ("c", None))
    with D.scheduler_context(backend="distributed"):
        pass
    assert called["n"] == 1
