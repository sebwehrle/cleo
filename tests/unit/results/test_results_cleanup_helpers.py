"""Unit tests for results cleanup I/O helpers."""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import pytest
import zarr

from cleo.results import (
    delete_result_store,
    list_result_stores,
    prune_empty_run_dirs,
    read_result_store_datetime,
)


def _mk_store(root: Path, run_id: str, metric: str, *, created_at: str | None = None) -> Path:
    path = root / run_id / f"{metric}.zarr"
    g = zarr.open_group(path, mode="w")
    if created_at is not None:
        g.attrs["created_at"] = created_at
    return path


def test_list_result_stores_filters_run_and_metric(tmp_path: Path) -> None:
    root = tmp_path / "results"
    s1 = _mk_store(root, "run1", "metric_a")
    _mk_store(root, "run1", "metric_b")
    _mk_store(root, "run2", "metric_a")

    all_stores = list_result_stores(root)
    assert [p.name for p in all_stores] == ["metric_a.zarr", "metric_b.zarr", "metric_a.zarr"]

    run_filtered = list_result_stores(root, run_id="run1")
    assert run_filtered == sorted([s1.parent / "metric_a.zarr", s1.parent / "metric_b.zarr"])

    metric_filtered = list_result_stores(root, metric_name="metric_b")
    assert [p.name for p in metric_filtered] == ["metric_b.zarr"]


def test_read_result_store_datetime_prefers_created_at_and_falls_back_to_mtime(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "results"
    with_created = _mk_store(root, "run1", "metric_a", created_at="2020-01-01T00:00:00+00:00")
    fallback = _mk_store(root, "run2", "metric_a")
    old_ts = 1_577_836_800  # 2020-01-01 UTC
    os.utime(fallback, (old_ts, old_ts))

    dt1 = read_result_store_datetime(with_created)
    assert dt1 == datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)

    real_open_group = zarr.open_group

    def _raise_for_fallback(path, *args, **kwargs):  # noqa: ANN002, ANN003
        if Path(path) == fallback:
            raise OSError("boom")
        return real_open_group(path, *args, **kwargs)

    monkeypatch.setattr("cleo.results.zarr.open_group", _raise_for_fallback)
    dt2 = read_result_store_datetime(fallback)
    assert dt2 == datetime.datetime.fromtimestamp(old_ts)


def test_delete_result_store_and_prune_empty_run_dirs(tmp_path: Path) -> None:
    root = tmp_path / "results"
    store = _mk_store(root, "run1", "metric_a")
    delete_result_store(store)
    assert not store.exists()

    removed = prune_empty_run_dirs(root)
    assert removed == 1
    assert not (root / "run1").exists()


def test_list_result_stores_rejects_invalid_tokens(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        list_result_stores(tmp_path / "results", run_id="../outside")
    with pytest.raises(ValueError, match="metric_name"):
        list_result_stores(tmp_path / "results", metric_name="../outside")

