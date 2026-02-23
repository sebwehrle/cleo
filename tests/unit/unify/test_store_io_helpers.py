"""Unit tests for shared store I/O helpers."""

from __future__ import annotations

from pathlib import Path

import datetime
import zarr

from cleo.unification.store_io import (
    delete_region_dir,
    list_region_dirs,
    read_region_store_meta,
    read_zarr_group_attrs,
)


def test_read_zarr_group_attrs_returns_root_attrs(tmp_path: Path) -> None:
    store = tmp_path / "sample.zarr"
    g = zarr.open_group(store, mode="w")
    g.attrs["store_state"] = "complete"
    g.attrs["grid_id"] = "abc123"

    attrs = read_zarr_group_attrs(store)
    assert attrs["store_state"] == "complete"
    assert attrs["grid_id"] == "abc123"


def test_list_region_dirs_returns_sorted_directories(tmp_path: Path) -> None:
    regions = tmp_path / "regions"
    (regions / "AT13").mkdir(parents=True)
    (regions / "AT11").mkdir(parents=True)
    (regions / "not_a_dir.txt").write_text("x", encoding="utf-8")

    out = list_region_dirs(regions)
    assert [p.name for p in out] == ["AT11", "AT13"]


def test_read_region_store_meta_prefers_created_at_and_completeness(tmp_path: Path) -> None:
    region_dir = tmp_path / "regions" / "AT13"
    wind = zarr.open_group(region_dir / "wind.zarr", mode="w")
    land = zarr.open_group(region_dir / "landscape.zarr", mode="w")
    wind.attrs["store_state"] = "complete"
    land.attrs["store_state"] = "complete"
    wind.attrs["created_at"] = "2020-01-01T00:00:00+00:00"

    meta = read_region_store_meta(region_dir)
    assert meta.is_complete is True
    assert meta.wind_exists is True
    assert meta.landscape_exists is True
    assert meta.created_at == datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)


def test_delete_region_dir_removes_directory(tmp_path: Path) -> None:
    region_dir = tmp_path / "regions" / "AT13"
    (region_dir / "wind.zarr").mkdir(parents=True)
    delete_region_dir(region_dir)
    assert not region_dir.exists()
