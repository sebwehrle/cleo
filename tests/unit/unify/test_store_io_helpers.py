"""Unit tests for shared store I/O helpers."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import datetime
import numpy as np
import xarray as xr
import zarr

from cleo.unification.store_io import (
    DEFAULT_CHUNK_POLICY,
    _read_stored_chunk_policy,
    delete_area_dir,
    list_area_dirs,
    open_zarr_dataset,
    read_area_store_meta,
    read_zarr_group_attrs,
    turbine_ids_from_json,
    write_netcdf_atomic,
)


def test_read_zarr_group_attrs_returns_root_attrs(tmp_path: Path) -> None:
    store = tmp_path / "sample.zarr"
    g = zarr.open_group(store, mode="w")
    g.attrs["store_state"] = "complete"
    g.attrs["grid_id"] = "abc123"

    attrs = read_zarr_group_attrs(store)
    assert attrs["store_state"] == "complete"
    assert attrs["grid_id"] == "abc123"


def test_list_area_dirs_returns_sorted_directories(tmp_path: Path) -> None:
    regions = tmp_path / "regions"
    (regions / "AT13").mkdir(parents=True)
    (regions / "AT11").mkdir(parents=True)
    (regions / "not_a_dir.txt").write_text("x", encoding="utf-8")

    out = list_area_dirs(regions)
    assert [p.name for p in out] == ["AT11", "AT13"]


def test_read_area_store_meta_prefers_created_at_and_completeness(tmp_path: Path) -> None:
    region_dir = tmp_path / "regions" / "AT13"
    wind = zarr.open_group(region_dir / "wind.zarr", mode="w")
    land = zarr.open_group(region_dir / "landscape.zarr", mode="w")
    wind.attrs["store_state"] = "complete"
    land.attrs["store_state"] = "complete"
    wind.attrs["created_at"] = "2020-01-01T00:00:00+00:00"

    meta = read_area_store_meta(region_dir)
    assert meta.is_complete is True
    assert meta.wind_exists is True
    assert meta.landscape_exists is True
    assert meta.created_at == datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)


def test_delete_area_dir_removes_directory(tmp_path: Path) -> None:
    region_dir = tmp_path / "regions" / "AT13"
    (region_dir / "wind.zarr").mkdir(parents=True)
    delete_area_dir(region_dir)
    assert not region_dir.exists()


def test_turbine_ids_from_json_decodes_ordered_ids() -> None:
    payload = '[{"id":"T2"},{"id":"T1"}]'
    assert turbine_ids_from_json(payload) == ("T2", "T1")


def test_turbine_ids_from_json_cache_is_payload_keyed() -> None:
    first = '[{"id":"A"}]'
    second = '[{"id":"B"}]'
    assert turbine_ids_from_json(first) == ("A",)
    assert turbine_ids_from_json(second) == ("B",)


# --- Chunk policy detection tests ---


def _create_zarr_store_with_data(store_path: Path, chunk_policy: dict | None = None) -> None:
    """Helper to create a minimal Zarr store with a data variable."""
    ds = xr.Dataset({"var": (["y", "x"], np.zeros((10, 10)))})
    ds.to_zarr(store_path, mode="w", consolidated=False)
    if chunk_policy is not None:
        g = zarr.open_group(store_path, mode="a")
        g.attrs["chunk_policy"] = json.dumps(chunk_policy)


def test_read_stored_chunk_policy_returns_policy_from_attrs(tmp_path: Path) -> None:
    """Test that _read_stored_chunk_policy reads chunk_policy from Zarr attrs."""
    store = tmp_path / "test.zarr"
    _create_zarr_store_with_data(store, chunk_policy={"y": 512, "x": 512})

    result = _read_stored_chunk_policy(store)
    assert result == {"y": 512, "x": 512}


def test_read_stored_chunk_policy_returns_none_when_no_attr(tmp_path: Path) -> None:
    """Test that _read_stored_chunk_policy returns None when no chunk_policy attr."""
    store = tmp_path / "test.zarr"
    _create_zarr_store_with_data(store, chunk_policy=None)

    result = _read_stored_chunk_policy(store)
    assert result is None


def test_open_zarr_dataset_uses_stored_chunks_when_mismatch(tmp_path: Path) -> None:
    """Test that open_zarr_dataset uses stored chunks and warns on mismatch."""
    store = tmp_path / "test.zarr"
    stored_policy = {"y": 512, "x": 512}
    _create_zarr_store_with_data(store, chunk_policy=stored_policy)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = open_zarr_dataset(store, chunk_policy={"y": 1024, "x": 1024})

        # Should have emitted a warning
        assert len(w) == 1
        assert "Stored chunk policy" in str(w[0].message)
        assert "{'y': 512, 'x': 512}" in str(w[0].message)
        assert "{'y': 1024, 'x': 1024}" in str(w[0].message)

    ds.close()


def test_open_zarr_dataset_no_warning_when_chunks_match(tmp_path: Path) -> None:
    """Test that open_zarr_dataset does not warn when chunks match."""
    store = tmp_path / "test.zarr"
    policy = {"y": 512, "x": 512}
    _create_zarr_store_with_data(store, chunk_policy=policy)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = open_zarr_dataset(store, chunk_policy=policy)

        # Should not have emitted a warning
        chunk_warnings = [x for x in w if "chunk policy" in str(x.message).lower()]
        assert len(chunk_warnings) == 0

    ds.close()


def test_open_zarr_dataset_warns_against_default_when_stored_differs(tmp_path: Path) -> None:
    """Test that open_zarr_dataset warns when stored differs from default."""
    store = tmp_path / "test.zarr"
    stored_policy = {"y": 512, "x": 512}
    _create_zarr_store_with_data(store, chunk_policy=stored_policy)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Pass None to use default
        ds = open_zarr_dataset(store, chunk_policy=None)

        # Should warn because stored (512) != default (1024)
        assert len(w) == 1
        assert "Stored chunk policy" in str(w[0].message)

    ds.close()


def test_open_zarr_dataset_no_warning_when_no_stored_policy(tmp_path: Path) -> None:
    """Test that open_zarr_dataset does not warn when store has no chunk_policy attr."""
    store = tmp_path / "test.zarr"
    _create_zarr_store_with_data(store, chunk_policy=None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = open_zarr_dataset(store, chunk_policy={"y": 1024, "x": 1024})

        # Should not warn - old stores without chunk_policy attr are backward compat
        chunk_warnings = [x for x in w if "chunk policy" in str(x.message).lower()]
        assert len(chunk_warnings) == 0

    ds.close()


def test_open_zarr_dataset_normalizes_boolean_root_attrs_for_netcdf(tmp_path: Path) -> None:
    """Boolean root attrs are normalized so public datasets export to NetCDF."""
    store = tmp_path / "test.zarr"
    _create_zarr_store_with_data(store, chunk_policy={"y": 512, "x": 512})

    root = zarr.open_group(store, mode="a")
    root.attrs["code_dirty"] = True
    root.attrs["requires_landscape_valid_mask"] = False

    ds = open_zarr_dataset(store, chunk_policy={"y": 512, "x": 512})
    try:
        assert ds.attrs["code_dirty"] == 1
        assert ds.attrs["requires_landscape_valid_mask"] == 0

        out_path = tmp_path / "normalized.nc"
        ds.to_netcdf(out_path)
        assert out_path.exists()
    finally:
        ds.close()


def test_write_netcdf_atomic_normalizes_boolean_root_attrs(tmp_path: Path) -> None:
    """Atomic NetCDF export accepts datasets with boolean root attrs."""
    ds = xr.Dataset({"var": (["y", "x"], np.zeros((2, 2), dtype=np.float64))})
    ds.attrs["code_dirty"] = True

    out_path = write_netcdf_atomic(ds, tmp_path / "atomic.nc")
    reopened = xr.open_dataset(out_path)
    try:
        assert reopened.attrs["code_dirty"] == np.int64(1)
    finally:
        reopened.close()


def test_default_chunk_policy_constant() -> None:
    """Test that DEFAULT_CHUNK_POLICY has expected value."""
    assert DEFAULT_CHUNK_POLICY == {"y": 1024, "x": 1024}
