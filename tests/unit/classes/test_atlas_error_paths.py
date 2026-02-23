"""Additional Atlas error/fallback path tests for branch coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import zarr

from cleo.atlas import Atlas, _safe_basename


def test_safe_basename_returns_question_on_unhandled_path_type() -> None:
    class BadPath:
        pass

    assert _safe_basename(BadPath()) == "?"


def test_repr_falls_back_to_minimal_when_repr_components_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")
    monkeypatch.setattr("cleo.atlas._safe_basename", lambda _p: (_ for _ in ()).throw(ValueError("boom")))
    assert repr(atlas) == "Atlas(?)"


def test_load_nuts_region_catalog_falls_back_when_legacy_index_json_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")
    g = zarr.open_group(str(tmp_path / "landscape.zarr"), mode="w")
    g.attrs["cleo_region_name_to_id_json"] = "{invalid-json"

    fallback_catalog = [
        {"name": "Wien", "name_norm": "wien", "nuts_id": "AT13", "level": 2},
    ]
    monkeypatch.setattr("cleo.atlas._read_nuts_region_catalog", lambda _path, _country: fallback_catalog)

    rows = atlas._load_nuts_region_catalog()
    assert rows == fallback_catalog


def test_clean_results_falls_back_to_mtime_when_store_attrs_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")
    store = tmp_path / "results" / "run_a" / "metric.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    zarr.open_group(str(store), mode="w")

    old_ts = 1_577_836_800  # 2020-01-01 UTC
    os.utime(store, (old_ts, old_ts))

    real_open_group = zarr.open_group

    def _open_group_fail_for_store(path, *args, **kwargs):
        if Path(path) == store:
            raise OSError("cannot read attrs")
        return real_open_group(path, *args, **kwargs)

    monkeypatch.setattr("cleo.results.zarr.open_group", _open_group_fail_for_store)

    deleted = atlas.clean_results(older_than="2021-01-01")
    assert deleted == 1
    assert not store.exists()


def test_clean_regions_treats_unreadable_store_state_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")
    region_dir = tmp_path / "regions" / "AT13"
    (region_dir / "wind.zarr").mkdir(parents=True, exist_ok=True)
    (region_dir / "landscape.zarr").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "cleo.unification.store_io.zarr.open_group",
        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")),
    )
    deleted = atlas.clean_regions(include_incomplete=False)
    assert deleted == 0
    assert region_dir.exists()


def test_ensure_region_stores_retries_with_region_name_and_migrates_legacy_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")
    atlas._region_name = "Niederösterreich"
    atlas._region_id = "AT12"

    class FakeUnifier:
        def __init__(self, *args, **kwargs):  # noqa: D401, ANN002, ANN003
            pass

        def materialize_region(self, atlas_obj, region):  # noqa: ANN001
            if region == "AT12":
                raise RuntimeError("legacy unifier expects region name")
            # Write to legacy region-name directory
            legacy = atlas_obj.path / "regions" / "Niederösterreich"
            (legacy / "wind.zarr").mkdir(parents=True, exist_ok=True)
            (legacy / "landscape.zarr").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("cleo.unification.Unifier", FakeUnifier)
    atlas._ensure_region_stores()

    expected = tmp_path / "regions" / "AT12"
    assert (expected / "wind.zarr").exists()
    assert (expected / "landscape.zarr").exists()
