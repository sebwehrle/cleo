"""Additional Atlas error/fallback path tests for branch coverage."""

from __future__ import annotations

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


def test_load_nuts_area_catalog_falls_back_when_catalog_json_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")
    g = zarr.open_group(str(tmp_path / "landscape.zarr"), mode="w")
    g.attrs["cleo_area_catalog_json"] = "{invalid-json"

    fallback_catalog = [
        {"name": "Wien", "name_norm": "wien", "nuts_id": "AT13", "level": 2},
    ]
    monkeypatch.setattr(
        "cleo.atlas.Atlas._ensure_nuts_shapefile",
        lambda self, *, auto_download=True: tmp_path / "data" / "nuts" / "dummy.shp",
    )
    monkeypatch.setattr("cleo.atlas._read_nuts_area_catalog", lambda _path, _country: fallback_catalog)

    rows = atlas._load_nuts_area_catalog()
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


def test_ensure_nuts_shapefile_downloads_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")
    calls = {"count": 0}

    def _fake_load_nuts(_atlas, **_kwargs):  # noqa: ANN001
        calls["count"] += 1
        shp = tmp_path / "data" / "nuts" / "NUTS_RG_03M_2021_4326.shp"
        shp.parent.mkdir(parents=True, exist_ok=True)
        shp.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr("cleo.loaders.load_nuts", _fake_load_nuts)

    shp = atlas._ensure_nuts_shapefile(auto_download=True)
    assert calls["count"] == 1
    assert shp.exists()


def test_build_with_pending_area_ensures_nuts_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035", area="Wien")
    calls = {"count": 0}

    def _fake_load_nuts(_atlas, **_kwargs):  # noqa: ANN001
        calls["count"] += 1
        shp = tmp_path / "data" / "nuts" / "NUTS_RG_03M_2021_4326.shp"
        shp.parent.mkdir(parents=True, exist_ok=True)
        shp.write_text("dummy", encoding="utf-8")

    def _fake_select(self, *, area=None, nuts_level=None, inplace=False):  # noqa: ANN001, ARG001
        self._area_name = area
        self._area_id = "AT13"
        return None

    monkeypatch.setattr("cleo.loaders.load_nuts", _fake_load_nuts)
    monkeypatch.setattr("cleo.atlas.Atlas.build_canonical", lambda self: setattr(self, "_canonical_ready", True))
    monkeypatch.setattr("cleo.atlas.Atlas.select", _fake_select)
    monkeypatch.setattr("cleo.atlas.Atlas._ensure_area_stores", lambda self: None)

    atlas.build()
    assert calls["count"] == 1


def test_clean_areas_treats_unreadable_store_state_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")
    region_dir = tmp_path / "areas" / "AT13"
    (region_dir / "wind.zarr").mkdir(parents=True, exist_ok=True)
    (region_dir / "landscape.zarr").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "cleo.unification.store_io.zarr.open_group",
        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")),
    )
    deleted = atlas.clean_areas(include_incomplete=False)
    assert deleted == 0
    assert region_dir.exists()


def test_ensure_area_stores_requires_area_id_store_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")
    atlas._area_name = "Niederösterreich"
    atlas._area_id = "AT12"

    class FakeUnifier:
        def __init__(self, *args, **kwargs):  # noqa: D401, ANN002, ANN003
            pass

        def materialize_area(self, atlas_obj, area):  # noqa: ANN001
            assert area == "AT12"
            # Create stores at WRONG path (legacy name instead of region_id)
            legacy = atlas_obj.path / "areas" / "Niederösterreich"
            (legacy / "wind.zarr").mkdir(parents=True, exist_ok=True)
            (legacy / "landscape.zarr").mkdir(parents=True, exist_ok=True)

        def ensure_area_stores(self, atlas_obj, region_id, *, logger):  # noqa: ANN001
            # Call materialize_area (which creates stores at wrong path)
            self.materialize_area(atlas_obj, region_id)
            # Check expected paths (which won't exist because stores are at wrong path)
            expected_root = atlas_obj.path / "areas" / region_id
            expected_wind = expected_root / "wind.zarr"
            expected_land = expected_root / "landscape.zarr"
            if not (expected_wind.exists() and expected_land.exists()):
                raise RuntimeError(
                    f"Region stores are still missing after materialize_area({region_id!r}). "
                    f"Details: {{'expected_root': '{expected_root}', "
                    f"'expected_wind_exists': {expected_wind.exists()}, "
                    f"'expected_landscape_exists': {expected_land.exists()}}}"
                )

    monkeypatch.setattr("cleo.unification.Unifier", FakeUnifier)
    with pytest.raises(RuntimeError, match="Region stores are still missing after materialize_area\\('AT12'\\)"):
        atlas._ensure_area_stores()
