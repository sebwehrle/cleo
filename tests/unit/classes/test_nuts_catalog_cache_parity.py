"""Atlas-level parity tests for NUTS catalog cache behavior."""

from __future__ import annotations

from pathlib import Path

import zarr

from cleo.atlas import Atlas


def test_load_nuts_area_catalog_uses_cache_and_returns_defensive_copies(tmp_path: Path, monkeypatch) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")
    calls = {"count": 0}
    fallback_catalog = [
        {"name": "Wien", "name_norm": "wien", "nuts_id": "AT13", "level": 2},
    ]

    def _fake_read_raw(_path: Path, _country: str) -> list[dict]:
        calls["count"] += 1
        return [dict(row) for row in fallback_catalog]

    monkeypatch.setattr(
        "cleo.atlas.Atlas._ensure_nuts_shapefile",
        lambda self, *, auto_download=True: tmp_path / "data" / "nuts" / "dummy.shp",
    )
    monkeypatch.setattr("cleo.atlas._read_nuts_area_catalog", _fake_read_raw)

    first = atlas._load_nuts_area_catalog()
    assert calls["count"] == 1
    assert first == fallback_catalog

    # Mutating returned rows must not poison cache.
    first[0]["name"] = "Mutated"
    first.append({"name": "Injected", "name_norm": "injected", "nuts_id": "ATXX", "level": 2})

    second = atlas._load_nuts_area_catalog()
    assert calls["count"] == 1, "raw fallback should not be called after cache is primed"
    assert second == fallback_catalog
    assert tuple(second) == atlas._nuts_area_catalog_cache


def test_load_nuts_area_catalog_ignores_attrs_for_other_country(tmp_path: Path, monkeypatch) -> None:
    g = zarr.open_group(str(tmp_path / "landscape.zarr"), mode="w")
    g.attrs["cleo_area_catalog_json"] = '[{"name":"Bucuresti","name_norm":"bucuresti","nuts_id":"RO32","level":2}]'
    g.attrs["cleo_area_catalog_country_iso3"] = "ROU"

    atlas = Atlas(tmp_path, "HUN", "epsg:3035")
    calls = {"count": 0}
    fallback_catalog = [{"name": "Budapest", "name_norm": "budapest", "nuts_id": "HU11", "level": 2}]

    def _fake_read_raw(_path: Path, _country: str) -> list[dict]:
        calls["count"] += 1
        return [dict(row) for row in fallback_catalog]

    monkeypatch.setattr(
        "cleo.atlas.Atlas._ensure_nuts_shapefile",
        lambda self, *, auto_download=True: tmp_path / "data" / "nuts" / "dummy.shp",
    )
    monkeypatch.setattr("cleo.atlas._read_nuts_area_catalog", _fake_read_raw)

    rows = atlas._load_nuts_area_catalog()

    assert calls["count"] == 1
    assert rows == fallback_catalog
    assert atlas._nuts_area_catalog_cache_country == "HUN"
