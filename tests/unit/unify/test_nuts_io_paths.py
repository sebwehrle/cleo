"""Phase 3 coverage tests for NUTS I/O helper branches."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers.optional import requires_geopandas

requires_geopandas()

import geopandas as gpd
from shapely.geometry import Point

from cleo.unification.nuts_io import _read_nuts_region_catalog


def test_read_nuts_region_catalog_raises_when_no_shapefile(tmp_path: Path) -> None:
    atlas = SimpleNamespace(path=tmp_path, country="AUT")
    with pytest.raises(FileNotFoundError, match="NUTS shapefile not found"):
        _read_nuts_region_catalog(atlas)


def test_read_nuts_region_catalog_returns_empty_when_country_has_no_alpha2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shp = tmp_path / "data" / "nuts" / "dummy.shp"
    shp.parent.mkdir(parents=True, exist_ok=True)
    shp.write_text("x", encoding="utf-8")

    atlas = SimpleNamespace(path=tmp_path, country="AUT")
    fake_gdf = gpd.GeoDataFrame(
        [{"CNTR_CODE": "AT", "NAME_LATN": "Wien", "NUTS_ID": "AT13", "LEVL_CODE": 2}],
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr("cleo.unification.nuts_io._read_vector_file", lambda _p: fake_gdf)
    monkeypatch.setattr("cleo.unification.nuts_io.pct.countries.get", lambda **_k: SimpleNamespace(alpha_2=None))

    rows = _read_nuts_region_catalog(atlas)
    assert rows == []


def test_read_nuts_region_catalog_filters_deduplicates_and_sorts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shp = tmp_path / "data" / "nuts" / "dummy.shp"
    shp.parent.mkdir(parents=True, exist_ok=True)
    shp.write_text("x", encoding="utf-8")

    atlas = SimpleNamespace(path=tmp_path, country="AUT")
    gdf = gpd.GeoDataFrame(
        [
            {"CNTR_CODE": "AT", "NAME_LATN": "Wien", "NUTS_ID": "AT13", "LEVL_CODE": 2},
            {"CNTR_CODE": "AT", "NAME_LATN": "Wien", "NUTS_ID": "AT13", "LEVL_CODE": 2},  # dup
            {"CNTR_CODE": "AT", "NAME_LATN": "Niederösterreich", "NUTS_ID": "AT12", "LEVL_CODE": 2},
            {"CNTR_CODE": "AT", "NAME_LATN": "BadLevel", "NUTS_ID": "ATXX", "LEVL_CODE": "bad"},
            {"CNTR_CODE": "AT", "NAME_LATN": "TooHigh", "NUTS_ID": "AT99", "LEVL_CODE": 4},
            {"CNTR_CODE": "DE", "NAME_LATN": "Berlin", "NUTS_ID": "DE30", "LEVL_CODE": 1},
        ],
        geometry=[Point(i, i) for i in range(6)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr("cleo.unification.nuts_io._read_vector_file", lambda _p: gdf)

    rows = _read_nuts_region_catalog(atlas)
    assert [r["nuts_id"] for r in rows] == ["AT12", "AT13"]
    assert rows[0]["name_norm"] == "niederösterreich"
    assert rows[1]["name_norm"] == "wien"


def test_read_nuts_region_catalog_accepts_explicit_path_and_country(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shp = tmp_path / "data" / "nuts" / "dummy.shp"
    shp.parent.mkdir(parents=True, exist_ok=True)
    shp.write_text("x", encoding="utf-8")

    gdf = gpd.GeoDataFrame(
        [{"CNTR_CODE": "AT", "NAME_LATN": "Wien", "NUTS_ID": "AT13", "LEVL_CODE": 2}],
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr("cleo.unification.nuts_io._read_vector_file", lambda _p: gdf)

    rows = _read_nuts_region_catalog(tmp_path, "AUT")
    assert rows == [{"name": "Wien", "name_norm": "wien", "nuts_id": "AT13", "level": 2}]
