"""Unit tests for nuts catalog policy helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cleo.atlas_policies.nuts_catalog import load_nuts_area_catalog


def test_load_nuts_area_catalog_returns_cached_rows_copy(tmp_path: Path) -> None:
    cache = ({"name": "Wien", "name_norm": "wien", "nuts_id": "AT13", "level": 2},)

    rows, returned_cache, cache_country = load_nuts_area_catalog(
        cached_rows=cache,
        cached_country_iso3="AUT",
        expected_country_iso3="AUT",
        landscape_store_path=tmp_path / "landscape.zarr",
        valid_levels=(1, 2, 3),
        read_store_attrs=lambda _p: {},
        read_raw_catalog=lambda: [],
        log_debug=lambda _msg: None,
    )

    assert rows == [dict(cache[0])]
    assert rows is not list(cache)
    assert returned_cache is cache
    assert cache_country == "AUT"


def test_load_nuts_area_catalog_uses_attrs_fast_path(tmp_path: Path) -> None:
    landscape = tmp_path / "landscape.zarr"
    landscape.mkdir(parents=True, exist_ok=True)
    payload = [
        {"name": "Wien", "name_norm": "wien", "nuts_id": "AT13", "level": 2},
        {"name": "Bad", "name_norm": "", "nuts_id": "ATXX", "level": 2},
    ]

    rows, cache, cache_country = load_nuts_area_catalog(
        cached_rows=None,
        cached_country_iso3=None,
        expected_country_iso3="AUT",
        landscape_store_path=landscape,
        valid_levels=(1, 2, 3),
        read_store_attrs=lambda _p: {
            "cleo_area_catalog_json": json.dumps(payload),
            "cleo_area_catalog_country_iso3": "AUT",
        },
        read_raw_catalog=lambda: pytest.fail("fallback should not be used"),
        log_debug=lambda _msg: None,
    )

    assert rows == [{"name": "Wien", "name_norm": "wien", "nuts_id": "AT13", "level": 2}]
    assert cache == tuple(rows)
    assert cache_country == "AUT"


def test_load_nuts_area_catalog_falls_back_when_attr_country_mismatches(tmp_path: Path) -> None:
    landscape = tmp_path / "landscape.zarr"
    landscape.mkdir(parents=True, exist_ok=True)
    fallback = [{"name": "Budapest", "name_norm": "budapest", "nuts_id": "HU11", "level": 2}]

    rows, cache, cache_country = load_nuts_area_catalog(
        cached_rows=None,
        cached_country_iso3=None,
        expected_country_iso3="HUN",
        landscape_store_path=landscape,
        valid_levels=(1, 2, 3),
        read_store_attrs=lambda _p: {
            "cleo_area_catalog_json": json.dumps(
                [{"name": "Wien", "name_norm": "wien", "nuts_id": "AT13", "level": 2}]
            ),
            "cleo_area_catalog_country_iso3": "AUT",
        },
        read_raw_catalog=lambda: fallback,
        log_debug=lambda _msg: None,
    )

    assert rows == fallback
    assert cache == tuple(fallback)
    assert cache_country == "HUN"


def test_load_nuts_area_catalog_falls_back_when_catalog_json_invalid(tmp_path: Path) -> None:
    landscape = tmp_path / "landscape.zarr"
    landscape.mkdir(parents=True, exist_ok=True)
    fallback = [{"name": "Wien", "name_norm": "wien", "nuts_id": "AT13", "level": 2}]

    rows, cache, cache_country = load_nuts_area_catalog(
        cached_rows=None,
        cached_country_iso3=None,
        expected_country_iso3="AUT",
        landscape_store_path=landscape,
        valid_levels=(1, 2, 3),
        read_store_attrs=lambda _p: {
            "cleo_area_catalog_json": "{invalid-json",
            "cleo_area_catalog_country_iso3": "AUT",
        },
        read_raw_catalog=lambda: fallback,
        log_debug=lambda _msg: None,
    )

    assert rows == fallback
    assert cache == tuple(fallback)
    assert cache_country == "AUT"


def test_load_nuts_area_catalog_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No NUTS areas available"):
        load_nuts_area_catalog(
            cached_rows=None,
            cached_country_iso3=None,
            expected_country_iso3="AUT",
            landscape_store_path=tmp_path / "landscape.zarr",
            valid_levels=(1, 2, 3),
            read_store_attrs=lambda _p: {},
            read_raw_catalog=lambda: [],
            log_debug=lambda _msg: None,
        )
