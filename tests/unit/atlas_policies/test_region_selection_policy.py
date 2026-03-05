"""Unit tests for area-selection policy helpers."""

from __future__ import annotations

import pytest

from cleo.atlas_policies.region_selection import (
    normalize_area_name,
    nuts_areas_level_names,
    resolve_area_name,
    select_area_decision,
    validate_nuts_level,
)


def _validate(level: int) -> int:
    return validate_nuts_level(level, valid_levels=(0, 1, 2, 3))


def test_normalize_area_name_collapses_whitespace_and_casefolds() -> None:
    assert normalize_area_name("  Nieder  Österreich  ") == "nieder österreich"


def test_validate_nuts_level_valid_and_invalid() -> None:
    assert _validate(2) == 2
    with pytest.raises(ValueError, match="must be an integer"):
        _validate("x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported NUTS level"):
        _validate(9)


def test_nuts_areas_level_names_deduplicates_and_sorts() -> None:
    rows = [
        {"name": "Wien", "name_norm": "wien", "nuts_id": "AT13", "level": 2},
        {"name": "Wien", "name_norm": "wien", "nuts_id": "AT13B", "level": 2},
        {"name": "Niederösterreich", "name_norm": "niederösterreich", "nuts_id": "AT12", "level": 2},
    ]

    names = nuts_areas_level_names(
        level=2,
        country="AUT",
        catalog_rows=rows,
        validate_level=_validate,
        make_area_name=lambda name, level: (name, level),
    )
    assert names == (("Niederösterreich", 2), ("Wien", 2))


def test_nuts_areas_level_names_raises_when_level_missing() -> None:
    with pytest.raises(ValueError, match="No NUTS level 3 areas"):
        nuts_areas_level_names(
            level=3,
            country="AUT",
            catalog_rows=[],
            validate_level=_validate,
            make_area_name=lambda name, level: (name, level),
        )


def test_resolve_area_name_nuts_level_explicit_paths() -> None:
    rows = [
        {"name": "SharedName", "name_norm": "sharedname", "nuts_id": "ATX1", "level": 1},
        {"name": "SharedName", "name_norm": "sharedname", "nuts_id": "ATX3", "level": 3},
    ]

    out = resolve_area_name(
        name="SharedName",
        nuts_level=1,
        default_level=2,
        country="AUT",
        catalog_rows=rows,
        normalize_name=normalize_area_name,
        validate_level=_validate,
        default_areas_supplier=lambda: tuple(),
    )
    assert out == ("sharedname", "ATX1", 1)

    with pytest.raises(ValueError, match="not found in NUTS level 2"):
        resolve_area_name(
            name="SharedName",
            nuts_level=2,
            default_level=2,
            country="AUT",
            catalog_rows=rows,
            normalize_name=normalize_area_name,
            validate_level=_validate,
            default_areas_supplier=lambda: tuple(),
        )


def test_resolve_area_name_default_and_fallback_paths() -> None:
    rows = [
        {"name": "Niederösterreich", "name_norm": "niederösterreich", "nuts_id": "AT12", "level": 2},
        {"name": "SharedName", "name_norm": "sharedname", "nuts_id": "ATX1", "level": 1},
    ]

    out = resolve_area_name(
        name="Niederösterreich",
        nuts_level=None,
        default_level=2,
        country="AUT",
        catalog_rows=rows,
        normalize_name=normalize_area_name,
        validate_level=_validate,
        default_areas_supplier=lambda: tuple(),
    )
    assert out == ("niederösterreich", "AT12", 2)

    fallback = resolve_area_name(
        name="SharedName",
        nuts_level=None,
        default_level=2,
        country="AUT",
        catalog_rows=rows,
        normalize_name=normalize_area_name,
        validate_level=_validate,
        default_areas_supplier=lambda: tuple(("Niederösterreich",)),
    )
    assert fallback == ("sharedname", "ATX1", 1)


def test_resolve_area_name_ambiguity_and_not_found_errors() -> None:
    rows = [
        {"name": "SharedName", "name_norm": "sharedname", "nuts_id": "ATX1", "level": 1},
        {"name": "SharedName", "name_norm": "sharedname", "nuts_id": "ATX3", "level": 3},
    ]

    with pytest.raises(ValueError, match="ambiguous across NUTS levels"):
        resolve_area_name(
            name="SharedName",
            nuts_level=None,
            default_level=2,
            country="AUT",
            catalog_rows=rows,
            normalize_name=normalize_area_name,
            validate_level=_validate,
            default_areas_supplier=lambda: tuple(("Wien", "Niederösterreich")),
        )

    with pytest.raises(ValueError, match="not found at default NUTS level 2"):
        resolve_area_name(
            name="Nope",
            nuts_level=None,
            default_level=2,
            country="AUT",
            catalog_rows=rows,
            normalize_name=normalize_area_name,
            validate_level=_validate,
            default_areas_supplier=lambda: tuple(("Wien", "Niederösterreich")),
        )


class _TaggedRegion(str):
    def __new__(cls, value: str, level: int):
        obj = str.__new__(cls, value)
        obj.level = int(level)
        return obj


def test_select_area_decision_paths() -> None:
    resolver_calls: list[tuple[str, int | None]] = []

    def _resolve(name: str, level: int | None):
        resolver_calls.append((name, level))
        return "norm", "AT13", 2

    cleared = select_area_decision(
        area=None,
        nuts_level=None,
        nuts_area_name_type=_TaggedRegion,
        validate_level=_validate,
        resolve_name=_resolve,
    )
    assert cleared.area_name is None
    assert cleared.area_id == "__all__"

    picked = select_area_decision(
        area="Wien",
        nuts_level=None,
        nuts_area_name_type=_TaggedRegion,
        validate_level=_validate,
        resolve_name=_resolve,
    )
    assert picked.area_name == "Wien"
    assert picked.area_id == "AT13"
    assert resolver_calls[-1] == ("Wien", None)

    tagged = select_area_decision(
        area=_TaggedRegion("Sankt Pölten", 3),
        nuts_level=None,
        nuts_area_name_type=_TaggedRegion,
        validate_level=_validate,
        resolve_name=_resolve,
    )
    assert tagged.area_name == "Sankt Pölten"
    assert resolver_calls[-1] == ("Sankt Pölten", 3)

    with pytest.raises(ValueError, match="Conflicting NUTS levels"):
        select_area_decision(
            area=_TaggedRegion("Sankt Pölten", 3),
            nuts_level=1,
            nuts_area_name_type=_TaggedRegion,
            validate_level=_validate,
            resolve_name=_resolve,
        )

    with pytest.raises(ValueError, match="area cannot be empty"):
        select_area_decision(
            area="   ",
            nuts_level=None,
            nuts_area_name_type=_TaggedRegion,
            validate_level=_validate,
            resolve_name=_resolve,
        )

    with pytest.raises(ValueError, match="area must be a string or None"):
        select_area_decision(
            area=123,
            nuts_level=None,
            nuts_area_name_type=_TaggedRegion,
            validate_level=_validate,
            resolve_name=_resolve,
        )
