"""Region-selection policy helpers for Atlas delegation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


CatalogRow = dict[str, Any]


@dataclass(frozen=True)
class SelectionDecision:
    """Resolved region-selection target for Atlas state application."""

    region_name: str | None
    region_id: str


def normalize_region_name(name: str) -> str:
    """Normalize region name for lookup semantics."""
    return re.sub(r"\s+", " ", name.strip()).casefold()


def validate_nuts_level(level: int, *, valid_levels: tuple[int, ...]) -> int:
    """Validate and normalize NUTS level."""
    try:
        level_i = int(level)
    except (TypeError, ValueError) as e:
        raise ValueError(f"NUTS level must be an integer, got {level!r}.") from e
    if level_i not in valid_levels:
        raise ValueError(
            f"Unsupported NUTS level {level_i!r}; expected one of {valid_levels}."
        )
    return level_i


def nuts_regions_level_names(
    *,
    level: int,
    country: str,
    catalog_rows: list[CatalogRow],
    validate_level: Callable[[int], int],
    make_region_name: Callable[[str, int], Any],
) -> tuple[Any, ...]:
    """Return deterministic unique names for the requested NUTS level."""
    level_i = validate_level(level)
    rows = [row for row in catalog_rows if int(row["level"]) == level_i]
    if not rows:
        raise ValueError(
            f"No NUTS level {level_i} regions found for atlas country {country!r}."
        )

    seen = set()
    out: list[Any] = []
    for row in sorted(rows, key=lambda r: (str(r["name"]).casefold(), str(r["nuts_id"]))):
        key = str(row["name_norm"])
        if key in seen:
            continue
        seen.add(key)
        out.append(make_region_name(str(row["name"]), level_i))
    return tuple(out)


def resolve_region_name(
    *,
    name: str,
    region_level: int | None,
    default_level: int,
    country: str,
    catalog_rows: list[CatalogRow],
    normalize_name: Callable[[str], str],
    validate_level: Callable[[int], int],
    default_regions_supplier: Callable[[], tuple[Any, ...]],
) -> tuple[str, str, int]:
    """Resolve region to ``(normalized_name, region_id, level)``."""
    name_norm = normalize_name(name)
    matches = [row for row in catalog_rows if row["name_norm"] == name_norm]

    if region_level is not None:
        level_i = validate_level(region_level)
        matches_level = [row for row in matches if int(row["level"]) == level_i]
        if len(matches_level) == 1:
            row = matches_level[0]
            return name_norm, str(row["nuts_id"]), level_i
        if len(matches_level) > 1:
            raise ValueError(
                f"Region '{name}' is ambiguous within NUTS level {level_i}; "
                f"matches IDs: {[str(r['nuts_id']) for r in matches_level]}."
            )
        raise ValueError(
            f"Region '{name}' not found in NUTS level {level_i} for country {country!r}."
        )

    matches_default = [row for row in matches if int(row["level"]) == default_level]
    if len(matches_default) == 1:
        row = matches_default[0]
        return name_norm, str(row["nuts_id"]), default_level
    if len(matches_default) > 1:
        raise ValueError(
            f"Region '{name}' is ambiguous at default NUTS level {default_level}; "
            f"pass region_level explicitly."
        )

    if len(matches) == 1:
        row = matches[0]
        return name_norm, str(row["nuts_id"]), int(row["level"])
    if len(matches) > 1:
        levels = sorted({int(r["level"]) for r in matches})
        raise ValueError(
            f"Region '{name}' is ambiguous across NUTS levels {levels}; "
            f"pass region_level explicitly."
        )

    available_regions = default_regions_supplier()
    available = [str(r) for r in available_regions[:20]]
    suffix = "..." if len(available_regions) > 20 else ""
    raise ValueError(
        f"Region '{name}' not found at default NUTS level {default_level}. "
        f"Available level-{default_level} regions: {available}{suffix}"
    )


def select_region_decision(
    *,
    region: object,
    region_level: int | None,
    nuts_region_name_type: type,
    validate_level: Callable[[int], int],
    resolve_name: Callable[[str, int | None], tuple[str, str, int]],
) -> SelectionDecision:
    """Resolve user selection arguments into Atlas state decision."""
    if region is None:
        return SelectionDecision(region_name=None, region_id="__all__")

    inferred_level: int | None = None
    if isinstance(region, nuts_region_name_type):
        inferred_level = int(region.level)
    elif not isinstance(region, str):
        raise ValueError(f"region must be a string or None, got {type(region).__name__}")

    if region_level is not None:
        region_level = validate_level(region_level)
    if inferred_level is not None and region_level is None:
        region_level = inferred_level
    elif inferred_level is not None and region_level != inferred_level:
        raise ValueError(
            f"Conflicting NUTS levels for region {region!r}: "
            f"region carries level={inferred_level}, but region_level={region_level}."
        )

    stripped = region.strip()
    if not stripped:
        raise ValueError("region cannot be empty or whitespace-only")

    _name_norm, region_id, _resolved_level = resolve_name(stripped, region_level)
    return SelectionDecision(region_name=stripped, region_id=region_id)
