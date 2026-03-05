"""NUTS catalog policy helpers for Atlas delegation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping


CatalogRow = dict[str, Any]
_AREA_CATALOG_ATTR = "cleo_area_catalog_json"
_AREA_CATALOG_COUNTRY_ATTR = "cleo_area_catalog_country_iso3"


def _catalog_rows_copy(rows: tuple[CatalogRow, ...]) -> list[CatalogRow]:
    """Return defensive mutable copies of cached catalog rows."""
    return [dict(row) for row in rows]


def _normalize_iso3(value: Any) -> str:
    """Normalize an ISO3-like value for equality checks."""
    return str(value).strip().upper()


def _coerce_catalog_rows(
    rows: list[Any],
    *,
    valid_levels: tuple[int, ...],
) -> list[CatalogRow]:
    """Normalize ``cleo_area_catalog_json`` rows to the contract shape."""
    catalog: list[CatalogRow] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            level = int(row.get("level"))
        except (TypeError, ValueError):
            continue
        if level not in valid_levels:
            continue
        name = str(row.get("name", "")).strip()
        name_norm = str(row.get("name_norm", "")).strip()
        nuts_id = str(row.get("nuts_id", "")).strip()
        if name and name_norm and nuts_id:
            catalog.append(
                {
                    "name": name,
                    "name_norm": name_norm,
                    "nuts_id": nuts_id,
                    "level": level,
                }
            )
    return catalog


def load_nuts_area_catalog(
    *,
    cached_rows: tuple[CatalogRow, ...] | None,
    cached_country_iso3: str | None,
    expected_country_iso3: str,
    landscape_store_path: Path,
    valid_levels: tuple[int, ...],
    read_store_attrs: Callable[[Path], Mapping[str, Any]],
    read_raw_catalog: Callable[[], list[CatalogRow]],
    log_debug: Callable[[str], None],
) -> tuple[list[CatalogRow], tuple[CatalogRow, ...], str]:
    """Load NUTS area catalog with country-scoped cache/attrs validation.

    :param cached_rows: In-memory cached rows from a previous call.
    :type cached_rows: tuple[dict[str, Any], ...] | None
    :param cached_country_iso3: Country code associated with ``cached_rows``.
    :type cached_country_iso3: str | None
    :param expected_country_iso3: Country code requested by the current atlas.
    :type expected_country_iso3: str
    :param landscape_store_path: Base landscape store path.
    :type landscape_store_path: pathlib.Path
    :param valid_levels: Accepted NUTS levels.
    :type valid_levels: tuple[int, ...]
    :param read_store_attrs: Store-attrs reader dependency.
    :type read_store_attrs: collections.abc.Callable
    :param read_raw_catalog: Raw NUTS fallback loader dependency.
    :type read_raw_catalog: collections.abc.Callable
    :param log_debug: Debug logger callback.
    :type log_debug: collections.abc.Callable
    :returns: ``(catalog_rows, cache_rows, cache_country_iso3)``.
    :rtype: tuple[list[dict[str, Any]], tuple[dict[str, Any], ...], str]
    """
    expected_iso3 = _normalize_iso3(expected_country_iso3)

    if cached_rows is not None and _normalize_iso3(cached_country_iso3) == expected_iso3:
        return _catalog_rows_copy(cached_rows), cached_rows, expected_iso3

    # Fast path: precomputed attrs on landscape store.
    if landscape_store_path.exists():
        try:
            attrs = read_store_attrs(landscape_store_path)
            attrs_country_iso3 = _normalize_iso3(attrs.get(_AREA_CATALOG_COUNTRY_ATTR, ""))
            if attrs_country_iso3 == expected_iso3:
                catalog_json = attrs.get(_AREA_CATALOG_ATTR)
                if catalog_json:
                    rows = json.loads(catalog_json)
                    if isinstance(rows, list):
                        catalog = _coerce_catalog_rows(rows, valid_levels=valid_levels)
                        if catalog:
                            cache = tuple(catalog)
                            return _catalog_rows_copy(cache), cache, expected_iso3

        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            log_debug(
                "Failed to load NUTS area catalog from landscape store attrs; falling back to raw NUTS catalog loading."
            )

    catalog = read_raw_catalog()
    if not catalog:
        raise ValueError(
            "No NUTS areas available for this atlas. Ensure NUTS data is present (e.g. run cleo.loaders.load_nuts)."
        )

    cache = tuple(dict(row) for row in catalog)
    return _catalog_rows_copy(cache), cache, expected_iso3
