"""NUTS catalog policy helpers for Atlas delegation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping


CatalogRow = dict[str, Any]


def _catalog_rows_copy(rows: tuple[CatalogRow, ...]) -> list[CatalogRow]:
    """Return defensive mutable copies of cached catalog rows."""
    return [dict(row) for row in rows]


def _coerce_catalog_rows(
    rows: list[Any],
    *,
    valid_levels: tuple[int, ...],
) -> list[CatalogRow]:
    """Normalize ``cleo_region_catalog_json`` rows to the contract shape."""
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


def load_nuts_region_catalog(
    *,
    cached_rows: tuple[CatalogRow, ...] | None,
    landscape_store_path: Path,
    valid_levels: tuple[int, ...],
    read_store_attrs: Callable[[Path], Mapping[str, Any]],
    read_raw_catalog: Callable[[], list[CatalogRow]],
    log_debug: Callable[[str], None],
) -> tuple[list[CatalogRow], tuple[CatalogRow, ...]]:
    """Load catalog from attrs fast-path or raw fallback, with cache plumbing."""
    if cached_rows is not None:
        return _catalog_rows_copy(cached_rows), cached_rows

    # Fast path: precomputed attrs on landscape store.
    if landscape_store_path.exists():
        try:
            attrs = read_store_attrs(landscape_store_path)

            catalog_json = attrs.get("cleo_region_catalog_json")
            if catalog_json:
                rows = json.loads(catalog_json)
                if isinstance(rows, list):
                    catalog = _coerce_catalog_rows(rows, valid_levels=valid_levels)
                    if catalog:
                        cache = tuple(catalog)
                        return _catalog_rows_copy(cache), cache

        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            log_debug(
                "Failed to load NUTS region catalog from landscape store attrs; "
                "falling back to raw NUTS catalog loading."
            )

    catalog = read_raw_catalog()
    if not catalog:
        raise ValueError(
            "No NUTS regions available for this atlas. "
            "Ensure NUTS data is present (e.g. run cleo.loaders.load_nuts)."
        )

    cache = tuple(dict(row) for row in catalog)
    return _catalog_rows_copy(cache), cache
