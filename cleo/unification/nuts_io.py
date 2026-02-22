"""NUTS vector and catalog I/O helpers extracted from ``cleo.unify``."""

from __future__ import annotations

import re as _re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pycountry as pct


def _read_vector_file(path: Path | str) -> gpd.GeoDataFrame:
    """Read a vector file (shapefile, GeoJSON, etc.).

    Args:
        path: Path to vector file.

    Returns:
        GeoDataFrame with loaded features.
    """
    return gpd.read_file(path)


def _read_nuts_region_catalog(atlas) -> list[dict[str, Any]]:
    """Read NUTS region catalog for the atlas country from raw NUTS files.

    Returns rows with keys: ``name``, ``name_norm``, ``nuts_id``, ``level``.
    Only levels 1, 2, and 3 are included.

    :param atlas: Atlas-like object exposing ``path`` and ``country``.
    :returns: Deterministically sorted region catalog rows.
    :raises FileNotFoundError: If no NUTS shapefile is available.
    """
    nuts_dir = Path(atlas.path) / "data" / "nuts"
    shp_files = sorted(nuts_dir.rglob("*.shp")) if nuts_dir.exists() else []
    if not shp_files:
        raise FileNotFoundError(
            f"NUTS shapefile not found under {nuts_dir}. "
            "Run NUTS download/extract first (e.g. cleo.loaders.load_nuts)."
        )

    nuts = _read_vector_file(shp_files[0])
    alpha_2 = pct.countries.get(alpha_3=atlas.country).alpha_2
    if alpha_2 is None:
        return []

    country_regions = nuts[nuts["CNTR_CODE"] == alpha_2]

    out: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for _, row in country_regions.iterrows():
        name = str(row.get("NAME_LATN", "")).strip()
        nuts_id = str(row.get("NUTS_ID", "")).strip()
        level_raw = row.get("LEVL_CODE")
        if not name or not nuts_id:
            continue
        try:
            level = int(level_raw)
        except (TypeError, ValueError):
            continue
        if level not in (1, 2, 3):
            continue

        key = (level, nuts_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "name": name,
                "name_norm": _re.sub(r"\s+", " ", name).casefold(),
                "nuts_id": nuts_id,
                "level": level,
            }
        )

    out.sort(key=lambda r: (int(r["level"]), str(r["name"]).casefold(), str(r["nuts_id"])))
    return out
