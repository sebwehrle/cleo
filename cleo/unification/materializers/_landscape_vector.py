"""Vector source handling helpers for landscape materialization.

This private module contains functions for loading, normalizing, hashing,
and managing vector (GeoDataFrame) sources for landscape variables.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cleo.spatial import canonical_crs_str, to_crs_if_needed


def _load_vector_shape(shape):
    """Load a vector shape from path-like input or GeoDataFrame-like object."""
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - exercised in optional-deps envs
        raise RuntimeError("geopandas is required for vector rasterization support.") from exc

    if isinstance(shape, (str, Path)):
        return gpd.read_file(shape)

    if isinstance(shape, gpd.GeoDataFrame):
        return shape.copy()

    raise TypeError("shape must be a path (str|Path) or a geopandas.GeoDataFrame.")


def _vector_values_for_column(gdf, *, column: str | None) -> list[float]:
    """Return per-feature burn values for vector rasterization."""
    if column is None:
        return [1.0] * len(gdf)

    if column not in gdf.columns:
        available = [c for c in gdf.columns if c != "geometry"]
        raise ValueError(f"Column {column!r} not found in shape. Available columns: {available!r}")

    vals = gdf[column].to_numpy()
    if not np.issubdtype(vals.dtype, np.number):
        raise TypeError(f"Column {column!r} must be numeric, got dtype={vals.dtype!r}.")
    out: list[float] = []
    for idx, raw in enumerate(vals.tolist()):
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"Column {column!r} contains non-finite value at row {idx}: {value!r}.")
        out.append(value)
    return out


def _vector_semantic_payload(
    gdf,
    *,
    column: str | None,
    all_touched: bool,
) -> dict[str, Any]:
    """Build deterministic semantic payload for vector source hashing."""
    if gdf.crs is None:
        raise ValueError("Input shape has no CRS; cannot rasterize safely.")

    values = _vector_values_for_column(gdf, column=column)
    features: list[dict[str, Any]] = []
    for geom, value in zip(gdf.geometry.tolist(), values, strict=True):
        if geom is None:
            raise ValueError("Input shape contains null geometry; cannot rasterize.")
        features.append(
            {
                "geometry_wkb_hex": geom.wkb_hex,
                "value": value,
            }
        )

    return {
        "schema_version": 1,
        "crs": canonical_crs_str(gdf.crs),
        "column": column,
        "all_touched": bool(all_touched),
        "features": features,
    }


def _vector_semantic_hash(
    gdf,
    *,
    column: str | None,
    all_touched: bool,
) -> str:
    """Compute deterministic semantic hash for a vector source."""
    payload = _vector_semantic_payload(
        gdf,
        column=column,
        all_touched=all_touched,
    )
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _canonical_vector_source_artifact(
    atlas,
    *,
    shape,
    column: str | None,
    all_touched: bool,
) -> tuple[Path, str]:
    """Normalize vector input to canonical artifact path + semantic fingerprint."""
    gdf = _load_vector_shape(shape)
    if gdf.crs is None:
        raise ValueError("Input shape has no CRS; cannot rasterize safely.")

    gdf = to_crs_if_needed(gdf, atlas.crs)
    gdf = gdf.reset_index(drop=True)

    semantic_hash = _vector_semantic_hash(
        gdf,
        column=column,
        all_touched=all_touched,
    )

    out_dir = Path(atlas.path) / "intermediates" / "vector_sources"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{semantic_hash}.geojson"

    if not out_path.exists():
        cols = ["geometry"] if column is None else [column, "geometry"]
        gdf.loc[:, cols].to_file(out_path, driver="GeoJSON")

    return out_path, semantic_hash
