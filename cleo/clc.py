"""Corine Land Cover (CLC) source handling and cache preparation."""

from __future__ import annotations

import re
from pathlib import Path

from cleo.net import download_to_path
from cleo.unification.clc_io import (
    load_clc_codes,
    prepare_clc_to_wind_grid,
    wind_reference_template,
)


CLC_SOURCES: dict[str, dict[str, str]] = {
    "clc2018": {
        "filename": "U2018_CLC2018_V2020_20u1.tif",
        # Direct download URL is user/environment specific in many setups.
        # Pass url=... to Atlas.build_clc(...) when first populating cache.
        "url": "",
        "crs": "EPSG:3035",
    }
}


def source_cache_path(atlas_path: Path, source: str) -> Path:
    """Return cache path for the full CLC source raster."""
    if source not in CLC_SOURCES:
        raise ValueError(
            f"Unsupported CLC source {source!r}. Supported: {sorted(CLC_SOURCES)}"
        )
    return Path(atlas_path) / "data" / "raw" / "clc" / CLC_SOURCES[source]["filename"]


def prepared_country_path(atlas_path: Path, country: str, source: str) -> Path:
    """Return cache path for country-prepared CLC raster aligned to wind grid."""
    return (
        Path(atlas_path)
        / "data"
        / "raw"
        / str(country)
        / "clc"
        / f"{country}_{source}_windgrid.tif"
    )


def default_category_name(atlas_path: Path, code: int) -> str | None:
    """Return default variable name for a CLC code, or ``None`` if unknown."""
    labels = load_clc_codes(Path(atlas_path))
    if int(code) not in labels:
        return None
    label = labels[int(code)].strip().casefold()
    name = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return name or None


def materialize_clc(
    atlas,
    *,
    source: str = "clc2018",
    url: str | None = None,
    force_download: bool = False,
    force_prepare: bool = False,
) -> Path:
    """Prepare a country-cropped CLC raster aligned to wind/GWA grid.

    Returns path to prepared GeoTIFF in ``<atlas>/data/raw/<ISO3>/clc/``.
    """
    if source not in CLC_SOURCES:
        raise ValueError(
            f"Unsupported CLC source {source!r}. Supported: {sorted(CLC_SOURCES)}"
        )

    source_path = source_cache_path(Path(atlas.path), source)
    source_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = prepared_country_path(Path(atlas.path), atlas.country, source)
    prepared.parent.mkdir(parents=True, exist_ok=True)

    if prepared.exists() and not force_prepare:
        return prepared

    if (not source_path.exists()) or force_download:
        download_url = url or CLC_SOURCES[source].get("url")
        if not download_url:
            raise RuntimeError(
                f"No download URL configured for source {source!r}; "
                "provide url=... explicitly."
            )
        download_to_path(download_url, source_path)

    if not getattr(atlas, "_canonical_ready", False):
        atlas.build_canonical()

    ref = wind_reference_template(atlas)
    return prepare_clc_to_wind_grid(
        source_path=source_path,
        prepared_path=prepared,
        ref=ref,
        source_crs=CLC_SOURCES[source]["crs"],
    )
