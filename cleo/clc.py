"""Corine Land Cover (CLC) source handling and cache preparation."""

from __future__ import annotations

import re
from urllib.parse import urlencode
from pathlib import Path

import numpy as np

from cleo.net import download_to_path
from cleo.unification.clc_io import (
    load_clc_codes,
    prepare_clc_to_wind_grid,
    wind_reference_template,
)


_CLC2018_LAEA_EXPORT_URL = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_LAEA/MapServer/export"
)
_CLC2018_LAEA_MAX_IMAGE_SIZE = 4100


CLC_SOURCES: dict[str, dict[str, str]] = {
    "clc2018": {
        "filename": "U2018_CLC2018_V2020_20u1.tif",
        # If left empty, materialize_clc() computes a deterministic default URL
        # against the official EEA CLC2018 ArcGIS raster service.
        "url": "",
        "crs": "EPSG:3035",
    }
}


def source_cache_path(atlas_path: Path, source: str) -> Path:
    """Return cache path for the full CLC source raster."""
    if source not in CLC_SOURCES:
        raise ValueError(f"Unsupported CLC source {source!r}. Supported: {sorted(CLC_SOURCES)}")
    return Path(atlas_path) / "data" / "raw" / "clc" / CLC_SOURCES[source]["filename"]


def prepared_country_path(atlas_path: Path, country: str, source: str) -> Path:
    """Return cache path for country-prepared CLC raster aligned to wind grid."""
    return Path(atlas_path) / "data" / "raw" / str(country) / "clc" / f"{country}_{source}_windgrid.tif"


def default_category_name(atlas_path: Path, code: int) -> str | None:
    """Return default variable name for a CLC code, or ``None`` if unknown."""
    labels = load_clc_codes(Path(atlas_path))
    if int(code) not in labels:
        return None
    label = labels[int(code)].strip().casefold()
    name = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return name or None


def _axis_resolution(values: np.ndarray) -> float:
    """Return absolute pixel spacing inferred from axis coordinates.

    :param values: 1D axis coordinate values.
    :type values: numpy.ndarray
    :returns: Positive pixel spacing in axis units.
    :rtype: float
    :raises RuntimeError: If spacing cannot be inferred.
    """
    if values.size < 2:
        raise RuntimeError("Unable to infer CLC export pixel spacing from single-cell axis.")
    diffs = np.diff(values.astype(float))
    diffs = np.abs(diffs[np.nonzero(diffs)])
    if diffs.size == 0:
        raise RuntimeError("Unable to infer CLC export pixel spacing from degenerate coordinates.")
    spacing = float(np.median(diffs))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise RuntimeError("Unable to infer a positive finite CLC export pixel spacing.")
    return spacing


def _epsg_from_crs(crs_obj) -> int | None:
    """Extract EPSG integer code from a CRS-like object.

    :param crs_obj: CRS-like object from ``ref.rio.crs``.
    :returns: EPSG integer code, or ``None`` if unavailable.
    :rtype: int | None
    """
    if crs_obj is None:
        return None
    to_epsg = getattr(crs_obj, "to_epsg", None)
    if callable(to_epsg):
        code = to_epsg()
        if code is not None:
            return int(code)
    match = re.search(r"EPSG:(\d+)", str(crs_obj), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _default_clc2018_download_url(ref) -> str:
    """Build deterministic default CLC2018 export URL from wind-grid geometry.

    :param ref: Wind reference raster aligned to atlas grid.
    :returns: ArcGIS export URL returning CLC2018 raster in GeoTIFF format.
    :rtype: str
    :raises RuntimeError: If grid axes/CRS are unsupported for deterministic export.
    """
    if "x" not in ref.coords or "y" not in ref.coords:
        raise RuntimeError("wind reference template must expose x/y coordinates for CLC default download.")
    width = int(ref.sizes.get("x", 0))
    height = int(ref.sizes.get("y", 0))
    if width <= 0 or height <= 0:
        raise RuntimeError("wind reference template has empty x/y axes; cannot build CLC default download URL.")
    if width > _CLC2018_LAEA_MAX_IMAGE_SIZE or height > _CLC2018_LAEA_MAX_IMAGE_SIZE:
        raise RuntimeError(
            "Wind grid exceeds CLC export service size limit "
            f"({_CLC2018_LAEA_MAX_IMAGE_SIZE}x{_CLC2018_LAEA_MAX_IMAGE_SIZE}); provide url=... explicitly."
        )

    crs_epsg = _epsg_from_crs(getattr(ref.rio, "crs", None))
    if crs_epsg is None:
        raise RuntimeError("Unable to infer EPSG code for wind reference template; provide url=... explicitly.")

    x_vals = np.asarray(ref.coords["x"].data, dtype=float)
    y_vals = np.asarray(ref.coords["y"].data, dtype=float)
    x_res = _axis_resolution(x_vals)
    y_res = _axis_resolution(y_vals)
    xmin = float(np.min(x_vals) - 0.5 * x_res)
    xmax = float(np.max(x_vals) + 0.5 * x_res)
    ymin = float(np.min(y_vals) - 0.5 * y_res)
    ymax = float(np.max(y_vals) + 0.5 * y_res)

    params = {
        "f": "image",
        "format": "tiff",
        "layers": "show:1",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": str(crs_epsg),
        "imageSR": str(crs_epsg),
        "size": f"{width},{height}",
        "transparent": "false",
    }
    return f"{_CLC2018_LAEA_EXPORT_URL}?{urlencode(params)}"


def materialize_clc(
    atlas,
    *,
    source: str = "clc2018",
    url: str | None = None,
    force_download: bool = False,
    force_prepare: bool = False,
) -> Path:
    """Prepare a country-cropped CLC raster aligned to wind/GWA grid.

    :param atlas: Atlas-like object with path/country/build state.
    :param source: CLC source identifier.
    :param url: Optional explicit source URL override.
    :param force_download: If ``True``, re-download source raster.
    :param force_prepare: If ``True``, rebuild prepared country cache.
    :returns: Path to prepared GeoTIFF in ``<atlas>/data/raw/<ISO3>/clc/``.
    :rtype: pathlib.Path
    :raises ValueError: If ``source`` is unknown.
    :raises RuntimeError: If required source download metadata is unavailable.
    """
    if source not in CLC_SOURCES:
        raise ValueError(f"Unsupported CLC source {source!r}. Supported: {sorted(CLC_SOURCES)}")

    source_path = source_cache_path(Path(atlas.path), source)
    source_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = prepared_country_path(Path(atlas.path), atlas.country, source)
    prepared.parent.mkdir(parents=True, exist_ok=True)

    if prepared.exists() and not force_prepare:
        return prepared

    canonical_ready = bool(getattr(atlas, "_canonical_ready", False))
    ref = None
    if (not source_path.exists()) or force_download:
        download_url = url or CLC_SOURCES[source].get("url")
        if not download_url and source == "clc2018":
            if not canonical_ready:
                atlas.build_canonical()
                canonical_ready = True
            ref = wind_reference_template(atlas)
            download_url = _default_clc2018_download_url(ref)
        if not download_url:
            raise RuntimeError(f"No download URL configured for source {source!r}; provide url=... explicitly.")
        download_to_path(download_url, source_path)

    if not canonical_ready:
        atlas.build_canonical()
        canonical_ready = True

    if ref is None:
        ref = wind_reference_template(atlas)
    return prepare_clc_to_wind_grid(
        source_path=source_path,
        prepared_path=prepared,
        ref=ref,
        source_crs=CLC_SOURCES[source]["crs"],
    )
