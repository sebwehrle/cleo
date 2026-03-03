"""GWA-specific I/O and CRS helper functions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr
from rasterio.crs import CRS
from rasterio.enums import Resampling
import cleo.net

logger = logging.getLogger(__name__)

GWA_HEIGHTS = [10, 50, 100, 150, 200]
GWA_GIS_API_BASE_URL = "https://globalwindatlas.info/api/gis/country"

_GWA_LAYER_NAME_BY_INTERNAL = {
    "weibull_A": "combined-Weibull-A",
    "weibull_k": "combined-Weibull-k",
    "rho": "air-density",
}


def fetch_gwa_crs(iso3: str) -> str:
    """Fetch CRS from Global Wind Atlas GeoJSON API for a given country."""
    url = f"https://globalwindatlas.info/api/gdal/country/geojson?areaId={iso3}"
    headers = {
        "Accept": "application/geo+json,application/json;q=0.9,*/*;q=0.8",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://globalwindatlas.info",
        "Referer": "https://globalwindatlas.info/en/download/gis-files",
        "User-Agent": "Mozilla/5.0",
    }

    try:
        response = cleo.net.http_get(url, headers=headers, timeout=(10, 60), stream=False)
        response.raise_for_status()
        payload = response.json()
    except cleo.net.RequestException as e:
        raise RuntimeError(f"Failed to fetch GWA CRS for {iso3} from {url}: {e}") from e

    # Handle double-encoded JSON (response.json() returns a string containing JSON)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse GWA GeoJSON response for country {iso3}: {e}") from e

    try:
        crs = payload["crs"]["properties"]["name"]
    except (KeyError, TypeError):
        raise ValueError(f"CRS missing from GWA GeoJSON response for country {iso3}")

    return crs


def ensure_crs_from_gwa(ds, iso3: str):
    """Ensure a raster DataArray/Dataset has a CRS set, fetching from GWA if needed."""
    if ds.rio.crs is not None:
        return ds

    crs = fetch_gwa_crs(iso3)
    return ds.rio.write_crs(crs)


def _crs_cache_path(atlas, iso3: str) -> Path:
    """Get the path for CRS cache file.

    Args:
        atlas: Atlas instance.
        iso3: ISO3 country code.

    Returns:
        Path to the CRS cache file.
    """
    return Path(atlas.path) / "intermediates" / "crs_cache" / f"{iso3}.wkt"


def _load_or_fetch_gwa_crs(atlas, iso3: str) -> CRS:
    """Load CRS from cache or fetch from GWA API.

    Fetch-once semantics: fetches from network only if cache is missing,
    then persists to cache. Subsequent calls read from cache.

    Args:
        atlas: Atlas instance.
        iso3: ISO3 country code.

    Returns:
        rasterio CRS object.

    Raises:
        RuntimeError: If fetch fails or cache is empty/corrupt.
    """
    cache = _crs_cache_path(atlas, iso3)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        wkt = cache.read_text(encoding="utf-8").strip()
        if not wkt:
            raise RuntimeError(f"Empty CRS cache file: {cache}")
        return CRS.from_wkt(wkt)

    # Fetch from network
    try:
        crs_str = fetch_gwa_crs(iso3)
        crs = CRS.from_string(crs_str)
    except (cleo.net.RequestException, RuntimeError, ValueError, TypeError, OSError) as e:
        logger.error(
            "Failed to fetch GWA CRS and cache is unavailable.",
            extra={"iso3": iso3, "cache_path": str(cache)},
            exc_info=True,
        )
        raise RuntimeError(f"Failed to fetch GWA CRS for {iso3}; cache missing at {cache}. Error: {e}") from e

    # Persist to cache
    try:
        cache.write_text(crs.to_wkt(), encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as e:
        logger.error(
            "Fetched CRS but failed to persist CRS cache.",
            extra={"iso3": iso3, "cache_path": str(cache)},
            exc_info=True,
        )
        raise RuntimeError(f"Fetched CRS but failed to persist cache at {cache}: {e}") from e

    return crs


def _required_gwa_files(atlas) -> list[tuple[str, Path]]:
    """Get list of required GWA files for wind unification.

    Args:
        atlas: Atlas instance.

    Returns:
        List of (source_id, path) tuples for all required files.
    """
    raw_dir = Path(atlas.path) / "data" / "raw" / atlas.country
    required = []

    for h in GWA_HEIGHTS:
        required.extend(
            [
                (f"gwa:file:weibull_A:{h}", raw_dir / f"{atlas.country}_combined-Weibull-A_{h}.tif"),
                (f"gwa:file:weibull_k:{h}", raw_dir / f"{atlas.country}_combined-Weibull-k_{h}.tif"),
                (f"gwa:file:rho:{h}", raw_dir / f"{atlas.country}_air-density_{h}.tif"),
            ]
        )

    return required


def _assert_all_required_gwa_present(atlas) -> list[tuple[str, Path]]:
    """Assert all required GWA files exist. Fail fast with all missing paths.

    Args:
        atlas: Atlas instance.

    Returns:
        List of (source_id, path) tuples for all required files.

    Raises:
        FileNotFoundError: If any required files are missing (lists ALL missing paths).
    """
    req = _required_gwa_files(atlas)
    missing = [str(p) for (_sid, p) in req if not p.exists()]

    if missing:
        raise FileNotFoundError("Missing required GWA files:\n" + "\n".join(missing))

    return req


def _missing_required_gwa_files(atlas) -> list[tuple[str, Path]]:
    """Return required GWA source entries that are missing on disk.

    :param atlas: Atlas-like object with ``path`` and ``country`` attributes.
    :type atlas: object
    :returns: Missing ``(source_id, file_path)`` entries.
    :rtype: list[tuple[str, pathlib.Path]]
    """
    return [(sid, path) for sid, path in _required_gwa_files(atlas) if not path.exists()]


def _download_url_for_required_source(*, country: str, source_id: str) -> str:
    """Build Global Wind Atlas GIS download URL for a required source ID.

    :param country: ISO3 country code.
    :type country: str
    :param source_id: Required source identifier in ``gwa:file:<layer>:<height>`` format.
    :type source_id: str
    :returns: Download URL for the requested GWA layer/height.
    :rtype: str
    :raises ValueError: If source ID format or layer key is invalid.
    """
    parts = source_id.split(":")
    if len(parts) != 4 or parts[0] != "gwa" or parts[1] != "file":
        raise ValueError(f"Invalid required GWA source_id format: {source_id!r}")

    layer_key = parts[2]
    height = parts[3]
    layer_name = _GWA_LAYER_NAME_BY_INTERNAL.get(layer_key)
    if layer_name is None:
        raise ValueError(f"Unknown required GWA layer key {layer_key!r} in source_id={source_id!r}")

    return f"{GWA_GIS_API_BASE_URL}/{country}/{layer_name}/{height}"


def ensure_required_gwa_files(atlas, *, auto_download: bool = True) -> list[tuple[str, Path]]:
    """Ensure all required GWA rasters are present for wind materialization.

    This helper supports the canonical build flow:
    - if required files already exist, return immediately,
    - otherwise optionally attempt to download missing files from GWA GIS API,
    - finally fail fast with a complete missing-file list if still incomplete.

    :param atlas: Atlas-like object with ``path`` and ``country`` attributes.
    :type atlas: object
    :param auto_download: If ``True`` (default), attempt downloading missing
        files before raising an error.
    :type auto_download: bool
    :returns: Full required source list when all files are present.
    :rtype: list[tuple[str, pathlib.Path]]
    :raises FileNotFoundError: If required files remain missing after optional
        download attempts.
    """
    req = _required_gwa_files(atlas)
    missing = _missing_required_gwa_files(atlas)

    if missing and auto_download:
        logger.info(
            "Required GWA files are missing; attempting automatic download.",
            extra={"country": atlas.country, "missing_count": len(missing)},
        )
        downloaded = 0
        for source_id, path in missing:
            url = _download_url_for_required_source(country=atlas.country, source_id=source_id)
            try:
                cleo.net.download_to_path(url, path, overwrite=False)
                downloaded += 1
            except Exception:
                logger.warning(
                    "Failed to auto-download required GWA file.",
                    extra={"source_id": source_id, "path": str(path), "url": url},
                    exc_info=True,
                )
        logger.info(
            "Finished GWA auto-download attempt.",
            extra={"country": atlas.country, "downloaded_count": downloaded, "attempted_count": len(missing)},
        )

    remaining = [str(path) for _sid, path in req if not path.exists()]
    if remaining:
        raise FileNotFoundError("Missing required GWA files:\n" + "\n".join(remaining))
    return req


def _open_gwa_raster(
    atlas,
    path: Path,
    *,
    iso3: str,
    target_crs,
    ref_da: xr.DataArray | None = None,
    clip_geom=None,
    resampling: str = "bilinear",
) -> xr.DataArray:
    """Open a GWA raster with CRS cache support.

    Args:
        atlas: Atlas instance.
        path: Path to the raster file.
        iso3: ISO3 country code for CRS lookup.
        target_crs: Target CRS to reproject to.
        ref_da: Reference DataArray to match grid to (optional).
        clip_geom: Geometry to clip to (optional).
        resampling: Resampling method (default "bilinear").

    Returns:
        DataArray with CRS set, reprojected/clipped as specified.
    """
    da = rxr.open_rasterio(path, chunks=None).squeeze(drop=True)

    # Ensure CRS is set (use cache if needed)
    if da.rio.crs is None:
        crs = _load_or_fetch_gwa_crs(atlas, iso3)
        da = da.rio.write_crs(crs)

    # Map resampling string to enum
    resampling_enum = getattr(Resampling, resampling, Resampling.bilinear)

    # Reproject to target CRS if needed
    from cleo.spatial import crs_equal

    if not crs_equal(da.rio.crs, target_crs):
        da = da.rio.reproject(target_crs, nodata=np.nan, resampling=resampling_enum)

    # Clip to geometry if provided
    if clip_geom is not None:
        from cleo.spatial import to_crs_if_needed

        if hasattr(clip_geom, "geometry"):
            # GeoDataFrame
            clip_geom = to_crs_if_needed(clip_geom, target_crs)
            da = da.rio.clip(clip_geom.geometry, drop=True)
        else:
            # Assume iterable of geometries
            da = da.rio.clip(clip_geom, drop=True)

    # Match to reference grid if provided
    if ref_da is not None:
        da = da.rio.reproject_match(ref_da, nodata=np.nan, resampling=resampling_enum)

    # Convert nodata to NaN
    nodata = da.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        da = da.where(da != nodata, np.nan)

    return da
