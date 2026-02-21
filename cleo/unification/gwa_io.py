"""GWA-specific I/O and CRS helper functions extracted from ``cleo.unify``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr
from rasterio.crs import CRS
from rasterio.enums import Resampling

# Public constant kept for backward-compatible imports from ``cleo.unify``
GWA_HEIGHTS = [10, 50, 100, 150, 200]


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
    import cleo.loaders

    cache = _crs_cache_path(atlas, iso3)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        wkt = cache.read_text(encoding="utf-8").strip()
        if not wkt:
            raise RuntimeError(f"Empty CRS cache file: {cache}")
        return CRS.from_wkt(wkt)

    # Fetch from network
    try:
        crs_str = cleo.loaders.fetch_gwa_crs(iso3)
        crs = CRS.from_string(crs_str)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch GWA CRS for {iso3}; cache missing at {cache}. Error: {e}"
        ) from e

    # Persist to cache
    try:
        cache.write_text(crs.to_wkt(), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(
            f"Fetched CRS but failed to persist cache at {cache}: {e}"
        ) from e

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
        required.extend([
            (f"gwa:file:weibull_A:{h}", raw_dir / f"{atlas.country}_combined-Weibull-A_{h}.tif"),
            (f"gwa:file:weibull_k:{h}", raw_dir / f"{atlas.country}_combined-Weibull-k_{h}.tif"),
            (f"gwa:file:rho:{h}", raw_dir / f"{atlas.country}_air-density_{h}.tif"),
        ])

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
        raise FileNotFoundError(
            "Missing required GWA files:\n" + "\n".join(missing)
        )

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
