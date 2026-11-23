"""Raster and local elevation I/O helpers."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)


def copdem_tile_id(lat_deg: int, lon_deg: int, resolution_arcsec: int = 10) -> str:
    """Generate Copernicus DEM tile ID for a given integer lat/lon degree."""
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    abs_lat = abs(lat_deg)
    abs_lon = abs(lon_deg)
    return f"Copernicus_DSM_COG_{resolution_arcsec}_{ns}{abs_lat:02d}_00_{ew}{abs_lon:03d}_00_DEM"


def tiles_for_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list[str]:
    """Return sorted CopDEM tile IDs intersecting bbox (min inclusive, max exclusive)."""
    if max_lon <= min_lon or max_lat <= min_lat:
        raise ValueError(
            f"Degenerate bbox: require max_lon>min_lon and max_lat>min_lat "
            f"(got min_lon={min_lon}, max_lon={max_lon}, min_lat={min_lat}, max_lat={max_lat}). "
            "Max bounds are treated as exclusive."
        )

    lon0 = math.floor(min_lon)
    lon1 = math.ceil(max_lon) - 1
    lat0 = math.floor(min_lat)
    lat1 = math.ceil(max_lat) - 1

    out: list[str] = []
    for lat_deg in range(lat0, lat1 + 1):
        for lon_deg in range(lon0, lon1 + 1):
            out.append(copdem_tile_id(lat_deg, lon_deg))
    out.sort()
    return out


def copdem_tile_url(tile_id: str) -> str:
    """Return HTTPS URL for CopDEM tile on public S3 bucket."""
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{tile_id}/{tile_id}.tif"


def copdem_tile_cache_path(base_dir: Path | str, iso3: str, tile_id: str) -> Path:
    """Return canonical local cache path for a CopDEM tile."""
    return Path(base_dir) / "data" / "raw" / iso3 / "copdem" / tile_id / f"{tile_id}.tif"


def download_copdem_tile(
    base_dir: Path | str,
    iso3: str,
    tile_id: str,
    *,
    timeout_s: float = 60.0,
    overwrite: bool = False,
) -> Path:
    """Download a CopDEM tile using the canonical network helper."""
    from cleo.net import download_to_path

    dest = copdem_tile_cache_path(base_dir, iso3, tile_id)
    url = copdem_tile_url(tile_id)

    try:
        return download_to_path(
            url,
            dest,
            timeout=timeout_s,
            overwrite=overwrite,
        )
    except FileNotFoundError:
        # Re-raise with tile-specific context (historic behavior).
        raise FileNotFoundError(f"Copernicus DEM tile not found: {tile_id} (HTTP 404)")


def download_copdem_tiles_for_bbox(
    base_dir: Path | str,
    iso3: str,
    *,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overwrite: bool = False,
) -> list[Path]:
    """Download CopDEM tiles for a bbox in deterministic lexicographic order.

    :param base_dir: Atlas base directory for cache storage.
    :type base_dir: Path | str
    :param iso3: ISO3 country code used in cache paths.
    :type iso3: str
    :param min_lon: Minimum longitude (EPSG:4326).
    :type min_lon: float
    :param min_lat: Minimum latitude (EPSG:4326).
    :type min_lat: float
    :param max_lon: Maximum longitude (EPSG:4326, exclusive for tile planning).
    :type max_lon: float
    :param max_lat: Maximum latitude (EPSG:4326, exclusive for tile planning).
    :type max_lat: float
    :param overwrite: Whether to force re-download of cached tiles.
    :type overwrite: bool
    :returns: Existing/downloading tile paths that were resolved successfully.
    :rtype: list[Path]
    :raises FileNotFoundError: If no tile could be resolved for the requested bbox.
    """
    tile_ids = tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)
    paths: list[Path] = []
    missing_tile_ids: list[str] = []
    for tile_id in tile_ids:
        try:
            paths.append(download_copdem_tile(base_dir, iso3, tile_id, overwrite=overwrite))
        except FileNotFoundError:
            missing_tile_ids.append(tile_id)

    if missing_tile_ids and paths:
        logger.warning(
            "CopDEM: skipped %d unavailable tiles (HTTP 404); proceeding with %d tiles. Missing tile IDs: %s",
            len(missing_tile_ids),
            len(paths),
            ", ".join(missing_tile_ids),
        )

    if not paths:
        raise FileNotFoundError(
            "No Copernicus DEM tiles could be downloaded for bbox "
            f"({min_lon}, {min_lat}, {max_lon}, {max_lat}). "
            f"Missing tile IDs ({len(missing_tile_ids)}): {', '.join(missing_tile_ids)}"
        )

    return paths


def build_copdem_elevation_like(reference_da: xr.DataArray, tile_paths: list[Path]) -> xr.DataArray:
    """Build CopDEM elevation aligned to reference DataArray."""
    return _build_copdem_mosaic(tile_paths, reference_da)


def _open_local_elevation(
    atlas,
    elev_path: Path,
    wind_ref: xr.DataArray,
) -> xr.DataArray:
    """Open local elevation GeoTIFF and align to wind grid.

    This function loads an existing elevation raster and reprojects it
    to match the wind reference grid. Does NOT fetch GWA CRS.

    Args:
        atlas: Atlas instance.
        elev_path: Path to the elevation GeoTIFF.
        wind_ref: Reference DataArray from wind for grid alignment.

    Returns:
        DataArray with elevation values aligned to wind grid.
    """
    import warnings

    da_raw = rxr.open_rasterio(elev_path, parse_coordinates=True)
    da: xr.DataArray = da_raw.squeeze(drop=True)

    if da.rio.crs is None:
        # Safer default: assume wind CRS but warn clearly
        warnings.warn(
            f"Elevation raster {elev_path} has no CRS; assuming wind CRS {wind_ref.rio.crs}.",
            RuntimeWarning,
        )
        da = da.rio.write_crs(wind_ref.rio.crs)

    # Match to wind grid
    da = da.rio.reproject_match(wind_ref, resampling=Resampling.bilinear, nodata=np.nan)

    # Convert nodata to NaN
    nodata = da.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        da = da.where(da != nodata, np.nan)

    return da


def _resolve_copdem_tiles(
    atlas,
    wind_ref: xr.DataArray,
) -> tuple[list[Path], dict[str, Any]]:
    """Resolve the CopDEM tiles covering the wind grid (downloading missing ones) without reading them.

    Args:
        atlas: Atlas instance.
        wind_ref: Reference DataArray from wind for grid alignment.

    Returns:
        Tuple of (available tile paths, metadata dict for ``inputs_id``).

    Raises:
        FileNotFoundError: If CopDEM tiles cannot be downloaded.
    """
    from rasterio.warp import transform_bounds

    # Determine bbox in EPSG:4326
    bounds = wind_ref.rio.bounds()
    wind_crs = wind_ref.rio.crs
    if str(wind_crs) != "EPSG:4326":
        bbox_4326 = transform_bounds(wind_crs, "EPSG:4326", *bounds, densify_pts=21)
    else:
        bbox_4326 = bounds

    min_lon, min_lat, max_lon, max_lat = bbox_4326

    # Download tiles (uses cache; deterministic order)
    tile_paths = download_copdem_tiles_for_bbox(
        atlas.path,
        atlas.country,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
    )

    downloaded_tile_ids = [path.stem for path in tile_paths]

    # Build metadata for deterministic inputs_id
    meta = {
        "provider": "copernicus",
        "version": "GLO-30",
        "bbox_4326": list(bbox_4326),
        "tile_ids": downloaded_tile_ids,
        "clip": "none",  # kept for inputs_id stability; the base store is never clipped
    }

    return tile_paths, meta


def _open_raster(
    path: Path | str,
    *,
    parse_coordinates: bool = True,
    chunks: dict | None = None,
) -> xr.DataArray:
    """
    Open a raster file (GeoTIFF etc.) as an xarray DataArray.

    This is the centralized raw I/O entry point for raster reads.
    Other modules must delegate here instead of calling rxr.open_rasterio directly.

    Args:
        path: Path to the raster file.
        parse_coordinates: Whether to parse coordinates from the raster.
        chunks: Chunking specification for dask (None for eager).

    Returns:
        DataArray with raster data.
    """
    da_raw = rxr.open_rasterio(path, parse_coordinates=parse_coordinates, chunks=chunks)
    da: xr.DataArray = da_raw.squeeze(drop=True)
    return da


def _open_dataset(path: Path | str, **kwargs) -> xr.Dataset:
    """
    Open a NetCDF/HDF5 dataset as an xarray Dataset.

    This is the centralized raw I/O entry point for dataset reads.
    Other modules must delegate here instead of calling xr.open_dataset directly.

    Args:
        path: Path to the dataset file.
        **kwargs: Additional arguments passed to xr.open_dataset.

    Returns:
        Dataset with the loaded data.
    """
    return xr.open_dataset(path, **kwargs)


# Output cells per block side when mosaicking CopDEM: bounds peak memory by the block, not the country.
# At the ~250 m GWA grid resolution a block spans about 1.2 degrees, i.e. about 4500 x 4500 source pixels.
_COPDEM_BLOCK_CELLS = 512


def _build_copdem_mosaic(
    tile_paths: list[Path],
    reference_da: xr.DataArray,
) -> xr.DataArray:
    """
    Mosaic Copernicus DEM tiles and reproject to match a reference raster.

    This function performs raw I/O (rasterio.open) and is centralized here.
    Other modules must delegate here for CopDEM mosaicing.

    The reference grid is processed in blocks of ``_COPDEM_BLOCK_CELLS`` cells per side. For each
    block, only the tile window around it (plus a margin for the resampling kernel) is mosaicked at
    native resolution, on the pixel grid of a whole-country mosaic, and warped onto the block.

    Contracts:
    - tile_paths must be non-empty
    - CRS must be present in both tiles and reference
    - tile nodata (if defined) is masked to NaN before reprojection
    - elevation is continuous -> bilinear resampling

    Args:
        tile_paths: List of Paths to Copernicus DEM tile GeoTIFFs.
        reference_da: Reference xarray DataArray with rioxarray metadata.

    Returns:
        DataArray with elevation data matching reference grid.

    Raises:
        ValueError: If tile_paths is empty or CRS is missing.
        RuntimeError: If mosaicking fails.
    """
    import rasterio

    if not tile_paths:
        raise ValueError("tile_paths cannot be empty")
    if reference_da.rio.crs is None:
        raise ValueError("CRS missing in reference DataArray")

    ref_crs = reference_da.rio.crs
    ref_transform = reference_da.rio.transform(recalc=True)
    y_dim, x_dim = reference_da.rio.y_dim, reference_da.rio.x_dim
    height, width = reference_da.sizes[y_dim], reference_da.sizes[x_dim]
    out = np.full((height, width), np.nan, dtype="float32")

    tile_datasets = []
    try:
        for path in tile_paths:
            ds = rasterio.open(path)
            if ds.crs is None:
                raise ValueError(f"CRS missing in tile: {path}")
            tile_datasets.append(ds)
        # Pixel grid and extent of a whole-country mosaic, as rasterio.merge would build it: anchored at the
        # union's west/north edges, first tile's resolution, whole pixels.
        bounds = [ds.bounds for ds in tile_datasets]
        res_x, res_y = tile_datasets[0].res
        west, north = min(b.left for b in bounds), max(b.top for b in bounds)
        east = west + round((max(b.right for b in bounds) - west) / res_x) * res_x
        south = north - round((north - min(b.bottom for b in bounds)) / res_y) * res_y
        extent = (west, south, east, north)

        for row0 in range(0, height, _COPDEM_BLOCK_CELLS):
            for col0 in range(0, width, _COPDEM_BLOCK_CELLS):
                window = rasterio.windows.Window(
                    col0, row0, min(_COPDEM_BLOCK_CELLS, width - col0), min(_COPDEM_BLOCK_CELLS, height - row0)
                )
                out[window.toslices()] = _copdem_block(tile_datasets, extent, ref_crs, ref_transform, window)
    finally:
        for ds in tile_datasets:
            try:
                ds.close()
            except (OSError, RuntimeError, ValueError):
                logger.debug("Failed to close temporary raster tile dataset.", exc_info=True)

    result = xr.DataArray(
        out,
        dims=(y_dim, x_dim),
        coords={y_dim: reference_da.coords[y_dim].values, x_dim: reference_da.coords[x_dim].values},  # 1D
        name="elevation",
    )
    result = result.rio.write_crs(ref_crs).rio.write_transform(ref_transform)
    return result.rio.write_nodata(np.nan)


def _copdem_block(tile_datasets: list, extent: tuple, ref_crs, ref_transform, window) -> np.ndarray:
    """Mosaic the tiles around one reference-grid window and warp them onto it (bilinear).

    The window's mosaic lies on the pixel grid and within the extent of a whole-country mosaic, so
    coverage matches it. Values can differ slightly: GDAL's resampling weights depend on the source
    extent, and with mixed tile resolutions merge may pick a neighbouring source pixel for a window.

    :param tile_datasets: Open CopDEM tile datasets; the first one sets the mosaic resolution.
    :param extent: ``(west, south, east, north)`` of the whole-country mosaic (see caller).
    :param ref_crs: Reference CRS.
    :param ref_transform: Affine transform of the full reference grid.
    :param window: Reference-grid window to fill.
    :returns: float32 elevation for the window, NaN where no tile data.
    """
    from rasterio.merge import merge
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import transform as window_transform

    dst = np.full((int(window.height), int(window.width)), np.nan, dtype="float32")
    first = tile_datasets[0]
    res_x, res_y = first.res

    corners = [
        ref_transform * (col, row)
        for col in (window.col_off, window.col_off + window.width)
        for row in (window.row_off, window.row_off + window.height)
    ]
    xs, ys = zip(*corners)
    left, bottom, right, top = transform_bounds(ref_crs, first.crs, min(xs), min(ys), max(xs), max(ys), densify_pts=21)
    # Margin for the bilinear kernel, which spans about one output cell when downsampling.
    pad_x = 2 * max(res_x, (right - left) / window.width)
    pad_y = 2 * max(res_y, (top - bottom) / window.height)
    west, south, east, north = extent
    left = max(west, west + math.floor((left - pad_x - west) / res_x) * res_x)
    right = min(east, west + math.ceil((right + pad_x - west) / res_x) * res_x)
    top = min(north, north - math.floor((north - top - pad_y) / res_y) * res_y)
    bottom = max(south, north - math.ceil((north - bottom + pad_y) / res_y) * res_y)
    if left >= right or bottom >= top:
        return dst

    mosaic, mosaic_transform = merge(tile_datasets, bounds=(left, bottom, right, top))
    block = mosaic[0].astype("float32", copy=False)
    nodata = first.nodata
    if nodata is not None and not np.isnan(nodata):
        block = np.where(block == np.float32(nodata), np.float32(np.nan), block)

    reproject(
        source=block,
        destination=dst,
        src_transform=mosaic_transform,
        src_crs=first.crs,
        src_nodata=np.nan,
        dst_transform=window_transform(window, ref_transform),
        dst_crs=ref_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return dst


def _atomic_replace_variable_dir(store_path: Path, variable_name: str) -> None:
    """Atomically replace a variable directory in a zarr store.

    Removes the existing variable directory if present. This is used when
    if_exists="replace" to ensure clean replacement of variable data.

    The operation is atomic: the directory is removed in one operation.

    Args:
        store_path: Path to the zarr store root.
        variable_name: Name of the variable directory to remove.
    """
    import shutil

    var_dir = store_path / variable_name
    if var_dir.exists():
        shutil.rmtree(var_dir)
