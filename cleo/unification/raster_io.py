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
    aoi_gdf,
) -> xr.DataArray:
    """Open local elevation GeoTIFF and align to wind grid.

    This function loads an existing elevation raster and reprojects/clips
    it to match the wind reference grid. Does NOT fetch GWA CRS.

    Args:
        atlas: Atlas instance.
        elev_path: Path to the elevation GeoTIFF.
        wind_ref: Reference DataArray from wind for grid alignment.
        aoi_gdf: GeoDataFrame for AOI clipping (or None).

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

    # Clip/mask to AOI BEFORE matching grid
    if aoi_gdf is not None:
        from cleo.spatial import to_crs_if_needed

        aoi_in_da_crs = to_crs_if_needed(aoi_gdf, da.rio.crs)
        da = da.rio.clip(aoi_in_da_crs.geometry, drop=False)

    # Match to wind grid
    da = da.rio.reproject_match(wind_ref, resampling=Resampling.bilinear, nodata=np.nan)

    # Convert nodata to NaN
    nodata = da.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        da = da.where(da != nodata, np.nan)

    return da


def _build_copdem_elevation(
    atlas,
    wind_ref: xr.DataArray,
    aoi_gdf,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Build CopDEM elevation data aligned to wind grid.

    Downloads CopDEM tiles, mosaics them, and reprojects to match the wind
    reference grid.

    Args:
        atlas: Atlas instance.
        wind_ref: Reference DataArray from wind for grid alignment.
        aoi_gdf: GeoDataFrame for AOI clipping (or None).

    Returns:
        Tuple of (elevation DataArray, metadata dict).

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

    # Build mosaic aligned to wind_ref
    elevation = build_copdem_elevation_like(wind_ref, tile_paths)
    downloaded_tile_ids = [path.stem for path in tile_paths]

    # Build metadata for deterministic inputs_id
    meta = {
        "provider": "copernicus",
        "version": "GLO-30",
        "bbox_4326": list(bbox_4326),
        "tile_ids": downloaded_tile_ids,
        "clip": "aoi" if aoi_gdf is not None else "none",
    }

    return elevation, meta


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


def _build_copdem_mosaic(
    tile_paths: list[Path],
    reference_da: xr.DataArray,
) -> xr.DataArray:
    """
    Mosaic Copernicus DEM tiles and reproject to match a reference raster.

    This function performs raw I/O (rasterio.open) and is centralized here.
    Other modules must delegate here for CopDEM mosaicing.

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
    from rasterio.merge import merge

    if not tile_paths:
        raise ValueError("tile_paths cannot be empty")

    tile_datasets = []
    nodata = None
    try:
        for path in tile_paths:
            ds = rasterio.open(path)
            if ds.crs is None:
                raise ValueError(f"CRS missing in tile: {path}")
            tile_datasets.append(ds)

        nodata = tile_datasets[0].nodata
        mosaic_arr, mosaic_transform = merge(tile_datasets)
        mosaic_crs = tile_datasets[0].crs

    finally:
        for ds in tile_datasets:
            try:
                ds.close()
            except (OSError, RuntimeError, ValueError):
                logger.debug("Failed to close temporary raster tile dataset.", exc_info=True)

    mosaic_2d = mosaic_arr[0]

    # Mask nodata / masked arrays to NaN
    if np.ma.isMaskedArray(mosaic_2d):
        mosaic_2d = mosaic_2d.filled(np.nan).astype("float32")
    else:
        mosaic_2d = mosaic_2d.astype("float32", copy=False)

    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        mosaic_2d = np.where(mosaic_2d == float(nodata), np.nan, mosaic_2d)

    height, width = mosaic_2d.shape
    cols = np.arange(width)
    rows = np.arange(height)
    xs = mosaic_transform.c + mosaic_transform.a * (cols + 0.5)
    ys = mosaic_transform.f + mosaic_transform.e * (rows + 0.5)

    mosaic_da = xr.DataArray(
        mosaic_2d,
        dims=["y", "x"],
        coords={"y": ys, "x": xs},
        name="elevation",
    )
    mosaic_da = mosaic_da.rio.write_crs(mosaic_crs)
    mosaic_da = mosaic_da.rio.write_transform(mosaic_transform)
    mosaic_da = mosaic_da.rio.write_nodata(np.nan)

    if reference_da.rio.crs is None:
        raise ValueError("CRS missing in reference DataArray")

    result = mosaic_da.rio.reproject_match(
        reference_da,
        resampling=Resampling.bilinear,
        nodata=np.nan,
    )
    result.name = "elevation"
    return result


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
