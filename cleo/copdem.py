# Copernicus DEM helpers
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import rioxarray as rxr
import xarray as xr
import requests


def copdem_tile_id(lat_deg: int, lon_deg: int, resolution_arcsec: int = 10) -> str:
    """
    Generate Copernicus DEM tile ID for a given lat/lon degree.

    Format matches the Copernicus DEM S3 naming:
        "Copernicus_DSM_COG_{resolution}_{NS}{abs_lat:02d}_00_{EW}{abs_lon:03d}_00_DEM"

    :param lat_deg: Latitude in integer degrees (tile's SW corner)
    :param lon_deg: Longitude in integer degrees (tile's SW corner)
    :param resolution_arcsec: Resolution in arcseconds (default 10)
    :return: Tile ID string
    :rtype: str
    """
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    abs_lat = abs(lat_deg)
    abs_lon = abs(lon_deg)
    return f"Copernicus_DSM_COG_{resolution_arcsec}_{ns}{abs_lat:02d}_00_{ew}{abs_lon:03d}_00_DEM"


def tiles_for_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list:
    """
    Return list of Copernicus DEM tile IDs that intersect the given bounding box.

    Convention:
    - `min_*` are inclusive
    - `max_*` are exclusive
    - bbox must have strictly positive width and height

    Uses:
        lon0 = floor(min_lon), lon1 = ceil(max_lon) - 1
        lat0 = floor(min_lat), lat1 = ceil(max_lat) - 1

    Returns all tile_ids for lat_deg in [lat0..lat1], lon_deg in [lon0..lon1],
    sorted lexicographically.

    :raises ValueError: If bbox is degenerate (max <= min)
    """
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

    tiles = []
    for lat_deg in range(lat0, lat1 + 1):
        for lon_deg in range(lon0, lon1 + 1):
            tiles.append(copdem_tile_id(lat_deg, lon_deg))

    tiles.sort()
    return tiles


def copdem_tile_url(tile_id: str) -> str:
    """
    Return the HTTPS URL for a Copernicus DEM tile on S3.

    :param tile_id: Tile ID string
    :return: Full URL to the tile TIF file
    :rtype: str
    """
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{tile_id}/{tile_id}.tif"


def copdem_tile_cache_path(base_dir, iso3: str, tile_id: str) -> Path:
    """
    Return the local cache path for a Copernicus DEM tile.

    Canonical cache layout (project-wide consistency):
        <base_dir>/data/raw/<iso3>/copdem/<tile_id>/<tile_id>.tif

    Parent directories are NOT created by this function.

    :param base_dir: Base directory for the project workdir
    :param iso3: ISO 3166-1 alpha-3 country code
    :param tile_id: Tile ID string
    :return: Path to the cached tile file
    :rtype: Path
    """
    return Path(base_dir) / "data" / "raw" / iso3 / "copdem" / tile_id / f"{tile_id}.tif"


def download_copdem_tile(
    base_dir,
    iso3: str,
    tile_id: str,
    *,
    timeout_s: float = 60.0,
    overwrite: bool = False,
) -> Path:
    """
    Download a Copernicus DEM tile to the local cache.

    Atomicity/cleanup contract:
    - Write to a temporary *.part file and rename on success.
    - On any failure, delete the *.part file (no stale partial artifacts).
    - Always close the HTTP response.

    :param base_dir: Base directory for the cache
    :param iso3: ISO 3166-1 alpha-3 country code
    :param tile_id: CopDEM tile ID
    :param timeout_s: Requests timeout in seconds
    :param overwrite: If True, re-download even if file exists
    :return: Path to the downloaded tile file
    :rtype: Path
    :raises FileNotFoundError: If tile does not exist (HTTP status == 404)
    :raises requests.RequestException: For other request failures
    """
    dest = copdem_tile_cache_path(base_dir, iso3, tile_id)

    if dest.exists() and not overwrite:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    url = copdem_tile_url(tile_id)
    response = None
    part_path = dest.with_suffix(".tif.part")

    try:
        response = requests.get(url, stream=True, timeout=timeout_s)

        if response.status_code == 404:
            raise FileNotFoundError(
                f"Copernicus DEM tile not found: {tile_id} (HTTP 404)"
            )
        response.raise_for_status()

        with open(part_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)

        part_path.replace(dest)
        return dest

    except Exception:
        # Best-effort cleanup of partial file
        try:
            if part_path.exists():
                part_path.unlink()
        except Exception:
            pass
        raise

    finally:
        if response is not None and hasattr(response, "close"):
            try:
                response.close()
            except Exception:
                pass


def download_copdem_tiles_for_bbox(
    base_dir,
    iso3: str,
    *,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overwrite: bool = False,
) -> list:
    """
    Download all Copernicus DEM tiles needed for a bounding box.

    Computes the required tiles using tiles_for_bbox, then downloads each
    tile in lexicographic order using download_copdem_tile.

    :param base_dir: Base directory for the cache
    :param iso3: ISO 3166-1 alpha-3 country code
    :param min_lon: Minimum longitude of bounding box
    :param min_lat: Minimum latitude of bounding box
    :param max_lon: Maximum longitude of bounding box
    :param max_lat: Maximum latitude of bounding box
    :param overwrite: If True, re-download even if files exist
    :return: List of Paths to downloaded tiles in the same order as tile_ids
    :rtype: list[Path]
    :raises FileNotFoundError: If any tile does not exist (propagated from download_copdem_tile)
    """
    tile_ids = tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)
    paths = []
    for tile_id in tile_ids:
        path = download_copdem_tile(base_dir, iso3, tile_id, overwrite=overwrite)
        paths.append(path)
    return paths


def build_copdem_elevation_like(reference_da, tile_paths):
    """
    Mosaic Copernicus DEM tiles and reproject to match a reference raster.

    Contracts:
    - tile_paths must be non-empty
    - CRS must be present in both tiles and reference
    - tile nodata (if defined) must be masked to NaN before reprojection
    - elevation is continuous -> bilinear resampling by default

    :param reference_da: Reference xarray DataArray with rioxarray metadata
    :type reference_da: xarray.DataArray
    :param tile_paths: List of Paths to Copernicus DEM tile GeoTIFFs
    :type tile_paths: list[Path]
    :return: DataArray with elevation data matching reference grid
    :rtype: xarray.DataArray
    :raises ValueError: If tile_paths is empty or CRS is missing
    :raises RuntimeError: If mosaicking fails
    """
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
            except Exception:
                pass

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
