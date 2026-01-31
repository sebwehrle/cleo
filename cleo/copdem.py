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

    Uses:
        lon0 = floor(min_lon), lon1 = ceil(max_lon) - 1
        lat0 = floor(min_lat), lat1 = ceil(max_lat) - 1

    Returns all tile_ids for lat_deg in [lat0..lat1], lon_deg in [lon0..lon1],
    sorted lexicographically.

    :param min_lon: Minimum longitude of bounding box
    :param min_lat: Minimum latitude of bounding box
    :param max_lon: Maximum longitude of bounding box
    :param max_lat: Maximum latitude of bounding box
    :return: List of tile IDs sorted lexicographically
    :rtype: list[str]
    """
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

    Cache layout (option B):
        <base_dir>/<iso3>/copdem/<tile_id>/<tile_id>.tif

    Parent directories are NOT created by this function.

    :param base_dir: Base directory for the cache
    :param iso3: ISO 3166-1 alpha-3 country code
    :param tile_id: Tile ID string
    :return: Path to the cached tile file
    :rtype: Path
    """
    return Path(base_dir) / iso3 / "copdem" / tile_id / f"{tile_id}.tif"


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

    If the file already exists and overwrite is False, returns immediately
    without making a network call.

    Downloads atomically: writes to .tif.part then renames to .tif.

    :param base_dir: Base directory for the cache
    :param iso3: ISO 3166-1 alpha-3 country code
    :param tile_id: Tile ID string
    :param timeout_s: Request timeout in seconds (default 60.0)
    :param overwrite: If True, re-download even if file exists
    :return: Path to the downloaded tile file
    :rtype: Path
    :raises FileNotFoundError: If tile does not exist (HTTP status != 200)
    """
    dest = copdem_tile_cache_path(base_dir, iso3, tile_id)

    # Return cached file if exists and not overwriting
    if dest.exists() and not overwrite:
        return dest

    # Ensure parent directories exist
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = copdem_tile_url(tile_id)
    response = requests.get(url, stream=True, timeout=timeout_s)

    if response.status_code != 200:
        raise FileNotFoundError(
            f"Copernicus DEM tile not found: {tile_id} (HTTP {response.status_code})"
        )

    # Download atomically: write to .part file then rename
    part_path = dest.with_suffix(".tif.part")
    with open(part_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    part_path.replace(dest)
    return dest


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

    Opens all tile GeoTIFFs, mosaics them into one raster, then reprojects
    to match the reference DataArray's grid (CRS, transform, shape).

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

    # Open all tile datasets for mosaicking
    tile_datasets = []
    try:
        for path in tile_paths:
            ds = rasterio.open(path)
            if ds.crs is None:
                raise ValueError(f"CRS missing in tile: {path}")
            tile_datasets.append(ds)

        # Mosaic all tiles
        mosaic_arr, mosaic_transform = merge(tile_datasets)

        # Get CRS from first tile (all should have same CRS)
        mosaic_crs = tile_datasets[0].crs

    finally:
        # Close all datasets
        for ds in tile_datasets:
            ds.close()

    # Convert mosaic to xarray DataArray with rioxarray metadata
    # mosaic_arr has shape (bands, height, width), we take first band
    mosaic_2d = mosaic_arr[0]

    # Create coordinates based on transform
    height, width = mosaic_2d.shape
    # For pixel-center coordinates
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

    # Check reference has CRS
    if reference_da.rio.crs is None:
        raise ValueError("CRS missing in reference DataArray")

    # Reproject to match reference using nearest-neighbor for determinism
    result = mosaic_da.rio.reproject_match(
        reference_da,
        resampling=Resampling.nearest,
    )
    result.name = "elevation"

    return result
