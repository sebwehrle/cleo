"""CopDEM compatibility facade.

Canonical CopDEM planning/download/mosaic ownership lives in
``cleo.unification.raster_io``. This module keeps stable helper names while
delegating behavior to canonical unification helpers.
"""

from __future__ import annotations

from pathlib import Path


def copdem_tile_id(lat_deg: int, lon_deg: int, resolution_arcsec: int = 10) -> str:
    from cleo.unification.raster_io import copdem_tile_id as _copdem_tile_id

    return _copdem_tile_id(lat_deg, lon_deg, resolution_arcsec=resolution_arcsec)


def tiles_for_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list[str]:
    from cleo.unification.raster_io import tiles_for_bbox as _tiles_for_bbox

    return _tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)


def copdem_tile_url(tile_id: str) -> str:
    from cleo.unification.raster_io import copdem_tile_url as _copdem_tile_url

    return _copdem_tile_url(tile_id)


def copdem_tile_cache_path(base_dir: Path | str, iso3: str, tile_id: str) -> Path:
    from cleo.unification.raster_io import copdem_tile_cache_path as _copdem_tile_cache_path

    return _copdem_tile_cache_path(base_dir, iso3, tile_id)


def download_copdem_tile(
    base_dir: Path | str,
    iso3: str,
    tile_id: str,
    *,
    timeout_s: float = 60.0,
    overwrite: bool = False,
) -> Path:
    from cleo.unification.raster_io import download_copdem_tile as _download_copdem_tile

    return _download_copdem_tile(
        base_dir,
        iso3,
        tile_id,
        timeout_s=timeout_s,
        overwrite=overwrite,
    )


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
    from cleo.unification.raster_io import download_copdem_tiles_for_bbox as _download_copdem_tiles_for_bbox

    return _download_copdem_tiles_for_bbox(
        base_dir,
        iso3,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        overwrite=overwrite,
    )


def build_copdem_elevation_like(reference_da, tile_paths: list[Path]):
    from cleo.unification.raster_io import build_copdem_elevation_like as _build_copdem_elevation_like

    return _build_copdem_elevation_like(reference_da, tile_paths)
