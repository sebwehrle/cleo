"""The block-wise CopDEM mosaic matches mosaicking all tiles at once and only ever merges a window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin

from cleo.unification import raster_io


def _write_tiles(tmp_path: Path, *, like_copdem: bool) -> list[Path]:
    """Tiles of 1 x 1 degree over lon 10-13, lat 49-51 with smooth terrain; tile (12, 51) is missing (sea).

    ``like_copdem=False``: uniform 0.01 degree pixels, nodata -9999 with a nodata patch.
    ``like_copdem=True``: as cached CopDEM tiles, no nodata value, edges offset by half a pixel, and the
    northern row with a coarser longitude spacing (as above 50 N); the first (lexicographic) tile is fine.
    """
    paths = []
    for lat in (50, 51):
        x_res = 0.015 if like_copdem and lat == 51 else 0.01
        for lon in (10, 11, 12):
            if (lon, lat) == (12, 51):
                continue
            west = lon - x_res / 2 if like_copdem else lon
            north = lat + 0.005 if like_copdem else lat
            width = round(1 / x_res)
            xs = west + x_res * (np.arange(width) + 0.5)
            ys = north - 0.01 * (np.arange(100) + 0.5)
            data = (300 + 200 * np.sin(xs * 3)[None, :] * np.cos(ys * 4)[:, None]).astype("float32")
            nodata = None if like_copdem else -9999.0
            if nodata is not None and (lon, lat) == (11, 50):
                data[40:60, 40:60] = nodata
            path = tmp_path / f"N{lat}_E{lon:03d}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=100,
                width=width,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(west, north, x_res, 0.01),
                nodata=nodata,
            ) as dst:
                dst.write(data, 1)
            paths.append(path)
    return paths


def _reference(*, ncols: int, south_up: bool = False) -> xr.DataArray:
    """2 km EPSG:3035 grid from (10.2 E, 50.8 N) across tile seams, the missing tile and the nodata patch."""
    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform(10.2, 50.8)
    x = x0 + 2_000 * np.arange(ncols)
    y = y0 - 2_000 * np.arange(75)
    if south_up:
        y = y[::-1]
    ref = xr.DataArray(np.zeros((75, ncols), "float32"), dims=("y", "x"), coords={"y": y, "x": x})
    return ref.rio.write_crs("EPSG:3035")


def _one_shot(tile_paths: list[Path], reference: xr.DataArray) -> np.ndarray:
    """The previous implementation: mosaic every tile in memory, then warp."""
    datasets = [rasterio.open(p) for p in tile_paths]
    nodata = datasets[0].nodata
    mosaic, transform = merge(datasets)
    for ds in datasets:
        ds.close()
    arr = mosaic[0].astype("float32")
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    h, w = arr.shape
    da = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "y": transform.f + transform.e * (np.arange(h) + 0.5),
            "x": transform.c + transform.a * (np.arange(w) + 0.5),
        },
    )
    da = da.rio.write_crs("EPSG:4326").rio.write_transform(transform).rio.write_nodata(np.nan)
    return da.rio.reproject_match(reference, resampling=Resampling.bilinear, nodata=np.nan).values


@pytest.mark.parametrize(
    ("like_copdem", "ncols", "south_up"),
    [
        (False, 80, False),  # inside the tiles, with a nodata patch
        (True, 130, False),  # CopDEM-like tiles; the grid extends east beyond the tiles
        (True, 75, True),  # south-up reference, partial last block
    ],
)
def test_block_mosaic_matches_one_shot_mosaic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, like_copdem: bool, ncols: int, south_up: bool
) -> None:
    tile_paths = _write_tiles(tmp_path, like_copdem=like_copdem)
    reference = _reference(ncols=ncols, south_up=south_up)
    expected = _one_shot(tile_paths, reference)

    # One block covering the grid reproduces the whole-country mosaic exactly (same extent, grid, nodata).
    monkeypatch.setattr(raster_io, "_COPDEM_BLOCK_CELLS", 10_000)
    np.testing.assert_array_equal(raster_io.build_copdem_elevation_like(reference, tile_paths).values, expected)

    merged_cells: list[int] = []

    def _merge(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        mosaic, transform = merge(*args, **kwargs)
        merged_cells.append(mosaic[0].size)
        return mosaic, transform

    monkeypatch.setattr("rasterio.merge.merge", _merge)
    monkeypatch.setattr(raster_io, "_COPDEM_BLOCK_CELLS", 16)
    result = raster_io.build_copdem_elevation_like(reference, tile_paths)

    # Block-wise: every merge covers a window, never the whole tile set (5 tiles of about 100 x 100 cells);
    # blocks entirely beyond the tiles (grid extending east) are skipped without a merge.
    n_blocks = -(-75 // 16) * -(-ncols // 16)
    assert 1 < len(merged_cells) <= n_blocks and max(merged_cells) < 5 * 100 * 100 / 4
    if ncols == 130:
        assert len(merged_cells) < n_blocks
    assert result.dims == ("y", "x") and result.name == "elevation"
    assert result.rio.crs == reference.rio.crs
    # Coverage is identical; values differ slightly because GDAL's resampling weights depend on the source
    # extent and, with mixed tile resolutions, merge may pick a neighbouring source pixel for a window.
    assert np.isfinite(expected).any()
    np.testing.assert_array_equal(np.isnan(result.values), np.isnan(expected))
    assert np.nanmean(np.abs(result.values - expected)) < 0.5
