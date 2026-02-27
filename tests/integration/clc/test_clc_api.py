"""Integration tests for CLC materialization and Landscape CLC category API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import rioxarray as rxr
from rasterio.crs import CRS

from cleo import Atlas
from cleo.clc import CLC_SOURCES
from cleo.unification.gwa_io import GWA_HEIGHTS


def _create_raster(
    path: Path,
    *,
    data: np.ndarray,
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (4000000, 2600000, 4100000, 2700000),
    nodata: float = np.nan,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], data.shape[1], data.shape[0])
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": CRS.from_epsg(crs_epsg),
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def _create_all_gwa_rasters(atlas_path: Path, country: str = "AUT") -> None:
    raw_dir = atlas_path / "data" / "raw" / country
    for h in GWA_HEIGHTS:
        _create_raster(
            raw_dir / f"{country}_combined-Weibull-A_{h}.tif",
            data=np.full((20, 20), 8.0 + h * 0.01, dtype=np.float32),
        )
        _create_raster(
            raw_dir / f"{country}_combined-Weibull-k_{h}.tif",
            data=np.full((20, 20), 2.0 + h * 0.001, dtype=np.float32),
        )
        rho = np.full((20, 20), 1.2 - h * 0.0001, dtype=np.float32)
        rho[:2, :2] = np.nan
        _create_raster(
            raw_dir / f"{country}_air-density_{h}.tif",
            data=rho,
        )


def _create_elevation_raster(atlas_path: Path, country: str = "AUT") -> None:
    _create_raster(
        atlas_path / "data" / "raw" / country / f"{country}_elevation_w_bathymetry.tif",
        data=np.full((24, 24), 300.0, dtype=np.float32),
        bounds=(3995000, 2595000, 4105000, 2705000),
    )


def _create_clc_source(atlas_path: Path) -> Path:
    filename = CLC_SOURCES["clc2018"]["filename"]
    path = atlas_path / "data" / "raw" / "clc" / filename
    clc = np.full((50, 50), 231.0, dtype=np.float32)
    clc[10:30, 10:30] = 311.0
    clc[30:45, 30:45] = 312.0
    _create_raster(
        path,
        data=clc,
        bounds=(3990000, 2590000, 4110000, 2710000),
    )
    return path


def _build_atlas(tmp_path: Path) -> Atlas:
    atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
    _create_all_gwa_rasters(tmp_path)
    _create_elevation_raster(tmp_path)
    return atlas


def test_materialize_clc_and_add_categories(tmp_path: Path) -> None:
    atlas = _build_atlas(tmp_path)
    _create_clc_source(tmp_path)

    prepared = atlas.build_clc()
    assert prepared.exists()

    atlas.landscape.add_clc_category("all")
    ds = atlas.landscape.data
    assert "land_cover" in ds.data_vars

    atlas.landscape.add_clc_category(311)
    ds = atlas.landscape.data
    assert "broad_leaved_forest" in ds.data_vars
    values = ds["broad_leaved_forest"].values
    finite = values[np.isfinite(values)]
    assert np.all(np.isin(np.unique(finite), np.array([0.0, 1.0], dtype=np.float32)))

    atlas.landscape.add_clc_category([311, 312], name="forest")
    ds = atlas.landscape.data
    assert "forest" in ds.data_vars


def test_add_clc_category_requires_name_for_multi_code(tmp_path: Path) -> None:
    atlas = _build_atlas(tmp_path)
    _create_clc_source(tmp_path)
    atlas.build_clc()

    with pytest.raises(ValueError, match="name is required"):
        atlas.landscape.add_clc_category([311, 312])


def test_materialize_clc_downloads_when_source_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = _build_atlas(tmp_path)

    def _fake_download(url: str, out_path: Path) -> None:
        del url
        clc = np.full((60, 60), 231.0, dtype=np.float32)
        clc[20:40, 20:40] = 311.0
        _create_raster(
            out_path,
            data=clc,
            bounds=(3990000, 2590000, 4110000, 2710000),
        )

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)
    prepared = atlas.build_clc(url="https://example.test/clc.tif")
    assert prepared.exists()

    # Prepared raster should be reprojected/cropped relative to wind grid.
    da = rxr.open_rasterio(prepared).squeeze(drop=True)
    assert da.sizes["y"] <= 20
    assert da.sizes["x"] <= 20
