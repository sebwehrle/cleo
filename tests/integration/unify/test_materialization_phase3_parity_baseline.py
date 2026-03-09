"""Phase-3 deterministic parity baseline for unifier materialization paths.

This suite locks golden expectations before PR3/PR4/PR5 behavior moves.
The fixture is deterministic (fixed synthetic rasters + fixed mtimes).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS
from shapely.geometry import box

from cleo import Atlas


_FIXED_MTIME_NS = 1704067200000000000  # 2024-01-01T00:00:00Z
_GWA_HEIGHTS = [10, 50, 100, 150, 200]
_FIXED_TURBINES = [
    "Enercon.E101.3050",
    "Enercon.E115.3000",
    "Enercon.E138.3500",
    "Enercon.E160.5560",
    "Enercon.E40.500",
    "Enercon.E82.3000",
    "Vestas.V100.1800",
    "Vestas.V100.2000",
    "Vestas.V112.3075",
    "Vestas.V150.4200",
]
_GOLDEN = {
    "wind_grid_id": "e3c31174a7256f36",
    "wind_inputs_id": "f657fcb7a043c9ee",
    "wind_subset_checksum": "d83475c443ac887100a9f7164978dabc06d0666c9093cf541d53f6388c185da0",
    "wind_template_checksum": "d83475c443ac887100a9f7164978dabc06d0666c9093cf541d53f6388c185da0",
    "land_valid_mask_checksum": "834a709ba2534ebe3ee1397fd4f7bd288b2acc1d20a08d6c862dcd99b6f04400",
    "land_elevation_checksum": "d83475c443ac887100a9f7164978dabc06d0666c9093cf541d53f6388c185da0",
    "region_grid_id": "e8a0b0cd5ad8d179",
    "region_wind_subset_checksum": "6f3cb4f2fc142458842a0deb85b1d25a2252382bad6296716b99981951a99c57",
    "region_land_elevation_checksum": "2a8f937d682719c9f7f16119cdfe7a15a274512c66ca2b72aa896c9ad46829cd",
}


def _set_fixed_mtime(root: Path) -> None:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            os.utime(p, ns=(_FIXED_MTIME_NS, _FIXED_MTIME_NS))


def _create_gwa_raster(
    path: Path,
    *,
    shape: tuple[int, int] = (40, 40),
    bounds: tuple[float, float, float, float] = (4000000, 2600000, 4200000, 2800000),
    crs_epsg: int = 3035,
    fill_value: float | None = None,
    add_nodata_region: bool = True,
) -> None:
    if fill_value is not None:
        data = np.full(shape, fill_value, dtype=np.float32)
    else:
        data = np.random.rand(*shape).astype(np.float32) * 10 + 1
    if add_nodata_region:
        data[:3, :3] = np.nan

    transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], shape[1], shape[0])
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "crs": CRS.from_epsg(crs_epsg),
        "transform": transform,
        "nodata": np.nan,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_all_gwa_rasters(
    atlas_path: Path,
    *,
    country: str,
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> None:
    raw_dir = atlas_path / "data" / "raw" / country
    for h in _GWA_HEIGHTS:
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-A_{h}.tif",
            shape=shape,
            bounds=bounds,
            fill_value=8.0 + h * 0.01,
        )
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-k_{h}.tif",
            shape=shape,
            bounds=bounds,
            fill_value=2.0 + h * 0.001,
        )
        _create_gwa_raster(
            raw_dir / f"{country}_air-density_{h}.tif",
            shape=shape,
            bounds=bounds,
            fill_value=1.2 - h * 0.0001,
        )


def _create_elevation_raster(
    path: Path,
    *,
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> None:
    _create_gwa_raster(
        path,
        shape=shape,
        bounds=bounds,
        fill_value=500.0,
        add_nodata_region=False,
    )


def _create_nuts_shapefile(
    atlas_path: Path,
    *,
    bounds: tuple[float, float, float, float],
) -> None:
    nuts_dir = atlas_path / "data" / "nuts"
    nuts_dir.mkdir(parents=True, exist_ok=True)

    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    width = (bounds[2] - bounds[0]) / 4
    height = (bounds[3] - bounds[1]) / 4
    region_bounds = (
        center_x - width,
        center_y - height,
        center_x + width,
        center_y + height,
    )

    gdf = gpd.GeoDataFrame(
        {
            "NUTS_ID": ["AT", "Niederösterreich"],
            "NUTS_NAME": ["Österreich", "Niederösterreich"],
            "NAME_LATN": ["Österreich", "Niederösterreich"],
            "CNTR_CODE": ["AT", "AT"],
            "LEVL_CODE": [0, 2],
            "geometry": [box(*bounds), box(*region_bounds)],
        },
        crs="EPSG:3035",
    )
    gdf.to_file(nuts_dir / "NUTS_RG_01M_2021_3035.shp")


def _checksum_subset(da: xr.DataArray) -> str:
    arr = np.asarray(da.values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


@pytest.fixture
def phase3_baseline_atlas(tmp_path: Path) -> Atlas:
    """Deterministic atlas fixture used for parity baseline assertions."""
    root = tmp_path / "phase3_parity_baseline"
    country = "AUT"
    shape = (40, 40)
    bounds = (4000000, 2600000, 4200000, 2800000)

    _create_all_gwa_rasters(root, country=country, shape=shape, bounds=bounds)
    _create_elevation_raster(
        root / "data" / "raw" / country / f"{country}_elevation_w_bathymetry.tif",
        shape=shape,
        bounds=bounds,
    )
    _create_nuts_shapefile(root, bounds=bounds)

    atlas = Atlas(root, country, "epsg:3035")
    atlas.configure_turbines(_FIXED_TURBINES)

    # Freeze raw/resource mtimes so path_mtime_size fingerprints are deterministic.
    # Pin turbine inventory so this golden remains stable when new packaged turbines are added.
    _set_fixed_mtime(root / "data")
    _set_fixed_mtime(root / "resources")
    return atlas


class TestPhase3MaterializationBaseline:
    """Golden baseline tests used for pre/post extraction parity."""

    def test_base_materialization_golden(self, phase3_baseline_atlas: Atlas) -> None:
        atlas = phase3_baseline_atlas
        atlas.build()

        wind = xr.open_zarr(Path(atlas.path) / "wind.zarr", consolidated=False)
        land = xr.open_zarr(Path(atlas.path) / "landscape.zarr", consolidated=False)

        # Wind: stable attrs + schema + deterministic numeric subset.
        assert wind.attrs.get("store_state") == "complete"
        assert wind.attrs.get("grid_id") == _GOLDEN["wind_grid_id"]
        assert wind.attrs.get("inputs_id") == _GOLDEN["wind_inputs_id"]
        assert json.loads(wind.attrs.get("chunk_policy")) == {"x": 1024, "y": 1024}
        assert len(json.loads(wind.attrs.get("cleo_manifest_sources_json"))) == 30
        assert len(json.loads(wind.attrs.get("cleo_manifest_variables_json"))) == 9

        assert (
            _checksum_subset(wind["weibull_A"].sel(height=100).isel(y=slice(0, 3), x=slice(0, 3)))
            == _GOLDEN["wind_subset_checksum"]
        )
        assert (
            _checksum_subset(wind["template"].isel(y=slice(0, 3), x=slice(0, 3))) == _GOLDEN["wind_template_checksum"]
        )

        # Landscape: stable structure/attrs plus deterministic subsets.
        assert land.attrs.get("store_state") == "complete"
        assert land.attrs.get("grid_id") == _GOLDEN["wind_grid_id"]
        assert json.loads(land.attrs.get("chunk_policy")) == {"x": 1024, "y": 1024}
        assert len(json.loads(land.attrs.get("cleo_manifest_sources_json"))) == 2
        assert len(json.loads(land.attrs.get("cleo_manifest_variables_json"))) == 2

        # NOTE: landscape inputs_id includes absolute elevation path in current implementation.
        land_inputs_id = str(land.attrs.get("inputs_id", ""))
        assert len(land_inputs_id) == 16

        assert (
            _checksum_subset(land["valid_mask"].astype(np.float64).isel(y=slice(0, 3), x=slice(0, 3)))
            == _GOLDEN["land_valid_mask_checksum"]
        )
        assert (
            _checksum_subset(land["elevation"].isel(y=slice(0, 3), x=slice(0, 3))) == _GOLDEN["land_elevation_checksum"]
        )

    def test_region_materialization_golden(self, phase3_baseline_atlas: Atlas) -> None:
        atlas = phase3_baseline_atlas
        atlas.build()

        atlas.select(area="Niederösterreich", inplace=True)
        atlas.build()

        region_root = Path(atlas.path) / "areas" / atlas._area_id
        wind = xr.open_zarr(region_root / "wind.zarr", consolidated=False)
        land = xr.open_zarr(region_root / "landscape.zarr", consolidated=False)

        assert wind.attrs.get("store_state") == "complete"
        assert land.attrs.get("store_state") == "complete"
        assert wind.attrs.get("grid_id") == _GOLDEN["region_grid_id"]
        assert land.attrs.get("grid_id") == _GOLDEN["region_grid_id"]

        # Region inputs_id currently derives from base inputs and path-sensitive landscape id.
        wind_inputs_id = str(wind.attrs.get("inputs_id", ""))
        land_inputs_id = str(land.attrs.get("inputs_id", ""))
        assert len(wind_inputs_id) == 16
        assert wind_inputs_id == land_inputs_id

        assert [int(wind.sizes["y"]), int(wind.sizes["x"])] == [20, 20]
        assert [int(land.sizes["y"]), int(land.sizes["x"])] == [20, 20]

        assert (
            _checksum_subset(wind["weibull_A"].sel(height=100).isel(y=slice(0, 3), x=slice(0, 3)))
            == _GOLDEN["region_wind_subset_checksum"]
        )
        assert (
            _checksum_subset(land["elevation"].isel(y=slice(0, 3), x=slice(0, 3)))
            == _GOLDEN["region_land_elevation_checksum"]
        )
