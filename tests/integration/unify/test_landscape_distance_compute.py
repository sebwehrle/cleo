"""Integration tests for landscape distance compute/materialize API."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from rasterio.crs import CRS

import cleo
from cleo.domains import LandscapeDomain
from cleo.unification.gwa_io import GWA_HEIGHTS
from cleo.unification.raster_io import _atomic_replace_variable_dir
from cleo.unification.unifier import Unifier


def _copy_default_turbine(atlas_path: Path) -> None:
    resources_src = Path(cleo.__file__).resolve().parent / "resources"
    resources_dest = atlas_path / "resources"
    resources_dest.mkdir(parents=True, exist_ok=True)
    turbine_name = "Enercon.E40.500"
    src = resources_src / f"{turbine_name}.yml"
    if src.exists():
        shutil.copy(src, resources_dest / f"{turbine_name}.yml")


class MockAtlas:
    def __init__(
        self,
        path: Path,
        country: str = "AUT",
        crs: str = "epsg:3035",
    ):
        self.path = path
        self.country = country
        self.crs = crs
        self.area = None
        self._area_name = None
        self._area_id = "__all__"
        self.turbines_configured = None
        self.chunk_policy = {"y": 64, "x": 64}
        self.fingerprint_method = "path_mtime_size"
        self._canonical_ready = False
        self.wind_store_path = path / "wind.zarr"
        self.landscape_store_path = path / "landscape.zarr"
        self._landscape_domain = None

        (path / "data" / "raw" / country).mkdir(parents=True, exist_ok=True)
        (path / "resources").mkdir(parents=True, exist_ok=True)
        (path / "intermediates" / "crs_cache").mkdir(parents=True, exist_ok=True)
        _copy_default_turbine(path)

    def get_nuts_area(self, area: str):
        del area
        return None

    def _active_wind_store_path(self) -> Path:
        if self._area_name is None:
            return self.wind_store_path
        return self.path / "areas" / self._area_id / "wind.zarr"

    def _active_landscape_store_path(self) -> Path:
        if self._area_name is None:
            return self.landscape_store_path
        return self.path / "areas" / self._area_id / "landscape.zarr"

    def build_canonical(self) -> None:
        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        u.materialize_wind(self)
        u.materialize_landscape(self)
        self._canonical_ready = True
        self._landscape_domain = None

    @property
    def landscape(self) -> LandscapeDomain:
        if self._landscape_domain is None:
            self._landscape_domain = LandscapeDomain(self)
        return self._landscape_domain


def _create_gwa_raster(
    path: Path,
    shape: tuple[int, int] = (20, 20),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (4000000, 2600000, 4100000, 2700000),
    fill_value: float | None = None,
) -> None:
    if fill_value is not None:
        data = np.full(shape, fill_value, dtype=np.float32)
    else:
        data = np.random.rand(*shape).astype(np.float32) * 10 + 1
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
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_elevation_raster(
    path: Path,
    shape: tuple[int, int] = (25, 25),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (3990000, 2590000, 4110000, 2710000),
) -> None:
    data = np.random.rand(*shape).astype(np.float32) * 3000
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
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_binary_layer(
    path: Path,
    *,
    shape: tuple[int, int] = (30, 30),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (3980000, 2580000, 4120000, 2720000),
    mode: str = "vertical",
) -> None:
    data = np.zeros(shape, dtype=np.float32)
    if mode == "vertical":
        data[:, shape[1] // 2] = 1.0
    elif mode == "horizontal":
        data[shape[0] // 2, :] = 1.0
    else:
        raise ValueError(f"Unsupported binary layer mode: {mode!r}")

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
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_all_gwa_rasters(atlas_path: Path, country: str = "AUT") -> None:
    raw_dir = atlas_path / "data" / "raw" / country
    for h in GWA_HEIGHTS:
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-A_{h}.tif",
            fill_value=8.0 + h * 0.01,
        )
        _create_gwa_raster(
            raw_dir / f"{country}_combined-Weibull-k_{h}.tif",
            fill_value=2.0 + h * 0.001,
        )
        _create_gwa_raster(
            raw_dir / f"{country}_air-density_{h}.tif",
            fill_value=1.2 - h * 0.0001,
        )


def test_landscape_distance_compute_batch_materialize(tmp_path: Path) -> None:
    atlas = MockAtlas(tmp_path)
    _create_all_gwa_rasters(tmp_path)
    _create_elevation_raster(tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif")
    atlas.build_canonical()

    roads_path = tmp_path / "data" / "raw" / "AUT" / "AUT_roads_mask.tif"
    water_path = tmp_path / "data" / "raw" / "AUT" / "AUT_water_mask.tif"
    _create_binary_layer(roads_path, mode="vertical")
    _create_binary_layer(water_path, mode="horizontal")

    atlas.landscape.add("roads_mask", roads_path, params={"categorical": True}).materialize()
    atlas.landscape.add("water_mask", water_path, params={"categorical": True}).materialize()

    result = atlas.landscape.compute(
        "distance",
        source=["roads_mask", "water_mask"],
        name=["distance_roads", "distance_water"],
        if_exists="error",
    )

    assert list(result.data.data_vars) == ["distance_roads", "distance_water"]
    out = result.materialize()

    assert list(out.data_vars) == ["distance_roads", "distance_water"]
    assert out["distance_roads"].attrs["units"] == "m"
    assert out["distance_water"].attrs["units"] == "m"
    assert np.isfinite(out["distance_roads"].values).any()
    assert np.isfinite(out["distance_water"].values).any()

    ds = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
    assert "distance_roads" in ds.data_vars
    assert "distance_water" in ds.data_vars


def test_region_local_distance_not_guaranteed_after_region_store_rebuild(tmp_path: Path) -> None:
    atlas = MockAtlas(tmp_path)
    _create_all_gwa_rasters(tmp_path)
    _create_elevation_raster(tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif")
    atlas.build_canonical()

    roads_path = tmp_path / "data" / "raw" / "AUT" / "AUT_roads_mask.tif"
    _create_binary_layer(roads_path, mode="vertical")
    atlas.landscape.add("roads_mask", roads_path, params={"categorical": True}).materialize()

    region_id = "AT-R1"
    region_root = tmp_path / "areas" / region_id
    region_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp_path / "wind.zarr", region_root / "wind.zarr")
    shutil.copytree(tmp_path / "landscape.zarr", region_root / "landscape.zarr")

    atlas._area_name = "Region-1"
    atlas._area_id = region_id
    atlas._landscape_domain = None

    atlas.landscape.compute("distance", source="roads_mask").materialize()
    ds_region = xr.open_zarr(region_root / "landscape.zarr", consolidated=False)
    assert "distance_roads_mask" in ds_region.data_vars

    shutil.rmtree(region_root / "landscape.zarr")
    shutil.copytree(tmp_path / "landscape.zarr", region_root / "landscape.zarr")
    atlas._landscape_domain = None

    ds_rebuilt = xr.open_zarr(region_root / "landscape.zarr", consolidated=False)
    assert "distance_roads_mask" not in ds_rebuilt.data_vars


def test_distance_noop_raises_when_existing_spec_is_invalid(tmp_path: Path) -> None:
    atlas = MockAtlas(tmp_path)
    _create_all_gwa_rasters(tmp_path)
    _create_elevation_raster(tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif")
    atlas.build_canonical()

    roads_path = tmp_path / "data" / "raw" / "AUT" / "AUT_roads_mask.tif"
    _create_binary_layer(roads_path, mode="vertical")
    atlas.landscape.add("roads_mask", roads_path, params={"categorical": True}).materialize()

    atlas.landscape.compute("distance", source="roads_mask").materialize()

    store_path = tmp_path / "landscape.zarr"
    preserve_attrs = dict(zarr.open_group(store_path, mode="r").attrs)
    ds = xr.open_zarr(store_path, consolidated=False)
    broken = ds["distance_roads_mask"].copy()
    broken.attrs["cleo:distance_spec_json"] = "{invalid-json"
    _atomic_replace_variable_dir(store_path, "distance_roads_mask")
    broken.to_dataset(name="distance_roads_mask").to_zarr(store_path, mode="a", consolidated=False)
    ds.close()
    root = zarr.open_group(store_path, mode="a")
    for key, value in preserve_attrs.items():
        root.attrs[key] = value

    atlas._landscape_domain = None
    with pytest.raises(ValueError, match="different distance spec"):
        atlas.landscape.compute("distance", source="roads_mask", if_exists="noop")
