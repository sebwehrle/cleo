"""Integration tests for in-memory raster landscape ingestion."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
import zarr
from rasterio.crs import CRS

import cleo
from cleo.domains import LandscapeDomain
from cleo.unification.gwa_io import GWA_HEIGHTS
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
    """Minimal Atlas-like object for in-memory raster integration tests."""

    def __init__(self, path: Path, country: str = "AUT", crs: str = "epsg:3035"):
        self.path = path
        self.country = country
        self.crs = crs
        self.area = None
        self.turbines_configured = None
        self.chunk_policy = {"y": 1024, "x": 1024}
        self.fingerprint_method = "path_mtime_size"
        self._canonical_ready = False

        (path / "data" / "raw" / country).mkdir(parents=True, exist_ok=True)
        (path / "resources").mkdir(parents=True, exist_ok=True)
        (path / "intermediates" / "crs_cache").mkdir(parents=True, exist_ok=True)
        _copy_default_turbine(path)

    def get_nuts_area(self, area: str):  # noqa: ANN001
        del area
        return None

    def build_canonical(self) -> None:
        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        u.materialize_wind(self)
        u.materialize_landscape(self)
        self._canonical_ready = True


def _create_raster(
    path: Path,
    *,
    shape: tuple[int, int] = (20, 20),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (4000000, 2600000, 4100000, 2700000),
    fill_value: float = 1.0,
    add_nodata: bool = True,
) -> None:
    data = np.full(shape, fill_value, dtype=np.float32)
    if add_nodata:
        data[:2, :2] = np.nan

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


def _create_all_gwa_rasters(atlas_path: Path, country: str = "AUT") -> None:
    raw_dir = atlas_path / "data" / "raw" / country
    for h in GWA_HEIGHTS:
        _create_raster(raw_dir / f"{country}_combined-Weibull-A_{h}.tif", fill_value=8.0 + h * 0.01)
        _create_raster(raw_dir / f"{country}_combined-Weibull-k_{h}.tif", fill_value=2.0 + h * 0.001)
        _create_raster(raw_dir / f"{country}_air-density_{h}.tif", fill_value=1.2 - h * 0.0001)


def _create_elevation_raster(atlas_path: Path, country: str = "AUT") -> None:
    _create_raster(
        atlas_path / "data" / "raw" / country / f"{country}_elevation_w_bathymetry.tif",
        shape=(24, 24),
        bounds=(3995000, 2595000, 4105000, 2705000),
        fill_value=300.0,
    )


def _aligned_candidate(domain: LandscapeDomain, *, value: float) -> xr.DataArray:
    valid_mask = domain.data["valid_mask"]
    values = np.full(valid_mask.shape, value, dtype=np.float32)
    values[~np.asarray(valid_mask.values, dtype=bool)] = np.nan
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": valid_mask.coords["y"].values, "x": valid_mask.coords["x"].values},
        name="candidate",
    )


def test_add_dataarray_stage_and_materialize_updates_manifest(tmp_path: Path) -> None:
    atlas = MockAtlas(tmp_path)
    _create_all_gwa_rasters(tmp_path)
    _create_elevation_raster(tmp_path)
    atlas.build_canonical()

    domain = LandscapeDomain(atlas)
    source_da = _aligned_candidate(domain, value=7.5)

    op = domain.add_dataarray("slope", source_da, if_exists="error")

    assert "slope" in domain.data.data_vars
    assert op.data.name == "slope"

    written = op.materialize()
    assert written.name == "slope"
    assert written.rio.crs is not None

    ds = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
    assert "slope" in ds.data_vars
    ds.close()

    root = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
    sources = json.loads(root.attrs.get("cleo_manifest_sources_json", "[]"))
    vars_ = json.loads(root.attrs.get("cleo_manifest_variables_json", "[]"))

    source = next(s for s in sources if s["source_id"] == "land:raster_cached:slope")
    assert source["kind"] == "raster_cached"
    assert source["path"].endswith(".tif")
    assert Path(source["path"]).exists()
    var = next(v for v in vars_ if v["variable_name"] == "slope")
    assert var["source_id"] == "land:raster_cached:slope"


def test_add_dataarray_noop_succeeds_for_equivalent_materialized_source(tmp_path: Path) -> None:
    atlas = MockAtlas(tmp_path)
    _create_all_gwa_rasters(tmp_path)
    _create_elevation_raster(tmp_path)
    atlas.build_canonical()

    domain = LandscapeDomain(atlas)
    source_da = _aligned_candidate(domain, value=4.0)
    domain.add_dataarray("slope", source_da, if_exists="error").materialize()

    op = domain.add_dataarray("slope", source_da, if_exists="noop")
    assert op.data.name == "slope"

    written = op.materialize()
    assert written.name == "slope"

    root = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
    sources = json.loads(root.attrs.get("cleo_manifest_sources_json", "[]"))
    assert any(s["source_id"] == "land:raster_cached:slope" for s in sources)


def test_add_dataarray_aligns_non_exact_grid_input(tmp_path: Path) -> None:
    atlas = MockAtlas(tmp_path)
    _create_all_gwa_rasters(tmp_path)
    _create_elevation_raster(tmp_path)
    atlas.build_canonical()

    domain = LandscapeDomain(atlas)
    source_path = tmp_path / "misaligned_input.tif"
    _create_raster(
        source_path,
        shape=(24, 24),
        bounds=(3997500, 2597500, 4102500, 2702500),
        fill_value=9.0,
        add_nodata=False,
    )
    source_da = rxr.open_rasterio(source_path, parse_coordinates=True).squeeze(drop=True)

    written = domain.add_dataarray("aligned_surface", source_da, if_exists="error").materialize()

    valid_mask = domain.data["valid_mask"]
    np.testing.assert_array_equal(written.coords["y"].values, valid_mask.coords["y"].values)
    np.testing.assert_array_equal(written.coords["x"].values, valid_mask.coords["x"].values)
    assert written.rio.crs is not None
    assert np.isclose(
        float(written.where(valid_mask).mean().compute()),
        9.0,
        atol=1e-6,
    )
