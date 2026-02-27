"""Integration tests for vector-source landscape materializer flow."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from rasterio.crs import CRS

import cleo
from tests.helpers.optional import requires_geopandas

requires_geopandas()

import geopandas as gpd
from shapely.geometry import Polygon

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
    """Minimal Atlas-like object for unifier integration tests."""

    def __init__(
        self,
        path: Path,
        country: str = "AUT",
        crs: str = "epsg:3035",
    ):
        self.path = path
        self.country = country
        self.crs = crs
        self.region = None
        self.turbines_configured = None
        self.chunk_policy = {"y": 1024, "x": 1024}
        self.fingerprint_method = "path_mtime_size"
        self._canonical_ready = False

        (path / "data" / "raw" / country).mkdir(parents=True, exist_ok=True)
        (path / "resources").mkdir(parents=True, exist_ok=True)
        (path / "intermediates" / "crs_cache").mkdir(parents=True, exist_ok=True)
        _copy_default_turbine(path)

    def get_nuts_region(self, region: str):  # noqa: ANN001
        del region
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
        _create_raster(
            raw_dir / f"{country}_combined-Weibull-A_{h}.tif",
            fill_value=8.0 + h * 0.01,
        )
        _create_raster(
            raw_dir / f"{country}_combined-Weibull-k_{h}.tif",
            fill_value=2.0 + h * 0.001,
        )
        _create_raster(
            raw_dir / f"{country}_air-density_{h}.tif",
            fill_value=1.2 - h * 0.0001,
        )


def _create_elevation_raster(atlas_path: Path, country: str = "AUT") -> None:
    _create_raster(
        atlas_path / "data" / "raw" / country / f"{country}_elevation_w_bathymetry.tif",
        shape=(24, 24),
        bounds=(3995000, 2595000, 4105000, 2705000),
        fill_value=300.0,
    )


def _sample_vector() -> gpd.GeoDataFrame:
    left = Polygon([(4000000, 2600000), (4000000, 2700000), (4050000, 2700000), (4050000, 2600000)])
    right = Polygon([(4050000, 2600000), (4050000, 2700000), (4100000, 2700000), (4100000, 2600000)])
    return gpd.GeoDataFrame(
        {"overnight_stays": [10.0, 25.0]},
        geometry=[left, right],
        crs="EPSG:3035",
    )


def test_vector_register_prepare_materialize_flow(tmp_path: Path) -> None:
    atlas = MockAtlas(tmp_path)
    _create_all_gwa_rasters(tmp_path)
    _create_elevation_raster(tmp_path)
    atlas.build_canonical()

    u = Unifier(chunk_policy=atlas.chunk_policy, fingerprint_method=atlas.fingerprint_method)
    gdf = _sample_vector()

    changed = u.register_landscape_vector_source(
        atlas,
        name="overnight_stays",
        shape=gdf,
        column="overnight_stays",
        all_touched=False,
        if_exists="error",
    )
    assert changed is True

    staged = u.prepare_landscape_variable_data(atlas, "overnight_stays")
    assert staged.name == "overnight_stays"
    assert staged.dims == ("y", "x")
    assert staged.dtype == np.float32
    finite = staged.values[np.isfinite(staged.values)]
    assert finite.size > 0
    unique = set(np.unique(finite).tolist())
    assert {10.0, 25.0}.issubset(unique)
    assert unique.issubset({0.0, 10.0, 25.0})

    written = u.materialize_landscape_variable(atlas, "overnight_stays", if_exists="error")
    assert written is True

    ds = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
    assert "overnight_stays" in ds.data_vars
    ds.close()

    root = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
    sources = json.loads(root.attrs.get("cleo_manifest_sources_json", "[]"))
    vars_ = json.loads(root.attrs.get("cleo_manifest_variables_json", "[]"))

    source = next(s for s in sources if s["source_id"] == "land:vector:overnight_stays")
    assert source["kind"] == "vector"
    assert source["path"].endswith(".geojson")
    var = next(v for v in vars_ if v["variable_name"] == "overnight_stays")
    assert var["source_id"] == "land:vector:overnight_stays"


def test_vector_noop_requires_exact_match(tmp_path: Path) -> None:
    atlas = MockAtlas(tmp_path)
    _create_all_gwa_rasters(tmp_path)
    _create_elevation_raster(tmp_path)
    atlas.build_canonical()

    u = Unifier(chunk_policy=atlas.chunk_policy, fingerprint_method=atlas.fingerprint_method)
    gdf = _sample_vector()
    u.register_landscape_vector_source(
        atlas,
        name="overnight_stays",
        shape=gdf,
        column="overnight_stays",
        all_touched=False,
        if_exists="error",
    )
    u.materialize_landscape_variable(atlas, "overnight_stays", if_exists="error")

    changed = u.register_landscape_vector_source(
        atlas,
        name="overnight_stays",
        shape=gdf,
        column="overnight_stays",
        all_touched=False,
        if_exists="noop",
    )
    assert changed is False

    written = u.materialize_landscape_variable(atlas, "overnight_stays", if_exists="noop")
    assert written is False

    modified = gdf.copy()
    modified.loc[0, "overnight_stays"] = 99.0
    with pytest.raises(ValueError, match="if_exists='replace'"):
        u.register_landscape_vector_source(
            atlas,
            name="overnight_stays",
            shape=modified,
            column="overnight_stays",
            all_touched=False,
            if_exists="noop",
        )


def test_landscape_domain_rasterize_stage_and_materialize(tmp_path: Path) -> None:
    atlas = MockAtlas(tmp_path)
    _create_all_gwa_rasters(tmp_path)
    _create_elevation_raster(tmp_path)
    atlas.build_canonical()

    domain = LandscapeDomain(atlas)
    op = domain.rasterize(
        _sample_vector(),
        name="overnight_stays",
        column="overnight_stays",
        all_touched=False,
        if_exists="error",
    )

    assert "overnight_stays" in domain.data.data_vars
    staged = domain.data["overnight_stays"]
    finite_staged = staged.values[np.isfinite(staged.values)]
    assert finite_staged.size > 0

    written = op.materialize()
    assert written.name == "overnight_stays"
    finite_written = written.values[np.isfinite(written.values)]
    assert finite_written.size > 0
    assert not np.isnan(finite_written).all()

    ds = xr.open_zarr(tmp_path / "landscape.zarr", consolidated=False)
    assert "overnight_stays" in ds.data_vars
    ds.close()

    root = zarr.open_group(tmp_path / "landscape.zarr", mode="r")
    sources = json.loads(root.attrs.get("cleo_manifest_sources_json", "[]"))
    source = next(s for s in sources if s["source_id"] == "land:vector:overnight_stays")
    assert source["kind"] == "vector"
