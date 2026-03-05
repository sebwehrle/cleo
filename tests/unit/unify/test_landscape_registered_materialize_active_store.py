from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from tests.helpers.optional import requires_rasterio, requires_rioxarray

from cleo.unification.unifier import Unifier

rasterio = requires_rasterio()
requires_rioxarray()


class _AtlasStub:
    """Atlas-like stub with active area store routing."""

    def __init__(self, root: Path) -> None:
        self.path = root
        self.country = "AUT"
        self.crs = "epsg:3035"
        self.area = None
        self.chunk_policy = {"y": 32, "x": 32}
        self.fingerprint_method = "path_mtime_size"
        self._area_name = "R1"
        self._area_id = "R1"
        self.wind_store_path = root / "wind.zarr"
        self.landscape_store_path = root / "landscape.zarr"

    def _active_wind_store_path(self) -> Path:
        """Return active wind store path for selected area."""
        return self.path / "areas" / self._area_id / "wind.zarr"

    def _active_landscape_store_path(self) -> Path:
        """Return active landscape store path for selected area."""
        return self.path / "areas" / self._area_id / "landscape.zarr"


def _make_wind_store(path: Path, *, grid_id: str) -> None:
    """Create a minimal complete wind store with ``weibull_A``."""
    y = np.array([1.5, 0.5], dtype=np.float64)
    x = np.array([0.5, 1.5], dtype=np.float64)
    h = np.array([100], dtype=np.int64)
    da = xr.DataArray(
        np.full((1, 2, 2), 8.0, dtype=np.float32),
        dims=("height", "y", "x"),
        coords={"height": h, "y": y, "x": x},
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    ds = xr.Dataset({"weibull_A": da})
    ds.to_zarr(path, mode="w", consolidated=False)
    root = zarr.open_group(path, mode="a")
    root.attrs["store_state"] = "complete"
    root.attrs["grid_id"] = grid_id
    root.attrs["inputs_id"] = f"wind-inputs-{grid_id}"


def _make_landscape_store(path: Path, *, grid_id: str) -> None:
    """Create a minimal complete landscape store with ``valid_mask``."""
    y = np.array([1.5, 0.5], dtype=np.float64)
    x = np.array([0.5, 1.5], dtype=np.float64)
    ds = xr.Dataset(
        data_vars={
            "valid_mask": xr.DataArray(
                np.ones((2, 2), dtype=bool),
                dims=("y", "x"),
                coords={"y": y, "x": x},
            )
        },
        coords={"y": y, "x": x},
    )
    ds.to_zarr(path, mode="w", consolidated=False)
    root = zarr.open_group(path, mode="a")
    root.attrs["store_state"] = "complete"
    root.attrs["grid_id"] = grid_id


def _write_source_raster(path: Path) -> None:
    """Write a tiny CLC-like source raster in EPSG:3035."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = rasterio.transform.from_bounds(0.0, 0.0, 2.0, 2.0, 2, 2)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": 2,
        "height": 2,
        "count": 1,
        "crs": rasterio.crs.CRS.from_epsg(3035),
        "transform": transform,
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.array([[311.0, 231.0], [311.0, 311.0]], dtype=np.float32), 1)


def test_registered_landscape_materialization_targets_active_region_store(tmp_path: Path) -> None:
    """Registered-variable flow should use active area stores, not base stores."""
    atlas = _AtlasStub(tmp_path)

    _make_wind_store(atlas.wind_store_path, grid_id="BASE")
    _make_landscape_store(atlas.landscape_store_path, grid_id="BASE")

    active_wind = atlas._active_wind_store_path()
    active_land = atlas._active_landscape_store_path()
    active_wind.parent.mkdir(parents=True, exist_ok=True)
    active_land.parent.mkdir(parents=True, exist_ok=True)
    _make_wind_store(active_wind, grid_id="REGION")
    _make_landscape_store(active_land, grid_id="REGION")

    source = tmp_path / "source" / "clc_region.tif"
    _write_source_raster(source)

    u = Unifier(chunk_policy=atlas.chunk_policy, fingerprint_method=atlas.fingerprint_method)
    changed = u.register_landscape_source(
        atlas,
        name="broad_leaved_forest",
        source_path=source,
        kind="raster",
        params={"categorical": True, "clc_codes": [311]},
        if_exists="error",
    )
    assert changed is True

    region_root = zarr.open_group(active_land, mode="r")
    base_root = zarr.open_group(atlas.landscape_store_path, mode="r")
    region_sources = json.loads(region_root.attrs.get("cleo_manifest_sources_json", "[]"))
    base_sources = json.loads(base_root.attrs.get("cleo_manifest_sources_json", "[]"))
    assert any(s.get("source_id") == "land:raster:broad_leaved_forest" for s in region_sources)
    assert not any(s.get("source_id") == "land:raster:broad_leaved_forest" for s in base_sources)

    staged = u.prepare_landscape_variable_data(atlas, "broad_leaved_forest")
    assert staged.name == "broad_leaved_forest"
    assert staged.dims == ("y", "x")

    written = u.materialize_landscape_variable(atlas, "broad_leaved_forest", if_exists="error")
    assert written is True

    ds_region = xr.open_zarr(active_land, consolidated=False)
    ds_base = xr.open_zarr(atlas.landscape_store_path, consolidated=False)
    assert "broad_leaved_forest" in ds_region.data_vars
    assert "broad_leaved_forest" not in ds_base.data_vars
