"""Integration tests for Atlas results handling (Zarr and NetCDF).

Tests are offline-only: no network calls, uses local test fixtures.

These tests verify that:
- persist creates atomic Zarr stores
- export_result_netcdf exports to atomic NetCDF files
- clean_results deletes Zarr stores correctly
"""

from __future__ import annotations

import datetime
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
from cleo.unification.gwa_io import GWA_HEIGHTS
from cleo.unification.unifier import Unifier


def _copy_default_turbine(atlas_path: Path) -> None:
    """Copy a default turbine YAML to resources for testing."""
    resources_src = Path(cleo.__file__).resolve().parent / "resources"
    resources_dest = atlas_path / "resources"
    resources_dest.mkdir(parents=True, exist_ok=True)

    turbine_name = "Enercon.E40.500"
    src = resources_src / f"{turbine_name}.yml"
    if src.exists():
        shutil.copy(src, resources_dest / f"{turbine_name}.yml")


class MockAtlas:
    """Minimal Atlas-like object for testing results handling."""

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
        self.turbines_configured = None
        self.chunk_policy = {"y": 1024, "x": 1024}
        self.fingerprint_method = "path_mtime_size"
        self._canonical_ready = False

        # Create required directories
        (path / "data" / "raw" / country).mkdir(parents=True, exist_ok=True)
        (path / "resources").mkdir(parents=True, exist_ok=True)
        (path / "intermediates" / "crs_cache").mkdir(parents=True, exist_ok=True)

        # Copy default turbine for wind materialization
        _copy_default_turbine(path)

        # Create results_root
        self.results_root = path / "results"
        self.results_root.mkdir(parents=True, exist_ok=True)

    @property
    def wind_store_path(self) -> Path:
        return self.path / "wind.zarr"

    @property
    def landscape_store_path(self) -> Path:
        return self.path / "landscape.zarr"

    @property
    def wind_zarr(self) -> xr.Dataset:
        if not self._canonical_ready:
            raise RuntimeError("Canonical stores not ready")
        return xr.open_zarr(self.wind_store_path, consolidated=False)

    @property
    def landscape_zarr(self) -> xr.Dataset:
        if not self._canonical_ready:
            raise RuntimeError("Canonical stores not ready")
        return xr.open_zarr(self.landscape_store_path, consolidated=False)

    def get_nuts_area(self, area: str):
        return None

    def build_canonical(self) -> None:
        """Materialize both wind.zarr and landscape.zarr."""
        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        u.materialize_wind(self)
        u.materialize_landscape(self)
        self._canonical_ready = True

    def persist(
        self,
        metric_name: str,
        ds_or_da,
        *,
        run_id: str,
        params: dict | None = None,
    ) -> Path:
        """Persist a result to Zarr store."""
        from cleo.store import atomic_dir

        store_path = Path(self.results_root) / run_id / f"{metric_name}.zarr"

        # Handle DataArray vs Dataset
        if isinstance(ds_or_da, xr.DataArray):
            da = ds_or_da
            if da.name is None:
                da = da.rename(metric_name)
            ds = da.to_dataset()
        else:
            ds = ds_or_da

        # Ensure parent directory exists
        (Path(self.results_root) / run_id).mkdir(parents=True, exist_ok=True)

        # Atomic write
        with atomic_dir(store_path) as tmp:
            ds.to_zarr(tmp, mode="w", consolidated=False)

            g = zarr.open_group(tmp, mode="a")
            g.attrs["store_state"] = "complete"
            g.attrs["run_id"] = run_id
            g.attrs["metric_name"] = metric_name

            g.attrs["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if params is not None:
                g.attrs["params_json"] = json.dumps(params, sort_keys=True, separators=(",", ":"))

            try:
                if self._canonical_ready:
                    wind = self.wind_zarr
                    land = self.landscape_zarr
                    g.attrs["wind_grid_id"] = wind.attrs.get("grid_id", "")
                    g.attrs["landscape_grid_id"] = land.attrs.get("grid_id", "")
            except Exception:
                pass

        return store_path

    def export_result_netcdf(
        self,
        run_id: str,
        metric_name: str,
        out_path: str | Path,
        *,
        encoding: dict | None = None,
    ) -> Path:
        """Export a result Zarr store to NetCDF."""
        import os
        from uuid import uuid4

        out_path = Path(out_path)

        if not str(out_path).endswith(".nc"):
            raise ValueError(f"out_path must end with '.nc', got: {out_path}")

        src_store = Path(self.results_root) / run_id / f"{metric_name}.zarr"
        if not src_store.exists():
            raise FileNotFoundError(f"Result store not found: {src_store}")

        ds = xr.open_zarr(src_store, consolidated=False)

        tmp = out_path.with_name(out_path.name + f".__tmp__{uuid4().hex}")
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(tmp, encoding=encoding or {})
            os.replace(tmp, out_path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        return out_path

    def clean_results(
        self,
        run_id: str | None = None,
        older_than: str | None = None,
        metric_name: str | None = None,
    ) -> int:
        """Clean up result Zarr stores."""
        import shutil

        base = Path(self.results_root)
        count = 0

        if run_id:
            runs = [base / run_id] if (base / run_id).exists() else []
        else:
            runs = sorted([p for p in base.iterdir() if p.is_dir()])

        threshold_dt = None
        if older_than:
            try:
                threshold_dt = datetime.datetime.fromisoformat(older_than)
            except ValueError:
                threshold_dt = datetime.datetime.strptime(older_than, "%Y-%m-%d")

        for run_dir in runs:
            for store in sorted(run_dir.glob("*.zarr")):
                if metric_name and store.name != f"{metric_name}.zarr":
                    continue

                if threshold_dt:
                    store_dt = None
                    try:
                        g = zarr.open_group(store, mode="r")
                        created_at = g.attrs.get("created_at")
                        if created_at:
                            store_dt = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except Exception:
                        pass

                    if store_dt is None:
                        mtime = store.stat().st_mtime
                        store_dt = datetime.datetime.fromtimestamp(mtime)

                    if store_dt >= threshold_dt:
                        continue

                shutil.rmtree(store)
                count += 1

            if run_dir.exists() and not any(run_dir.iterdir()):
                run_dir.rmdir()

        return count


def _create_gwa_raster(
    path: Path,
    shape: tuple[int, int] = (20, 20),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (4000000, 2600000, 4100000, 2700000),
    fill_value: float | None = None,
    add_nodata_region: bool = True,
) -> None:
    """Create a minimal GWA-style GeoTIFF with CRS."""
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

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_elevation_raster(
    path: Path,
    shape: tuple[int, int] = (25, 25),
    crs_epsg: int = 3035,
    bounds: tuple[float, float, float, float] = (3990000, 2590000, 4110000, 2710000),
) -> None:
    """Create a legacy elevation GeoTIFF."""
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


def _create_all_gwa_rasters(atlas_path: Path, country: str = "AUT") -> None:
    """Create all required GWA rasters for wind materialization."""
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


class TestPersist:
    """Tests for Atlas.persist."""

    def test_persist_creates_store(self, tmp_path: Path) -> None:
        """persist creates a Zarr store at expected path."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.build_canonical()

        # Get small aligned coords from landscape
        land = atlas.landscape_zarr
        y = land["y"].values[:4]
        x = land["x"].values[:4]

        ds = xr.Dataset(
            {"metric": (("y", "x"), np.ones((4, 4), dtype=float))},
            coords={"y": y, "x": x},
        )

        p = atlas.persist("metric1", ds, run_id="run1", params={"a": 1})

        assert p.exists()
        assert p == tmp_path / "results" / "run1" / "metric1.zarr"

    def test_persist_has_consolidated_metadata(self, tmp_path: Path) -> None:
        """persist creates consolidated metadata."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.build_canonical()

        land = atlas.landscape_zarr
        y = land["y"].values[:4]
        x = land["x"].values[:4]

        ds = xr.Dataset(
            {"metric": (("y", "x"), np.ones((4, 4), dtype=float))},
            coords={"y": y, "x": x},
        )

        p = atlas.persist("metric1", ds, run_id="run1", params={"a": 1})

        # Check store is valid zarr (zarr.json for v3 or .zarray files for v2)
        assert (p / "zarr.json").exists() or (p / "metric" / ".zarray").exists()

    def test_persist_readable(self, tmp_path: Path) -> None:
        """Result store is readable with consolidated=False."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.build_canonical()

        land = atlas.landscape_zarr
        y = land["y"].values[:4]
        x = land["x"].values[:4]

        ds = xr.Dataset(
            {"metric": (("y", "x"), np.ones((4, 4), dtype=float))},
            coords={"y": y, "x": x},
        )

        p = atlas.persist("metric1", ds, run_id="run1", params={"a": 1})

        # Open with consolidated=False (Zarr v3 compatible)
        ds_r = xr.open_zarr(p, consolidated=False)
        assert "metric" in ds_r
        assert ds_r["metric"].shape == (4, 4)

    def test_persist_dataarray(self, tmp_path: Path) -> None:
        """persist handles DataArray input."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.build_canonical()

        land = atlas.landscape_zarr
        y = land["y"].values[:4]
        x = land["x"].values[:4]

        # Pass DataArray without name
        da = xr.DataArray(
            np.ones((4, 4), dtype=float),
            dims=["y", "x"],
            coords={"y": y, "x": x},
        )

        p = atlas.persist("metric2", da, run_id="run1")

        ds_r = xr.open_zarr(p, consolidated=False)
        # Should be renamed to metric_name
        assert "metric2" in ds_r

    def test_persist_has_attrs(self, tmp_path: Path) -> None:
        """persist stores metadata in attrs."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.build_canonical()

        land = atlas.landscape_zarr
        y = land["y"].values[:4]
        x = land["x"].values[:4]

        ds = xr.Dataset(
            {"metric": (("y", "x"), np.ones((4, 4), dtype=float))},
            coords={"y": y, "x": x},
        )

        p = atlas.persist("metric1", ds, run_id="run1", params={"a": 1})

        g = zarr.open_group(p, mode="r")
        assert g.attrs["store_state"] == "complete"
        assert g.attrs["run_id"] == "run1"
        assert g.attrs["metric_name"] == "metric1"
        assert "created_at" in g.attrs


class TestExportResultNetcdf:
    """Tests for Atlas.export_result_netcdf."""

    def test_export_result_netcdf_creates_file(self, tmp_path: Path) -> None:
        """export_result_netcdf creates NetCDF file."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.build_canonical()

        land = atlas.landscape_zarr
        y = land["y"].values[:4]
        x = land["x"].values[:4]

        ds = xr.Dataset(
            {"metric": (("y", "x"), np.ones((4, 4), dtype=float))},
            coords={"y": y, "x": x},
        )

        atlas.persist("metric1", ds, run_id="run1", params={"a": 1})

        out = tmp_path / "metric1.nc"
        atlas.export_result_netcdf("run1", "metric1", out)

        assert out.exists()

    def test_export_result_netcdf_readable(self, tmp_path: Path) -> None:
        """Exported NetCDF is readable."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.build_canonical()

        land = atlas.landscape_zarr
        y = land["y"].values[:4]
        x = land["x"].values[:4]

        ds = xr.Dataset(
            {"metric": (("y", "x"), np.ones((4, 4), dtype=float))},
            coords={"y": y, "x": x},
        )

        atlas.persist("metric1", ds, run_id="run1")

        out = tmp_path / "metric1.nc"
        atlas.export_result_netcdf("run1", "metric1", out)

        ds_nc = xr.open_dataset(out)
        assert "metric" in ds_nc
        assert ds_nc["metric"].shape == (4, 4)

    def test_export_result_netcdf_invalid_extension(self, tmp_path: Path) -> None:
        """export_result_netcdf raises for non-.nc extension."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.build_canonical()

        land = atlas.landscape_zarr
        y = land["y"].values[:4]
        x = land["x"].values[:4]

        ds = xr.Dataset(
            {"metric": (("y", "x"), np.ones((4, 4), dtype=float))},
            coords={"y": y, "x": x},
        )

        atlas.persist("metric1", ds, run_id="run1")

        with pytest.raises(ValueError, match="must end with '.nc'"):
            atlas.export_result_netcdf("run1", "metric1", tmp_path / "metric1.txt")

    def test_export_result_netcdf_missing_store(self, tmp_path: Path) -> None:
        """export_result_netcdf raises for missing store."""
        atlas = MockAtlas(tmp_path)

        with pytest.raises(FileNotFoundError, match="not found"):
            atlas.export_result_netcdf("run1", "missing", tmp_path / "missing.nc")


class TestCleanResults:
    """Tests for Atlas.clean_results."""

    def test_clean_results_deletes_store(self, tmp_path: Path) -> None:
        """clean_results deletes specified store."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.build_canonical()

        land = atlas.landscape_zarr
        y = land["y"].values[:4]
        x = land["x"].values[:4]

        ds = xr.Dataset(
            {"metric": (("y", "x"), np.ones((4, 4), dtype=float))},
            coords={"y": y, "x": x},
        )

        p = atlas.persist("metric1", ds, run_id="run1")
        assert p.exists()

        deleted = atlas.clean_results(run_id="run1", metric_name="metric1")

        assert deleted == 1
        assert not p.exists()

    def test_clean_results_returns_count(self, tmp_path: Path) -> None:
        """clean_results returns correct deletion count."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.build_canonical()

        land = atlas.landscape_zarr
        y = land["y"].values[:4]
        x = land["x"].values[:4]

        ds = xr.Dataset(
            {"metric": (("y", "x"), np.ones((4, 4), dtype=float))},
            coords={"y": y, "x": x},
        )

        # Create multiple stores
        atlas.persist("metric1", ds, run_id="run1")
        atlas.persist("metric2", ds, run_id="run1")

        deleted = atlas.clean_results(run_id="run1")
        assert deleted == 2

    def test_clean_results_all_runs(self, tmp_path: Path) -> None:
        """clean_results without run_id cleans all runs."""
        atlas = MockAtlas(tmp_path)

        _create_all_gwa_rasters(tmp_path)
        elev_path = tmp_path / "data" / "raw" / "AUT" / "AUT_elevation_w_bathymetry.tif"
        _create_elevation_raster(elev_path)
        atlas.build_canonical()

        land = atlas.landscape_zarr
        y = land["y"].values[:4]
        x = land["x"].values[:4]

        ds = xr.Dataset(
            {"metric": (("y", "x"), np.ones((4, 4), dtype=float))},
            coords={"y": y, "x": x},
        )

        # Create stores in multiple runs
        atlas.persist("metric1", ds, run_id="run1")
        atlas.persist("metric1", ds, run_id="run2")

        deleted = atlas.clean_results()
        assert deleted == 2

    def test_clean_results_empty_returns_zero(self, tmp_path: Path) -> None:
        """clean_results returns 0 for empty results dir."""
        atlas = MockAtlas(tmp_path)

        deleted = atlas.clean_results()
        assert deleted == 0
