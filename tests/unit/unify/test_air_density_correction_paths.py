"""Phase 4 coverage tests for Unifier.compute_air_density_correction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from cleo.unification.unifier import Unifier


def _write_canonical_stores(
    root: Path,
    *,
    wind_state: str = "complete",
    land_state: str = "complete",
    with_weibull: bool = True,
    with_elevation: bool = True,
    aligned: bool = True,
) -> None:
    y = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([10.0, 11.0], dtype=np.float64)
    x_land = x if aligned else np.array([20.0, 21.0], dtype=np.float64)
    height = np.array([100.0], dtype=np.float64)

    wind_vars = {}
    if with_weibull:
        wind_vars["weibull_A"] = (("height", "y", "x"), np.ones((1, 2, 2), dtype=np.float32))
    wind = xr.Dataset(wind_vars, coords={"height": height, "y": y, "x": x})
    wind.attrs["store_state"] = wind_state
    wind.to_zarr(root / "wind.zarr", mode="w", consolidated=False)

    land_vars = {}
    if with_elevation:
        land_vars["elevation"] = (("y", "x"), np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32))
    land = xr.Dataset(land_vars, coords={"y": y, "x": x_land})
    land.attrs["store_state"] = land_state
    land.to_zarr(root / "landscape.zarr", mode="w", consolidated=False)


def test_compute_air_density_correction_raises_when_wind_missing(tmp_path: Path) -> None:
    atlas = SimpleNamespace(path=tmp_path)
    with pytest.raises(FileNotFoundError, match="wind.zarr not found"):
        Unifier().compute_air_density_correction(atlas)


def test_compute_air_density_correction_raises_when_landscape_missing(tmp_path: Path) -> None:
    y = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([10.0, 11.0], dtype=np.float64)
    height = np.array([100.0], dtype=np.float64)
    wind = xr.Dataset(
        {"weibull_A": (("height", "y", "x"), np.ones((1, 2, 2), dtype=np.float32))},
        coords={"height": height, "y": y, "x": x},
    )
    wind.attrs["store_state"] = "complete"
    wind.to_zarr(tmp_path / "wind.zarr", mode="w", consolidated=False)

    atlas = SimpleNamespace(path=tmp_path)
    with pytest.raises(FileNotFoundError, match="landscape.zarr not found"):
        Unifier().compute_air_density_correction(atlas)


def test_compute_air_density_correction_raises_when_store_incomplete(tmp_path: Path) -> None:
    _write_canonical_stores(tmp_path, wind_state="skeleton", land_state="complete")
    atlas = SimpleNamespace(path=tmp_path)
    with pytest.raises(RuntimeError, match="store_state='skeleton'"):
        Unifier().compute_air_density_correction(atlas)


def test_compute_air_density_correction_raises_when_required_vars_missing(tmp_path: Path) -> None:
    _write_canonical_stores(tmp_path, with_weibull=False, with_elevation=True)
    atlas = SimpleNamespace(path=tmp_path)
    with pytest.raises(RuntimeError, match="missing 'weibull_A'"):
        Unifier().compute_air_density_correction(atlas)


def test_compute_air_density_correction_raises_when_elevation_missing(tmp_path: Path) -> None:
    _write_canonical_stores(tmp_path, with_weibull=True, with_elevation=False)
    atlas = SimpleNamespace(path=tmp_path)
    with pytest.raises(RuntimeError, match="missing 'elevation'"):
        Unifier().compute_air_density_correction(atlas)


def test_compute_air_density_correction_raises_on_grid_mismatch(tmp_path: Path) -> None:
    _write_canonical_stores(tmp_path, aligned=False)
    atlas = SimpleNamespace(path=tmp_path)
    with pytest.raises(RuntimeError, match="Elevation not aligned to wind grid"):
        Unifier().compute_air_density_correction(atlas)


def test_compute_air_density_correction_success_path(tmp_path: Path) -> None:
    _write_canonical_stores(tmp_path, aligned=True)
    atlas = SimpleNamespace(path=tmp_path)
    out = Unifier(chunk_policy={"y": 2, "x": 2}).compute_air_density_correction(atlas)
    assert out.name == "air_density_correction"
    assert out.dims == ("y", "x")
    assert out.shape == (2, 2)
    assert np.isfinite(out.values).all()
