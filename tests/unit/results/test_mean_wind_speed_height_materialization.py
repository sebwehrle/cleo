"""Tests for height-aware mean_wind_speed materialization semantics."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cleo.results import DomainResult


class _DomainDouble:
    """Minimal wind-domain double exposing atlas and active-store access."""

    def __init__(self, atlas):
        """Initialize a domain double.

        :param atlas: Atlas-like object exposing ``_active_wind_store_path``.
        :type atlas: object
        """
        self._atlas = atlas
        self._data = None

    @property
    def data(self) -> xr.Dataset:
        """Open and cache active wind store data.

        :returns: Active wind-store dataset.
        :rtype: xarray.Dataset
        """
        if self._data is None:
            self._data = xr.open_zarr(self._atlas._active_wind_store_path(), consolidated=False)
        return self._data


def _seed_wind_store(path: Path, *, include_legacy_mean_wind_speed: bool = False) -> None:
    """Create a minimal wind store with canonical ``height,y,x`` coordinates.

    :param path: Destination Zarr store path.
    :type path: pathlib.Path
    :param include_legacy_mean_wind_speed: If ``True``, seed legacy 2D
        ``mean_wind_speed(y,x)`` for migration-failure testing.
    :type include_legacy_mean_wind_speed: bool
    :returns: ``None``.
    :rtype: None
    """
    coords = {
        "height": np.array([50.0, 100.0, 150.0], dtype=np.float64),
        "y": np.array([1.0, 0.0], dtype=np.float64),
        "x": np.array([0.0, 1.0], dtype=np.float64),
    }
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "weibull_A": (("height", "y", "x"), np.ones((3, 2, 2), dtype=np.float32)),
        "template": (("y", "x"), np.ones((2, 2), dtype=np.float32)),
    }
    if include_legacy_mean_wind_speed:
        data_vars["mean_wind_speed"] = (("y", "x"), np.full((2, 2), 6.0, dtype=np.float32))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "store_state": "complete",
            "cleo_turbines_json": json.dumps([{"id": "Enercon.E40.500"}], sort_keys=True, separators=(",", ":")),
        },
    )
    ds.to_zarr(path, mode="w", consolidated=False)


def _mean_ws_result(*, domain: _DomainDouble, height: int, value: float) -> DomainResult:
    """Build a DomainResult for a singleton-height mean_wind_speed slice.

    :param domain: Wind-domain double.
    :type domain: _DomainDouble
    :param height: Requested height in meters.
    :type height: int
    :param value: Constant fill value for the test slice.
    :type value: float
    :returns: DomainResult configured for ``mean_wind_speed``.
    :rtype: cleo.results.DomainResult
    """
    da = xr.DataArray(
        np.full((1, 2, 2), value, dtype=np.float32),
        dims=("height", "y", "x"),
        coords={"height": np.array([float(height)]), "y": np.array([1.0, 0.0]), "x": np.array([0.0, 1.0])},
        name="mean_wind_speed",
        attrs={"units": "m/s"},
    )
    return DomainResult(domain, "mean_wind_speed", da, {"height": height})


def test_materialize_mean_wind_speed_keeps_height_dimension(tmp_path: Path) -> None:
    """First materialization writes ``mean_wind_speed(height,y,x)`` with singleton data slice."""
    store_path = tmp_path / "wind.zarr"
    _seed_wind_store(store_path)
    domain = _DomainDouble(SimpleNamespace(_active_wind_store_path=lambda: store_path))

    _mean_ws_result(domain=domain, height=100, value=7.0).materialize()

    ds = xr.open_zarr(store_path, consolidated=False)
    try:
        mean_ws = ds["mean_wind_speed"]
        assert mean_ws.dims == ("height", "y", "x")
        assert mean_ws.sizes["height"] == 3
        np.testing.assert_allclose(mean_ws.sel(height=100).values, np.full((2, 2), 7.0, dtype=np.float32))
        assert bool(np.isnan(mean_ws.sel(height=50).values).all())
        assert bool(np.isnan(mean_ws.sel(height=150).values).all())
    finally:
        ds.close()


def test_materialize_mean_wind_speed_aggregates_multiple_heights(tmp_path: Path) -> None:
    """Materializing different heights accumulates values in one variable."""
    store_path = tmp_path / "wind.zarr"
    _seed_wind_store(store_path)
    domain = _DomainDouble(SimpleNamespace(_active_wind_store_path=lambda: store_path))

    _mean_ws_result(domain=domain, height=100, value=7.0).materialize()
    _mean_ws_result(domain=domain, height=150, value=8.0).materialize(overwrite=False)

    ds = xr.open_zarr(store_path, consolidated=False)
    try:
        mean_ws = ds["mean_wind_speed"]
        np.testing.assert_allclose(mean_ws.sel(height=100).values, np.full((2, 2), 7.0, dtype=np.float32))
        np.testing.assert_allclose(mean_ws.sel(height=150).values, np.full((2, 2), 8.0, dtype=np.float32))
    finally:
        ds.close()


def test_materialize_mean_wind_speed_overwrite_false_blocks_existing_height(tmp_path: Path) -> None:
    """``overwrite=False`` rejects writes when requested height already has data."""
    store_path = tmp_path / "wind.zarr"
    _seed_wind_store(store_path)
    domain = _DomainDouble(SimpleNamespace(_active_wind_store_path=lambda: store_path))

    _mean_ws_result(domain=domain, height=100, value=7.0).materialize()

    with pytest.raises(ValueError, match="already exists"):
        _mean_ws_result(domain=domain, height=100, value=9.0).materialize(overwrite=False)


def test_materialize_mean_wind_speed_fails_on_legacy_2d_store(tmp_path: Path) -> None:
    """Legacy ``mean_wind_speed(y,x)`` in existing stores fails materialization."""
    store_path = tmp_path / "wind.zarr"
    _seed_wind_store(store_path, include_legacy_mean_wind_speed=True)
    domain = _DomainDouble(SimpleNamespace(_active_wind_store_path=lambda: store_path))

    with pytest.raises(RuntimeError, match="legacy 2D"):
        _mean_ws_result(domain=domain, height=100, value=7.0).materialize()
