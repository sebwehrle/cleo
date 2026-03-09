"""Contract tests for CF integrators: interpolation semantics, alignment, endpoint stability."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cleo.assess import (
    _integrate_cf_no_density,
    _integrate_cf_with_density_correction,
    _interp_power_curve,
)


def _pdf_with_coords(values: np.ndarray, wind_speed: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": wind_speed, "y": [0], "x": [0]},
    )


def test_interp_power_curve_is_zero_outside_knots() -> None:
    u = np.array([0.0, 5.0, 10.0], dtype=np.float64)
    p = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u_eq = xr.DataArray(
        np.array([-1.0, 0.0, 2.5, 5.0, 10.0, 12.0], dtype=np.float64),
        dims=("wind_speed",),
        coords={"wind_speed": np.arange(6)},
    )
    out = _interp_power_curve(u_eq, u, p).values
    assert out[0] == 0.0
    assert out[-1] == 0.0
    assert out[3] == 1.0


def test_integrator_alignment_requires_exact_wind_speed_match() -> None:
    u_grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    # Drifted coordinate at first knot (0.1 instead of 0.0).
    pdf = _pdf_with_coords(
        np.array([[[0.0]], [[0.5]], [[0.5]]], dtype=np.float64),
        wind_speed=np.array([0.1, 1.0, 2.0], dtype=np.float64),
    )
    p_curve = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="do not exactly match integration grid"):
        _integrate_cf_no_density(pdf=pdf, u_grid=u_grid, p_curve=p_curve, loss_factor=1.0)


def test_integrator_u0_guard_prevents_nan_when_pdf_is_infinite_at_zero_no_density() -> None:
    u_grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    # Simulate k<1 endpoint behavior: +inf at u=0.
    pdf = _pdf_with_coords(
        np.array([[[np.inf]], [[0.3]], [[0.2]]], dtype=np.float64),
        wind_speed=u_grid,
    )
    p_curve = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    out = _integrate_cf_no_density(pdf=pdf, u_grid=u_grid, p_curve=p_curve, loss_factor=1.0)
    assert np.isfinite(float(out.values.item()))


def test_integrator_u0_guard_prevents_nan_when_pdf_is_infinite_at_zero_with_density() -> None:
    u_grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    pdf = _pdf_with_coords(
        np.array([[[np.inf]], [[0.3]], [[0.2]]], dtype=np.float64),
        wind_speed=u_grid,
    )
    p_curve = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    c = xr.DataArray(np.array([[1.0]], dtype=np.float64), dims=("y", "x"), coords={"y": [0], "x": [0]})
    out = _integrate_cf_with_density_correction(
        pdf=pdf,
        u_grid=u_grid,
        p_curve=p_curve,
        c=c,
        loss_factor=1.0,
    )
    assert np.isfinite(float(out.values.item()))
