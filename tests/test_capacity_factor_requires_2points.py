"""Test that capacity_factor raises ValueError (not IndexError) for insufficient points."""

import numpy as np
import xarray as xr
import pytest

from cleo.assess import capacity_factor


def _create_minimal_inputs(n_wind_speeds):
    """Create inputs with specified number of wind speed points."""
    u = np.linspace(0, 20, n_wind_speeds) if n_wind_speeds > 0 else np.array([])
    p = np.ones(n_wind_speeds) * 0.5 if n_wind_speeds > 0 else np.array([])

    # Create minimal spatial grid
    if n_wind_speeds > 0:
        pdf = xr.DataArray(
            np.ones((n_wind_speeds, 2, 2)) * 0.1,
            dims=["wind_speed", "y", "x"],
            coords={"wind_speed": u, "y": [0.0, 1.0], "x": [0.0, 1.0]},
        )
    else:
        pdf = xr.DataArray(
            np.ones((2, 2)) * 0.1,
            dims=["y", "x"],
            coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        )

    wind_shear = xr.DataArray(
        np.full((2, 2), 0.14),
        dims=["y", "x"],
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )

    return pdf, wind_shear, u, p


def test_capacity_factor_single_point_raises_valueerror():
    """capacity_factor with 1 point should raise ValueError, not IndexError."""
    pdf, wind_shear, u, p = _create_minimal_inputs(1)

    with pytest.raises(ValueError) as exc_info:
        capacity_factor(
            weibull_pdf=pdf,
            wind_shear=wind_shear,
            u_power_curve=u,
            p_power_curve=p,
            h_turbine=100,
        )

    assert "at least 2 points" in str(exc_info.value)


def test_capacity_factor_zero_points_raises_valueerror():
    """capacity_factor with 0 points should raise ValueError."""
    pdf, wind_shear, u, p = _create_minimal_inputs(0)

    with pytest.raises(ValueError) as exc_info:
        capacity_factor(
            weibull_pdf=pdf,
            wind_shear=wind_shear,
            u_power_curve=u,
            p_power_curve=p,
            h_turbine=100,
        )

    assert "at least 2 points" in str(exc_info.value)


def test_capacity_factor_two_points_works():
    """capacity_factor with exactly 2 points should work."""
    pdf, wind_shear, u, p = _create_minimal_inputs(2)

    # Should not raise
    cf = capacity_factor(
        weibull_pdf=pdf,
        wind_shear=wind_shear,
        u_power_curve=u,
        p_power_curve=p,
        h_turbine=100,
    )

    assert cf.shape == wind_shear.shape
    assert np.all(np.isfinite(cf.values))


def test_capacity_factor_mismatched_lengths_raises_valueerror():
    """capacity_factor should raise ValueError if u and p have different lengths."""
    u = np.array([0.0, 5.0, 10.0])
    p = np.array([0.0, 0.5])  # Only 2 elements

    pdf = xr.DataArray(
        np.ones((3, 2, 2)) * 0.1,
        dims=["wind_speed", "y", "x"],
        coords={"wind_speed": u, "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    wind_shear = xr.DataArray(
        np.full((2, 2), 0.14),
        dims=["y", "x"],
    )

    with pytest.raises(ValueError) as exc_info:
        capacity_factor(pdf, wind_shear, u, p, h_turbine=100)

    assert "length" in str(exc_info.value).lower()
