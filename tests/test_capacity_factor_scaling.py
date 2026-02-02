"""Test capacity_factor returns ~1.0 when power curve is always at rated power."""

import numpy as np
import xarray as xr
from unittest.mock import MagicMock

from cleo.assess import capacity_factor, compute_optimal_power_energy


def test_capacity_factor_scaling_always_rated():
    """
    Verify that capacity_factor ≈ 1.0 when the power curve is identically
    rated power (p_power_curve = 1.0 everywhere), and that compute_optimal_power_energy
    correctly uses this capacity factor and power.

    Oracle:
        expected_capacity_factor = 1.0
        expected_power_kw = rated_power_kw
    """
    # Fixed Weibull parameters
    k = 2.0
    A = 8.0

    # Power curve grid
    u_power_curve = np.linspace(0.0, 25.0, 501)
    p_power_curve = np.ones_like(u_power_curve)  # Always at rated power (fraction = 1.0)

    # Weibull PDF: f(u; k, A) = (k/A) * (u/A)^(k-1) * exp(-(u/A)^k)
    u_over_A = u_power_curve / A
    pdf_1d = (k / A) * (u_over_A ** (k - 1)) * np.exp(-(u_over_A ** k))

    # Normalize so that trapz(pdf, x=u) == 1.0 within tolerance
    norm = np.trapezoid(pdf_1d, x=u_power_curve)
    pdf_1d = pdf_1d / norm

    # Verify normalization
    assert np.isclose(np.trapezoid(pdf_1d, x=u_power_curve), 1.0, atol=1e-9)

    # Broadcast to (wind_speed, y, x) with y=1, x=1
    pdf_3d = pdf_1d[:, None, None].copy()
    weibull_pdf = xr.DataArray(
        data=pdf_3d,
        dims=["wind_speed", "y", "x"],
        coords={"wind_speed": u_power_curve, "y": [0], "x": [0]},
    )

    # wind_shear: zeros with dims (y, x)
    wind_shear = xr.DataArray(
        data=[[0.0]],
        dims=["y", "x"],
        coords={"y": [0], "x": [0]},
    )

    # Compute capacity factor using the production function
    cf_result = capacity_factor(
        weibull_pdf,
        wind_shear,
        u_power_curve,
        p_power_curve,
        h_turbine=100,
        h_reference=100,
        correction_factor=1.0,
    )

    # Oracle: capacity_factor should be 1.0 (integral of pdf * 1.0 = 1.0)
    expected_capacity_factor = 1.0
    assert np.allclose(cf_result.values, expected_capacity_factor, rtol=0, atol=1e-6), (
        f"Capacity factor {cf_result.values} does not match oracle {expected_capacity_factor}"
    )

    # Now test compute_optimal_power_energy with this capacity factor
    rated_power_kw = 1234.0
    turbine_name = f"TestTurbine.{int(rated_power_kw)}"

    mock_self = MagicMock()

    # lcoe with single turbine
    lcoe = xr.DataArray(
        data=[[[50.0]]],
        dims=["turbine", "y", "x"],
        coords={"turbine": [turbine_name], "y": [0], "x": [0]},
    )

    # capacity_factors from our computed result
    capacity_factors = cf_result.expand_dims(turbine=[turbine_name])

    mock_self.data = xr.Dataset({
        "lcoe": lcoe,
        "capacity_factors": capacity_factors,
    })

    # Mock get_turbine_attribute to return rated power
    def mock_get_turbine_attribute(turbine, attr):
        if attr == "capacity":
            return rated_power_kw
        return None

    mock_self.get_turbine_attribute = mock_get_turbine_attribute

    # Call compute_optimal_power_energy
    compute_optimal_power_energy(mock_self)

    # Assert: returned power is close to rated_power_kw
    result_power = float(mock_self.data["optimal_power"].values.item())
    assert np.isclose(result_power, rated_power_kw, rtol=0, atol=1e-6), (
        f"Optimal power {result_power} does not match rated_power_kw {rated_power_kw}"
    )

    # Assert: energy calculation is consistent with capacity_factor ≈ 1.0
    # expected_energy = capacity_factor * power * 8766 / 1e6
    expected_energy = expected_capacity_factor * rated_power_kw * 8766.0 / 1e6
    result_energy = float(mock_self.data["optimal_energy"].values.item())
    assert np.isclose(result_energy, expected_energy, rtol=0, atol=1e-6), (
        f"Optimal energy {result_energy} does not match expected {expected_energy}"
    )
