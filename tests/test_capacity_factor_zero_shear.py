"""Test capacity_factor reduces to direct integral when wind_shear is zero."""

import numpy as np
import xarray as xr

from cleo.assess import capacity_factor


def test_capacity_factor_zero_shear():
    """
    Verify that capacity_factor with zero wind_shear equals the direct integral
    of weibull_pdf * power_curve over wind speed.
    """
    # Fixed parameters
    k = 2.0
    A = 8.0

    # Power curve grid
    u_power_curve = np.linspace(0.0, 25.0, 501)
    p_power_curve = np.clip(u_power_curve / 25.0, 0, 1)

    # wind_shear: zeros with dims (y, x)
    wind_shear = xr.DataArray(
        data=np.zeros((2, 3)),
        dims=["y", "x"],
        coords={"y": [0, 1], "x": [0, 1, 2]},
    )

    # weibull_pdf: analytic Weibull PDF with dims (wind_speed, y, x)
    # f(u; k, A) = (k/A) * (u/A)^(k-1) * exp(-(u/A)^k) for u >= 0
    u_over_A = u_power_curve / A
    pdf_1d = (k / A) * (u_over_A ** (k - 1)) * np.exp(-(u_over_A ** k))
    # Broadcast to (wind_speed, y, x)
    pdf_3d = np.broadcast_to(pdf_1d[:, None, None], (len(u_power_curve), 2, 3))
    weibull_pdf = xr.DataArray(
        data=pdf_3d.copy(),
        dims=["wind_speed", "y", "x"],
        coords={"wind_speed": u_power_curve, "y": [0, 1], "x": [0, 1, 2]},
    )

    # Oracle: direct integral of weibull_pdf * power_curve
    # expected shape: (y, x)
    expected = np.trapezoid(
        weibull_pdf.values * p_power_curve[:, None, None],
        x=u_power_curve,
        axis=0,
    )

    # Call capacity_factor
    result = capacity_factor(
        weibull_pdf,
        wind_shear,
        u_power_curve,
        p_power_curve,
        h_turbine=100,
        h_reference=100,
        correction_factor=1,
    )

    # Assertions
    assert isinstance(result, xr.DataArray), "Result must be an xarray DataArray"
    assert result.dims == ("y", "x"), f"Expected dims ('y', 'x'), got {result.dims}"
    assert result.name == "capacity_factor", f"Expected name 'capacity_factor', got {result.name}"
    assert np.allclose(result.values, expected, rtol=0, atol=1e-6), (
        f"Result values {result.values} do not match expected {expected}"
    )
