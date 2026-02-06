"""Oracle-driven capacity_factor test with step power curve and wind shear.

This test uses the correct analytic Weibull-CDF oracle for wind-shear scaling,
where the power curve is evaluated at u_hub = u_ref * s.
"""

import numpy as np
import xarray as xr

from cleo.assess import capacity_factor


def test_capacity_factor_step_curve_with_shear():
    """
    Verify capacity_factor with a step power curve and nonzero wind shear
    matches the analytic Weibull CDF oracle.

    Physics:
    - Wind at hub height is scaled: u_hub = u_ref * s where s = (h_turbine/h_ref)^alpha
    - Step power curve at threshold u0 (hub height): power = 1 when u_hub >= u0
    - This means power = 1 when u_ref >= u0/s

    Oracle for normalized/truncated PDF on [0, u_max]:
        CF = P(u_ref * s >= u0 | u_ref in [0, u_max])
           = P(u_ref >= u0/s | u_ref in [0, u_max])
           = (F(u_max) - F(u0/s)) / F(u_max)

    where F(u) = 1 - exp(-(u/A)^k) is the Weibull CDF.
    """
    # Weibull parameters
    k = 2.0
    A = 8.0

    # Grid parameters - fine grid to minimize discretization error
    u_max = 5 * A  # 40.0
    n_points = 20001
    u_power_curve = np.linspace(0.0, u_max, n_points)

    # Step power curve threshold (in hub height wind speed terms)
    u0 = 5.0

    # Wind shear and heights
    alpha = 0.14
    h_turbine = 150.0
    h_reference = 100.0
    s = (h_turbine / h_reference) ** alpha

    # Sanity check: s > 1 for h_turbine > h_reference with positive alpha
    assert s > 1.0, f"Expected s > 1, got s = {s}"

    # Build step power curve: p(u) = 1 for u >= u0, else 0
    # This is evaluated at hub height wind speed
    p_power_curve = np.where(u_power_curve >= u0, 1.0, 0.0)

    # Build wind_shear DataArray with dims (y, x)
    wind_shear = xr.DataArray(
        data=np.full((2, 3), alpha),
        dims=["y", "x"],
        coords={"y": [0, 1], "x": [0, 1, 2]},
    )

    # Build Weibull PDF at reference height, normalized over [0, u_max]
    u_over_A = u_power_curve / A
    pdf_raw = (k / A) * (u_over_A ** (k - 1)) * np.exp(-(u_over_A ** k))
    pdf_raw[0] = 0.0  # Handle u=0 for k=2

    # Normalize by trapz to enforce total mass = 1
    total_mass = np.trapezoid(pdf_raw, u_power_curve)
    pdf_1d = pdf_raw / total_mass

    # Verify normalization
    assert np.isclose(np.trapezoid(pdf_1d, u_power_curve), 1.0, atol=1e-10)

    # Broadcast to (wind_speed, y, x)
    pdf_3d = np.broadcast_to(pdf_1d[:, None, None], (n_points, 2, 3))
    weibull_pdf = xr.DataArray(
        data=pdf_3d.copy(),
        dims=["wind_speed", "y", "x"],
        coords={"wind_speed": u_power_curve, "y": [0, 1], "x": [0, 1, 2]},
    )

    # Analytic oracle using Weibull CDF
    # F(u) = 1 - exp(-(u/A)^k)
    def weibull_cdf(u):
        return 1.0 - np.exp(-((u / A) ** k))

    F_u_max = weibull_cdf(u_max)
    F_u0_over_s = weibull_cdf(u0 / s)

    # Oracle: CF = (F(u_max) - F(u0/s)) / F(u_max)
    CF_expected = (F_u_max - F_u0_over_s) / F_u_max

    # Sanity check: CF should be in (0, 1)
    assert 0 < CF_expected < 1, f"Expected CF in (0,1), got {CF_expected}"

    # Call capacity_factor
    result = capacity_factor(
        weibull_pdf,
        wind_shear,
        u_power_curve,
        p_power_curve,
        h_turbine=h_turbine,
        h_reference=h_reference,
        correction_factor=1.0,
    )

    # Assertions
    assert isinstance(result, xr.DataArray), "Result must be xarray DataArray"
    assert result.dims == ("y", "x"), f"Expected dims ('y', 'x'), got {result.dims}"
    assert result.name == "capacity_factor", f"Expected name 'capacity_factor', got {result.name}"

    # Check values match analytic oracle
    assert np.allclose(result.values, CF_expected, atol=5e-4, rtol=0), (
        f"Result {result.values.flatten()[0]:.6f} does not match "
        f"analytic oracle {CF_expected:.6f}"
    )
