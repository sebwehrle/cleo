"""Test that capacity_factor correctly applies wind shear scaling."""

import numpy as np
import xarray as xr

from cleo.assess import capacity_factor


def test_capacity_factor_shear_scaling():
    """
    Verify wind shear scaling is applied correctly.

    With wind_shear=1 and h_turbine/h_reference=2, scaling factor s=2.
    Power curve p(u)=u^2 (normalized), so p(2*u) = 4*u^2.
    CF = integral over u_ref of pdf(u_ref) * p(s * u_ref).
    """
    # Simple grid
    u = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    # Uniform PDF (for simplicity, unnormalized)
    pdf_vals = np.ones((len(u), 1, 1)) * 0.2  # Constant PDF
    pdf = xr.DataArray(
        pdf_vals,
        dims=["wind_speed", "y", "x"],
        coords={"wind_speed": u, "y": [0.0], "x": [0.0]},
    )

    # Wind shear = 1 everywhere
    wind_shear = xr.DataArray(
        np.array([[1.0]]),
        dims=["y", "x"],
        coords={"y": [0.0], "x": [0.0]},
    )

    # Power curve: p(u) = (u/5)^2, capped at 1
    # This gives p = [0, 0.04, 0.16, 0.36, 0.64, 1.0]
    p = np.minimum((u / 5.0) ** 2, 1.0)

    # With h_turbine=200, h_reference=100, s = 2^1 = 2
    # So we evaluate p at scaled speeds: 2*u = [0, 2, 4, 6, 8, 10]
    # p(2*u) = [(0/5)^2, (2/5)^2, (4/5)^2, 1, 1, 1] = [0, 0.16, 0.64, 1, 1, 1]
    h_turbine = 200
    h_reference = 100

    cf = capacity_factor(
        weibull_pdf=pdf,
        wind_shear=wind_shear,
        u_power_curve=u,
        p_power_curve=p,
        h_turbine=h_turbine,
        h_reference=h_reference,
    )

    # Compute expected value manually using trapezoid
    s = (h_turbine / h_reference) ** 1.0  # s = 2
    u_scaled = u * s
    p_scaled = np.interp(u_scaled, u, p, left=0.0, right=0.0)
    integrand = 0.2 * p_scaled  # pdf * p_scaled
    expected_cf = np.trapezoid(integrand, x=u)

    np.testing.assert_allclose(
        cf.values.flatten()[0],
        expected_cf,
        rtol=1e-10,
        err_msg=f"CF mismatch: got {cf.values.flatten()[0]}, expected {expected_cf}",
    )


def test_capacity_factor_no_shear():
    """With wind_shear=0, s=1 regardless of height ratio."""
    u = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    p = np.array([0.0, 0.2, 0.6, 0.9, 1.0])

    pdf = xr.DataArray(
        np.ones((len(u), 2, 2)) * 0.05,
        dims=["wind_speed", "y", "x"],
        coords={"wind_speed": u, "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )

    wind_shear = xr.DataArray(
        np.zeros((2, 2)),
        dims=["y", "x"],
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )

    # With wind_shear=0, s = (h_turbine/h_reference)^0 = 1, so no scaling
    cf = capacity_factor(
        weibull_pdf=pdf,
        wind_shear=wind_shear,
        u_power_curve=u,
        p_power_curve=p,
        h_turbine=150,  # Different from reference
        h_reference=100,
    )

    # Compute expected: integral of pdf * p with no scaling
    integrand = 0.05 * p
    expected_cf = np.trapezoid(integrand, x=u)

    np.testing.assert_allclose(
        cf.values,
        expected_cf,
        rtol=1e-10,
    )
