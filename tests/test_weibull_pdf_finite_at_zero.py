"""Test that weibull_probability_density returns finite values at u=0, even for k<1."""

import numpy as np
import xarray as xr

from cleo.assess import weibull_probability_density


def test_weibull_pdf_finite_at_zero_k_less_than_1():
    """PDF should be finite (0) at u=0 even when k<1 (which would otherwise give inf)."""
    k = 0.5  # k < 1 causes (u/a)^(k-1) = (0/a)^(-0.5) = inf without protection
    a = 2.0
    u = np.array([0.0, 1.0, 2.0, 3.0])

    # Create tiny spatial grids for weibull_k and weibull_a
    weibull_k = xr.DataArray(
        np.full((2, 2), k),
        dims=["y", "x"],
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    weibull_a = xr.DataArray(
        np.full((2, 2), a),
        dims=["y", "x"],
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )

    pdf = weibull_probability_density(u, weibull_k, weibull_a)

    # All values should be finite
    assert np.all(np.isfinite(pdf.values)), f"PDF contains non-finite values: {pdf.values}"

    # PDF at u=0 should be exactly 0
    pdf_at_zero = pdf.sel(wind_speed=0.0)
    assert float(pdf_at_zero.max()) == 0.0, f"PDF at u=0 should be 0, got {pdf_at_zero.values}"


def test_weibull_pdf_finite_at_zero_k_equal_1():
    """PDF should be finite at u=0 for k=1 (exponential distribution)."""
    k = 1.0
    a = 2.0
    u = np.array([0.0, 1.0, 2.0])

    weibull_k = xr.DataArray(np.full((2, 2), k), dims=["y", "x"])
    weibull_a = xr.DataArray(np.full((2, 2), a), dims=["y", "x"])

    pdf = weibull_probability_density(u, weibull_k, weibull_a)

    assert np.all(np.isfinite(pdf.values)), "PDF should be finite everywhere"
    # For k=1, PDF at u=0 is (1/a)*exp(0) = 1/a, but we set it to 0 for consistency
    pdf_at_zero = pdf.sel(wind_speed=0.0)
    assert float(pdf_at_zero.max()) == 0.0, f"PDF at u=0 should be 0, got {pdf_at_zero.values}"


def test_weibull_pdf_finite_at_zero_k_greater_than_1():
    """PDF should be finite at u=0 for k>1 (naturally 0 anyway)."""
    k = 2.0
    a = 8.0
    u = np.array([0.0, 1.0, 5.0, 10.0])

    weibull_k = xr.DataArray(np.full((2, 2), k), dims=["y", "x"])
    weibull_a = xr.DataArray(np.full((2, 2), a), dims=["y", "x"])

    pdf = weibull_probability_density(u, weibull_k, weibull_a)

    assert np.all(np.isfinite(pdf.values)), "PDF should be finite everywhere"
    pdf_at_zero = pdf.sel(wind_speed=0.0)
    assert float(pdf_at_zero.max()) == 0.0, f"PDF at u=0 should be 0, got {pdf_at_zero.values}"


def test_weibull_pdf_integration_still_valid():
    """Numerical integration of PDF should still approximate CDF correctly."""
    k = 2.0
    a = 8.0
    u = np.linspace(0, 40, 201)  # Include 0

    # Use 2D grid to avoid squeeze removing the y dimension
    weibull_k = xr.DataArray(np.full((2, 2), k), dims=["y", "x"])
    weibull_a = xr.DataArray(np.full((2, 2), a), dims=["y", "x"])

    pdf = weibull_probability_density(u, weibull_k, weibull_a)

    # Extract 1D array for integration (pick any spatial point)
    pdf_1d = pdf.isel(y=0, x=0).values

    # Numerical integration using trapezoid
    numerical_mass = np.trapezoid(pdf_1d, x=u)

    # Analytic CDF at u_max
    u_max = u[-1]
    oracle_mass = 1.0 - np.exp(-((u_max / a) ** k))

    assert abs(numerical_mass - oracle_mass) < 1e-3, (
        f"Numerical mass {numerical_mass} differs from oracle {oracle_mass}"
    )
