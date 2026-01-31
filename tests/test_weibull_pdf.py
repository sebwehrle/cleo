"""Minimal unit test for Weibull PDF numerical integration."""

import numpy as np


def test_weibull_pdf_integration_mass():
    """
    Verify that numerically integrating the Weibull PDF over [0, u_max]
    matches the analytic CDF (oracle) within tolerance.

    Weibull PDF: f(u; k, A) = (k/A) * (u/A)^(k-1) * exp(-(u/A)^k)  for u >= 0
    Weibull CDF: F(u) = 1 - exp(-(u/A)^k)
    """
    # Fixed parameters
    k = 2.0
    A = 8.0
    u_max = 5 * A  # 40.0, so exp(-(u_max/A)^k) = exp(-25) is tiny

    # Grid: 20001 points from 0 to u_max
    u = np.linspace(0, u_max, 20001, dtype=np.float64)

    # Compute Weibull PDF at each grid point
    # f(u; k, A) = (k/A) * (u/A)^(k-1) * exp(-(u/A)^k)
    u_over_A = u / A
    pdf = (k / A) * (u_over_A ** (k - 1)) * np.exp(-(u_over_A ** k))

    # Numerical integration using trapezoidal rule
    numerical_mass = np.trapz(pdf, x=u)

    # Analytic oracle: CDF at u_max
    # F(u_max) = 1 - exp(-(u_max/A)^k)
    oracle_mass = 1.0 - np.exp(-((u_max / A) ** k))

    # Assertion: numerical integral matches oracle within tolerance
    assert abs(numerical_mass - oracle_mass) <= 5e-4, (
        f"Numerical mass {numerical_mass} differs from oracle {oracle_mass} "
        f"by {abs(numerical_mass - oracle_mass)}"
    )
