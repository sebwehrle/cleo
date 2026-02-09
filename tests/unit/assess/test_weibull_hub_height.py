r"""test_weibull_hub_height.py

Oracle tests for hub-height Weibull interpolation and capacity factor computation.

Tests are independent of raster IO - they construct xarray objects directly.

Oracle formulas (LaTeX notation):
- Test A (interpolation): A(h) = a_0 + a_1 \ln(h), k(h) = b_0 + b_1 \ln(h)
- Test B (PDF): f(u) = \frac{k}{A} \left(\frac{u}{A}\right)^{k-1} \exp\left(-\left(\frac{u}{A}\right)^k\right) for u > 0, f(0) = 0
- Test C (step curve CF): CF = P(U \geq u_0) = \exp\left(-\left(\frac{u_0}{A}\right)^k\right)
- Test D (top-hat CF): CF = \exp\left(-\left(\frac{u_{in}}{A}\right)^k\right) - \exp\left(-\left(\frac{u_{out}}{A}\right)^k\right)
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tests.helpers.oracles import weibull_pdf as weibull_pdf_oracle, weibull_tail as weibull_tail_probability
from tests.helpers.factories import wind_speed_axis

from cleo.assess import (
    interpolate_weibull_params_to_height,
    weibull_probability_density,
    capacity_factor,
)


# ============================================================================
# Test A: Interpolation Oracle
# ============================================================================

def test_interpolate_weibull_log_height_linear_oracle():
    """
    Test interpolation using log-height-linear model.

    Oracle: A(h) = a_0 + a_1 * ln(h), k(h) = b_0 + b_1 * ln(h)
    For target_height=125:
    A_oracle = a_0 + a_1 * ln(125)
    k_oracle = b_0 + b_1 * ln(125)
    """
    # Setup: small spatial grid (y=2, x=3)
    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(3)

    # Define per-cell coefficients
    # a_0, a_1 for A(h); b_0, b_1 for k(h)
    # Each has shape (y, x)
    np.random.seed(42)
    a_0 = np.random.uniform(3.0, 5.0, size=(2, 3))
    a_1 = np.random.uniform(0.5, 1.5, size=(2, 3))
    b_0 = np.random.uniform(1.0, 1.5, size=(2, 3))
    b_1 = np.random.uniform(0.1, 0.3, size=(2, 3))

    # Construct A and k at each height using oracle formula
    A_vals = np.stack([a_0 + a_1 * np.log(h) for h in heights], axis=0)  # (4, 2, 3)
    k_vals = np.stack([b_0 + b_1 * np.log(h) for h in heights], axis=0)  # (4, 2, 3)

    # Create DataArrays
    A_da = xr.DataArray(
        A_vals,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="weibull_A",
    )
    k_da = xr.DataArray(
        k_vals,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="weibull_k",
    )

    # Interpolate to target height
    target_height = 125.0
    A_hub, k_hub = interpolate_weibull_params_to_height(A_da, k_da, target_height)

    # Oracle values at target_height
    A_oracle = a_0 + a_1 * np.log(target_height)
    k_oracle = b_0 + b_1 * np.log(target_height)

    # Assert max abs diff < 1e-12
    A_diff = np.abs(A_hub.values - A_oracle).max()
    k_diff = np.abs(k_hub.values - k_oracle).max()

    assert A_diff < 1e-12, f"A interpolation error {A_diff} exceeds tolerance"
    assert k_diff < 1e-12, f"k interpolation error {k_diff} exceeds tolerance"

    # Check hub_height coord is present
    assert "hub_height" in A_hub.coords
    assert float(A_hub.coords["hub_height"]) == target_height


def test_interpolate_weibull_extrapolation_raises():
    """Test that extrapolation outside height range raises ValueError."""
    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(3)

    A_da = xr.DataArray(
        np.ones((4, 2, 3)) * 8.0,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
    )
    k_da = xr.DataArray(
        np.ones((4, 2, 3)) * 2.0,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
    )

    # Below minimum
    with pytest.raises(ValueError, match="outside available height range"):
        interpolate_weibull_params_to_height(A_da, k_da, 40.0)

    # Above maximum
    with pytest.raises(ValueError, match="outside available height range"):
        interpolate_weibull_params_to_height(A_da, k_da, 250.0)


def test_interpolate_weibull_unsupported_method_raises():
    """Test that unsupported interpolation method raises ValueError."""
    heights = np.array([50.0, 100.0])
    A_da = xr.DataArray(np.ones((2, 2, 2)), dims=("height", "y", "x"), coords={"height": heights})
    k_da = xr.DataArray(np.ones((2, 2, 2)), dims=("height", "y", "x"), coords={"height": heights})

    with pytest.raises(ValueError, match="Unsupported interpolation method"):
        interpolate_weibull_params_to_height(A_da, k_da, 75.0, method="cubic")


def test_interpolate_weibull_insufficient_heights_raises():
    """Test that fewer than 2 heights raises ValueError."""
    heights = np.array([100.0])
    A_da = xr.DataArray(np.ones((1, 2, 2)), dims=("height", "y", "x"), coords={"height": heights})
    k_da = xr.DataArray(np.ones((1, 2, 2)), dims=("height", "y", "x"), coords={"height": heights})

    with pytest.raises(ValueError, match="At least 2 heights required"):
        interpolate_weibull_params_to_height(A_da, k_da, 100.0)


# ============================================================================
# Test B: Hub-height PDF Oracle
# ============================================================================

def test_weibull_pdf_oracle_pointwise():
    """
    Test weibull_probability_density against closed-form oracle.

    Oracle: f(u) = (k/A) * (u/A)^(k-1) * exp(-(u/A)^k) for u > 0
            f(0) = 0 (package convention)
    """
    # Use synthetic A and k from log-height model
    target_height = 125.0
    a_0, a_1 = 4.0, 1.0
    b_0, b_1 = 1.5, 0.2

    A_hub = a_0 + a_1 * np.log(target_height)
    k_hub = b_0 + b_1 * np.log(target_height)

    # Wind speed grid (include 0)
    u = wind_speed_axis()

    # Create scalar DataArrays for A and k
    A_da = xr.DataArray(A_hub)
    k_da = xr.DataArray(k_hub)

    # Compute PDF via package function
    pdf_package = weibull_probability_density(u, k_da, A_da)

    # Compute oracle PDF
    pdf_oracle = weibull_pdf_oracle(u, k_hub, A_hub)

    # Check f(0) = 0 exactly
    assert pdf_package.sel(wind_speed=0.0).values == 0.0, "f(0) must be exactly 0"

    # Check pointwise closeness for u > 0
    u_positive = u[u > 0]
    pdf_pkg_positive = pdf_package.sel(wind_speed=u_positive).values
    pdf_oracle_positive = pdf_oracle[u > 0]

    np.testing.assert_allclose(
        pdf_pkg_positive, pdf_oracle_positive, rtol=1e-12, atol=1e-12,
        err_msg="PDF values do not match oracle for u > 0"
    )


def test_weibull_pdf_spatial_matches_oracle():
    """Test PDF computation with spatial (y, x) Weibull parameters."""
    y = np.arange(2)
    x = np.arange(3)

    # Spatially varying A and k
    A_vals = np.array([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    k_vals = np.array([[1.5, 1.8, 2.0], [2.2, 2.5, 2.8]])

    A_da = xr.DataArray(A_vals, dims=("y", "x"), coords={"y": y, "x": x})
    k_da = xr.DataArray(k_vals, dims=("y", "x"), coords={"y": y, "x": x})

    u = np.array([0.0, 5.0, 10.0, 15.0])

    pdf = weibull_probability_density(u, k_da, A_da)

    # Verify for each cell
    for iy in range(2):
        for ix in range(3):
            A_cell = A_vals[iy, ix]
            k_cell = k_vals[iy, ix]
            pdf_cell = pdf.isel(y=iy, x=ix).values
            pdf_oracle = weibull_pdf_oracle(u, k_cell, A_cell)

            np.testing.assert_allclose(
                pdf_cell, pdf_oracle, rtol=1e-12, atol=1e-12,
                err_msg=f"PDF mismatch at cell (y={iy}, x={ix})"
            )


# ============================================================================
# Test C: CF Oracle with Step Power Curve (Tail Probability)
# ============================================================================

def test_capacity_factor_step_curve_tail_probability_oracle():
    """
    Test CF with step power curve equals Weibull tail probability.

    Step curve: CF(u) = 0 for u < u_0, CF(u) = 1 for u >= u_0
    Oracle: CF = P(U >= u_0) = exp(-(u_0/A)^k)
    """
    # Target hub height and Weibull params from log-height model
    target_height = 125.0
    a_0, a_1 = 4.0, 1.0
    b_0, b_1 = 1.5, 0.2

    A_hub = a_0 + a_1 * np.log(target_height)
    k_hub = b_0 + b_1 * np.log(target_height)

    # Wind speed grid and step threshold (on grid)
    u = wind_speed_axis()
    u_0 = 6.0  # Step threshold

    # Step power curve: 0 below u_0, 1 at and above
    p_step = np.where(u >= u_0, 1.0, 0.0)

    # Compute PDF at hub height
    A_da = xr.DataArray([[A_hub]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    k_da = xr.DataArray([[k_hub]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    pdf = weibull_probability_density(u, k_da, A_da)

    # Dummy wind shear (zeros for no scaling)
    wind_shear = xr.DataArray([[0.0]], dims=("y", "x"), coords={"y": [0], "x": [0]})

    # Compute CF using package function
    cf_numeric = capacity_factor(
        weibull_pdf=pdf,
        wind_shear=wind_shear,
        u_power_curve=u,
        p_power_curve=p_step,
        h_turbine=target_height,
        h_reference=target_height,  # No scaling
        correction_factor=1.0,
    )

    # Oracle: tail probability
    cf_oracle = weibull_tail_probability(u_0, k_hub, A_hub)

    cf_value = float(cf_numeric.isel(y=0, x=0).values)
    # Tolerance: trapezoidal integration with h=0.5 grid has O(h²) ≈ O(0.25) discretization error.
    # For Weibull integrals, observed error is ~3-4% with standard grid.
    assert abs(cf_value - cf_oracle) < 0.03, (
        f"Step curve CF {cf_value:.6f} differs from oracle {cf_oracle:.6f} "
        f"by {abs(cf_value - cf_oracle):.6f}"
    )


# ============================================================================
# Test D: CF Oracle with Top-hat Curve (Difference of Tails)
# ============================================================================

def test_capacity_factor_tophat_curve_difference_of_tails_oracle():
    """
    Test CF with top-hat power curve equals difference of tail probabilities.

    Top-hat curve: CF(u) = 1 for u_in <= u < u_out, else 0
    Oracle: CF = exp(-(u_in/A)^k) - exp(-(u_out/A)^k)
    """
    # Target hub height and Weibull params
    target_height = 125.0
    a_0, a_1 = 4.0, 1.0
    b_0, b_1 = 1.5, 0.2

    A_hub = a_0 + a_1 * np.log(target_height)
    k_hub = b_0 + b_1 * np.log(target_height)

    # Wind speed grid
    u = wind_speed_axis()
    u_in = 6.0
    u_out = 12.0

    # Top-hat power curve: 1 for u_in <= u < u_out
    p_tophat = np.where((u >= u_in) & (u < u_out), 1.0, 0.0)

    # Compute PDF
    A_da = xr.DataArray([[A_hub]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    k_da = xr.DataArray([[k_hub]], dims=("y", "x"), coords={"y": [0], "x": [0]})
    pdf = weibull_probability_density(u, k_da, A_da)

    # Dummy wind shear
    wind_shear = xr.DataArray([[0.0]], dims=("y", "x"), coords={"y": [0], "x": [0]})

    # Compute CF
    cf_numeric = capacity_factor(
        weibull_pdf=pdf,
        wind_shear=wind_shear,
        u_power_curve=u,
        p_power_curve=p_tophat,
        h_turbine=target_height,
        h_reference=target_height,
        correction_factor=1.0,
    )

    # Oracle: difference of tails
    cf_oracle = weibull_tail_probability(u_in, k_hub, A_hub) - weibull_tail_probability(u_out, k_hub, A_hub)

    cf_value = float(cf_numeric.isel(y=0, x=0).values)
    # Tolerance: trapezoidal integration with h=0.5 grid has O(h²) discretization error
    assert abs(cf_value - cf_oracle) < 0.03, (
        f"Top-hat curve CF {cf_value:.6f} differs from oracle {cf_oracle:.6f} "
        f"by {abs(cf_value - cf_oracle):.6f}"
    )


# ============================================================================
# Test E: Equivalence when shear model matches interpolation
# ============================================================================

def test_hub_mode_equals_shear_mode_when_power_law_holds():
    """
    Test that hub-height mode equals 100m_shear mode when Weibull A follows power law.

    Setup:
    - k is constant across all heights (k = k_0)
    - A follows power law: A(h) = A_100 * (h/100)^alpha

    For a step power curve at threshold u_0:
    - Hub mode: PDF at h* -> CF = exp(-(u_0/A(h*))^k)
    - Shear mode: PDF at 100m, scale u by (h*/100)^alpha
      -> equivalent to CF = exp(-(u_0/A(h*))^k)

    Both should give same result.
    """
    from cleo.assess import simulate_capacity_factors
    from dataclasses import dataclass
    from types import SimpleNamespace

    # Setup: constant k, power-law A
    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(3)

    # Spatially varying A_100 and alpha
    np.random.seed(123)
    A_100 = np.random.uniform(7.0, 9.0, size=(2, 3))
    alpha = np.random.uniform(0.1, 0.2, size=(2, 3))
    k_0 = 2.0  # Constant k everywhere

    # Construct A at each height using power law
    A_vals = np.stack([A_100 * (h / 100.0) ** alpha for h in heights], axis=0)
    k_vals = np.ones((4, 2, 3)) * k_0

    # Mean wind speed (∝ A for fixed k via Gamma function)
    # For shear calculation: alpha = (ln(u_100) - ln(u_50)) / (ln(100) - ln(50))
    # With power law A: u_mean ∝ A -> alpha_shear should equal our alpha
    from scipy.special import gamma as gamma_func
    mean_factor = gamma_func(1 + 1/k_0)  # E[U] = A * Gamma(1 + 1/k)
    u_mean_50 = A_100 * (50/100)**alpha * mean_factor
    u_mean_100 = A_100 * mean_factor

    # Wind speed grid
    wind_speed = wind_speed_axis()

    # Hub height and step threshold
    h_star = 125.0
    u_0 = 6.0

    # Step power curve
    p_step = np.where(wind_speed >= u_0, 1.0, 0.0)

    # Create DataArrays for multi-height Weibull params
    A_da = xr.DataArray(
        A_vals,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
    )
    k_da = xr.DataArray(
        k_vals,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
    )

    # Compute wind_shear from mean wind speeds
    with np.errstate(divide="ignore", invalid="ignore"):
        wind_shear_vals = (np.log(u_mean_100) - np.log(u_mean_50)) / (np.log(100) - np.log(50))
    wind_shear = xr.DataArray(
        wind_shear_vals, dims=("y", "x"), coords={"y": y, "x": x}, name="wind_shear"
    )

    # ====== Hub mode: interpolate A to h_star, compute CF ======
    A_hub, k_hub = interpolate_weibull_params_to_height(A_da, k_da, h_star)
    pdf_hub = weibull_probability_density(wind_speed, k_hub, A_hub)

    dummy_shear = xr.zeros_like(wind_shear)
    cf_hub = capacity_factor(
        weibull_pdf=pdf_hub,
        wind_shear=dummy_shear,
        u_power_curve=wind_speed,
        p_power_curve=p_step,
        h_turbine=h_star,
        h_reference=h_star,
        correction_factor=1.0,
    )

    # ====== Shear mode: PDF at 100m, apply shear scaling ======
    # A and k at 100m
    A_100_da = xr.DataArray(A_100, dims=("y", "x"), coords={"y": y, "x": x})
    k_100_da = xr.DataArray(np.full((2, 3), k_0), dims=("y", "x"), coords={"y": y, "x": x})
    pdf_100 = weibull_probability_density(wind_speed, k_100_da, A_100_da)

    cf_shear = capacity_factor(
        weibull_pdf=pdf_100,
        wind_shear=wind_shear,
        u_power_curve=wind_speed,
        p_power_curve=p_step,
        h_turbine=h_star,
        h_reference=100.0,
        correction_factor=1.0,
    )

    # ====== Oracle: direct calculation ======
    A_at_hstar = A_100 * (h_star / 100.0) ** alpha
    cf_oracle = np.exp(-((u_0 / A_at_hstar) ** k_0))

    # Tolerance: trapezoidal integration with h=0.5 grid has discretization error
    tol = 0.03

    # Compare hub mode to oracle
    max_diff_hub = np.abs(cf_hub.values - cf_oracle).max()
    assert max_diff_hub < tol, f"Hub mode differs from oracle by {max_diff_hub}"

    # Compare shear mode to oracle
    max_diff_shear = np.abs(cf_shear.values - cf_oracle).max()
    assert max_diff_shear < tol, f"Shear mode differs from oracle by {max_diff_shear}"

    # Compare hub mode to shear mode (should match closely when power law holds)
    max_diff_modes = np.abs(cf_hub.values - cf_shear.values).max()
    assert max_diff_modes < tol, f"Hub and shear modes differ by {max_diff_modes}"


# ============================================================================
# Test F: Exact-height identity (interpolator returns exact slice at grid point)
# ============================================================================

def test_interpolate_exact_height_identity():
    """
    If target_height equals a known height in the grid, interpolator must return
    the exact slice (no drift from interpolation).
    """
    heights = np.array([50.0, 100.0, 150.0, 200.0])
    y = np.arange(2)
    x = np.arange(3)

    # Use arbitrary nonlinear values (not following any simple model)
    np.random.seed(999)
    A_vals = np.random.uniform(5.0, 12.0, size=(4, 2, 3))
    k_vals = np.random.uniform(1.5, 3.0, size=(4, 2, 3))

    A_da = xr.DataArray(
        A_vals,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="weibull_A",
    )
    k_da = xr.DataArray(
        k_vals,
        dims=("height", "y", "x"),
        coords={"height": heights, "y": y, "x": x},
        name="weibull_k",
    )

    for h in heights:
        A_hub, k_hub = interpolate_weibull_params_to_height(A_da, k_da, h)

        # Expected: exact slice at height h
        A_expected = A_da.sel(height=h).values
        k_expected = k_da.sel(height=h).values

        # Must match exactly (or within floating point epsilon)
        np.testing.assert_allclose(
            A_hub.values, A_expected, rtol=0, atol=1e-15,
            err_msg=f"A at exact height {h} differs from slice"
        )
        np.testing.assert_allclose(
            k_hub.values, k_expected, rtol=0, atol=1e-15,
            err_msg=f"k at exact height {h} differs from slice"
        )


# ============================================================================
# Test G: PDF probability mass sanity check
# ============================================================================

def test_pdf_mass_close_to_one_on_grid():
    """
    Sanity test: PDF integrated over wind_speed grid (0..40 step 0.5) should be
    close to 1 for reasonable Weibull params (minimal tail truncation at 40 m/s).
    """
    # Use params where truncation at 40 m/s is negligible
    # A=8, k=2 gives mean ~7.1 m/s, so 40 m/s is well into the tail
    A = 8.0
    k = 2.0

    # Standard wind speed grid
    u = wind_speed_axis()

    # Scalar case
    A_da = xr.DataArray(A)
    k_da = xr.DataArray(k)

    pdf = weibull_probability_density(u, k_da, A_da)

    # Integrate using trapezoidal rule
    mass = float(np.trapezoid(pdf.values, x=u))

    # The analytic CDF at u=40 for A=8, k=2 is 1 - exp(-(40/8)^2) = 1 - exp(-25) ≈ 1.0
    # So mass should be very close to 1
    assert mass >= 0.999, f"PDF mass {mass} is too low (expected >= 0.999)"
    assert mass <= 1.001, f"PDF mass {mass} exceeds 1 (numerical issue)"
