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

from tests.helpers.oracles import weibull_pdf as weibull_pdf_oracle
from tests.helpers.factories import wind_speed_axis

from cleo.assess import (
    interpolate_weibull_params_to_height,
    weibull_probability_density,
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
        pdf_pkg_positive,
        pdf_oracle_positive,
        rtol=1e-12,
        atol=1e-12,
        err_msg="PDF values do not match oracle for u > 0",
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
                pdf_cell, pdf_oracle, rtol=1e-12, atol=1e-12, err_msg=f"PDF mismatch at cell (y={iy}, x={ix})"
            )


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
            A_hub.values, A_expected, rtol=0, atol=1e-15, err_msg=f"A at exact height {h} differs from slice"
        )
        np.testing.assert_allclose(
            k_hub.values, k_expected, rtol=0, atol=1e-15, err_msg=f"k at exact height {h} differs from slice"
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
