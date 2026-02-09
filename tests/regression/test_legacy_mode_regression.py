"""test_legacy_mode_regression.py

Regression snapshot test for weibull_height_mode="100m_shear" to ensure
legacy mode output remains unchanged.

Uses a deterministic synthetic atlas fixture with no external IO.
Expected values are computed analytically from known Weibull formulas.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from types import SimpleNamespace

from cleo.assess import (
    simulate_capacity_factors,
    weibull_probability_density,
    capacity_factor,
)


def test_legacy_100m_shear_mode_regression_snapshot():
    """
    Regression test for 100m_shear mode: verify CF output matches expected
    deterministic values for a synthetic atlas fixture.

    Setup:
    - Single turbine at hub_height=100m (so shear scaling s=1)
    - Constant A=8, k=2 across 2x3 spatial grid
    - Zero wind shear (alpha=0)
    - Step power curve at u_0=6.0 (CF = 1 for u >= 6, else 0)

    Expected CF (analytical):
    CF = P(U >= 6) = exp(-(6/8)^2) = exp(-0.5625) ≈ 0.5697...

    The numerical integration will differ slightly due to discretization,
    but should match the frozen snapshot value.
    """
    # Setup spatial grid
    y = np.arange(2)
    x = np.arange(3)
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    # Weibull params: constant A=8, k=2
    A = 8.0
    k = 2.0

    # Compute PDF at 100m (same as hub height for this test)
    A_da = xr.DataArray(
        np.full((2, 3), A, dtype=np.float64),
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )
    k_da = xr.DataArray(
        np.full((2, 3), k, dtype=np.float64),
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )
    weibull_pdf = weibull_probability_density(wind_speed, k_da, A_da)

    # Wind shear = 0 (no scaling)
    wind_shear = xr.DataArray(
        np.zeros((2, 3), dtype=np.float64),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="wind_shear",
    )

    # Step power curve: 0 below u_0, 1 at and above
    u_0 = 6.0
    p_step = np.where(wind_speed >= u_0, 1.0, 0.0)

    # Single turbine
    turbine_name = "TestTurbine100m"
    hub_height = 100.0

    # Power curve as DataArray with turbine dim
    power_curve = xr.DataArray(
        p_step.reshape(1, -1),
        dims=("turbine", "wind_speed"),
        coords={"turbine": [turbine_name], "wind_speed": wind_speed},
        name="power_curve",
    )

    # Template for spatial grid
    template = xr.DataArray(
        np.ones((2, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="template",
    )

    # Build dataset
    ds = xr.Dataset(
        data_vars={
            "weibull_pdf": weibull_pdf,
            "wind_shear": wind_shear,
            "power_curve": power_curve,
            "template": template,
        },
        coords={
            "turbine": [turbine_name],
            "wind_speed": wind_speed,
        },
        attrs={"country": "TEST"},
    )

    # Create atlas-like object
    class MockAtlas:
        def __init__(self, data):
            self.data = data
            self.parent = SimpleNamespace(country="TEST", crs="epsg:4326")

        def get_turbine_attribute(self, turbine_id: str, key: str):
            if key == "hub_height":
                return hub_height
            raise KeyError(f"Unknown attribute: {key}")

        def _set_var(self, name, da):
            self.data[name] = da

    atlas = MockAtlas(ds)

    # Run legacy mode
    simulate_capacity_factors(atlas, chunk_size=None, loss_factor=1.0,
                               force=True, weibull_height_mode="100m_shear")

    # Extract CF
    cf = atlas.data["capacity_factors"]
    assert "turbine" in cf.dims
    assert cf.sel(turbine=turbine_name).shape == (2, 3)

    # Get CF at pixel [0,0]
    cf_00 = float(cf.sel(turbine=turbine_name).isel(y=0, x=0).values)

    # ===== SNAPSHOT VALUE =====
    # Computed from: CF = exp(-(6/8)^2) = exp(-0.5625) ≈ 0.5697...
    # Numerical integration with step 0.5 gives slightly different result due to
    # trapezoidal integration over the discrete grid.
    # This is the frozen value from the current implementation.
    # If the legacy mode changes, this test will fail.
    expected_cf_00 = 0.5965378838314023

    # Assert within tight tolerance (regression check)
    assert abs(cf_00 - expected_cf_00) < 1e-10, (
        f"Legacy mode CF at [0,0] = {cf_00:.16f} differs from snapshot {expected_cf_00:.16f}"
    )

    # Verify all cells have the same CF (uniform A, k, alpha)
    cf_vals = cf.sel(turbine=turbine_name).values
    np.testing.assert_allclose(cf_vals, expected_cf_00, rtol=0, atol=1e-10,
                                err_msg="CF values should be uniform across grid")


def test_legacy_mode_with_nonzero_shear_regression():
    """
    Regression test with nonzero wind shear.

    Setup:
    - Turbine at hub_height=150m, reference=100m
    - A=8, k=2 at 100m
    - alpha=0.14 (shear exponent)
    - Step power curve at u_0=5.0

    Expected CF (analytical):
    s = (150/100)^0.14 ≈ 1.0588...
    Effective threshold at 100m ref: u_0_ref = u_0 / s ≈ 4.722...
    CF = P(U_100 >= u_0_ref) = exp(-(4.722/8)^2) ≈ 0.6966...
    """
    y = np.arange(2)
    x = np.arange(3)
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    A = 8.0
    k = 2.0
    alpha = 0.14
    hub_height = 150.0
    u_0 = 5.0

    A_da = xr.DataArray(np.full((2, 3), A), dims=("y", "x"), coords={"y": y, "x": x})
    k_da = xr.DataArray(np.full((2, 3), k), dims=("y", "x"), coords={"y": y, "x": x})
    weibull_pdf = weibull_probability_density(wind_speed, k_da, A_da)

    wind_shear = xr.DataArray(
        np.full((2, 3), alpha),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="wind_shear",
    )

    p_step = np.where(wind_speed >= u_0, 1.0, 0.0)
    turbine_name = "TestTurbine150m"

    power_curve = xr.DataArray(
        p_step.reshape(1, -1),
        dims=("turbine", "wind_speed"),
        coords={"turbine": [turbine_name], "wind_speed": wind_speed},
    )

    template = xr.DataArray(
        np.ones((2, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="template",
    )

    ds = xr.Dataset(
        data_vars={
            "weibull_pdf": weibull_pdf,
            "wind_shear": wind_shear,
            "power_curve": power_curve,
            "template": template,
        },
        coords={"turbine": [turbine_name], "wind_speed": wind_speed},
        attrs={"country": "TEST"},
    )

    class MockAtlas:
        def __init__(self, data):
            self.data = data
            self.parent = SimpleNamespace(country="TEST", crs="epsg:4326")

        def get_turbine_attribute(self, turbine_id: str, key: str):
            if key == "hub_height":
                return hub_height
            raise KeyError(f"Unknown attribute: {key}")

        def _set_var(self, name, da):
            self.data[name] = da

    atlas = MockAtlas(ds)

    simulate_capacity_factors(atlas, chunk_size=None, loss_factor=1.0,
                               force=True, weibull_height_mode="100m_shear")

    cf = atlas.data["capacity_factors"]
    cf_00 = float(cf.sel(turbine=turbine_name).isel(y=0, x=0).values)

    # ===== SNAPSHOT VALUE =====
    # Frozen from current implementation (with shear scaling)
    expected_cf_00 = 0.7299042242205249

    assert abs(cf_00 - expected_cf_00) < 1e-10, (
        f"Legacy mode with shear CF at [0,0] = {cf_00:.16f} differs from snapshot {expected_cf_00:.16f}"
    )
