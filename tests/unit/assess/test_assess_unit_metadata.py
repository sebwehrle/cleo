"""Tests for unit metadata contract compliance in assess module.

Verifies that all metric outputs from assess.py have the canonical 'units' attr
as defined in docs/CONTRACT_UNIFIED_ATLAS.md section B9.2.
"""

import numpy as np
import pytest
import xarray as xr

from cleo.assess import (
    capacity_factors_v1,
    mean_wind_speed_from_weibull,
    rews_mps_v1,
)


@pytest.fixture
def weibull_stack():
    """Create minimal Weibull A/k stack for testing."""
    heights = [50, 100, 150]
    y = [0, 1]
    x = [0, 1]

    A_data = np.ones((len(heights), len(y), len(x))) * 7.0
    k_data = np.ones((len(heights), len(y), len(x))) * 2.0

    A = xr.DataArray(
        A_data,
        dims=["height", "y", "x"],
        coords={"height": heights, "y": y, "x": x},
        name="weibull_A",
    )
    k = xr.DataArray(
        k_data,
        dims=["height", "y", "x"],
        coords={"height": heights, "y": y, "x": x},
        name="weibull_k",
    )
    return A, k


@pytest.fixture
def turbine_params():
    """Create minimal turbine parameters for testing."""
    u_grid = np.arange(0.0, 30.0, 0.5)
    # Simple power curve: ramps from 0 to 1 between 3-12 m/s, rated 12-25, then 0
    power_curve = np.zeros_like(u_grid)
    mask_ramp = (u_grid >= 3) & (u_grid < 12)
    mask_rated = (u_grid >= 12) & (u_grid <= 25)
    power_curve[mask_ramp] = (u_grid[mask_ramp] - 3) / 9
    power_curve[mask_rated] = 1.0

    return {
        "u_grid": u_grid,
        "turbine_ids": ("TestTurbine.100",),
        "hub_heights_m": np.array([100.0]),
        "rotor_diameters_m": np.array([80.0]),
        "power_curves": power_curve.reshape(1, -1),
    }


class TestCapacityFactorsUnitsAttr:
    """Tests for capacity_factors units attr."""

    def test_capacity_factors_has_units_attr(self, weibull_stack, turbine_params):
        """capacity_factors output has 'units' attr."""
        A, k = weibull_stack

        cf = capacity_factors_v1(
            A_stack=A,
            k_stack=k,
            u_grid=turbine_params["u_grid"],
            turbine_ids=turbine_params["turbine_ids"],
            hub_heights_m=turbine_params["hub_heights_m"],
            power_curves=turbine_params["power_curves"],
            rotor_diameters_m=turbine_params["rotor_diameters_m"],
            mode="direct_cf_quadrature",
        )

        assert "units" in cf.attrs
        assert cf.attrs["units"] == "1"  # dimensionless

    def test_capacity_factors_units_attr_all_modes(self, weibull_stack, turbine_params):
        """capacity_factors has units attr for all computation modes."""
        A, k = weibull_stack

        for mode in ["direct_cf_quadrature", "hub", "rews", "momentmatch_weibull"]:
            cf = capacity_factors_v1(
                A_stack=A,
                k_stack=k,
                u_grid=turbine_params["u_grid"],
                turbine_ids=turbine_params["turbine_ids"],
                hub_heights_m=turbine_params["hub_heights_m"],
                power_curves=turbine_params["power_curves"],
                rotor_diameters_m=turbine_params["rotor_diameters_m"],
                mode=mode,
            )

            assert "units" in cf.attrs, f"Missing units attr for mode={mode}"
            assert cf.attrs["units"] == "1", f"Wrong units for mode={mode}"


class TestRewsMpsUnitsAttr:
    """Tests for rews_mps units attr."""

    def test_rews_mps_has_units_attr(self, weibull_stack, turbine_params):
        """rews_mps output has 'units' attr."""
        A, k = weibull_stack

        rews = rews_mps_v1(
            A_stack=A,
            k_stack=k,
            turbine_ids=turbine_params["turbine_ids"],
            hub_heights_m=turbine_params["hub_heights_m"],
            rotor_diameters_m=turbine_params["rotor_diameters_m"],
        )

        assert "units" in rews.attrs
        assert rews.attrs["units"] == "m/s"


class TestMeanWindSpeedUnitsAttr:
    """Tests for mean_wind_speed units attr."""

    def test_mean_wind_speed_has_units_attr(self):
        """mean_wind_speed output has 'units' attr."""
        A = xr.DataArray([7.0, 8.0], dims=["x"])
        k = xr.DataArray([2.0, 2.1], dims=["x"])

        mws = mean_wind_speed_from_weibull(A=A, k=k)

        assert "units" in mws.attrs
        assert mws.attrs["units"] == "m/s"
