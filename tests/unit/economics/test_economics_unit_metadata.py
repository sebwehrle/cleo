"""Tests for unit metadata contract compliance in economics module.

Verifies that all metric outputs from economics.py have the canonical 'units' attr
as defined in CONTRACT_UNIFIED_ATLAS.md section B9.2.
"""

import numpy as np
import pytest
import xarray as xr

from cleo.economics import (
    lcoe_v1_from_capacity_factors,
    min_lcoe_turbine_idx,
    optimal_energy_gwh_a,
    optimal_power_kw,
)


@pytest.fixture
def capacity_factors():
    """Create minimal capacity factors for testing."""
    cf_data = np.array([[[0.3, 0.35], [0.32, 0.38]]])  # (turbine=1, y=2, x=2)
    cf = xr.DataArray(
        cf_data,
        dims=["turbine", "y", "x"],
        coords={"turbine": ["TestTurbine.100"], "y": [0, 1], "x": [0, 1]},
        name="capacity_factors",
    )
    cf.attrs["cleo:cf_mode"] = "direct_cf_quadrature"
    cf.attrs["cleo:air_density"] = 0
    cf.attrs["cleo:rews_n"] = 12
    cf.attrs["cleo:loss_factor"] = 1.0
    cf.attrs["units"] = "1"
    return cf


@pytest.fixture
def turbine_params():
    """Create minimal turbine parameters for testing."""
    return {
        "turbine_ids": ("TestTurbine.100",),
        "power_kw": np.array([2000.0]),
        "overnight_cost_eur_per_kw": np.array([1200.0]),
    }


@pytest.fixture
def economics_params():
    """Create minimal economics parameters for testing."""
    return {
        "bos_cost_share": 0.0,
        "om_fixed_eur_per_kw_a": 20.0,
        "om_variable_eur_per_kwh": 0.008,
        "discount_rate": 0.05,
        "lifetime_a": 25,
        "hours_per_year": 8766.0,
    }


class TestLcoeUnitsAttr:
    """Tests for LCOE units attr."""

    def test_lcoe_has_units_attr(self, capacity_factors, turbine_params, economics_params):
        """lcoe output has 'units' attr."""
        lcoe = lcoe_v1_from_capacity_factors(
            cf=capacity_factors,
            turbine_ids=turbine_params["turbine_ids"],
            power_kw=turbine_params["power_kw"],
            overnight_cost_eur_per_kw=turbine_params["overnight_cost_eur_per_kw"],
            **economics_params,
        )

        assert "units" in lcoe.attrs
        assert lcoe.attrs["units"] == "EUR/MWh"


class TestOptimalPowerUnitsAttr:
    """Tests for optimal_power units attr."""

    def test_optimal_power_has_units_attr(self, capacity_factors, turbine_params, economics_params):
        """optimal_power output has 'units' attr."""
        lcoe = lcoe_v1_from_capacity_factors(
            cf=capacity_factors,
            turbine_ids=turbine_params["turbine_ids"],
            power_kw=turbine_params["power_kw"],
            overnight_cost_eur_per_kw=turbine_params["overnight_cost_eur_per_kw"],
            **economics_params,
        )

        opt_power = optimal_power_kw(
            lcoe=lcoe,
            power_kw=turbine_params["power_kw"],
        )

        assert "units" in opt_power.attrs
        assert opt_power.attrs["units"] == "kW"


class TestOptimalEnergyUnitsAttr:
    """Tests for optimal_energy units attr."""

    def test_optimal_energy_has_units_attr(self, capacity_factors, turbine_params, economics_params):
        """optimal_energy output has 'units' attr."""
        lcoe = lcoe_v1_from_capacity_factors(
            cf=capacity_factors,
            turbine_ids=turbine_params["turbine_ids"],
            power_kw=turbine_params["power_kw"],
            overnight_cost_eur_per_kw=turbine_params["overnight_cost_eur_per_kw"],
            **economics_params,
        )

        opt_energy = optimal_energy_gwh_a(
            lcoe=lcoe,
            cf=capacity_factors,
            power_kw=turbine_params["power_kw"],
            hours_per_year=economics_params["hours_per_year"],
        )

        assert "units" in opt_energy.attrs
        assert opt_energy.attrs["units"] == "GWh/a"


class TestMinLcoeTurbineNoUnitsAttr:
    """Tests for min_lcoe_turbine units attr (should be None/dimensionless)."""

    def test_min_lcoe_turbine_no_units_attr(self, capacity_factors, turbine_params, economics_params):
        """min_lcoe_turbine output has no 'units' attr (index is dimensionless)."""
        lcoe = lcoe_v1_from_capacity_factors(
            cf=capacity_factors,
            turbine_ids=turbine_params["turbine_ids"],
            power_kw=turbine_params["power_kw"],
            overnight_cost_eur_per_kw=turbine_params["overnight_cost_eur_per_kw"],
            **economics_params,
        )

        min_idx = min_lcoe_turbine_idx(
            lcoe=lcoe,
            turbine_ids=turbine_params["turbine_ids"],
        )

        # min_lcoe_turbine is an index - no units expected
        # Per contract B9.2, min_lcoe_turbine has canonical unit of None (dimensionless)
        assert min_idx.attrs.get("units") is None
