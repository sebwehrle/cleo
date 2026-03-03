"""Contract: economics outputs carry timebase provenance.

This test documents and enforces that all LCOE-family metrics carry
cleo:hours_per_year attrs for provenance tracking.

Covered metrics:
- lcoe
- min_lcoe_turbine
- optimal_power
- optimal_energy
"""

import warnings

import numpy as np
import xarray as xr

from cleo.economics import (
    lcoe_v1_from_capacity_factors,
    min_lcoe_turbine_idx,
    optimal_power_kw,
    optimal_energy_gwh_a,
)


def _make_minimal_cf():
    """Create minimal capacity factors for economics functions."""
    y = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([0.0, 1.0], dtype=np.float64)
    turbine_ids = ("T1", "T2")

    cf = xr.DataArray(
        np.full((2, 2, 2), 0.3, dtype=np.float64),
        dims=("turbine", "y", "x"),
        coords={"turbine": list(turbine_ids), "y": y, "x": x},
        name="capacity_factors",
    )
    cf.attrs["cleo:cf_mode"] = "hub"

    return cf, turbine_ids


def _lcoe_params():
    return {
        "bos_cost_share": 0.0,
        "om_fixed_eur_per_kw_a": 25.0,
        "om_variable_eur_per_kwh": 0.01,
        "discount_rate": 0.05,
        "lifetime_a": 20,
        "hours_per_year": 8766.0,
    }


class TestTimebaseAttrsPresence:
    """Tests that verify cleo:hours_per_year is present on all economics outputs."""

    def test_lcoe_has_hours_per_year_attr(self):
        cf, turbine_ids = _make_minimal_cf()
        power_kw = np.array([3000.0, 3500.0], dtype=np.float64)
        overnight_cost = np.array([1300.0, 1400.0], dtype=np.float64)

        result = lcoe_v1_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=power_kw,
            overnight_cost_eur_per_kw=overnight_cost,
            **_lcoe_params(),
        )

        assert "cleo:hours_per_year" in result.attrs
        assert result.attrs["cleo:hours_per_year"] == 8766.0

    def test_min_lcoe_turbine_has_hours_per_year_attr(self):
        cf, turbine_ids = _make_minimal_cf()
        power_kw = np.array([3000.0, 3500.0], dtype=np.float64)
        overnight_cost = np.array([1300.0, 1400.0], dtype=np.float64)

        lcoe = lcoe_v1_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=power_kw,
            overnight_cost_eur_per_kw=overnight_cost,
            **_lcoe_params(),
        )

        result = min_lcoe_turbine_idx(lcoe=lcoe, turbine_ids=turbine_ids)

        assert "cleo:hours_per_year" in result.attrs
        assert result.attrs["cleo:hours_per_year"] == 8766.0

    def test_optimal_power_has_hours_per_year_attr(self):
        cf, turbine_ids = _make_minimal_cf()
        power_kw = np.array([3000.0, 3500.0], dtype=np.float64)
        overnight_cost = np.array([1300.0, 1400.0], dtype=np.float64)

        lcoe = lcoe_v1_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=power_kw,
            overnight_cost_eur_per_kw=overnight_cost,
            **_lcoe_params(),
        )

        result = optimal_power_kw(lcoe=lcoe, power_kw=power_kw)

        assert "cleo:hours_per_year" in result.attrs
        assert result.attrs["cleo:hours_per_year"] == 8766.0

    def test_optimal_energy_has_hours_per_year_attr(self):
        cf, turbine_ids = _make_minimal_cf()
        power_kw = np.array([3000.0, 3500.0], dtype=np.float64)
        overnight_cost = np.array([1300.0, 1400.0], dtype=np.float64)
        hours_per_year = 8766.0

        lcoe = lcoe_v1_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=power_kw,
            overnight_cost_eur_per_kw=overnight_cost,
            **_lcoe_params(),
        )

        result = optimal_energy_gwh_a(
            lcoe=lcoe,
            cf=cf,
            power_kw=power_kw,
            hours_per_year=hours_per_year,
        )

        assert "cleo:hours_per_year" in result.attrs
        assert result.attrs["cleo:hours_per_year"] == hours_per_year


class TestTimebaseValuesPropagate:
    """Tests that verify timebase values are correctly propagated."""

    def test_lcoe_uses_provided_hours_per_year_value(self):
        cf, turbine_ids = _make_minimal_cf()
        power_kw = np.array([3000.0, 3500.0], dtype=np.float64)
        overnight_cost = np.array([1300.0, 1400.0], dtype=np.float64)
        custom_hours = 8760.0  # Non-default value

        params = _lcoe_params()
        params["hours_per_year"] = custom_hours

        result = lcoe_v1_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=power_kw,
            overnight_cost_eur_per_kw=overnight_cost,
            **params,
        )

        assert result.attrs["cleo:hours_per_year"] == custom_hours

    def test_different_timebase_produces_different_lcoe(self):
        cf, turbine_ids = _make_minimal_cf()
        power_kw = np.array([3000.0, 3500.0], dtype=np.float64)
        overnight_cost = np.array([1300.0, 1400.0], dtype=np.float64)

        params1 = _lcoe_params()
        params1["hours_per_year"] = 8766.0

        params2 = _lcoe_params()
        params2["hours_per_year"] = 8760.0

        lcoe1 = lcoe_v1_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=power_kw,
            overnight_cost_eur_per_kw=overnight_cost,
            **params1,
        )

        lcoe2 = lcoe_v1_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=power_kw,
            overnight_cost_eur_per_kw=overnight_cost,
            **params2,
        )

        # LCOE should differ with different timebase
        assert not np.allclose(lcoe1.values, lcoe2.values)
        # Attrs should reflect the different values
        assert lcoe1.attrs["cleo:hours_per_year"] == 8766.0
        assert lcoe2.attrs["cleo:hours_per_year"] == 8760.0


def test_lcoe_concat_has_no_futurewarning_for_coords_default_change():
    """lcoe_v1_from_capacity_factors should not emit xarray concat coords FutureWarning."""
    cf, turbine_ids = _make_minimal_cf()
    power_kw = np.array([3000.0, 3500.0], dtype=np.float64)
    overnight_cost = np.array([1300.0, 1400.0], dtype=np.float64)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _ = lcoe_v1_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=power_kw,
            overnight_cost_eur_per_kw=overnight_cost,
            **_lcoe_params(),
        )

    matching = [
        w
        for w in rec
        if issubclass(w.category, FutureWarning)
        and "default value for coords will change" in str(w.message)
    ]
    assert not matching


def test_lcoe_concat_handles_aux_turbine_id_coord() -> None:
    """LCOE concat tolerates auxiliary turbine coords carried on CF input."""
    cf, turbine_ids = _make_minimal_cf()
    cf = cf.assign_coords(turbine_id=("turbine", np.array([0, 1], dtype=np.int64)))
    power_kw = np.array([3000.0, 3500.0], dtype=np.float64)
    overnight_cost = np.array([1300.0, 1400.0], dtype=np.float64)

    out = lcoe_v1_from_capacity_factors(
        cf=cf,
        turbine_ids=turbine_ids,
        power_kw=power_kw,
        overnight_cost_eur_per_kw=overnight_cost,
        **_lcoe_params(),
    )

    assert out.dims == ("turbine", "y", "x")
    assert out.sizes["turbine"] == 2
