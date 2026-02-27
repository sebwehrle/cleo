"""Tests for grid_connect_cost_eur_per_kw parameter in LCOE computations.

PR8a: Paper reproducibility gap - make grid_connect_cost configurable (allow 0).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from cleo.economics import lcoe_v1_from_capacity_factors


def _make_cf(turbine_ids: tuple[str, ...]) -> xr.DataArray:
    """Create a minimal CF DataArray for testing."""
    cf = xr.DataArray(
        np.full((2, 2, len(turbine_ids)), 0.3, dtype=np.float64),
        dims=("y", "x", "turbine"),
        coords={
            "y": np.arange(2),
            "x": np.arange(2),
            "turbine": list(turbine_ids),
        },
    )
    cf.attrs["cleo:cf_mode"] = "direct_cf_quadrature"
    cf.attrs["cleo:air_density"] = 0
    cf.attrs["cleo:rews_n"] = 12
    cf.attrs["cleo:loss_factor"] = 1.0
    return cf


class TestLcoeGridConnectCostParameter:
    """Tests for grid_connect_cost_eur_per_kw parameter in LCOE."""

    def test_default_grid_connect_cost_is_50(self) -> None:
        """Default grid_connect_cost_eur_per_kw is 50.0 EUR/kW."""
        turbine_ids = ("T1",)
        cf = _make_cf(turbine_ids)
        power_kw = np.array([1000.0])
        overnight_cost = np.array([1000.0])

        lcoe = lcoe_v1_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=power_kw,
            overnight_cost_eur_per_kw=overnight_cost,
            bos_cost_share=0.0,
            om_fixed_eur_per_kw_a=20.0,
            om_variable_eur_per_kwh=0.008,
            discount_rate=0.05,
            lifetime_a=25,
            hours_per_year=8766.0,
            # Not passing grid_connect_cost_eur_per_kw - should default to 50.0
        )

        # Check that economics_json includes the default grid_connect_cost
        economics = json.loads(lcoe.attrs["cleo:economics_json"])
        assert economics["grid_connect_cost_eur_per_kw"] == 50.0

    def test_grid_connect_cost_zero_reduces_lcoe(self) -> None:
        """Setting grid_connect_cost_eur_per_kw=0 reduces LCOE."""
        turbine_ids = ("T1",)
        cf = _make_cf(turbine_ids)
        power_kw = np.array([1000.0])
        overnight_cost = np.array([1000.0])

        common_kwargs = {
            "cf": cf,
            "turbine_ids": turbine_ids,
            "power_kw": power_kw,
            "overnight_cost_eur_per_kw": overnight_cost,
            "bos_cost_share": 0.0,
            "om_fixed_eur_per_kw_a": 20.0,
            "om_variable_eur_per_kwh": 0.008,
            "discount_rate": 0.05,
            "lifetime_a": 25,
            "hours_per_year": 8766.0,
        }

        lcoe_with_grid = lcoe_v1_from_capacity_factors(
            **common_kwargs,
            grid_connect_cost_eur_per_kw=50.0,
        )

        lcoe_without_grid = lcoe_v1_from_capacity_factors(
            **common_kwargs,
            grid_connect_cost_eur_per_kw=0.0,
        )

        # LCOE without grid cost should be lower
        assert (lcoe_without_grid.values < lcoe_with_grid.values).all()

        # Check provenance
        economics_with = json.loads(lcoe_with_grid.attrs["cleo:economics_json"])
        economics_without = json.loads(lcoe_without_grid.attrs["cleo:economics_json"])
        assert economics_with["grid_connect_cost_eur_per_kw"] == 50.0
        assert economics_without["grid_connect_cost_eur_per_kw"] == 0.0

    def test_grid_connect_cost_delta_is_exact(self) -> None:
        """The LCOE delta from grid cost equals expected present-value contribution."""
        turbine_ids = ("T1",)
        cf = _make_cf(turbine_ids)
        power_kw = np.array([1000.0])  # 1000 kW turbine
        overnight_cost = np.array([1000.0])
        grid_rate = 50.0  # EUR/kW

        common_kwargs = {
            "cf": cf,
            "turbine_ids": turbine_ids,
            "power_kw": power_kw,
            "overnight_cost_eur_per_kw": overnight_cost,
            "bos_cost_share": 0.0,
            "om_fixed_eur_per_kw_a": 20.0,
            "om_variable_eur_per_kwh": 0.008,
            "discount_rate": 0.05,
            "lifetime_a": 25,
            "hours_per_year": 8766.0,
        }

        lcoe_with_grid = lcoe_v1_from_capacity_factors(
            **common_kwargs,
            grid_connect_cost_eur_per_kw=grid_rate,
        )

        lcoe_without_grid = lcoe_v1_from_capacity_factors(
            **common_kwargs,
            grid_connect_cost_eur_per_kw=0.0,
        )

        # Calculate expected delta
        # Grid cost contribution = (grid_rate * power_kw) / npv_electricity * 1000 (for EUR/MWh)
        # npv_electricity = CF * hours_per_year * power_kw * npv_factor
        # npv_factor = (1 - (1 + r)^-n) / r
        r = 0.05
        n = 25
        npv_factor = (1 - (1 + r) ** (-n)) / r
        cf_val = 0.3
        hours = 8766.0
        p_kw = 1000.0

        npv_electricity = cf_val * hours * p_kw * npv_factor
        grid_cost_abs = grid_rate * p_kw
        expected_delta = (grid_cost_abs / npv_electricity) * 1000  # EUR/MWh

        actual_delta = lcoe_with_grid.values - lcoe_without_grid.values

        np.testing.assert_allclose(
            actual_delta,
            expected_delta,
            rtol=1e-10,
            err_msg="LCOE delta from grid cost doesn't match expected value",
        )

    def test_grid_connect_cost_custom_rate(self) -> None:
        """Custom grid_connect_cost_eur_per_kw value is respected."""
        turbine_ids = ("T1",)
        cf = _make_cf(turbine_ids)
        power_kw = np.array([1000.0])
        overnight_cost = np.array([1000.0])

        lcoe = lcoe_v1_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=power_kw,
            overnight_cost_eur_per_kw=overnight_cost,
            bos_cost_share=0.0,
            om_fixed_eur_per_kw_a=20.0,
            om_variable_eur_per_kwh=0.008,
            discount_rate=0.05,
            lifetime_a=25,
            hours_per_year=8766.0,
            grid_connect_cost_eur_per_kw=75.0,  # Custom rate
        )

        economics = json.loads(lcoe.attrs["cleo:economics_json"])
        assert economics["grid_connect_cost_eur_per_kw"] == 75.0


class TestAtlasConfigureEconomicsGridCost:
    """Tests for grid_connect_cost_eur_per_kw in Atlas.configure_economics."""

    def test_configure_economics_accepts_grid_cost(self, tmp_path) -> None:
        """Atlas.configure_economics accepts grid_connect_cost_eur_per_kw."""
        from cleo import Atlas

        atlas_path = tmp_path / "atlas"
        atlas_path.mkdir()
        atlas = Atlas(atlas_path, country="AUT", crs="epsg:3035")

        # Should not raise
        atlas.configure_economics(
            discount_rate=0.05,
            lifetime_a=25,
            om_fixed_eur_per_kw_a=20.0,
            om_variable_eur_per_kwh=0.008,
            grid_connect_cost_eur_per_kw=0.0,  # Paper qLCOE mode
        )

        assert atlas.economics_configured is not None
        assert atlas.economics_configured["grid_connect_cost_eur_per_kw"] == 0.0

    def test_configure_economics_default_grid_cost(self, tmp_path) -> None:
        """Default grid_connect_cost_eur_per_kw is 50.0 when not configured."""
        from cleo import Atlas

        atlas_path = tmp_path / "atlas"
        atlas_path.mkdir()
        atlas = Atlas(atlas_path, country="AUT", crs="epsg:3035")

        atlas.configure_economics(
            discount_rate=0.05,
            lifetime_a=25,
            om_fixed_eur_per_kw_a=20.0,
            om_variable_eur_per_kwh=0.008,
            # Not configuring grid_connect_cost_eur_per_kw
        )

        # _effective_economics should provide the default
        effective = atlas._effective_economics()
        assert effective["grid_connect_cost_eur_per_kw"] == 50.0

    def test_configure_economics_rejects_negative_grid_cost(self, tmp_path) -> None:
        """Negative grid_connect_cost_eur_per_kw raises ValueError."""
        from cleo import Atlas

        atlas_path = tmp_path / "atlas"
        atlas_path.mkdir()
        atlas = Atlas(atlas_path, country="AUT", crs="epsg:3035")

        with pytest.raises(ValueError, match="grid_connect_cost_eur_per_kw must be finite and >= 0"):
            atlas.configure_economics(
                grid_connect_cost_eur_per_kw=-10.0,
            )

    def test_configure_economics_rejects_non_numeric_grid_cost(self, tmp_path) -> None:
        """Non-numeric grid_connect_cost_eur_per_kw raises TypeError."""
        from cleo import Atlas

        atlas_path = tmp_path / "atlas"
        atlas_path.mkdir()
        atlas = Atlas(atlas_path, country="AUT", crs="epsg:3035")

        with pytest.raises(TypeError, match="grid_connect_cost_eur_per_kw must be numeric"):
            atlas.configure_economics(
                grid_connect_cost_eur_per_kw="fifty",
            )
