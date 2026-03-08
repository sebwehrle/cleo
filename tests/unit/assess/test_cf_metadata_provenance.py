"""Unit tests for enhanced CF metadata provenance (PR 5).

Verifies:
- CF outputs carry cleo:loss_factor
- CF outputs carry cleo:turbines_json
- LCOE outputs carry cleo:cf_spec_json
"""

import json
import numpy as np
import pytest
import xarray as xr

from cleo.assess import capacity_factors
from cleo.economics import lcoe_from_capacity_factors


class TestCFMetadataProvenance:
    """Tests for enhanced CF metadata."""

    @pytest.fixture
    def minimal_cf_inputs(self):
        """Create minimal inputs for capacity_factors."""
        y = np.array([0.0, 1.0], dtype=np.float64)
        x = np.array([0.0, 1.0], dtype=np.float64)
        # Height range must cover rotor span (hub_height +/- rotor_diameter/2)
        # Hub = 100m, rotor = 80m -> range is 60-140m, so heights [50, 150] work
        height = np.array([50.0, 100.0, 150.0], dtype=np.float64)

        A_stack = xr.DataArray(
            np.full((3, 2, 2), 8.0, dtype=np.float64),
            dims=("height", "y", "x"),
            coords={"height": height, "y": y, "x": x},
        )
        k_stack = xr.DataArray(
            np.full((3, 2, 2), 2.0, dtype=np.float64),
            dims=("height", "y", "x"),
            coords={"height": height, "y": y, "x": x},
        )
        u_grid = np.linspace(0.0, 25.0, 26, dtype=np.float64)
        turbine_ids = ("T1", "T2")
        hub_heights_m = np.array([100.0, 100.0], dtype=np.float64)
        # Smaller rotor diameters to fit within height range
        rotor_diameters_m = np.array([80.0, 80.0], dtype=np.float64)
        power_curves = np.vstack(
            [
                np.clip((u_grid - 3.0) / 10.0, 0.0, 1.0),
                np.clip((u_grid - 4.0) / 9.0, 0.0, 1.0),
            ]
        ).astype(np.float64)

        return {
            "A_stack": A_stack,
            "k_stack": k_stack,
            "u_grid": u_grid,
            "turbine_ids": turbine_ids,
            "hub_heights_m": hub_heights_m,
            "rotor_diameters_m": rotor_diameters_m,
            "power_curves": power_curves,
        }

    def test_cf_has_loss_factor_attr(self, minimal_cf_inputs):
        """CF output carries cleo:loss_factor attr."""
        result = capacity_factors(
            **minimal_cf_inputs,
            method="rotor_node_average",
            loss_factor=0.95,
        )

        assert "cleo:loss_factor" in result.attrs
        assert result.attrs["cleo:loss_factor"] == 0.95

    def test_cf_has_loss_factor_default(self, minimal_cf_inputs):
        """CF output carries cleo:loss_factor=1.0 when not specified."""
        result = capacity_factors(
            **minimal_cf_inputs,
            method="rotor_node_average",
        )

        assert "cleo:loss_factor" in result.attrs
        assert result.attrs["cleo:loss_factor"] == 1.0

    def test_cf_has_turbines_json_attr(self, minimal_cf_inputs):
        """CF output carries cleo:turbines_json attr."""
        result = capacity_factors(
            **minimal_cf_inputs,
            method="rotor_node_average",
        )

        assert "cleo:turbines_json" in result.attrs
        turbines = json.loads(result.attrs["cleo:turbines_json"])
        assert turbines == ["T1", "T2"]

    def test_cf_algo_version_is_4(self, minimal_cf_inputs):
        """CF output algo_version is 4 (v4 stores method/interpolation metadata)."""
        result = capacity_factors(
            **minimal_cf_inputs,
            method="rotor_node_average",
        )

        assert result.attrs["cleo:algo_version"] == "4"


class TestLCOECFSpecMetadata:
    """Tests for LCOE cf_spec_json metadata."""

    def test_lcoe_has_cf_spec_json(self):
        """LCOE output carries cleo:cf_spec_json attr."""
        y = np.array([0.0, 1.0], dtype=np.float64)
        x = np.array([0.0, 1.0], dtype=np.float64)
        turbine_ids = ("T1",)

        # Create minimal CF with required attrs
        cf = xr.DataArray(
            np.full((1, 2, 2), 0.3, dtype=np.float64),
            dims=("turbine", "y", "x"),
            coords={"turbine": list(turbine_ids), "y": y, "x": x},
        )
        cf.attrs["cleo:cf_method"] = "hub_height_weibull"
        cf.attrs["cleo:interpolation"] = "ak_logz"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 0.95

        result = lcoe_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=np.array([3000.0]),
            overnight_cost_eur_per_kw=np.array([1300.0]),
            bos_cost_share=0.0,
            om_fixed_eur_per_kw_a=20.0,
            om_variable_eur_per_kwh=0.008,
            discount_rate=0.05,
            lifetime_a=25,
            hours_per_year=8766.0,
        )

        assert "cleo:cf_spec_json" in result.attrs
        cf_spec = json.loads(result.attrs["cleo:cf_spec_json"])
        assert cf_spec["method"] == "hub_height_weibull"
        assert cf_spec["interpolation"] == "ak_logz"
        assert cf_spec["air_density"] is False
        assert cf_spec["rews_n"] == 12
        assert cf_spec["loss_factor"] == 0.95

    def test_lcoe_algo_version_is_3(self):
        """LCOE output algo_version is 3 (v3 added cf_spec_json)."""
        y = np.array([0.0, 1.0], dtype=np.float64)
        x = np.array([0.0, 1.0], dtype=np.float64)
        turbine_ids = ("T1",)

        cf = xr.DataArray(
            np.full((1, 2, 2), 0.3, dtype=np.float64),
            dims=("turbine", "y", "x"),
            coords={"turbine": list(turbine_ids), "y": y, "x": x},
        )
        cf.attrs["cleo:cf_method"] = "hub_height_weibull"
        cf.attrs["cleo:interpolation"] = "ak_logz"

        result = lcoe_from_capacity_factors(
            cf=cf,
            turbine_ids=turbine_ids,
            power_kw=np.array([3000.0]),
            overnight_cost_eur_per_kw=np.array([1300.0]),
            bos_cost_share=0.0,
            om_fixed_eur_per_kw_a=20.0,
            om_variable_eur_per_kwh=0.008,
            discount_rate=0.05,
            lifetime_a=25,
            hours_per_year=8766.0,
        )

        assert result.attrs["cleo:algo_version"] == "3"
