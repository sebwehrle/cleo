"""Tests for CF reuse in LCOE-family metrics (PR 6)."""

import json
import numpy as np
import xarray as xr

from cleo.domains import _cf_spec_matches
from cleo.wind_metrics import resolve_cf_spec


class TestCfSpecMatches:
    """Tests for _cf_spec_matches helper function."""

    def test_returns_false_when_missing_metadata(self):
        """Returns False if existing CF lacks required metadata keys."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        # No cleo:* attrs
        spec = resolve_cf_spec(None)
        turbines = ("T1",)
        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_mode_mismatch(self):
        """Returns False if cf_mode doesn't match."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "hub_height_weibull"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec({"method": "rotor_node_average"})  # Different mode
        turbines = ("T1",)

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_air_density_mismatch(self):
        """Returns False if air_density doesn't match."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "hub_height_weibull"
        cf.attrs["cleo:air_density"] = 0  # False
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec({"air_density": True})  # Different
        turbines = ("T1",)

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_rews_n_mismatch(self):
        """Returns False if rews_n doesn't match."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec({"rews_n": 24})  # Different
        turbines = ("T1",)

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_loss_factor_mismatch(self):
        """Returns False if loss_factor doesn't match."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec({"loss_factor": 0.95})  # Different
        turbines = ("T1",)

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_turbines_mismatch(self):
        """Returns False if turbines don't match."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec(None)
        turbines = ("T1", "T2")  # Different

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_false_when_turbine_order_differs(self):
        """Returns False if turbine order differs (order matters)."""
        cf = xr.DataArray(np.ones((2, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1", "T2"])

        spec = resolve_cf_spec(None)
        turbines = ("T2", "T1")  # Same turbines, different order

        assert not _cf_spec_matches(cf, spec, turbines)

    def test_returns_true_when_all_match(self):
        """Returns True when all CF parameters and turbines match exactly."""
        cf = xr.DataArray(np.ones((2, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 0
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1", "T2"])

        spec = resolve_cf_spec(
            None
        )  # Defaults: method=rotor_node_average, air_density=False, rews_n=12, loss_factor=1.0
        turbines = ("T1", "T2")

        assert _cf_spec_matches(cf, spec, turbines)

    def test_returns_true_with_custom_spec_match(self):
        """Returns True when custom spec matches existing CF exactly."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "hub_height_weibull"
        cf.attrs["cleo:air_density"] = 1  # True
        cf.attrs["cleo:rews_n"] = 24
        cf.attrs["cleo:loss_factor"] = 0.95
        cf.attrs["cleo:turbines_json"] = json.dumps(["TurbineA"])

        spec = resolve_cf_spec(
            {
                "method": "hub_height_weibull",
                "air_density": True,
                "rews_n": 24,
                "loss_factor": 0.95,
            }
        )
        turbines = ("TurbineA",)

        assert _cf_spec_matches(cf, spec, turbines)

    def test_air_density_int_bool_conversion(self):
        """Handles air_density stored as int (0/1) for netCDF compatibility."""
        cf = xr.DataArray(np.ones((1, 2, 2)), dims=("turbine", "y", "x"))
        cf.attrs["cleo:cf_method"] = "rotor_node_average"
        cf.attrs["cleo:air_density"] = 1  # Stored as int
        cf.attrs["cleo:rews_n"] = 12
        cf.attrs["cleo:loss_factor"] = 1.0
        cf.attrs["cleo:turbines_json"] = json.dumps(["T1"])

        spec = resolve_cf_spec({"air_density": True})  # Bool in spec
        turbines = ("T1",)

        assert _cf_spec_matches(cf, spec, turbines)


class TestResolveCfSpecForReuse:
    """Tests for resolve_cf_spec used in CF reuse context."""

    def test_defaults_match_expected_values(self):
        """Default CF spec values match what assess module produces."""
        spec = resolve_cf_spec(None)
        assert spec["method"] == "rotor_node_average"
        assert spec["air_density"] is False
        assert spec["rews_n"] == 12
        assert spec["loss_factor"] == 1.0

    def test_partial_override_preserves_other_defaults(self):
        """Partial spec override keeps non-specified defaults."""
        spec = resolve_cf_spec({"method": "hub_height_weibull"})
        assert spec["method"] == "hub_height_weibull"
        assert spec["air_density"] is False  # Default preserved
        assert spec["rews_n"] == 12  # Default preserved
        assert spec["loss_factor"] == 1.0  # Default preserved
