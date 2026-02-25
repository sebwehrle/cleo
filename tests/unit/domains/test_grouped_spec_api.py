"""Unit tests for grouped spec API for composed metrics (LCOE-family).

Tests:
- Grouped cf={} and economics={} spec parsing
- Flat kwargs rejection for composed metrics
- Missing required economics fields error
- Override merge behavior with atlas baseline
- CF spec defaults
"""

import json
import pytest
import numpy as np
import xarray as xr

from cleo.wind_metrics import (
    _wind_metric_lcoe,
    _CF_SPEC_DEFAULTS,
    _REQUIRED_ECONOMICS_FIELDS,
    _FLAT_CF_KWARGS,
    _FLAT_ECONOMICS_KWARGS,
)
from cleo.domains import _resolve_cf_spec


class TestResolveCfSpec:
    """Tests for _resolve_cf_spec helper function."""

    def test_resolve_cf_spec_returns_defaults_when_none(self):
        """When cf is None, returns all defaults."""
        result = _resolve_cf_spec(None)

        assert result == _CF_SPEC_DEFAULTS
        assert result["mode"] == "direct_cf_quadrature"
        assert result["air_density"] is False
        assert result["rews_n"] == 12
        assert result["loss_factor"] == 1.0

    def test_resolve_cf_spec_merges_partial_override(self):
        """Partial cf dict merges with defaults."""
        cf = {"mode": "hub"}
        result = _resolve_cf_spec(cf)

        assert result["mode"] == "hub"
        assert result["air_density"] is False  # default
        assert result["rews_n"] == 12  # default
        assert result["loss_factor"] == 1.0  # default

    def test_resolve_cf_spec_full_override(self):
        """Full cf dict overrides all defaults."""
        cf = {
            "mode": "rews",
            "air_density": True,
            "rews_n": 7,
            "loss_factor": 0.9,
        }
        result = _resolve_cf_spec(cf)

        assert result == cf

    def test_resolve_cf_spec_returns_copy(self):
        """Returned dict is independent of defaults."""
        result = _resolve_cf_spec(None)
        result["mode"] = "hub"

        # Defaults should be unchanged
        assert _CF_SPEC_DEFAULTS["mode"] == "direct_cf_quadrature"


class TestGroupedSpecConstants:
    """Tests for grouped spec module-level constants."""

    def test_required_economics_fields_complete(self):
        """All required economics fields are defined."""
        assert "discount_rate" in _REQUIRED_ECONOMICS_FIELDS
        assert "lifetime_a" in _REQUIRED_ECONOMICS_FIELDS
        assert "om_fixed_eur_per_kw_a" in _REQUIRED_ECONOMICS_FIELDS
        assert "om_variable_eur_per_kwh" in _REQUIRED_ECONOMICS_FIELDS

    def test_flat_cf_kwargs_complete(self):
        """All flat CF kwargs that should be rejected are defined."""
        assert "mode" in _FLAT_CF_KWARGS
        assert "air_density" in _FLAT_CF_KWARGS
        assert "loss_factor" in _FLAT_CF_KWARGS
        assert "rews_n" in _FLAT_CF_KWARGS

    def test_flat_economics_kwargs_complete(self):
        """All flat economics kwargs that should be rejected are defined."""
        assert "discount_rate" in _FLAT_ECONOMICS_KWARGS
        assert "lifetime_a" in _FLAT_ECONOMICS_KWARGS
        assert "om_fixed_eur_per_kw_a" in _FLAT_ECONOMICS_KWARGS
        assert "om_variable_eur_per_kwh" in _FLAT_ECONOMICS_KWARGS
        assert "bos_cost_share" in _FLAT_ECONOMICS_KWARGS


class TestLcoeDirectCallWithGroupedSpec:
    """Tests for _wind_metric_lcoe with grouped spec unpacking."""

    @pytest.fixture
    def minimal_wind_land(self):
        """Create minimal wind/land datasets for testing."""
        y = np.array([0.0, 1.0], dtype=np.float64)
        x = np.array([0.0, 1.0], dtype=np.float64)
        height = np.array([50.0, 100.0], dtype=np.float64)
        wind_speed = np.linspace(0.0, 25.0, 26, dtype=np.float64)
        turbines_meta = [
            {"id": "T1", "capacity": 3000.0, "overnight_cost_eur_per_kw": 1300.0, "rotor_diameter": 120.0},
        ]

        wind = xr.Dataset(
            coords={
                "y": y,
                "x": x,
                "height": height,
                "wind_speed": wind_speed,
                "turbine": np.array([0], dtype=np.int64),
            },
            data_vars={
                "weibull_A": (("height", "y", "x"), np.full((2, 2, 2), 8.0, dtype=np.float64)),
                "weibull_k": (("height", "y", "x"), np.full((2, 2, 2), 2.0, dtype=np.float64)),
                "power_curve": (
                    ("turbine", "wind_speed"),
                    np.clip((wind_speed - 3.0) / 10.0, 0.0, 1.0).reshape(1, -1).astype(np.float64),
                ),
                "turbine_hub_height": (("turbine",), np.array([100.0], dtype=np.float64)),
                "turbine_rotor_diameter": (("turbine",), np.array([120.0], dtype=np.float64)),
            },
        )
        wind.attrs["cleo_turbines_json"] = json.dumps(turbines_meta)
        land = xr.Dataset(
            {"valid_mask": (("y", "x"), np.array([[True, True], [True, True]]))},
            coords={"y": y, "x": x},
        )
        return wind, land

    def test_lcoe_with_full_economics_spec(self, minimal_wind_land):
        """_wind_metric_lcoe works with all economics parameters."""
        wind, land = minimal_wind_land
        result = _wind_metric_lcoe(
            wind, land,
            turbines=("T1",),
            mode="hub",
            air_density=False,
            loss_factor=1.0,
            rews_n=12,
            bos_cost_share=0.0,
            om_fixed_eur_per_kw_a=20.0,
            om_variable_eur_per_kwh=0.008,
            discount_rate=0.05,
            lifetime_a=25,
            hours_per_year=8766.0,
        )

        assert result.name == "lcoe"
        assert "cleo:economics_json" in result.attrs

    def test_lcoe_economics_json_has_bos_cost_share(self, minimal_wind_land):
        """LCOE economics_json contains bos_cost_share."""
        wind, land = minimal_wind_land
        result = _wind_metric_lcoe(
            wind, land,
            turbines=("T1",),
            mode="hub",
            bos_cost_share=0.3,
            om_fixed_eur_per_kw_a=20.0,
            om_variable_eur_per_kwh=0.008,
            discount_rate=0.05,
            lifetime_a=25,
            hours_per_year=8766.0,
        )

        economics = json.loads(result.attrs["cleo:economics_json"])
        assert economics["bos_cost_share"] == 0.3
