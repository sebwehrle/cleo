"""Tests for unit metadata contract compliance in turbine ingestion.

Verifies that turbine metadata variables have the canonical 'units' attr
as defined in CONTRACT_UNIFIED_ATLAS.md section B9.2.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cleo.unification.turbines import _ingest_turbines_and_costs


class MockAtlas:
    """Minimal mock Atlas for turbine ingestion tests."""

    def __init__(self, path: Path, turbines: list[str] | None = None):
        self.path = path
        self._turbines_configured = turbines
        self.vertical_policy = None

    @property
    def turbines_configured(self):
        return self._turbines_configured


@pytest.fixture
def atlas_with_turbine(tmp_path):
    """Create a mock Atlas with a single turbine YAML."""
    resources_dir = tmp_path / "resources"
    resources_dir.mkdir()

    # Create a minimal turbine YAML
    turbine_yaml = resources_dir / "Test.Turbine.1000.yml"
    turbine_data = {
        "manufacturer": "Test",
        "model": "Turbine",
        "capacity": 1000,
        "hub_height": 80.0,
        "rotor_diameter": 60.0,
        "commissioning_year": 2020,
        "V": [0, 3, 12, 25, 26],
        "cf": [0, 0, 1.0, 1.0, 0],
        "cutout_wind_speed": 26.0,
    }
    with open(turbine_yaml, "w") as f:
        import yaml
        yaml.dump(turbine_data, f)

    atlas = MockAtlas(tmp_path, turbines=["Test.Turbine.1000"])
    return atlas


class TestTurbineMetadataUnitsAttr:
    """Tests for turbine metadata units attr."""

    def test_power_curve_has_units_attr(self, atlas_with_turbine):
        """power_curve variable has 'units' attr."""
        ds, sources, variables = _ingest_turbines_and_costs(atlas_with_turbine)

        assert "power_curve" in ds.data_vars
        assert "units" in ds["power_curve"].attrs
        assert ds["power_curve"].attrs["units"] == "1"  # dimensionless

    def test_turbine_capacity_has_units_attr(self, atlas_with_turbine):
        """turbine_capacity variable has 'units' attr."""
        ds, sources, variables = _ingest_turbines_and_costs(atlas_with_turbine)

        assert "turbine_capacity" in ds.data_vars
        assert "units" in ds["turbine_capacity"].attrs
        assert ds["turbine_capacity"].attrs["units"] == "kW"

    def test_turbine_hub_height_has_units_attr(self, atlas_with_turbine):
        """turbine_hub_height variable has 'units' attr."""
        ds, sources, variables = _ingest_turbines_and_costs(atlas_with_turbine)

        assert "turbine_hub_height" in ds.data_vars
        assert "units" in ds["turbine_hub_height"].attrs
        assert ds["turbine_hub_height"].attrs["units"] == "m"

    def test_turbine_rotor_diameter_has_units_attr(self, atlas_with_turbine):
        """turbine_rotor_diameter variable has 'units' attr."""
        ds, sources, variables = _ingest_turbines_and_costs(atlas_with_turbine)

        assert "turbine_rotor_diameter" in ds.data_vars
        assert "units" in ds["turbine_rotor_diameter"].attrs
        assert ds["turbine_rotor_diameter"].attrs["units"] == "m"

    def test_turbine_commissioning_year_no_units_attr(self, atlas_with_turbine):
        """turbine_commissioning_year has no 'units' attr (year is not a unit)."""
        ds, sources, variables = _ingest_turbines_and_costs(atlas_with_turbine)

        assert "turbine_commissioning_year" in ds.data_vars
        # Year number has no unit (not a duration)
        assert ds["turbine_commissioning_year"].attrs.get("units") is None
