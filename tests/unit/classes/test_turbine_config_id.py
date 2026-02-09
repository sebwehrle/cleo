"""test_turbine_config_id.py

Tests for turbine config ID (YAML stem) as turbine coordinate,
and turbine metadata storage in dataset (no runtime YAML reads).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tests.helpers.factories import wind_speed_axis

import cleo.classes as C


def _make_turbine_yaml(
    path: Path,
    filename: str,
    manufacturer: str,
    model: str,
    capacity: float,
    hub_height: float,
    rotor_diameter: float = 82.0,
    commissioning_year: int = 2020,
) -> Path:
    """Create a turbine YAML file with all required fields."""
    yaml_content = f"""manufacturer: {manufacturer}
model: {model}
capacity: {capacity}
hub_height: {hub_height}
rotor_diameter: {rotor_diameter}
commissioning_year: {commissioning_year}
V: [0, 3, 5, 8, 10, 12, 15, 20, 25, 30]
cf: [0.0, 0.0, 0.05, 0.25, 0.45, 0.65, 0.85, 0.70, 0.40, 0.0]
"""
    yaml_file = path / f"{filename}.yml"
    yaml_file.write_text(yaml_content)
    return yaml_file


def _create_minimal_windatlas() -> C._WindAtlas:
    """Create a minimal _WindAtlas instance for testing."""
    wind = C._WindAtlas.__new__(C._WindAtlas)
    wind.parent = SimpleNamespace(crs="EPSG:31287")

    wind_speed = wind_speed_axis()
    x_coords = [0.0, 1.0, 2.0]
    y_coords = [0.0, 1.0, 2.0]
    template = xr.DataArray(
        np.zeros((3, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={"x": x_coords, "y": y_coords},
    )
    wind.data = xr.Dataset(
        {"template": template},
        coords={"wind_speed": wind_speed, "x": x_coords, "y": y_coords},
    )
    return wind


class TestTurbineConfigId:
    """
    Tests for turbine config ID semantics:
    - Same turbine model (manufacturer/model/capacity) with different hub heights
    - Turbine coordinate is YAML file stem (config id)
    - All metadata stored in dataset
    - No runtime YAML reads after ingest
    """

    def test_same_model_different_hub_heights_coexist(self, tmp_path: Path) -> None:
        """
        Two turbines with SAME manufacturer/model/capacity but DIFFERENT hub_height
        can coexist in the dataset with DIFFERENT YAML stems (config IDs).
        """
        resources = tmp_path / "resources"
        resources.mkdir()

        # Create two YAML files: same model, different hub heights, different stems
        yaml1 = _make_turbine_yaml(
            resources,
            filename="e101_h99",
            manufacturer="Enercon",
            model="E101",
            capacity=3050.0,
            hub_height=99.0,
            rotor_diameter=101.0,
            commissioning_year=2015,
        )
        yaml2 = _make_turbine_yaml(
            resources,
            filename="e101_h135",
            manufacturer="Enercon",
            model="E101",
            capacity=3050.0,
            hub_height=135.0,
            rotor_diameter=101.0,
            commissioning_year=2015,
        )

        # Create WindAtlas and add both turbines
        wind = _create_minimal_windatlas()
        wind.add_turbine_data(yaml1)
        wind.add_turbine_data(yaml2)

        # Assertion 1: turbine coord contains both stems
        turbine_ids = list(wind.data.coords["turbine"].values)
        assert "e101_h99" in turbine_ids, f"Expected 'e101_h99' in {turbine_ids}"
        assert "e101_h135" in turbine_ids, f"Expected 'e101_h135' in {turbine_ids}"

        # Assertion 2: dataset size["turbine"] == 2
        assert wind.data.sizes["turbine"] == 2, (
            f"Expected 2 turbines, got {wind.data.sizes['turbine']}"
        )

        # Assertion 3: hub_height differs between the two turbine entries
        h1 = float(wind.data["turbine_hub_height"].sel(turbine="e101_h99").values)
        h2 = float(wind.data["turbine_hub_height"].sel(turbine="e101_h135").values)
        assert h1 == 99.0, f"Expected hub_height=99.0, got {h1}"
        assert h2 == 135.0, f"Expected hub_height=135.0, got {h2}"
        assert h1 != h2, "Hub heights should differ"

        # Assertion 4: turbine_model_key is equal between them
        key1 = str(wind.data["turbine_model_key"].sel(turbine="e101_h99").values)
        key2 = str(wind.data["turbine_model_key"].sel(turbine="e101_h135").values)
        assert key1 == "Enercon.E101.3050.0", f"Unexpected model key: {key1}"
        assert key1 == key2, f"Model keys should match: {key1} vs {key2}"

    def test_no_runtime_yaml_reads_after_ingest(self, tmp_path: Path) -> None:
        """
        After turbine ingest, get_turbine_attribute must work without YAML files.
        This proves metadata is stored in the dataset.
        """
        resources = tmp_path / "resources"
        resources.mkdir()

        # Create two YAML files
        yaml1 = _make_turbine_yaml(
            resources,
            filename="e101_h99",
            manufacturer="Enercon",
            model="E101",
            capacity=3050.0,
            hub_height=99.0,
        )
        yaml2 = _make_turbine_yaml(
            resources,
            filename="e101_h135",
            manufacturer="Enercon",
            model="E101",
            capacity=3050.0,
            hub_height=135.0,
        )

        # Create WindAtlas and add both turbines
        wind = _create_minimal_windatlas()
        wind.add_turbine_data(yaml1)
        wind.add_turbine_data(yaml2)

        # Purity check: delete the resources directory AFTER ingest
        shutil.rmtree(resources)
        assert not resources.exists(), "Resources dir should be deleted"

        # Call get_turbine_attribute for both turbine_ids - should still work
        # (proves no YAML IO at runtime)
        from cleo.loaders import get_turbine_attribute

        h1 = get_turbine_attribute(wind, "e101_h99", "hub_height")
        h2 = get_turbine_attribute(wind, "e101_h135", "hub_height")

        assert h1 == 99.0, f"Expected hub_height=99.0, got {h1}"
        assert h2 == 135.0, f"Expected hub_height=135.0, got {h2}"

        # Also verify other attributes work
        cap1 = get_turbine_attribute(wind, "e101_h99", "capacity")
        assert cap1 == 3050.0, f"Expected capacity=3050.0, got {cap1}"

        mfg1 = get_turbine_attribute(wind, "e101_h99", "manufacturer")
        assert mfg1 == "Enercon", f"Expected manufacturer='Enercon', got {mfg1}"

        rd1 = get_turbine_attribute(wind, "e101_h99", "rotor_diameter")
        assert rd1 == 82.0, f"Expected rotor_diameter=82.0, got {rd1}"

        year1 = get_turbine_attribute(wind, "e101_h99", "commissioning_year")
        assert year1 == 2020, f"Expected commissioning_year=2020, got {year1}"

    def test_duplicate_turbine_id_raises(self, tmp_path: Path) -> None:
        """Adding the same turbine_id twice should raise ValueError."""
        resources = tmp_path / "resources"
        resources.mkdir()

        yaml1 = _make_turbine_yaml(
            resources,
            filename="same_id",
            manufacturer="Test",
            model="T1",
            capacity=1000.0,
            hub_height=80.0,
        )

        wind = _create_minimal_windatlas()
        wind.add_turbine_data(yaml1)

        # Create another YAML with the same stem
        yaml2 = _make_turbine_yaml(
            resources,
            filename="same_id",  # Same stem = duplicate
            manufacturer="Test",
            model="T2",  # Different model, but same stem
            capacity=2000.0,
            hub_height=100.0,
        )

        with pytest.raises(ValueError, match="Duplicate turbine_id"):
            wind.add_turbine_data(yaml2)

    def test_missing_attribute_raises_helpful_error(self, tmp_path: Path) -> None:
        """
        Requesting a non-existent attribute should raise ValueError
        with helpful message including turbine_id and attribute name.
        """
        resources = tmp_path / "resources"
        resources.mkdir()

        yaml1 = _make_turbine_yaml(
            resources,
            filename="test_turbine",
            manufacturer="Test",
            model="T1",
            capacity=1000.0,
            hub_height=80.0,
        )

        wind = _create_minimal_windatlas()
        wind.add_turbine_data(yaml1)

        from cleo.loaders import get_turbine_attribute

        with pytest.raises(ValueError, match="not found in dataset"):
            get_turbine_attribute(wind, "test_turbine", "nonexistent_attr")

    def test_all_required_metadata_stored(self, tmp_path: Path) -> None:
        """Verify all required metadata fields are stored as data_vars."""
        resources = tmp_path / "resources"
        resources.mkdir()

        yaml1 = _make_turbine_yaml(
            resources,
            filename="test_turbine",
            manufacturer="TestMfg",
            model="TestModel",
            capacity=2500.0,
            hub_height=120.0,
            rotor_diameter=90.0,
            commissioning_year=2018,
        )

        wind = _create_minimal_windatlas()
        wind.add_turbine_data(yaml1)

        # All required metadata should be in data_vars
        required_vars = [
            "turbine_manufacturer",
            "turbine_model",
            "turbine_capacity",
            "turbine_hub_height",
            "turbine_rotor_diameter",
            "turbine_commissioning_year",
            "turbine_model_key",
        ]
        for var in required_vars:
            assert var in wind.data.data_vars, f"Missing metadata var: {var}"

        # Verify values
        t = "test_turbine"
        assert str(wind.data["turbine_manufacturer"].sel(turbine=t).values) == "TestMfg"
        assert str(wind.data["turbine_model"].sel(turbine=t).values) == "TestModel"
        assert float(wind.data["turbine_capacity"].sel(turbine=t).values) == 2500.0
        assert float(wind.data["turbine_hub_height"].sel(turbine=t).values) == 120.0
        assert float(wind.data["turbine_rotor_diameter"].sel(turbine=t).values) == 90.0
        assert int(wind.data["turbine_commissioning_year"].sel(turbine=t).values) == 2018
        assert str(wind.data["turbine_model_key"].sel(turbine=t).values) == "TestMfg.TestModel.2500.0"
