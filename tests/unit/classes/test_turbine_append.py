"""classes: test_turbine_append.

Regression tests for turbine append bug:
- Multiple turbines must accumulate in dataset (turbine dim grows)
- Atlas._wind_turbines must stay in sync with dataset coords
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tests.helpers.factories import wind_speed_axis

import cleo
import cleo.classes as C


def _make_turbine_yaml(
    path: Path,
    manufacturer: str,
    model: str,
    capacity: int,
    hub_height: float = 80.0,
    rotor_diameter: float = 40.0,
    commissioning_year: int = 2020,
    filename: str | None = None,
) -> Path:
    """Create a minimal turbine YAML file with all required fields."""
    yaml_content = f"""manufacturer: {manufacturer}
model: {model}
capacity: {capacity}
hub_height: {hub_height}
rotor_diameter: {rotor_diameter}
commissioning_year: {commissioning_year}
V: [0, 5, 10, 15, 20, 25]
cf: [0.0, 0.1, 0.3, 0.4, 0.3, 0.1]
"""
    if filename is None:
        filename = f"{manufacturer}.{model}.{capacity}"
    yaml_file = path / f"{filename}.yml"
    yaml_file.write_text(yaml_content)
    return yaml_file


def test_windatlas_add_turbine_data_appends_without_collapse(tmp_path: Path) -> None:
    """
    _WindAtlas.add_turbine_data must accumulate turbines in dataset.
    After adding two turbines, data.sizes["turbine"] must be 2.
    """
    # Create minimal _WindAtlas without full __init__
    wind = C._WindAtlas.__new__(C._WindAtlas)
    wind.parent = SimpleNamespace(crs="EPSG:31287")

    # Minimal dataset with required coords/vars
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

    # Create two distinct turbine YAML files
    yaml1 = _make_turbine_yaml(tmp_path, "TestMfg", "ModelA", 500)
    yaml2 = _make_turbine_yaml(tmp_path, "TestMfg", "ModelB", 1000)

    # Add first turbine
    wind.add_turbine_data(yaml1)
    assert "power_curve" in wind.data.data_vars, "power_curve should exist after first add"
    assert wind.data.sizes["turbine"] == 1, "Should have 1 turbine after first add"

    # Add second turbine
    wind.add_turbine_data(yaml2)
    assert wind.data.sizes["turbine"] == 2, f"Should have 2 turbines, got {wind.data.sizes['turbine']}"

    # Verify both turbine IDs (YAML stems) are present
    turbine_ids = set(wind.data.coords["turbine"].values.tolist())
    assert "TestMfg.ModelA.500" in turbine_ids  # YAML stem is the turbine ID
    assert "TestMfg.ModelB.1000" in turbine_ids


def test_atlas_add_turbine_keeps_list_synced_with_dataset(tmp_path: Path) -> None:
    """
    Atlas.add_turbine must keep _wind_turbines list synced with dataset turbine coord.
    After adding two turbines, both must match.
    """
    # Create minimal _WindAtlas
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

    # Create minimal Atlas without full __init__
    atlas = C.Atlas.__new__(C.Atlas)
    atlas._path = tmp_path
    atlas.country = "AUT"
    atlas._region = None
    atlas._crs = "EPSG:31287"
    atlas._wind_turbines = []
    atlas._wind = wind
    atlas._landscape = SimpleNamespace()  # dummy to satisfy _require_materialized
    atlas._materialized = True

    # Copy real turbine YAMLs from package resources
    resources_src = Path(cleo.__file__).resolve().parent / "resources"
    resources_dest = tmp_path / "resources"
    resources_dest.mkdir(parents=True, exist_ok=True)

    turbine1 = "Enercon.E40.500"
    turbine2 = "Enercon.E82.3000"

    shutil.copy(resources_src / f"{turbine1}.yml", resources_dest / f"{turbine1}.yml")
    shutil.copy(resources_src / f"{turbine2}.yml", resources_dest / f"{turbine2}.yml")

    # Add turbines via Atlas
    atlas.add_turbine(turbine1)
    atlas.add_turbine(turbine2)

    # Assertions
    ds_turbines = atlas.wind.data.coords["turbine"].values.tolist()
    assert atlas._wind_turbines == ds_turbines, (
        f"List/dataset mismatch: list={atlas._wind_turbines}, dataset={ds_turbines}"
    )
    assert atlas.wind.data.sizes["turbine"] == 2, (
        f"Should have 2 turbines in dataset, got {atlas.wind.data.sizes['turbine']}"
    )
    assert turbine1 in atlas._wind_turbines
    assert turbine2 in atlas._wind_turbines
