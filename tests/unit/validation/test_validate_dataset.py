"""Tests for validate_dataset function."""

import json
import numpy as np
import pytest
import xarray as xr

from cleo.validation import validate_dataset, ValidationError


class TestValidateDatasetWind:
    """Tests for wind store dataset validation."""

    def _make_valid_wind_ds(self) -> xr.Dataset:
        """Create a minimal valid wind dataset."""
        y = np.arange(3)
        x = np.arange(4)
        height = np.array([50, 100, 150])
        turbine = np.arange(2)
        wind_speed = np.linspace(0, 25, 26)

        turbines_json = json.dumps(
            [
                {"id": "T1", "capacity": 1000, "hub_height": 80, "rotor_diameter": 50},
                {"id": "T2", "capacity": 2000, "hub_height": 100, "rotor_diameter": 80},
            ]
        )

        ds = xr.Dataset(
            {
                "weibull_A": (["height", "y", "x"], np.ones((3, 3, 4))),
                "weibull_k": (["height", "y", "x"], np.ones((3, 3, 4)) * 2),
                "power_curve": (["turbine", "wind_speed"], np.ones((2, 26))),
            },
            coords={
                "y": y,
                "x": x,
                "height": height,
                "turbine": turbine,
                "wind_speed": wind_speed,
            },
            attrs={
                "store_state": "complete",
                "grid_id": "test_grid_id",
                "inputs_id": "test_inputs_id",
                "cleo_turbines_json": turbines_json,
            },
        )
        return ds

    def test_valid_wind_dataset_passes(self) -> None:
        """A properly formed wind dataset passes validation."""
        ds = self._make_valid_wind_ds()
        validate_dataset(ds, kind="wind")  # Should not raise

    def test_missing_store_state_fails(self) -> None:
        """Missing store_state attr fails validation."""
        ds = self._make_valid_wind_ds()
        del ds.attrs["store_state"]

        with pytest.raises(ValidationError, match="Missing required attrs.*store_state"):
            validate_dataset(ds, kind="wind")

    def test_incomplete_store_state_fails(self) -> None:
        """store_state != 'complete' fails validation."""
        ds = self._make_valid_wind_ds()
        ds.attrs["store_state"] = "incomplete"

        with pytest.raises(ValidationError, match="store_state is 'incomplete'"):
            validate_dataset(ds, kind="wind")

    def test_missing_required_variable_fails(self) -> None:
        """Missing required variable fails validation."""
        ds = self._make_valid_wind_ds()
        ds = ds.drop_vars("weibull_A")

        with pytest.raises(ValidationError, match="Missing required variables.*weibull_A"):
            validate_dataset(ds, kind="wind")

    def test_missing_required_dimension_fails(self) -> None:
        """Missing required dimension fails validation."""
        ds = self._make_valid_wind_ds()
        # Remove turbine dimension by dropping power_curve and turbine coord
        ds = ds.drop_vars("power_curve")
        ds = ds.drop_vars("turbine")

        with pytest.raises(ValidationError, match="Missing required dimensions.*turbine"):
            validate_dataset(ds, kind="wind")

    def test_invalid_turbines_json_fails(self) -> None:
        """Invalid cleo_turbines_json fails validation."""
        ds = self._make_valid_wind_ds()
        ds.attrs["cleo_turbines_json"] = "not valid json {"

        with pytest.raises(ValidationError, match="cleo_turbines_json is invalid JSON"):
            validate_dataset(ds, kind="wind")

    def test_turbines_json_not_array_fails(self) -> None:
        """cleo_turbines_json that is not an array fails validation."""
        ds = self._make_valid_wind_ds()
        ds.attrs["cleo_turbines_json"] = json.dumps({"turbine": "T1"})  # dict, not list

        with pytest.raises(ValidationError, match="cleo_turbines_json is not a JSON array"):
            validate_dataset(ds, kind="wind")

    def test_empty_coordinate_fails(self) -> None:
        """Empty coordinate fails validation."""
        ds = self._make_valid_wind_ds()
        # Create dataset with empty y coordinate
        ds = ds.isel(y=slice(0, 0))

        with pytest.raises(ValidationError, match="Coordinate 'y' is empty"):
            validate_dataset(ds, kind="wind")


class TestValidateDatasetLandscape:
    """Tests for landscape store dataset validation."""

    def _make_valid_landscape_ds(self) -> xr.Dataset:
        """Create a minimal valid landscape dataset."""
        y = np.arange(3)
        x = np.arange(4)

        ds = xr.Dataset(
            {
                "valid_mask": (["y", "x"], np.ones((3, 4), dtype=bool)),
            },
            coords={"y": y, "x": x},
            attrs={
                "store_state": "complete",
                "grid_id": "test_grid_id",
                "inputs_id": "test_inputs_id",
            },
        )
        return ds

    def test_valid_landscape_dataset_passes(self) -> None:
        """A properly formed landscape dataset passes validation."""
        ds = self._make_valid_landscape_ds()
        validate_dataset(ds, kind="landscape")  # Should not raise

    def test_missing_valid_mask_fails(self) -> None:
        """Missing valid_mask variable fails validation."""
        ds = self._make_valid_landscape_ds()
        ds = ds.drop_vars("valid_mask")

        with pytest.raises(ValidationError, match="Missing required variables.*valid_mask"):
            validate_dataset(ds, kind="landscape")

    def test_incomplete_store_state_fails(self) -> None:
        """store_state != 'complete' fails validation."""
        ds = self._make_valid_landscape_ds()
        ds.attrs["store_state"] = "building"

        with pytest.raises(ValidationError, match="store_state is 'building'"):
            validate_dataset(ds, kind="landscape")


class TestValidateDatasetDeep:
    """Tests for deep validation mode."""

    def test_deep_checks_coordinate_monotonicity(self) -> None:
        """Deep validation checks that coordinates are monotonic."""
        y = np.array([0, 2, 1])  # Non-monotonic
        x = np.arange(4)

        ds = xr.Dataset(
            {"valid_mask": (["y", "x"], np.ones((3, 4), dtype=bool))},
            coords={"y": y, "x": x},
            attrs={
                "store_state": "complete",
                "grid_id": "test_grid_id",
                "inputs_id": "test_inputs_id",
            },
        )

        # Shallow validation passes
        validate_dataset(ds, kind="landscape", deep=False)

        # Deep validation catches non-monotonic coordinate
        with pytest.raises(ValidationError, match="Coordinate 'y' is not monotonic"):
            validate_dataset(ds, kind="landscape", deep=True)

    def test_deep_checks_height_positive(self) -> None:
        """Deep validation checks height coordinate is positive."""
        y = np.arange(3)
        x = np.arange(4)
        height = np.array([-10, 50, 100])  # Contains negative
        turbine = np.arange(2)
        wind_speed = np.linspace(0, 25, 26)

        turbines_json = json.dumps(
            [
                {"id": "T1", "capacity": 1000, "hub_height": 80, "rotor_diameter": 50},
                {"id": "T2", "capacity": 2000, "hub_height": 100, "rotor_diameter": 80},
            ]
        )

        ds = xr.Dataset(
            {
                "weibull_A": (["height", "y", "x"], np.ones((3, 3, 4))),
                "weibull_k": (["height", "y", "x"], np.ones((3, 3, 4)) * 2),
                "power_curve": (["turbine", "wind_speed"], np.ones((2, 26))),
            },
            coords={
                "y": y,
                "x": x,
                "height": height,
                "turbine": turbine,
                "wind_speed": wind_speed,
            },
            attrs={
                "store_state": "complete",
                "grid_id": "test_grid_id",
                "inputs_id": "test_inputs_id",
                "cleo_turbines_json": turbines_json,
            },
        )

        # Deep validation catches negative height
        with pytest.raises(ValidationError, match="Height coordinate contains non-positive"):
            validate_dataset(ds, kind="wind", deep=True)

    def test_deep_checks_valid_mask_dtype(self) -> None:
        """Deep validation checks valid_mask is boolean."""
        y = np.arange(3)
        x = np.arange(4)

        ds = xr.Dataset(
            {
                "valid_mask": (["y", "x"], np.ones((3, 4), dtype=np.float64)),  # Wrong dtype
            },
            coords={"y": y, "x": x},
            attrs={
                "store_state": "complete",
                "grid_id": "test_grid_id",
                "inputs_id": "test_inputs_id",
            },
        )

        # Deep validation catches wrong dtype
        with pytest.raises(ValidationError, match="valid_mask dtype is float64"):
            validate_dataset(ds, kind="landscape", deep=True)


class TestValidateDatasetGeneric:
    """Tests for generic kind validation."""

    def test_generic_requires_store_state(self) -> None:
        """Generic validation requires only store_state."""
        ds = xr.Dataset(
            {"data": (["x"], np.arange(10))},
            attrs={"store_state": "complete"},
        )
        validate_dataset(ds, kind="generic")  # Should pass

    def test_generic_missing_store_state_fails(self) -> None:
        """Generic validation fails without store_state."""
        ds = xr.Dataset({"data": (["x"], np.arange(10))})

        with pytest.raises(ValidationError, match="Missing required attrs.*store_state"):
            validate_dataset(ds, kind="generic")


class TestValidateDatasetNoCompute:
    """Regression tests ensuring validate_dataset does not trigger compute."""

    def test_dask_array_not_computed_shallow(self) -> None:
        """Shallow validation does not compute dask arrays."""
        import dask.array as da

        y = np.arange(100)
        x = np.arange(100)

        # Create dask-backed arrays
        weibull_A_data = da.ones((3, 100, 100), chunks=(1, 50, 50))
        weibull_k_data = da.ones((3, 100, 100), chunks=(1, 50, 50)) * 2
        power_curve_data = da.ones((2, 26), chunks=(1, 26))

        turbines_json = json.dumps(
            [
                {"id": "T1", "capacity": 1000, "hub_height": 80, "rotor_diameter": 50},
                {"id": "T2", "capacity": 2000, "hub_height": 100, "rotor_diameter": 80},
            ]
        )

        ds = xr.Dataset(
            {
                "weibull_A": (["height", "y", "x"], weibull_A_data),
                "weibull_k": (["height", "y", "x"], weibull_k_data),
                "power_curve": (["turbine", "wind_speed"], power_curve_data),
            },
            coords={
                "y": y,
                "x": x,
                "height": [50, 100, 150],
                "turbine": [0, 1],
                "wind_speed": np.linspace(0, 25, 26),
            },
            attrs={
                "store_state": "complete",
                "grid_id": "test_grid_id",
                "inputs_id": "test_inputs_id",
                "cleo_turbines_json": turbines_json,
            },
        )

        # Verify arrays are dask-backed before validation
        assert hasattr(ds["weibull_A"].data, "dask")
        assert hasattr(ds["weibull_k"].data, "dask")
        assert hasattr(ds["power_curve"].data, "dask")

        # Run validation
        validate_dataset(ds, kind="wind", deep=False)

        # Verify arrays are still dask-backed (not computed)
        assert hasattr(ds["weibull_A"].data, "dask"), "weibull_A was computed"
        assert hasattr(ds["weibull_k"].data, "dask"), "weibull_k was computed"
        assert hasattr(ds["power_curve"].data, "dask"), "power_curve was computed"

    def test_dask_array_not_computed_deep(self) -> None:
        """Deep validation does not compute main data arrays (only coords)."""
        import dask.array as da

        y = np.arange(100)
        x = np.arange(100)

        # Create dask-backed arrays
        valid_mask_data = da.ones((100, 100), dtype=bool, chunks=(50, 50))

        ds = xr.Dataset(
            {"valid_mask": (["y", "x"], valid_mask_data)},
            coords={"y": y, "x": x},
            attrs={
                "store_state": "complete",
                "grid_id": "test_grid_id",
                "inputs_id": "test_inputs_id",
            },
        )

        # Verify array is dask-backed before validation
        assert hasattr(ds["valid_mask"].data, "dask")

        # Run deep validation
        validate_dataset(ds, kind="landscape", deep=True)

        # Verify array is still dask-backed (not computed)
        # Note: deep validation checks dtype which accesses .dtype but not .values
        assert hasattr(ds["valid_mask"].data, "dask"), "valid_mask was computed"
