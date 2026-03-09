"""Tests for validate_store function."""

import json
import numpy as np
import pytest
import xarray as xr
import zarr
from pathlib import Path

from cleo.validation import validate_store, ValidationError


class TestValidateStoreWind:
    """Tests for wind store validation."""

    def _create_valid_wind_store(self, store_path: Path) -> None:
        """Create a minimal valid wind zarr store."""
        store_path.mkdir(parents=True, exist_ok=True)

        turbines_json = json.dumps(
            [
                {"id": "T1", "capacity": 1000, "hub_height": 80, "rotor_diameter": 50},
                {"id": "T2", "capacity": 2000, "hub_height": 100, "rotor_diameter": 80},
            ]
        )

        root = zarr.open_group(store_path, mode="w")
        root.attrs["store_state"] = "complete"
        root.attrs["grid_id"] = "test_grid_id"
        root.attrs["inputs_id"] = "test_inputs_id"
        root.attrs["cleo_turbines_json"] = turbines_json

        # Create required arrays using create_array (zarr v3 API)
        root.create_array("weibull_A", data=np.ones((3, 10, 10)), chunks=(1, 5, 5))
        root.create_array("weibull_k", data=np.ones((3, 10, 10)) * 2, chunks=(1, 5, 5))
        root.create_array("power_curve", data=np.ones((2, 26)), chunks=(1, 26))

    def test_valid_wind_store_passes(self, tmp_path: Path) -> None:
        """A properly formed wind store passes validation."""
        store_path = tmp_path / "wind.zarr"
        self._create_valid_wind_store(store_path)

        validate_store(store_path, kind="wind")  # Should not raise

    def test_missing_store_fails(self, tmp_path: Path) -> None:
        """Non-existent store fails with FileNotFoundError."""
        store_path = tmp_path / "nonexistent.zarr"

        with pytest.raises(FileNotFoundError, match="Store not found"):
            validate_store(store_path, kind="wind")

    def test_file_instead_of_dir_fails(self, tmp_path: Path) -> None:
        """Store path pointing to a file fails validation."""
        store_path = tmp_path / "not_a_dir.zarr"
        store_path.write_text("not a zarr store")

        with pytest.raises(ValidationError, match="not a directory"):
            validate_store(store_path, kind="wind")

    def test_incomplete_store_fails(self, tmp_path: Path) -> None:
        """Store with store_state != 'complete' fails by default."""
        store_path = tmp_path / "wind.zarr"
        self._create_valid_wind_store(store_path)

        # Modify to incomplete
        root = zarr.open_group(store_path, mode="r+")
        root.attrs["store_state"] = "building"

        with pytest.raises(ValidationError, match="store_state is 'building'"):
            validate_store(store_path, kind="wind")

    def test_incomplete_store_allowed(self, tmp_path: Path) -> None:
        """Store with store_state != 'complete' passes with allow_incomplete=True."""
        store_path = tmp_path / "wind.zarr"
        self._create_valid_wind_store(store_path)

        # Modify to incomplete
        root = zarr.open_group(store_path, mode="r+")
        root.attrs["store_state"] = "building"

        validate_store(store_path, kind="wind", allow_incomplete=True)  # Should pass

    def test_missing_required_attr_fails(self, tmp_path: Path) -> None:
        """Store missing required attr fails validation."""
        store_path = tmp_path / "wind.zarr"
        self._create_valid_wind_store(store_path)

        # Remove required attr
        root = zarr.open_group(store_path, mode="r+")
        del root.attrs["grid_id"]

        with pytest.raises(ValidationError, match="Missing required attrs.*grid_id"):
            validate_store(store_path, kind="wind")

    def test_missing_required_array_fails(self, tmp_path: Path) -> None:
        """Store missing required array fails validation."""
        store_path = tmp_path / "wind.zarr"
        store_path.mkdir(parents=True, exist_ok=True)

        turbines_json = json.dumps([{"id": "T1", "capacity": 1000}])

        root = zarr.open_group(store_path, mode="w")
        root.attrs["store_state"] = "complete"
        root.attrs["grid_id"] = "test_grid_id"
        root.attrs["inputs_id"] = "test_inputs_id"
        root.attrs["cleo_turbines_json"] = turbines_json

        # Only create some required arrays
        root.create_array("weibull_A", data=np.ones((3, 10, 10)))
        # Missing weibull_k and power_curve

        with pytest.raises(ValidationError, match="Missing required arrays"):
            validate_store(store_path, kind="wind")

    def test_invalid_turbines_json_fails(self, tmp_path: Path) -> None:
        """Store with invalid cleo_turbines_json fails validation."""
        store_path = tmp_path / "wind.zarr"
        self._create_valid_wind_store(store_path)

        # Set invalid JSON
        root = zarr.open_group(store_path, mode="r+")
        root.attrs["cleo_turbines_json"] = "not valid json {"

        with pytest.raises(ValidationError, match="cleo_turbines_json is invalid JSON"):
            validate_store(store_path, kind="wind")


class TestValidateStoreLandscape:
    """Tests for landscape store validation."""

    def _create_valid_landscape_store(self, store_path: Path) -> None:
        """Create a minimal valid landscape zarr store."""
        store_path.mkdir(parents=True, exist_ok=True)

        root = zarr.open_group(store_path, mode="w")
        root.attrs["store_state"] = "complete"
        root.attrs["grid_id"] = "test_grid_id"
        root.attrs["inputs_id"] = "test_inputs_id"

        # Create required arrays using create_array (zarr v3 API)
        root.create_array("valid_mask", data=np.ones((10, 10), dtype=bool))

    def test_valid_landscape_store_passes(self, tmp_path: Path) -> None:
        """A properly formed landscape store passes validation."""
        store_path = tmp_path / "landscape.zarr"
        self._create_valid_landscape_store(store_path)

        validate_store(store_path, kind="landscape")  # Should not raise

    def test_missing_valid_mask_fails(self, tmp_path: Path) -> None:
        """Store missing valid_mask array fails validation."""
        store_path = tmp_path / "landscape.zarr"
        store_path.mkdir(parents=True, exist_ok=True)

        root = zarr.open_group(store_path, mode="w")
        root.attrs["store_state"] = "complete"
        root.attrs["grid_id"] = "test_grid_id"
        root.attrs["inputs_id"] = "test_inputs_id"
        # No valid_mask array

        with pytest.raises(ValidationError, match="Missing required arrays.*valid_mask"):
            validate_store(store_path, kind="landscape")


class TestValidateStoreExport:
    """Tests for export store validation."""

    def _create_valid_export_store(self, store_path: Path) -> None:
        """Create a minimal valid export zarr store."""
        ds = xr.Dataset(
            {"wind__capacity_factors": (("y", "x"), np.ones((2, 2), dtype=np.float32))},
            coords={"y": np.array([1.0, 0.0]), "x": np.array([0.0, 1.0])},
            attrs={
                "store_state": "complete",
                "schema_version": 1,
                "created_at": "2026-02-26T12:00:00Z",
            },
        )
        ds.to_zarr(store_path, mode="w", consolidated=False)

    def test_valid_export_store_passes(self, tmp_path: Path) -> None:
        """A properly formed export store passes validation."""
        store_path = tmp_path / "export.zarr"
        self._create_valid_export_store(store_path)

        validate_store(store_path, kind="export")  # Should not raise

    def test_missing_schema_version_fails(self, tmp_path: Path) -> None:
        """Export store missing schema_version fails validation."""
        store_path = tmp_path / "export.zarr"
        self._create_valid_export_store(store_path)
        root = zarr.open_group(store_path, mode="a")
        del root.attrs["schema_version"]

        with pytest.raises(ValidationError, match="Missing required attrs.*schema_version"):
            validate_store(store_path, kind="export")

    def test_invalid_schema_version_type_fails(self, tmp_path: Path) -> None:
        """Export store with non-integer schema_version fails validation."""
        store_path = tmp_path / "export.zarr"
        self._create_valid_export_store(store_path)
        root = zarr.open_group(store_path, mode="a")
        root.attrs["schema_version"] = "1"  # String instead of int

        with pytest.raises(ValidationError, match="schema_version is str, expected int"):
            validate_store(store_path, kind="export")

    def test_export_store_without_data_vars_fails(self, tmp_path: Path) -> None:
        """Export store must contain at least one data variable."""
        store_path = tmp_path / "export_empty.zarr"
        ds = xr.Dataset(
            coords={"y": np.array([1.0, 0.0]), "x": np.array([0.0, 1.0])},
            attrs={
                "store_state": "complete",
                "schema_version": 1,
                "created_at": "2026-02-26T12:00:00Z",
            },
        )
        ds.to_zarr(store_path, mode="w", consolidated=False)

        with pytest.raises(ValidationError, match="contains no data variables"):
            validate_store(store_path, kind="export")


class TestValidateStoreResult:
    """Tests for result store validation."""

    def _create_valid_result_store(self, store_path: Path) -> None:
        """Create a minimal valid result zarr store."""
        store_path.mkdir(parents=True, exist_ok=True)

        root = zarr.open_group(store_path, mode="w")
        root.attrs["store_state"] = "complete"
        root.attrs["run_id"] = "test_run_001"
        root.attrs["metric_name"] = "capacity_factors"
        root.attrs["created_at"] = "2026-02-26T12:00:00Z"

    def test_valid_result_store_passes(self, tmp_path: Path) -> None:
        """A properly formed result store passes validation."""
        store_path = tmp_path / "result.zarr"
        self._create_valid_result_store(store_path)

        validate_store(store_path, kind="result")  # Should not raise

    def test_missing_run_id_fails(self, tmp_path: Path) -> None:
        """Result store missing run_id fails validation."""
        store_path = tmp_path / "result.zarr"
        store_path.mkdir(parents=True, exist_ok=True)

        root = zarr.open_group(store_path, mode="w")
        root.attrs["store_state"] = "complete"
        root.attrs["metric_name"] = "capacity_factors"
        root.attrs["created_at"] = "2026-02-26T12:00:00Z"
        # No run_id

        with pytest.raises(ValidationError, match="Missing required attrs.*run_id"):
            validate_store(store_path, kind="result")


class TestValidateStoreGeneric:
    """Tests for generic store validation."""

    def test_generic_requires_only_store_state(self, tmp_path: Path) -> None:
        """Generic validation only requires store_state."""
        store_path = tmp_path / "generic.zarr"
        store_path.mkdir(parents=True, exist_ok=True)

        root = zarr.open_group(store_path, mode="w")
        root.attrs["store_state"] = "complete"

        validate_store(store_path, kind="generic")  # Should pass

    def test_generic_missing_store_state_fails(self, tmp_path: Path) -> None:
        """Generic validation fails without store_state."""
        store_path = tmp_path / "generic.zarr"
        store_path.mkdir(parents=True, exist_ok=True)

        root = zarr.open_group(store_path, mode="w")
        # No store_state

        with pytest.raises(ValidationError, match="store_state is None"):
            validate_store(store_path, kind="generic")
