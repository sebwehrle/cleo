"""Tests for cleo.unification.materializers._helpers module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from cleo.unification.materializers._helpers import (
    PreservedStoreAttrs,
    StoreNotReadyError,
    capture_store_state,
    check_if_exists_simple,
    check_if_exists_with_fingerprint,
    check_store_idempotent,
    delete_variable_from_store,
    read_store_attrs_safe,
    require_complete_store,
    restore_store_attrs,
    validate_if_exists_param,
    write_dataset_to_store,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def complete_zarr_store(tmp_path: Path) -> Path:
    """Create a complete zarr store for testing."""
    store_path = tmp_path / "test.zarr"
    ds = xr.Dataset(
        {"var1": (["y", "x"], np.ones((3, 3)))},
        coords={"y": [0, 1, 2], "x": [0, 1, 2]},
    )
    ds.to_zarr(store_path, mode="w")

    # Add required attrs
    g = zarr.open_group(store_path, mode="a")
    g.attrs["store_state"] = "complete"
    g.attrs["inputs_id"] = "test_inputs_123"
    g.attrs["grid_id"] = "test_grid_456"
    g.attrs["custom_attr"] = "custom_value"

    return store_path


@pytest.fixture
def incomplete_zarr_store(tmp_path: Path) -> Path:
    """Create an incomplete zarr store for testing."""
    store_path = tmp_path / "incomplete.zarr"
    ds = xr.Dataset({"var1": (["y", "x"], np.ones((3, 3)))})
    ds.to_zarr(store_path, mode="w")

    g = zarr.open_group(store_path, mode="a")
    g.attrs["store_state"] = "building"

    return store_path


# =============================================================================
# Tests for require_complete_store
# =============================================================================


class TestRequireCompleteStore:
    """Tests for require_complete_store function."""

    def test_returns_dataset_when_complete(self, complete_zarr_store: Path) -> None:
        """Complete store returns open dataset."""
        ds = require_complete_store(complete_zarr_store, "test store")
        assert isinstance(ds, xr.Dataset)
        assert "var1" in ds.data_vars
        ds.close()

    def test_raises_file_not_found_when_missing(self, tmp_path: Path) -> None:
        """Missing store raises FileNotFoundError."""
        missing_path = tmp_path / "nonexistent.zarr"
        with pytest.raises(FileNotFoundError, match="test store not found"):
            require_complete_store(missing_path, "test store")

    def test_raises_store_not_ready_when_incomplete(self, incomplete_zarr_store: Path) -> None:
        """Incomplete store raises StoreNotReadyError."""
        with pytest.raises(StoreNotReadyError, match="not complete"):
            require_complete_store(incomplete_zarr_store, "test store")


# =============================================================================
# Tests for read_store_attrs_safe
# =============================================================================


class TestReadStoreAttrsSafe:
    """Tests for read_store_attrs_safe function."""

    def test_reads_existing_attrs(self, complete_zarr_store: Path) -> None:
        """Reads attrs that exist in store."""
        result = read_store_attrs_safe(complete_zarr_store, "store_state", "inputs_id", "custom_attr")
        assert result["store_state"] == "complete"
        assert result["inputs_id"] == "test_inputs_123"
        assert result["custom_attr"] == "custom_value"

    def test_returns_default_for_missing_attrs(self, complete_zarr_store: Path) -> None:
        """Returns default for attrs that don't exist."""
        result = read_store_attrs_safe(complete_zarr_store, "nonexistent_attr", default="fallback")
        assert result["nonexistent_attr"] == "fallback"

    def test_returns_default_when_store_missing(self, tmp_path: Path) -> None:
        """Returns default when store doesn't exist."""
        missing_path = tmp_path / "nonexistent.zarr"
        result = read_store_attrs_safe(missing_path, "any_attr", default="default_val")
        assert result["any_attr"] == "default_val"


# =============================================================================
# Tests for check_store_idempotent
# =============================================================================


class TestCheckStoreIdempotent:
    """Tests for check_store_idempotent function."""

    def test_returns_false_when_store_missing(self, tmp_path: Path) -> None:
        """Returns False when store doesn't exist."""
        missing_path = tmp_path / "nonexistent.zarr"
        assert check_store_idempotent(missing_path, inputs_id="any") is False

    def test_returns_false_when_store_incomplete(self, incomplete_zarr_store: Path) -> None:
        """Returns False when store_state != complete."""
        assert check_store_idempotent(incomplete_zarr_store, inputs_id="any") is False

    def test_returns_false_when_inputs_id_mismatch(self, complete_zarr_store: Path) -> None:
        """Returns False when inputs_id doesn't match."""
        assert check_store_idempotent(complete_zarr_store, inputs_id="wrong_id") is False

    def test_returns_true_when_inputs_id_matches(self, complete_zarr_store: Path) -> None:
        """Returns True when inputs_id matches."""
        assert check_store_idempotent(complete_zarr_store, inputs_id="test_inputs_123") is True

    def test_returns_false_when_grid_id_mismatch(self, complete_zarr_store: Path) -> None:
        """Returns False when grid_id doesn't match."""
        result = check_store_idempotent(
            complete_zarr_store,
            inputs_id="test_inputs_123",
            grid_id="wrong_grid",
        )
        assert result is False

    def test_returns_true_when_grid_id_matches(self, complete_zarr_store: Path) -> None:
        """Returns True when both inputs_id and grid_id match."""
        result = check_store_idempotent(
            complete_zarr_store,
            inputs_id="test_inputs_123",
            grid_id="test_grid_456",
        )
        assert result is True

    def test_extra_checks_work(self, complete_zarr_store: Path) -> None:
        """Extra attr checks are evaluated."""
        # Matching extra check
        result = check_store_idempotent(
            complete_zarr_store,
            inputs_id="test_inputs_123",
            extra_checks={"custom_attr": "custom_value"},
        )
        assert result is True

        # Non-matching extra check
        result = check_store_idempotent(
            complete_zarr_store,
            inputs_id="test_inputs_123",
            extra_checks={"custom_attr": "wrong_value"},
        )
        assert result is False


# =============================================================================
# Tests for validate_if_exists_param
# =============================================================================


class TestValidateIfExistsParam:
    """Tests for validate_if_exists_param function."""

    def test_accepts_valid_values(self) -> None:
        """Valid values don't raise."""
        validate_if_exists_param("error")
        validate_if_exists_param("replace")
        validate_if_exists_param("noop")

    def test_rejects_invalid_values(self) -> None:
        """Invalid values raise ValueError."""
        with pytest.raises(ValueError, match="if_exists must be one of"):
            validate_if_exists_param("invalid")
        with pytest.raises(ValueError):
            validate_if_exists_param("overwrite")


# =============================================================================
# Tests for check_if_exists_simple
# =============================================================================


class TestCheckIfExistsSimple:
    """Tests for check_if_exists_simple function."""

    def test_should_write_when_not_exists(self) -> None:
        """Returns should_write=True when variable doesn't exist."""
        result = check_if_exists_simple("var", exists_in_store=False, if_exists="error")
        assert result.should_write is True
        assert result.should_delete_first is False

    def test_raises_on_error_when_exists(self) -> None:
        """Raises ValueError when exists and if_exists='error'."""
        with pytest.raises(ValueError, match="already exists"):
            check_if_exists_simple("var", exists_in_store=True, if_exists="error")

    def test_noop_when_exists(self) -> None:
        """Returns should_write=False when exists and if_exists='noop'."""
        result = check_if_exists_simple("var", exists_in_store=True, if_exists="noop")
        assert result.should_write is False
        assert result.skip_reason == "exists_noop"

    def test_replace_when_exists(self) -> None:
        """Returns should_delete_first=True when exists and if_exists='replace'."""
        result = check_if_exists_simple("var", exists_in_store=True, if_exists="replace")
        assert result.should_write is True
        assert result.should_delete_first is True


# =============================================================================
# Tests for check_if_exists_with_fingerprint
# =============================================================================


class TestCheckIfExistsWithFingerprint:
    """Tests for check_if_exists_with_fingerprint function."""

    def test_should_write_when_not_exists(self) -> None:
        """Returns should_write=True when variable doesn't exist."""
        result = check_if_exists_with_fingerprint(
            "var", exists_in_store=False, if_exists="error", fingerprint_matches=False
        )
        assert result.should_write is True

    def test_raises_on_error_when_exists(self) -> None:
        """Raises ValueError when exists and if_exists='error'."""
        with pytest.raises(ValueError, match="already exists"):
            check_if_exists_with_fingerprint("var", exists_in_store=True, if_exists="error", fingerprint_matches=True)

    def test_noop_with_matching_fingerprint(self) -> None:
        """Returns should_write=False when exists, noop, and fingerprint matches."""
        result = check_if_exists_with_fingerprint(
            "var", exists_in_store=True, if_exists="noop", fingerprint_matches=True
        )
        assert result.should_write is False
        assert result.skip_reason == "fingerprint_match"

    def test_raises_on_noop_with_mismatched_fingerprint(self) -> None:
        """Raises ValueError when exists, noop, but fingerprint doesn't match."""
        with pytest.raises(ValueError, match="fingerprint differs"):
            check_if_exists_with_fingerprint("var", exists_in_store=True, if_exists="noop", fingerprint_matches=False)

    def test_replace_ignores_fingerprint(self) -> None:
        """Replace works regardless of fingerprint."""
        result = check_if_exists_with_fingerprint(
            "var", exists_in_store=True, if_exists="replace", fingerprint_matches=False
        )
        assert result.should_write is True
        assert result.should_delete_first is True


# =============================================================================
# Tests for capture_store_state and PreservedStoreAttrs
# =============================================================================


class TestCaptureStoreState:
    """Tests for capture_store_state function."""

    def test_captures_attrs_and_sizes(self) -> None:
        """Captures dataset attrs and sizes."""
        ds = xr.Dataset(
            {"var1": (["y", "x"], np.ones((3, 4)))},
            attrs={"attr1": "value1", "attr2": 42},
        )
        preserved = capture_store_state(ds)

        assert isinstance(preserved, PreservedStoreAttrs)
        assert preserved.attrs["attr1"] == "value1"
        assert preserved.attrs["attr2"] == 42
        assert preserved.sizes["y"] == 3
        assert preserved.sizes["x"] == 4


# =============================================================================
# Tests for write_dataset_to_store
# =============================================================================


class TestWriteDatasetToStore:
    """Tests for write_dataset_to_store function."""

    def test_writes_dataset(self, tmp_path: Path) -> None:
        """Writes dataset to zarr store."""
        store_path = tmp_path / "output.zarr"
        ds = xr.Dataset({"var1": (["y", "x"], np.ones((3, 3)))})

        write_dataset_to_store(ds, store_path, mode="w", use_lock=False)

        assert store_path.exists()
        loaded = xr.open_zarr(store_path)
        assert "var1" in loaded.data_vars
        loaded.close()

    def test_writes_with_new_attrs(self, tmp_path: Path) -> None:
        """Writes dataset with new attrs."""
        store_path = tmp_path / "output.zarr"
        ds = xr.Dataset({"var1": (["y", "x"], np.ones((3, 3)))})

        write_dataset_to_store(ds, store_path, mode="w", new_attrs={"custom": "attr"}, use_lock=False)

        g = zarr.open_group(store_path, mode="r")
        assert g.attrs.get("custom") == "attr"

    def test_restores_preserved_attrs(self, tmp_path: Path) -> None:
        """Restores preserved attrs after write."""
        store_path = tmp_path / "output.zarr"
        ds = xr.Dataset({"var1": (["y", "x"], np.ones((3, 3)))})
        preserved = PreservedStoreAttrs(attrs={"preserved": "value"}, sizes={})

        write_dataset_to_store(ds, store_path, mode="w", preserved=preserved, use_lock=False)

        g = zarr.open_group(store_path, mode="r")
        assert g.attrs.get("preserved") == "value"


# =============================================================================
# Tests for delete_variable_from_store
# =============================================================================


class TestDeleteVariableFromStore:
    """Tests for delete_variable_from_store function."""

    def test_deletes_existing_variable(self, complete_zarr_store: Path) -> None:
        """Deletes variable that exists."""
        result = delete_variable_from_store(complete_zarr_store, "var1", use_lock=False)
        assert result is True

        # Verify variable is gone
        g = zarr.open_group(complete_zarr_store, mode="r")
        assert "var1" not in g

    def test_returns_false_for_nonexistent_variable(self, complete_zarr_store: Path) -> None:
        """Returns False when variable doesn't exist."""
        result = delete_variable_from_store(complete_zarr_store, "nonexistent", use_lock=False)
        assert result is False


# =============================================================================
# Tests for restore_store_attrs
# =============================================================================


class TestRestoreStoreAttrs:
    """Tests for restore_store_attrs function."""

    def test_restores_attrs(self, complete_zarr_store: Path) -> None:
        """Restores preserved attrs to store."""
        preserved = PreservedStoreAttrs(
            attrs={"new_attr": "new_value", "another": 123},
            sizes={},
        )

        restore_store_attrs(complete_zarr_store, preserved)

        g = zarr.open_group(complete_zarr_store, mode="r")
        assert g.attrs.get("new_attr") == "new_value"
        assert g.attrs.get("another") == 123

    def test_respects_exclude_keys(self, complete_zarr_store: Path) -> None:
        """Doesn't restore excluded keys."""
        preserved = PreservedStoreAttrs(
            attrs={"include_me": "yes", "exclude_me": "no"},
            sizes={},
        )

        restore_store_attrs(complete_zarr_store, preserved, exclude_keys={"exclude_me"})

        g = zarr.open_group(complete_zarr_store, mode="r")
        assert g.attrs.get("include_me") == "yes"
        assert "exclude_me" not in g.attrs

    def test_handles_empty_attrs(self, complete_zarr_store: Path) -> None:
        """Handles empty attrs gracefully."""
        preserved = PreservedStoreAttrs(attrs={}, sizes={})
        # Should not raise
        restore_store_attrs(complete_zarr_store, preserved)
