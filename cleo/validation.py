"""Validation utilities for CLEO stores and datasets.

This module provides centralized validation functions for wind, landscape,
export, and result stores. Validators are metadata-first by default and
do not trigger compute on dask-backed arrays.

Public API:
    validate_dataset(ds, *, kind, deep=False) -> None
    validate_store(path, *, kind, allow_incomplete=False) -> None
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
import zarr

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Schema definitions: required attributes and variables by store kind
# -----------------------------------------------------------------------------

# Required root attributes for complete stores (all kinds)
_COMMON_REQUIRED_ATTRS = frozenset({"store_state"})

# Required attributes by store kind
WIND_REQUIRED_ATTRS = frozenset(
    {
        "store_state",
        "grid_id",
        "inputs_id",
        "cleo_turbines_json",
    }
)

LANDSCAPE_REQUIRED_ATTRS = frozenset(
    {
        "store_state",
        "grid_id",
        "inputs_id",
    }
)

EXPORT_REQUIRED_ATTRS = frozenset(
    {
        "store_state",
        "schema_version",
        "created_at",
    }
)

RESULT_REQUIRED_ATTRS = frozenset(
    {
        "store_state",
        "run_id",
        "metric_name",
        "created_at",
    }
)

# Required data variables by store kind
WIND_REQUIRED_VARS = frozenset(
    {
        "weibull_A",
        "weibull_k",
        "power_curve",
    }
)

LANDSCAPE_REQUIRED_VARS = frozenset(
    {
        "valid_mask",
    }
)

# Required dimensions/coordinates by store kind
WIND_REQUIRED_DIMS = frozenset({"y", "x", "turbine", "height"})
LANDSCAPE_REQUIRED_DIMS = frozenset({"y", "x"})

# Store kind type
StoreKind = Literal["wind", "landscape", "export", "result", "generic"]


class ValidationError(ValueError):
    """Raised when store or dataset validation fails."""

    pass


# -----------------------------------------------------------------------------
# Dataset validation
# -----------------------------------------------------------------------------


def validate_dataset(
    ds: xr.Dataset,
    *,
    kind: StoreKind,
    deep: bool = False,
) -> None:
    """Validate an xarray Dataset against store schema expectations.

    This function performs metadata-first validation and does NOT trigger
    compute on dask-backed arrays by default.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to validate.
    kind : {"wind", "landscape", "export", "result", "generic"}
        The expected store kind, determining which schema to validate against.
    deep : bool, default False
        If True, perform additional checks that may sample small data windows.
        Default False performs only metadata/attribute checks.

    Raises
    ------
    ValidationError
        If validation fails (missing attrs, variables, or schema violations).

    Notes
    -----
    - ``deep=False`` (default) checks only attrs, dims, coords, and variable presence.
    - ``deep=True`` may check coordinate monotonicity and sample data validity.
    - This function is safe to call on dask-backed datasets without triggering compute.
    """
    errors: list[str] = []

    # Determine required attrs and vars based on kind
    if kind == "wind":
        required_attrs = WIND_REQUIRED_ATTRS
        required_vars = WIND_REQUIRED_VARS
        required_dims = WIND_REQUIRED_DIMS
    elif kind == "landscape":
        required_attrs = LANDSCAPE_REQUIRED_ATTRS
        required_vars = LANDSCAPE_REQUIRED_VARS
        required_dims = LANDSCAPE_REQUIRED_DIMS
    elif kind == "export":
        required_attrs = EXPORT_REQUIRED_ATTRS
        required_vars = frozenset()
        required_dims = frozenset({"y", "x"})
    elif kind == "result":
        required_attrs = RESULT_REQUIRED_ATTRS
        required_vars = frozenset()
        required_dims = frozenset()
    else:  # generic
        required_attrs = _COMMON_REQUIRED_ATTRS
        required_vars = frozenset()
        required_dims = frozenset()

    # Check required attributes
    missing_attrs = required_attrs - set(ds.attrs.keys())
    if missing_attrs:
        errors.append(f"Missing required attrs: {sorted(missing_attrs)}")

    # Check required variables
    missing_vars = required_vars - set(ds.data_vars.keys())
    if missing_vars:
        errors.append(f"Missing required variables: {sorted(missing_vars)}")

    # Check required dimensions (use .sizes for dict-like access)
    missing_dims = required_dims - set(ds.sizes.keys())
    if missing_dims:
        errors.append(f"Missing required dimensions: {sorted(missing_dims)}")

    # Check store_state if present
    store_state = ds.attrs.get("store_state")
    if store_state is not None and store_state != "complete":
        errors.append(f"store_state is {store_state!r}, expected 'complete'")

    # Wind-specific checks
    if kind == "wind":
        # Validate cleo_turbines_json is valid JSON array
        turbines_json = ds.attrs.get("cleo_turbines_json")
        if turbines_json is not None:
            try:
                parsed = json.loads(turbines_json)
                if not isinstance(parsed, list):
                    errors.append("cleo_turbines_json is not a JSON array")
            except json.JSONDecodeError as e:
                errors.append(f"cleo_turbines_json is invalid JSON: {e}")

    # Coordinate checks (metadata-only, no compute)
    for dim in ("y", "x"):
        if dim in ds.coords:
            coord = ds.coords[dim]
            if coord.size < 1:
                errors.append(f"Coordinate '{dim}' is empty")

    # Deep validation (optional, may access small data samples)
    if deep:
        _validate_dataset_deep(ds, kind, errors)

    if errors:
        raise ValidationError(
            f"Dataset validation failed for kind={kind!r}:\n  - " + "\n  - ".join(errors)
        )


def _validate_dataset_deep(
    ds: xr.Dataset, kind: StoreKind, errors: list[str]
) -> None:
    """Perform deep validation checks (may access small data samples)."""
    # Check coordinate monotonicity for spatial dims
    for dim in ("y", "x"):
        if dim in ds.coords:
            coord = ds.coords[dim]
            if coord.size > 1:
                # Access coordinate values (usually small, not chunked)
                vals = coord.values
                diffs = np.diff(vals)
                if not (np.all(diffs > 0) or np.all(diffs < 0)):
                    errors.append(f"Coordinate '{dim}' is not monotonic")

    # Check height coordinate for wind stores
    if kind == "wind" and "height" in ds.coords:
        height = ds.coords["height"]
        if height.size > 0:
            vals = height.values
            if not np.all(vals > 0):
                errors.append("Height coordinate contains non-positive values")
            if height.size > 1:
                diffs = np.diff(vals)
                if not np.all(diffs > 0):
                    errors.append("Height coordinate is not strictly increasing")

    # Check valid_mask dtype for landscape stores
    if kind == "landscape" and "valid_mask" in ds.data_vars:
        vm = ds["valid_mask"]
        if vm.dtype != bool and vm.dtype != np.bool_:
            errors.append(f"valid_mask dtype is {vm.dtype}, expected bool")


# -----------------------------------------------------------------------------
# Store validation
# -----------------------------------------------------------------------------


def validate_store(
    path: str | Path,
    *,
    kind: StoreKind,
    allow_incomplete: bool = False,
) -> None:
    """Validate a Zarr store at the given path.

    Opens the store metadata (not full arrays) and validates completeness
    and required attributes.

    Parameters
    ----------
    path : str or Path
        Path to the Zarr store directory.
    kind : {"wind", "landscape", "export", "result", "generic"}
        The expected store kind, determining which schema to validate against.
    allow_incomplete : bool, default False
        If True, allow stores with ``store_state != "complete"``.
        Default False requires stores to be complete.

    Raises
    ------
    FileNotFoundError
        If the store does not exist.
    ValidationError
        If validation fails (incomplete store, missing attrs, or schema violations).

    Notes
    -----
    This function opens only store metadata via zarr, not full xarray datasets,
    making it lightweight for pre-flight validation checks.
    """
    store_path = Path(path)

    if not store_path.exists():
        raise FileNotFoundError(f"Store not found: {store_path}")

    if not store_path.is_dir():
        raise ValidationError(f"Store path is not a directory: {store_path}")

    # Open zarr group to read attrs
    try:
        group = zarr.open_group(store_path, mode="r")
    except Exception as e:
        raise ValidationError(f"Failed to open store as Zarr group: {e}") from e

    attrs = dict(group.attrs)
    errors: list[str] = []

    # Check store_state
    store_state = attrs.get("store_state")
    if not allow_incomplete:
        if store_state != "complete":
            errors.append(
                f"store_state is {store_state!r}, expected 'complete' "
                "(use allow_incomplete=True to skip this check)"
            )

    # Determine required attrs based on kind
    if kind == "wind":
        required_attrs = WIND_REQUIRED_ATTRS
    elif kind == "landscape":
        required_attrs = LANDSCAPE_REQUIRED_ATTRS
    elif kind == "export":
        required_attrs = EXPORT_REQUIRED_ATTRS
    elif kind == "result":
        required_attrs = RESULT_REQUIRED_ATTRS
    else:  # generic
        required_attrs = _COMMON_REQUIRED_ATTRS

    # Check required attributes (skip store_state if allow_incomplete)
    check_attrs = required_attrs
    if allow_incomplete:
        check_attrs = check_attrs - {"store_state"}

    missing_attrs = check_attrs - set(attrs.keys())
    if missing_attrs:
        errors.append(f"Missing required attrs: {sorted(missing_attrs)}")

    # Wind-specific: validate turbines JSON
    if kind == "wind":
        turbines_json = attrs.get("cleo_turbines_json")
        if turbines_json is not None:
            try:
                parsed = json.loads(turbines_json)
                if not isinstance(parsed, list):
                    errors.append("cleo_turbines_json is not a JSON array")
            except json.JSONDecodeError as e:
                errors.append(f"cleo_turbines_json is invalid JSON: {e}")

    # Export-specific: validate schema_version is integer
    if kind == "export":
        schema_version = attrs.get("schema_version")
        if schema_version is not None and not isinstance(schema_version, int):
            errors.append(
                f"schema_version is {type(schema_version).__name__}, expected int"
            )

    # Check required arrays exist in store
    array_names = set(group.array_keys()) if hasattr(group, "array_keys") else set()

    if kind == "wind":
        missing_vars = WIND_REQUIRED_VARS - array_names
        if missing_vars:
            errors.append(f"Missing required arrays: {sorted(missing_vars)}")
    elif kind == "landscape":
        missing_vars = LANDSCAPE_REQUIRED_VARS - array_names
        if missing_vars:
            errors.append(f"Missing required arrays: {sorted(missing_vars)}")

    if errors:
        raise ValidationError(
            f"Store validation failed for {store_path} (kind={kind!r}):\n  - "
            + "\n  - ".join(errors)
        )
