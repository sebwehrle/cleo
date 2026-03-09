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


def _requirements_for_kind(kind: StoreKind) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Return required attrs, vars, and dims for a store kind.

    :param kind: Expected store kind.
    :type kind: StoreKind
    :returns: Tuple ``(required_attrs, required_vars, required_dims)``.
    :rtype: tuple[frozenset[str], frozenset[str], frozenset[str]]
    """
    if kind == "wind":
        return WIND_REQUIRED_ATTRS, WIND_REQUIRED_VARS, WIND_REQUIRED_DIMS
    if kind == "landscape":
        return LANDSCAPE_REQUIRED_ATTRS, LANDSCAPE_REQUIRED_VARS, LANDSCAPE_REQUIRED_DIMS
    if kind == "export":
        return EXPORT_REQUIRED_ATTRS, frozenset(), frozenset({"y", "x"})
    if kind == "result":
        return RESULT_REQUIRED_ATTRS, frozenset(), frozenset()
    return _COMMON_REQUIRED_ATTRS, frozenset(), frozenset()


def _validate_json_array_attr(raw_value: object, *, attr_name: str, errors: list[str]) -> None:
    """Append validation errors when a JSON-array attr is missing or malformed.

    :param raw_value: Raw attribute payload.
    :type raw_value: object
    :param attr_name: Attribute name for error reporting.
    :type attr_name: str
    :param errors: Mutable error accumulator.
    :type errors: list[str]
    :returns: ``None``
    :rtype: None
    """
    if raw_value is None:
        return
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as e:
        errors.append(f"{attr_name} is invalid JSON: {e}")
        return
    if not isinstance(parsed, list):
        errors.append(f"{attr_name} is not a JSON array")


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

    required_attrs, required_vars, required_dims = _requirements_for_kind(kind)

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
        _validate_json_array_attr(ds.attrs.get("cleo_turbines_json"), attr_name="cleo_turbines_json", errors=errors)

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
        raise ValidationError(f"Dataset validation failed for kind={kind!r}:\n  - " + "\n  - ".join(errors))


def _validate_dataset_deep(ds: xr.Dataset, kind: StoreKind, errors: list[str]) -> None:
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
    This function reads Zarr metadata for all store kinds. For
    ``kind="export"``, it also opens the store with xarray metadata to
    distinguish data variables from coordinates without materializing array
    payloads.
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
                f"store_state is {store_state!r}, expected 'complete' (use allow_incomplete=True to skip this check)"
            )

    required_attrs, required_vars, _required_dims = _requirements_for_kind(kind)

    # Check required attributes (skip store_state if allow_incomplete)
    check_attrs = required_attrs
    if allow_incomplete:
        check_attrs = check_attrs - {"store_state"}

    missing_attrs = check_attrs - set(attrs.keys())
    if missing_attrs:
        errors.append(f"Missing required attrs: {sorted(missing_attrs)}")

    # Wind-specific: validate turbines JSON
    if kind == "wind":
        _validate_json_array_attr(attrs.get("cleo_turbines_json"), attr_name="cleo_turbines_json", errors=errors)

    # Export-specific: validate schema_version is integer
    if kind == "export":
        schema_version = attrs.get("schema_version")
        if schema_version is not None and not isinstance(schema_version, int):
            errors.append(f"schema_version is {type(schema_version).__name__}, expected int")

    # Check required arrays exist in store
    array_names = set(group.array_keys()) if hasattr(group, "array_keys") else set()

    if kind == "wind":
        missing_vars = required_vars - array_names
        if missing_vars:
            errors.append(f"Missing required arrays: {sorted(missing_vars)}")
    elif kind == "landscape":
        missing_vars = required_vars - array_names
        if missing_vars:
            errors.append(f"Missing required arrays: {sorted(missing_vars)}")
    elif kind == "export":
        try:
            ds = xr.open_zarr(store_path, consolidated=False)
        except Exception as e:
            errors.append(f"Failed to open export store dataset: {e}")
        else:
            try:
                if len(ds.data_vars) == 0:
                    errors.append("Export store contains no data variables")
            finally:
                close = getattr(ds, "close", None)
                if callable(close):
                    close()

    if errors:
        raise ValidationError(
            f"Store validation failed for {store_path} (kind={kind!r}):\n  - " + "\n  - ".join(errors)
        )
