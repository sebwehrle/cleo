"""Shared helpers for materialize operations.

This module provides common utilities used across materialize functions
to reduce code duplication and cyclomatic complexity.

Public API:
    - require_complete_store: Open and validate a zarr store is complete.
    - read_store_attrs_safe: Safely read attrs without failing on errors.
    - get_wind_reference: Get wind reference grid for alignment.
    - check_store_idempotent: Check if store exists with matching identity.
    - check_if_exists: Apply if_exists semantics for variable writes.
    - write_dataset_to_store: Write dataset with locking and attr management.
    - delete_variable_from_store: Delete a variable from a zarr store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr
import zarr

from cleo.store import single_writer_lock, zarr_store_lock_dir
from cleo.unification.store_io import open_zarr_dataset

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class StoreNotReadyError(RuntimeError):
    """Raised when a required store is not complete."""

    pass


# =============================================================================
# Store State Validation
# =============================================================================


def require_complete_store(
    store_path: Path,
    store_name: str = "store",
    *,
    chunk_policy: dict[str, int] | None = None,
) -> xr.Dataset:
    """
    Open a zarr store and validate it is complete.

    :param store_path: Path to the zarr store.
    :param store_name: Human-readable name for error messages.
    :param chunk_policy: Optional chunk policy for opening.
    :returns: Opened xarray Dataset.
    :raises StoreNotReadyError: If store_state != "complete".
    :raises FileNotFoundError: If store does not exist.
    """
    if not store_path.exists():
        raise FileNotFoundError(f"{store_name} not found at {store_path}")

    ds = open_zarr_dataset(store_path, chunk_policy=chunk_policy)

    if ds.attrs.get("store_state") != "complete":
        ds.close()
        raise StoreNotReadyError(f"{store_name} at {store_path} is not complete; run atlas.build() first.")

    return ds


def read_store_attrs_safe(
    store_path: Path,
    *attrs: str,
    default: Any = None,
) -> dict[str, Any]:
    """
    Safely read attrs from a zarr store without failing on errors.

    :param store_path: Path to zarr store.
    :param attrs: Attribute names to read.
    :param default: Default value if attr missing or read fails.
    :returns: Dict of attr_name -> value (or default).
    """
    result = {attr: default for attr in attrs}
    try:
        if store_path.exists():
            g = zarr.open_group(store_path, mode="r")
            for attr in attrs:
                result[attr] = g.attrs.get(attr, default)
    except (OSError, ValueError, TypeError, KeyError):
        logger.debug(
            "Failed to read store attrs from %s",
            store_path,
            exc_info=True,
        )
    return result


# =============================================================================
# Wind Reference Grid
# =============================================================================


@dataclass
class WindReference:
    """Wind reference grid for alignment.

    :param ref_da: Reference DataArray for grid alignment.
    :param grid_id: Wind store grid_id.
    :param inputs_id: Wind store inputs_id.
    :param crs: Coordinate reference system.
    :param transform: Affine transform.
    :param dataset: Full wind dataset (caller must close).
    """

    ref_da: xr.DataArray
    grid_id: str
    inputs_id: str
    crs: Any
    transform: Any
    dataset: xr.Dataset


def get_wind_reference(
    wind_store_path: Path,
    atlas_crs: str,
    *,
    chunk_policy: dict[str, int] | None = None,
    preferred_height: int = 100,
) -> WindReference:
    """
    Get wind reference grid for landscape/region alignment.

    The caller is responsible for closing the returned dataset when done.

    :param wind_store_path: Path to wind.zarr.
    :param atlas_crs: Fallback CRS from atlas.
    :param chunk_policy: Optional chunk policy.
    :param preferred_height: Preferred height for reference (default 100m).
    :returns: WindReference with ref_da and metadata.
    :raises StoreNotReadyError: If wind store not complete.
    :raises RuntimeError: If weibull_A missing.
    """
    wind = require_complete_store(wind_store_path, "wind.zarr", chunk_policy=chunk_policy)

    grid_id = wind.attrs.get("grid_id", "")
    inputs_id = wind.attrs.get("inputs_id", "")

    if not grid_id or not inputs_id:
        wind.close()
        raise RuntimeError("wind.zarr missing grid_id/inputs_id; cannot get reference.")

    if "weibull_A" not in wind:
        wind.close()
        raise RuntimeError("wind.zarr missing weibull_A; cannot define canonical grid.")

    # Select reference height
    ref_da = wind["weibull_A"].isel(height=0)
    if "height" in wind["weibull_A"].dims:
        heights = wind["weibull_A"]["height"].values
        if preferred_height in heights:
            ref_da = wind["weibull_A"].sel(height=preferred_height)

    # Ensure CRS
    if ref_da.rio.crs is None:
        if "template" in wind and wind["template"].rio.crs is not None:
            ref_da = ref_da.rio.write_crs(wind["template"].rio.crs)
        else:
            ref_da = ref_da.rio.write_crs(atlas_crs)

    # Ensure transform
    if ref_da.rio.transform() is None:
        ref_da = ref_da.rio.write_transform(ref_da.rio.transform(recalc=True))

    return WindReference(
        ref_da=ref_da,
        grid_id=grid_id,
        inputs_id=inputs_id,
        crs=ref_da.rio.crs,
        transform=ref_da.rio.transform(),
        dataset=wind,
    )


# =============================================================================
# Idempotency Checking
# =============================================================================


def check_store_idempotent(
    store_path: Path,
    *,
    inputs_id: str,
    grid_id: str | None = None,
    extra_checks: dict[str, Any] | None = None,
) -> bool:
    """
    Check if store already exists with matching identity.

    :param store_path: Path to zarr store.
    :param inputs_id: Expected inputs_id.
    :param grid_id: Optional expected grid_id.
    :param extra_checks: Additional attr=value checks.
    :returns: True if store exists and matches (skip rebuild).
    """
    if not store_path.exists():
        return False

    try:
        g = zarr.open_group(store_path, mode="r")
        if g.attrs.get("store_state") != "complete":
            return False
        if g.attrs.get("inputs_id") != inputs_id:
            return False
        if grid_id is not None and g.attrs.get("grid_id") != grid_id:
            return False
        if extra_checks:
            for key, expected in extra_checks.items():
                if g.attrs.get(key) != expected:
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


# =============================================================================
# if_exists Semantics
# =============================================================================


@dataclass
class ExistsCheckResult:
    """Result of if_exists check.

    :param should_write: Whether to proceed with writing.
    :param should_delete_first: Whether to delete existing variable first.
    :param skip_reason: Reason for skipping if should_write is False.
    """

    should_write: bool
    should_delete_first: bool = False
    skip_reason: str | None = None


def validate_if_exists_param(if_exists: str) -> None:
    """
    Validate if_exists parameter value.

    :param if_exists: Value to validate.
    :raises ValueError: If invalid.
    """
    valid = {"error", "replace", "noop"}
    if if_exists not in valid:
        raise ValueError(f"if_exists must be one of {sorted(valid)!r}; got {if_exists!r}")


def check_if_exists_simple(
    variable_name: str,
    exists_in_store: bool,
    if_exists: str,
) -> ExistsCheckResult:
    """
    Apply if_exists semantics for simple variable materialization.

    This is for cases without fingerprint validation (use check_if_exists_with_fingerprint
    for cases that need source fingerprint matching on noop).

    :param variable_name: Name of variable being written.
    :param exists_in_store: Whether variable already exists.
    :param if_exists: Policy ("error", "replace", "noop").
    :returns: ExistsCheckResult indicating action to take.
    :raises ValueError: On conflict with if_exists="error".
    """
    validate_if_exists_param(if_exists)

    if not exists_in_store:
        return ExistsCheckResult(should_write=True)

    if if_exists == "error":
        raise ValueError(
            f"Variable {variable_name!r} already exists; "
            "use if_exists='replace' to overwrite or if_exists='noop' to skip."
        )

    if if_exists == "noop":
        return ExistsCheckResult(should_write=False, skip_reason="exists_noop")

    # if_exists == "replace"
    return ExistsCheckResult(should_write=True, should_delete_first=True)


def check_if_exists_with_fingerprint(
    variable_name: str,
    exists_in_store: bool,
    if_exists: str,
    *,
    fingerprint_matches: bool,
) -> ExistsCheckResult:
    """
    Apply if_exists semantics with fingerprint validation for noop.

    :param variable_name: Name of variable being written.
    :param exists_in_store: Whether variable already exists.
    :param if_exists: Policy ("error", "replace", "noop").
    :param fingerprint_matches: For noop, whether source fingerprint matches.
    :returns: ExistsCheckResult indicating action to take.
    :raises ValueError: On conflict or fingerprint mismatch with noop.
    """
    validate_if_exists_param(if_exists)

    if not exists_in_store:
        return ExistsCheckResult(should_write=True)

    if if_exists == "error":
        raise ValueError(
            f"Variable {variable_name!r} already exists; "
            "use if_exists='replace' to overwrite or if_exists='noop' to skip."
        )

    if if_exists == "noop":
        if fingerprint_matches:
            return ExistsCheckResult(should_write=False, skip_reason="fingerprint_match")
        else:
            raise ValueError(
                f"Variable {variable_name!r} exists but source fingerprint differs; "
                "use if_exists='replace' to overwrite."
            )

    # if_exists == "replace"
    return ExistsCheckResult(should_write=True, should_delete_first=True)


# =============================================================================
# Atomic Store Writing
# =============================================================================


@dataclass
class PreservedStoreAttrs:
    """Container for preserved store attributes.

    :param attrs: Dictionary of preserved attributes.
    :param sizes: Dictionary of dimension sizes.
    """

    attrs: dict[str, Any] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)


def capture_store_state(ds: xr.Dataset) -> PreservedStoreAttrs:
    """
    Capture store attributes and sizes before closing.

    :param ds: Open dataset to capture from.
    :returns: PreservedStoreAttrs with attrs and sizes.
    """
    return PreservedStoreAttrs(
        attrs=dict(ds.attrs),
        sizes={k: int(v) for k, v in ds.sizes.items()},
    )


def write_dataset_to_store(
    ds: xr.Dataset,
    store_path: Path,
    *,
    mode: str = "a",
    new_attrs: dict[str, Any] | None = None,
    preserved: PreservedStoreAttrs | None = None,
    use_lock: bool = True,
) -> None:
    """
    Write dataset to zarr store with optional locking and attr management.

    :param ds: Dataset to write.
    :param store_path: Target store path.
    :param mode: Zarr write mode ("w", "a", "r+").
    :param new_attrs: New attributes to set after write.
    :param preserved: Previously captured attrs to restore.
    :param use_lock: Whether to acquire single-writer lock.
    """

    def _do_write() -> None:
        ds.to_zarr(store_path, mode=mode, consolidated=False)

        if new_attrs or (preserved and preserved.attrs):
            g = zarr.open_group(store_path, mode="a")
            # Restore preserved attrs first (don't overwrite new attrs)
            if preserved:
                for key, val in preserved.attrs.items():
                    if key not in (new_attrs or {}):
                        g.attrs[key] = val
            # Then apply new attrs
            if new_attrs:
                g.attrs.update(new_attrs)

    if use_lock:
        with single_writer_lock(zarr_store_lock_dir(store_path)):
            _do_write()
    else:
        _do_write()


def delete_variable_from_store(
    store_path: Path,
    variable_name: str,
    *,
    use_lock: bool = True,
) -> bool:
    """
    Delete a variable from a zarr store.

    :param store_path: Path to zarr store.
    :param variable_name: Variable to delete.
    :param use_lock: Whether to acquire single-writer lock.
    :returns: True if deleted, False if not found.
    """

    def _do_delete() -> bool:
        try:
            root = zarr.open_group(store_path, mode="a")
            if variable_name in root:
                del root[variable_name]
                return True
            return False
        except (OSError, ValueError, KeyError):
            return False

    if use_lock:
        with single_writer_lock(zarr_store_lock_dir(store_path)):
            return _do_delete()
    else:
        return _do_delete()


def restore_store_attrs(
    store_path: Path,
    preserved: PreservedStoreAttrs,
    *,
    exclude_keys: set[str] | None = None,
) -> None:
    """
    Restore preserved attributes to a store.

    :param store_path: Path to zarr store.
    :param preserved: Preserved attributes to restore.
    :param exclude_keys: Attribute keys to skip.
    """
    if not preserved.attrs:
        return

    try:
        g = zarr.open_group(store_path, mode="a")
        for key, val in preserved.attrs.items():
            if exclude_keys and key in exclude_keys:
                continue
            g.attrs[key] = val
    except (OSError, ValueError):
        logger.debug("Failed to restore attrs to %s", store_path, exc_info=True)
