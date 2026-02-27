"""Consolidated analysis xarray export utilities.

This module provides the implementation for `Atlas.export_analysis_dataset_zarr()`,
creating schema-versioned xarray exports with provenance tracking.

Export stores include:
- schema_version: from ANALYSIS_EXPORT_SCHEMA_VERSION
- store_state: "complete" on successful write
- created_at: ISO 8601 timestamp
- cleo:package_version: version string
- export_spec_json: domain/include_only/prefixing/exclude_template settings
- upstream provenance: wind/landscape store attrs when available
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import xarray as xr
import zarr

from cleo.contracts import ANALYSIS_EXPORT_SCHEMA_VERSION
from cleo.dask_utils import compute as dask_compute
from cleo.store import atomic_dir
from cleo.validation import validate_dataset, validate_store

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Template variables excluded by default from exports
_TEMPLATE_VARS = frozenset({"template"})


def _get_package_version() -> str:
    """Get cleo package version string."""
    try:
        from importlib.metadata import version

        return version("cleo")
    except Exception:
        return "unknown"


def _build_export_spec(
    *,
    domain: str,
    include_only: Sequence[str] | None,
    prefix: bool,
    exclude_template: bool,
) -> dict:
    """Build export specification dictionary for attrs."""
    return {
        "domain": domain,
        "include_only": list(include_only) if include_only is not None else None,
        "prefix": prefix,
        "exclude_template": exclude_template,
    }


def _extract_upstream_provenance(
    wind_ds: xr.Dataset | None,
    landscape_ds: xr.Dataset | None,
) -> dict:
    """Extract upstream provenance attrs from wind/landscape stores."""
    provenance: dict = {}

    if wind_ds is not None:
        wind_attrs = {}
        for key in ("grid_id", "inputs_id", "unify_version", "created_at"):
            if key in wind_ds.attrs:
                wind_attrs[key] = wind_ds.attrs[key]
        if wind_attrs:
            provenance["wind_store"] = wind_attrs

    if landscape_ds is not None:
        land_attrs = {}
        for key in ("grid_id", "inputs_id", "unify_version", "created_at"):
            if key in landscape_ds.attrs:
                land_attrs[key] = landscape_ds.attrs[key]
        if land_attrs:
            provenance["landscape_store"] = land_attrs

    return provenance


def _filter_variables(
    ds: xr.Dataset,
    *,
    include_only: Sequence[str] | None,
    exclude_template: bool,
) -> xr.Dataset:
    """Filter dataset variables based on include_only and exclude_template."""
    vars_to_keep = set(ds.data_vars.keys())

    if exclude_template:
        vars_to_keep -= _TEMPLATE_VARS

    if include_only is not None:
        requested = set(include_only)
        missing = requested - set(ds.data_vars.keys())
        if missing:
            raise ValueError(
                f"include_only contains unknown variables: {sorted(missing)}. "
                f"Available: {sorted(ds.data_vars.keys())}"
            )
        vars_to_keep &= requested

    if not vars_to_keep:
        raise ValueError(
            "No variables to export after filtering. Check include_only and exclude_template settings."
        )

    return ds[sorted(vars_to_keep)]


def _prefix_variables(ds: xr.Dataset, prefix: str) -> xr.Dataset:
    """Rename variables with a prefix."""
    rename_map = {var: f"{prefix}{var}" for var in ds.data_vars}
    return ds.rename(rename_map)


def build_analysis_export_dataset(
    wind_ds: xr.Dataset | None,
    landscape_ds: xr.Dataset | None,
    *,
    domain: Literal["wind", "landscape", "both"],
    include_only: Sequence[str] | None = None,
    prefix: bool = True,
    exclude_template: bool = True,
) -> xr.Dataset:
    """Build a consolidated analysis export dataset.

    Parameters
    ----------
    wind_ds : xr.Dataset or None
        Wind store dataset (required for domain="wind" or "both").
    landscape_ds : xr.Dataset or None
        Landscape store dataset (required for domain="landscape" or "both").
    domain : {"wind", "landscape", "both"}
        Which domain(s) to include in the export.
    include_only : sequence of str, optional
        If provided, only export these variables. Variable names should be
        the prefixed names when domain="both" and prefix=True.
    prefix : bool, default True
        When domain="both", prefix variables with "wind__" / "landscape__"
        to avoid collisions. Ignored for single-domain exports.
    exclude_template : bool, default True
        If True, exclude template variables from the export.

    Returns
    -------
    xr.Dataset
        Export dataset with selected variables.

    Raises
    ------
    ValueError
        If required domain data is missing, if there are name collisions
        when prefix=False, or if include_only contains unknown variables.
    """
    if domain == "wind":
        if wind_ds is None:
            raise ValueError("wind_ds required for domain='wind'")
        result = _filter_variables(
            wind_ds,
            include_only=include_only,
            exclude_template=exclude_template,
        )

    elif domain == "landscape":
        if landscape_ds is None:
            raise ValueError("landscape_ds required for domain='landscape'")
        result = _filter_variables(
            landscape_ds,
            include_only=include_only,
            exclude_template=exclude_template,
        )

    elif domain == "both":
        if wind_ds is None:
            raise ValueError("wind_ds required for domain='both'")
        if landscape_ds is None:
            raise ValueError("landscape_ds required for domain='both'")

        # Filter each domain first (before prefixing, using raw variable names)
        # For include_only with prefixed names, we need to handle differently
        if include_only is not None and prefix:
            # Parse prefixed names to determine which vars from each domain
            wind_vars = []
            land_vars = []
            for name in include_only:
                if name.startswith("wind__"):
                    wind_vars.append(name[6:])  # Remove "wind__" prefix
                elif name.startswith("landscape__"):
                    land_vars.append(name[11:])  # Remove "landscape__" prefix
                else:
                    raise ValueError(
                        f"Variable {name!r} does not have expected prefix "
                        f"('wind__' or 'landscape__') for domain='both' with prefix=True"
                    )
            # None means "all vars", empty list means "no vars from this domain"
            wind_include = wind_vars if wind_vars else []
            land_include = land_vars if land_vars else []
        else:
            wind_include = include_only
            land_include = include_only

        # Filter variables - handle empty include_only by creating empty dataset
        if wind_include is not None and len(wind_include) == 0:
            wind_filtered = xr.Dataset(coords=wind_ds.coords)
        else:
            wind_filtered = _filter_variables(
                wind_ds,
                include_only=wind_include,
                exclude_template=exclude_template,
            )

        if land_include is not None and len(land_include) == 0:
            land_filtered = xr.Dataset(coords=landscape_ds.coords)
        else:
            land_filtered = _filter_variables(
                landscape_ds,
                include_only=land_include,
                exclude_template=exclude_template,
            )

        if prefix:
            wind_prefixed = _prefix_variables(wind_filtered, "wind__")
            land_prefixed = _prefix_variables(land_filtered, "landscape__")
        else:
            # Check for collisions
            wind_vars = set(wind_filtered.data_vars.keys())
            land_vars = set(land_filtered.data_vars.keys())
            overlap = wind_vars & land_vars
            if overlap:
                raise ValueError(
                    f"Variable name collision when exporting domain='both' with prefix=False: "
                    f"{sorted(overlap)}. Set prefix=True to avoid collisions."
                )
            wind_prefixed = wind_filtered
            land_prefixed = land_filtered

        # Merge datasets
        # Use the wind coords as primary (both should have same y/x)
        result = xr.merge([wind_prefixed, land_prefixed], compat="override")

    else:
        raise ValueError(
            f"Invalid domain {domain!r}. Expected 'wind', 'landscape', or 'both'."
        )

    return result


def export_analysis_dataset_zarr(
    atlas,
    path: str | Path,
    *,
    domain: Literal["wind", "landscape", "both"] = "both",
    include_only: Sequence[str] | None = None,
    prefix: bool = True,
    exclude_template: bool = True,
    compute: bool = True,
) -> Path:
    """Export consolidated analysis dataset to a Zarr store.

    Creates a schema-versioned Zarr store with provenance tracking.
    This is the implementation for `Atlas.export_analysis_dataset_zarr()`.

    Parameters
    ----------
    atlas : Atlas
        Atlas instance with built wind/landscape stores.
    path : str or Path
        Output path for the Zarr store. Must end with '.zarr'.
    domain : {"wind", "landscape", "both"}, default "both"
        Which domain(s) to include in the export.
    include_only : sequence of str, optional
        If provided, only export these variables. For domain="both" with
        prefix=True, use prefixed names ("wind__varname", "landscape__varname").
    prefix : bool, default True
        When domain="both", prefix variables with "wind__" / "landscape__"
        to avoid collisions. Ignored for single-domain exports.
    exclude_template : bool, default True
        If True, exclude template variables from the export.
    compute : bool, default True
        If True, compute dask arrays before writing. If False, write lazily
        (may fail if underlying data has issues).

    Returns
    -------
    Path
        Path to the created Zarr store.

    Raises
    ------
    ValueError
        If path doesn't end with '.zarr', if stores not ready, or if
        variable selection fails.
    FileExistsError
        If the export store already exists.
    """
    out_path = Path(path)

    if not str(out_path).endswith(".zarr"):
        raise ValueError(f"Export path must end with '.zarr', got: {out_path}")

    if out_path.exists():
        raise FileExistsError(
            f"Export store already exists: {out_path}. Remove it first or use a different path."
        )

    # Ensure stores are ready
    if not getattr(atlas, "_canonical_ready", False):
        raise ValueError(
            "Atlas stores not ready. Call atlas.build() before exporting."
        )

    # Get wind and landscape datasets
    wind_ds = None
    landscape_ds = None

    if domain in ("wind", "both"):
        try:
            wind_ds = atlas.wind_data
        except (FileNotFoundError, RuntimeError) as e:
            raise ValueError(f"Cannot access wind data for export: {e}") from e

    if domain in ("landscape", "both"):
        try:
            landscape_ds = atlas.landscape_data
        except (FileNotFoundError, RuntimeError) as e:
            raise ValueError(f"Cannot access landscape data for export: {e}") from e

    # Build export dataset
    export_ds = build_analysis_export_dataset(
        wind_ds,
        landscape_ds,
        domain=domain,
        include_only=include_only,
        prefix=prefix,
        exclude_template=exclude_template,
    )

    # Build export spec for attrs
    export_spec = _build_export_spec(
        domain=domain,
        include_only=include_only,
        prefix=prefix,
        exclude_template=exclude_template,
    )

    # Build upstream provenance
    upstream = _extract_upstream_provenance(wind_ds, landscape_ds)

    # Note: Pre-write dataset validation is not performed here because
    # export-kind validation requires store_state/schema_version/created_at
    # attrs which are added after the zarr write. Post-write validate_store()
    # handles full validation.

    # Compute if requested
    if compute:
        backend = getattr(atlas, "compute_backend", "serial")
        workers = getattr(atlas, "compute_workers", None)
        export_ds = dask_compute(export_ds, backend=backend, num_workers=workers)

    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write atomically
    with atomic_dir(out_path) as tmp:
        export_ds.to_zarr(tmp, mode="w", consolidated=False)

        # Add required export attrs
        g = zarr.open_group(tmp, mode="a")
        g.attrs["store_state"] = "complete"
        g.attrs["schema_version"] = ANALYSIS_EXPORT_SCHEMA_VERSION
        g.attrs["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        g.attrs["cleo:package_version"] = _get_package_version()
        g.attrs["export_spec_json"] = json.dumps(
            export_spec,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        if upstream:
            g.attrs["upstream_provenance_json"] = json.dumps(
                upstream,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )

    # Validate store after write
    validate_store(out_path, kind="export")

    logger.info(
        "Exported analysis dataset to %s (domain=%s, vars=%d)",
        out_path,
        domain,
        len(export_ds.data_vars),
    )

    return out_path
