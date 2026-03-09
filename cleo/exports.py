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

import numpy as np
import xarray as xr
import zarr

from cleo.contracts import ANALYSIS_EXPORT_SCHEMA_VERSION
from cleo.dask_utils import compute as dask_compute
from cleo.store import atomic_dir
from cleo.validation import validate_store

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
                f"include_only contains unknown variables: {sorted(missing)}. Available: {sorted(ds.data_vars.keys())}"
            )
        vars_to_keep &= requested

    if not vars_to_keep:
        raise ValueError("No variables to export after filtering. Check include_only and exclude_template settings.")

    return ds[sorted(vars_to_keep)]


def _prefix_variables(ds: xr.Dataset, prefix: str) -> xr.Dataset:
    """Rename variables with a prefix."""
    rename_map = {var: f"{prefix}{var}" for var in ds.data_vars}
    return ds.rename(rename_map)


def _empty_dataset() -> xr.Dataset:
    """Return an empty dataset that does not affect downstream alignment."""
    return xr.Dataset()


def _assert_exact_shared_spatial_grid(wind_ds: xr.Dataset, landscape_ds: xr.Dataset) -> None:
    """Raise when both contributing export datasets do not share identical spatial coordinates.

    :param wind_ds: Wind dataset candidate for combined export.
    :type wind_ds: xarray.Dataset
    :param landscape_ds: Landscape dataset candidate for combined export.
    :type landscape_ds: xarray.Dataset
    :raises ValueError: If one dataset has a spatial coordinate that the other
        lacks, or if shared ``x``/``y`` coordinate values differ.
    """
    for dim in ("y", "x"):
        wind_has_dim = dim in wind_ds.coords
        landscape_has_dim = dim in landscape_ds.coords
        if wind_has_dim != landscape_has_dim:
            raise ValueError(f"Combined export requires both datasets to define {dim!r} coordinates.")
        if not wind_has_dim:
            continue
        if not np.array_equal(np.asarray(wind_ds.coords[dim].values), np.asarray(landscape_ds.coords[dim].values)):
            raise ValueError(
                f"Combined export requires identical {dim!r} coordinates between wind and landscape datasets."
            )


def _build_single_domain_export(
    ds: xr.Dataset | None,
    *,
    domain: Literal["wind", "landscape"],
    include_only: Sequence[str] | None,
    exclude_template: bool,
) -> xr.Dataset:
    """Filter one export domain and raise if the source dataset is missing.

    :param ds: Source dataset for the requested domain.
    :param domain: Domain label used in error reporting.
    :param include_only: Optional variable subset.
    :param exclude_template: Whether to exclude template variables.
    :returns: Filtered export dataset.
    :raises ValueError: If the required source dataset is missing.
    """
    if ds is None:
        raise ValueError(f"{domain}_ds required for domain={domain!r}")
    return _filter_variables(ds, include_only=include_only, exclude_template=exclude_template)


def _split_combined_include_only(
    include_only: Sequence[str] | None,
    *,
    prefix: bool,
) -> tuple[Sequence[str] | None, Sequence[str] | None]:
    """Split combined-export selections into wind and landscape subsets.

    :param include_only: Optional combined export selection.
    :param prefix: Whether combined exports require domain prefixes.
    :returns: Tuple ``(wind_include, landscape_include)``.
    :raises ValueError: If a prefixed combined selection includes an invalid name.
    """
    if include_only is None or not prefix:
        return include_only, include_only

    wind_vars: list[str] = []
    land_vars: list[str] = []
    for name in include_only:
        if name.startswith("wind__"):
            wind_vars.append(name[6:])
        elif name.startswith("landscape__"):
            land_vars.append(name[11:])
        else:
            raise ValueError(
                f"Variable {name!r} does not have expected prefix "
                f"('wind__' or 'landscape__') for domain='both' with prefix=True"
            )
    return wind_vars, land_vars


def _filter_combined_domain(
    ds: xr.Dataset,
    *,
    include_only: Sequence[str] | None,
    exclude_template: bool,
) -> xr.Dataset:
    """Filter one side of a combined export.

    :param ds: Source dataset for one export domain.
    :param include_only: Optional variable subset for that domain.
    :param exclude_template: Whether to exclude template variables.
    :returns: Filtered dataset or an empty dataset for an intentionally skipped domain.
    """
    if include_only is not None and len(include_only) == 0:
        return _empty_dataset()
    return _filter_variables(ds, include_only=include_only, exclude_template=exclude_template)


def _merge_combined_export_domains(
    wind_filtered: xr.Dataset,
    landscape_filtered: xr.Dataset,
    *,
    prefix: bool,
) -> xr.Dataset:
    """Merge filtered wind and landscape export datasets.

    :param wind_filtered: Filtered wind export dataset.
    :param landscape_filtered: Filtered landscape export dataset.
    :param prefix: Whether to prefix variables before merging.
    :returns: Merged export dataset.
    :raises ValueError: If both datasets contribute data on mismatched grids, if
        unprefixed names collide, or if the final export has no data variables.
    """
    if wind_filtered.data_vars and landscape_filtered.data_vars:
        _assert_exact_shared_spatial_grid(wind_filtered, landscape_filtered)

    if prefix:
        wind_export = _prefix_variables(wind_filtered, "wind__")
        landscape_export = _prefix_variables(landscape_filtered, "landscape__")
    else:
        wind_var_names = set(wind_filtered.data_vars.keys())
        landscape_var_names = set(landscape_filtered.data_vars.keys())
        overlap = wind_var_names & landscape_var_names
        if overlap:
            raise ValueError(
                f"Variable name collision when exporting domain='both' with prefix=False: "
                f"{sorted(overlap)}. Set prefix=True to avoid collisions."
            )
        wind_export = wind_filtered
        landscape_export = landscape_filtered

    result = xr.merge([wind_export, landscape_export], compat="override")
    if len(result.data_vars) == 0:
        raise ValueError("No variables to export after filtering. Check include_only and exclude_template settings.")
    return result


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
        when prefix=False, if include_only contains unknown variables, if no
        variables remain after filtering, or if both contributing wind and
        landscape exports do not share the same ``x``/``y`` grid.
    """
    if domain == "wind":
        result = _build_single_domain_export(
            wind_ds,
            domain="wind",
            include_only=include_only,
            exclude_template=exclude_template,
        )

    elif domain == "landscape":
        result = _build_single_domain_export(
            landscape_ds,
            domain="landscape",
            include_only=include_only,
            exclude_template=exclude_template,
        )

    elif domain == "both":
        if wind_ds is None:
            raise ValueError("wind_ds required for domain='both'")
        if landscape_ds is None:
            raise ValueError("landscape_ds required for domain='both'")
        wind_include, land_include = _split_combined_include_only(include_only, prefix=prefix)
        wind_filtered = _filter_combined_domain(
            wind_ds,
            include_only=wind_include,
            exclude_template=exclude_template,
        )
        land_filtered = _filter_combined_domain(
            landscape_ds,
            include_only=land_include,
            exclude_template=exclude_template,
        )
        result = _merge_combined_export_domains(wind_filtered, land_filtered, prefix=prefix)

    else:
        raise ValueError(f"Invalid domain {domain!r}. Expected 'wind', 'landscape', or 'both'.")

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
        If True, compute dask arrays before writing using the Atlas compute
        backend for a more controlled execution path. If False, skip that
        explicit precompute step and write synchronously via
        :meth:`xarray.Dataset.to_zarr`.

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
        raise FileExistsError(f"Export store already exists: {out_path}. Remove it first or use a different path.")

    # Ensure stores are ready
    if not getattr(atlas, "_canonical_ready", False):
        raise ValueError("Atlas stores not ready. Call atlas.build() before exporting.")

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

    # `compute=False` skips only the explicit dataset precompute step.
    # The export write itself remains a normal synchronous to_zarr() call.
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
