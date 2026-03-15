"""Domain facades for wind and landscape data access.

This module provides domain-specific facades that expose computed metrics and
data variables through a consistent interface. Each domain wraps an underlying
Zarr store and provides:

- Lazy data access via ``.data`` property (xarray Dataset)
- Metric computation via ``.compute(metric=..., **kwargs)``
- Result materialization into active stores
- Variable selection and filtering

WindDomain
----------
Handles wind resource metrics including:

- ``capacity_factors``: Turbine-specific capacity factors
- ``wind_speed``: Height-specific or rotor-equivalent wind speed
- ``lcoe``: Levelized cost of electricity
- ``min_lcoe_turbine``: Optimal turbine selection
- ``optimal_power``, ``optimal_energy``: Optimal turbine outputs

LandscapeDomain
---------------
Handles landscape data including:

- ``valid_mask``: Land validity mask
- ``elevation``: Terrain elevation
- User-added rasters via ``.add(...)``
- Vector rasterization via ``.rasterize(...)``
- Distance transforms via ``.compute(metric="distance", ...)``

See Also
--------
cleo.atlas.Atlas : Main orchestration class
cleo.wind_metrics : Wind metric specifications and orchestration
cleo.results : Result wrapper and materialization
"""

# %% imports
import json

import numpy as np
import xarray as xr
from pathlib import Path

from cleo._turbine_validation import _normalize_turbine_ids, _validate_sequence_not_scalar
from cleo.results import DomainResult, normalize_metric_for_active_wind_store
from cleo.spatial import distance_to_positive_mask
from cleo.wind_metrics import (
    _FLAT_CF_KWARGS,
    _FLAT_ECONOMICS_KWARGS,
    _REQUIRED_ECONOMICS_FIELDS,
    _WIND_METRICS,
    resolve_cf_spec,
    resolved_wind_output_name,
)
from cleo.unification.materializers._landscape_api import (
    materialize_landscape_computed_variables,
    register_landscape_dataarray_source,
    materialize_landscape_variable,
    prepare_landscape_variable_data,
    register_landscape_source,
    register_landscape_vector_source,
)
from cleo.unification.store_io import (
    ActiveStoreTurbineAxis,
    active_store_turbine_axis,
    open_zarr_dataset,
    resolve_active_landscape_store_path,
    turbine_ids_from_json,
)
from cleo.validation import validate_dataset, ValidationError


def _validate_turbine_index_list(
    indices: list[object],
    available: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate turbine indices and convert to turbine IDs.

    :param indices: List of integer indices to validate.
    :param available: Available turbine IDs.
    :returns: Tuple of turbine IDs corresponding to validated indices.
    :raises ValueError: If any index is not an int, out of range, or duplicate.
    """
    n = len(available)
    selected: list[str] = []
    seen: set[int] = set()
    for idx in indices:
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise ValueError(f"Each turbine index must be an integer, got {type(idx).__name__}")
        if idx < 0 or idx >= n:
            raise ValueError(f"turbine index out of range: {idx}. Valid range is [0, {n - 1}]")
        if idx in seen:
            raise ValueError(f"Duplicate turbine index: {idx}")
        seen.add(idx)
        selected.append(available[idx])
    return tuple(selected)


_DISTANCE_SPEC_ALGO = "edt"
_DISTANCE_SPEC_ALGO_VERSION = "1"
_DISTANCE_SPEC_RULE = "isfinite_and_gt_zero"


def _distance_spec_json(source_var: str) -> str:
    """Build canonical distance spec JSON for attrs/noop checks."""
    payload = {
        "algo": _DISTANCE_SPEC_ALGO,
        "algo_version": _DISTANCE_SPEC_ALGO_VERSION,
        "rule": _DISTANCE_SPEC_RULE,
        "source_var": source_var,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _cf_spec_matches(
    existing_cf: xr.DataArray,
    requested_spec: dict,
    turbines: tuple[str, ...],
) -> bool:
    """Check if existing CF matches requested spec exactly.

    Used for CF reuse in LCOE-family metrics. Only materialized store CF
    is checked; exact spec match is required.

    Args:
        existing_cf: Existing capacity factors DataArray from store.
        requested_spec: Resolved CF spec dict (method, interpolation, air_density, rews_n, loss_factor).
        turbines: Requested turbine IDs tuple.

    Returns:
        True if existing CF can be reused, False otherwise.
    """
    if not _cf_spec_is_compatible(existing_cf, requested_spec):
        return False

    stored_turbines = tuple(json.loads(existing_cf.attrs["cleo:turbines_json"]))
    return stored_turbines == turbines


def _cf_spec_is_compatible(existing_cf: xr.DataArray, requested_spec: dict) -> bool:
    """Check whether an existing CF array matches requested physics settings.

    This helper intentionally ignores turbine breadth and ordering so callers
    can decide separately whether a requested turbine subset is extractable from
    the existing CF array. It still requires turbine provenance metadata to be
    present so reuse never proceeds on an untracked CF artifact.

    :param existing_cf: Existing capacity-factors array from the active store.
    :type existing_cf: xarray.DataArray
    :param requested_spec: Resolved CF spec dict.
    :type requested_spec: dict
    :returns: ``True`` when the stored CF metadata is compatible with the
        requested CF settings.
    :rtype: bool
    """
    # Required metadata keys
    required_keys = [
        "cleo:cf_method",
        "cleo:air_density",
        "cleo:rews_n",
        "cleo:loss_factor",
        # Required as a provenance guard; the turbine list value is not compared
        # here because subset eligibility is evaluated separately.
        "cleo:turbines_json",
    ]

    # Check all metadata present
    for key in required_keys:
        if key not in existing_cf.attrs:
            return False

    # Check exact match of CF parameters
    if existing_cf.attrs["cleo:cf_method"] != requested_spec["method"]:
        return False

    stored_interpolation = existing_cf.attrs.get("cleo:interpolation")
    if stored_interpolation is None:
        stored_method = existing_cf.attrs["cleo:cf_method"]
        if stored_method in {"hub_height_weibull", "hub_height_weibull_rews_scaled"}:
            stored_interpolation = "ak_logz"
        else:
            stored_interpolation = "mu_cv_loglog"
    if stored_interpolation != requested_spec["interpolation"]:
        return False

    # air_density is stored as int (0/1) for netCDF compat
    stored_air_density = bool(existing_cf.attrs["cleo:air_density"])
    if stored_air_density != requested_spec["air_density"]:
        return False

    if (
        requested_spec["method"] != "hub_height_weibull"
        and existing_cf.attrs["cleo:rews_n"] != requested_spec["rews_n"]
    ):
        return False

    if existing_cf.attrs["cleo:loss_factor"] != requested_spec["loss_factor"]:
        return False

    return True


def _extract_requested_cf_subset(
    existing_cf: xr.DataArray,
    store_ds: xr.Dataset,
    requested_turbines: tuple[str, ...],
) -> xr.DataArray | None:
    """Extract a requested turbine subset from stored full-axis CF data.

    The returned array is a transient compute input for economics metrics. It
    therefore advertises only the requested turbine subset even when the stored
    CF variable itself is normalized to the full active-store axis.

    :param existing_cf: Stored capacity-factors array from the active wind
        store.
    :type existing_cf: xarray.DataArray
    :param store_ds: Active wind-store dataset used to interpret the store axis.
    :type store_ds: xarray.Dataset
    :param requested_turbines: Requested turbine IDs in caller-defined order.
    :type requested_turbines: tuple[str, ...]
    :returns: Subset-scoped transient CF array, or ``None`` when subset reuse is
        not possible.
    :rtype: xarray.DataArray | None
    """
    if "turbine" not in existing_cf.dims:
        return None
    if not _requested_turbines_were_computed(existing_cf, requested_turbines):
        return None

    subset = _select_requested_cf_subset(existing_cf, store_ds, requested_turbines)
    if subset is None:
        return None
    return _finalize_requested_cf_subset(subset, requested_turbines)


def _requested_turbines_were_computed(existing_cf: xr.DataArray, requested_turbines: tuple[str, ...]) -> bool:
    """Return whether the requested turbines are backed by computed CF slices.

    New stores record computed-slice provenance in
    ``cleo:computed_turbines_json`` before full-axis alignment rewrites the
    advertised turbine list. Older stores lack that attr, so reuse falls back to
    exact turbine-list matching only.

    :param existing_cf: Stored capacity-factors array candidate for reuse.
    :type existing_cf: xarray.DataArray
    :param requested_turbines: Requested turbine IDs.
    :type requested_turbines: tuple[str, ...]
    :returns: ``True`` when the requested turbines are safe to reuse.
    :rtype: bool
    """
    computed_turbines_json = existing_cf.attrs.get("cleo:computed_turbines_json")
    if isinstance(computed_turbines_json, str) and computed_turbines_json:
        try:
            computed_turbines = tuple(str(tid) for tid in json.loads(computed_turbines_json))
        except (TypeError, ValueError):
            return False
        return all(tid in computed_turbines for tid in requested_turbines)

    try:
        stored_turbines = tuple(str(tid) for tid in json.loads(existing_cf.attrs["cleo:turbines_json"]))
    except (KeyError, TypeError, ValueError):
        return False
    return stored_turbines == requested_turbines


def _select_requested_cf_subset(
    existing_cf: xr.DataArray,
    store_ds: xr.Dataset,
    requested_turbines: tuple[str, ...],
) -> xr.DataArray | None:
    """Select requested turbine slices from a stored CF array.

    :param existing_cf: Stored capacity-factors array from the active wind
        store.
    :type existing_cf: xarray.DataArray
    :param store_ds: Active wind-store dataset used to interpret the store axis.
    :type store_ds: xarray.Dataset
    :param requested_turbines: Requested turbine IDs in caller-defined order.
    :type requested_turbines: tuple[str, ...]
    :returns: Selected subset before transient coord/attr finalization, or
        ``None`` when selection fails.
    :rtype: xarray.DataArray | None
    """
    coord = existing_cf.coords.get("turbine")
    if coord is not None and coord.dims == ("turbine",):
        try:
            return existing_cf.sel(turbine=list(requested_turbines))
        except (KeyError, ValueError):
            pass

    try:
        axis = active_store_turbine_axis(store_ds)
    except RuntimeError:
        return None

    if any(tid not in axis.label_by_id for tid in requested_turbines):
        return None
    if coord is not None and coord.dims == ("turbine",):
        return _select_requested_cf_subset_by_label(existing_cf, axis, requested_turbines)
    return _select_requested_cf_subset_by_index(existing_cf, axis, requested_turbines)


def _select_requested_cf_subset_by_label(
    existing_cf: xr.DataArray,
    axis: ActiveStoreTurbineAxis,
    requested_turbines: tuple[str, ...],
) -> xr.DataArray | None:
    """Select requested turbines from a labeled store axis.

    :param existing_cf: Stored capacity-factors array from the active wind
        store.
    :type existing_cf: xarray.DataArray
    :param axis: Active-store turbine-axis metadata.
    :type axis: cleo.unification.store_io.ActiveStoreTurbineAxis
    :param requested_turbines: Requested turbine IDs in caller-defined order.
    :type requested_turbines: tuple[str, ...]
    :returns: Selected subset, or ``None`` when the selection fails.
    :rtype: xarray.DataArray | None
    """
    labels = [axis.label_by_id[tid] for tid in requested_turbines]
    try:
        return existing_cf.sel(turbine=labels)
    except (KeyError, ValueError):
        return None


def _select_requested_cf_subset_by_index(
    existing_cf: xr.DataArray,
    axis: ActiveStoreTurbineAxis,
    requested_turbines: tuple[str, ...],
) -> xr.DataArray | None:
    """Select requested turbines from a positional store axis.

    :param existing_cf: Stored capacity-factors array from the active wind
        store.
    :type existing_cf: xarray.DataArray
    :param axis: Active-store turbine-axis metadata.
    :type axis: cleo.unification.store_io.ActiveStoreTurbineAxis
    :param requested_turbines: Requested turbine IDs in caller-defined order.
    :type requested_turbines: tuple[str, ...]
    :returns: Selected subset, or ``None`` when the selection fails.
    :rtype: xarray.DataArray | None
    """
    indices = [axis.index_by_id[tid] for tid in requested_turbines]
    try:
        return existing_cf.isel(turbine=indices)
    except (IndexError, ValueError):
        return None


def _finalize_requested_cf_subset(subset: xr.DataArray, requested_turbines: tuple[str, ...]) -> xr.DataArray:
    """Finalize a transient subset-scoped CF array for economics reuse.

    :param subset: Selected subset of the stored capacity-factors array.
    :type subset: xarray.DataArray
    :param requested_turbines: Requested turbine IDs in caller-defined order.
    :type requested_turbines: tuple[str, ...]
    :returns: Subset with requested turbine coordinates and truthful transient
        turbine attrs.
    :rtype: xarray.DataArray
    """
    subset = subset.assign_coords(turbine=np.asarray(requested_turbines, dtype=object))
    attrs = subset.attrs.copy()
    attrs["cleo:turbines_json"] = json.dumps(list(requested_turbines), ensure_ascii=True)
    subset.attrs = attrs
    return subset


def _distance_spec_matches(da: xr.DataArray, expected_spec_json: str) -> bool:
    """Return True when DataArray carries the exact expected distance spec payload."""
    raw = da.attrs.get("cleo:distance_spec_json")
    if not isinstance(raw, str) or not raw:
        return False
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False
    normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return normalized == expected_spec_json


# =============================================================================
# Shared compute validation helpers
# =============================================================================


def _reject_inplace_kwarg(kwargs: dict, method_name: str) -> None:
    """Reject inplace kwarg in compute methods."""
    if "inplace" in kwargs:
        raise ValueError(
            f"{method_name}(...) does not accept inplace. "
            f"Use atlas.wind.{method_name}(...).materialize(...) to write into the active wind store."
        )


# =============================================================================
# WindDomain compute helpers
# =============================================================================


def _validate_wind_compute_kwargs(kwargs: dict, metric: str, spec: dict) -> None:
    """Validate public compute kwargs for a wind metric.

    :param kwargs: User-supplied compute kwargs for the requested metric.
    :type kwargs: dict
    :param metric: Public wind metric name.
    :type metric: str
    :param spec: Internal metric specification with ``required`` and ``allowed`` keys.
    :type spec: dict
    :raises ValueError: If ``compute(...)`` receives inplace, materialize-only,
        timebase, missing, or unknown parameters.
    """
    _reject_inplace_kwarg(kwargs, "compute")

    materialize_only = tuple(key for key in ("overwrite", "allow_method_change") if key in kwargs)
    if materialize_only:
        keys_text = ", ".join(repr(key) for key in materialize_only)
        raise ValueError(
            f"materialize-only parameter(s) {keys_text} were passed to compute(...); "
            "pass them to .materialize(...), e.g. "
            "atlas.wind.compute(...).materialize(overwrite=True, allow_method_change=True)."
        )

    timebase_kwargs = tuple(key for key in ("hours_per_year",) if key in kwargs)
    if timebase_kwargs:
        keys_text = ", ".join(repr(key) for key in timebase_kwargs)
        raise ValueError(
            f"Timebase parameter(s) {keys_text} cannot be passed to compute(...). "
            f"Use atlas.configure_timebase(hours_per_year=...) to set timebase assumptions."
        )

    # Check required kwargs
    required = spec["required"]
    missing = required - kwargs.keys()
    if missing:
        raise ValueError(f"Missing required parameters for {metric}: {sorted(missing)}")

    allowed = spec["allowed"]
    unknown = set(kwargs.keys()) - allowed
    if unknown:
        internal_params = {"hours_per_year"}
        user_allowed = sorted(allowed - internal_params)
        raise ValueError(f"Unknown parameter(s) for metric {metric!r}: {sorted(unknown)}. Allowed: {user_allowed}")


def _resolve_composed_metric_kwargs(
    kwargs: dict,
    metric: str,
    spec: dict,
    atlas,
) -> dict:
    """Resolve grouped cf={} and economics={} specs for composed metrics.

    Returns resolved_cf dict for CF reuse checking.
    """
    if not spec.get("composed", False):
        return {}

    # Reject flat CF/economics kwargs - must use grouped specs
    flat_cf_present = set(kwargs.keys()) & _FLAT_CF_KWARGS
    if flat_cf_present:
        raise ValueError(
            f"For {metric!r}, CF parameters must be passed via cf={{...}}, "
            f"not as flat kwargs. Found: {sorted(flat_cf_present)}. "
            f'Use: compute({metric!r}, cf={{"method": ..., "air_density": ...}}, ...)'
        )

    flat_econ_present = set(kwargs.keys()) & _FLAT_ECONOMICS_KWARGS
    if flat_econ_present:
        raise ValueError(
            f"For {metric!r}, economics parameters must be passed via economics={{...}}, "
            f"not as flat kwargs. Found: {sorted(flat_econ_present)}. "
            f'Use: compute({metric!r}, economics={{"discount_rate": ..., ...}}, ...)'
        )

    # Extract grouped specs
    cf_spec = kwargs.pop("cf", None)
    economics_override = kwargs.pop("economics", None)

    # Resolve CF spec with defaults
    resolved_cf = resolve_cf_spec(cf_spec)

    # Resolve economics via atlas baseline + per-call overrides
    effective_economics = atlas._effective_economics(economics_override)

    # Validate required economics fields
    missing_econ = _REQUIRED_ECONOMICS_FIELDS - set(effective_economics.keys())
    if missing_econ:
        raise ValueError(
            f"Missing required economics parameters for {metric!r}: {sorted(missing_econ)}. "
            f"Configure via atlas.configure_economics(...) or pass economics={{...}}."
        )

    # Unpack grouped specs into flat kwargs for underlying metric function
    kwargs.update(resolved_cf)
    kwargs.update(effective_economics)

    return resolved_cf


def _resolve_turbines_for_metric(
    kwargs: dict,
    domain: "WindDomain",
) -> None:
    """Resolve turbines for metrics that require them."""
    turbines = kwargs.get("turbines", None)
    if turbines is None:
        if domain.selected_turbines is not None:
            turbines = domain.selected_turbines
        else:
            raise ValueError("turbines required; use atlas.wind.select(...) or pass turbines=.")
    else:
        if len(turbines) == 0:
            raise ValueError("turbines must be non-empty; see atlas.wind.turbines")
        turbines = _normalize_turbine_ids(list(turbines), available=domain.turbines)
    kwargs["turbines"] = turbines


def _inject_economics_params_and_cf_reuse(
    kwargs: dict,
    metric: str,
    spec: dict,
    resolved_cf: dict,
    domain: "WindDomain",
) -> None:
    """Inject timebase and check for CF reuse in economics metrics."""
    _ECONOMICS_METRICS = {"lcoe", "min_lcoe_turbine", "optimal_power", "optimal_energy"}

    if metric in _ECONOMICS_METRICS:
        kwargs["hours_per_year"] = domain._atlas._effective_hours_per_year()

    # CF reuse check for LCOE-family metrics
    if spec.get("composed", False) and metric in _ECONOMICS_METRICS:
        turbines = kwargs["turbines"]
        store_ds = domain._store_data()
        if "capacity_factors" in store_ds.data_vars:
            candidate_cf = store_ds["capacity_factors"]
            if _cf_spec_is_compatible(candidate_cf, resolved_cf):
                reusable_cf = _extract_requested_cf_subset(candidate_cf, store_ds, turbines)
                if reusable_cf is not None:
                    kwargs["_precomputed_cf"] = reusable_cf


# =============================================================================
# LandscapeDomain compute helpers
# =============================================================================


def _validate_landscape_compute_kwargs(kwargs: dict, metric: str) -> None:
    """Validate kwargs for landscape compute."""
    _reject_inplace_kwarg(kwargs, "compute")

    allowed = {"source", "name", "if_exists"}
    unknown = sorted(set(kwargs.keys()) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) for landscape metric {metric!r}: {unknown!r}. Supported keys: {sorted(allowed)!r}"
        )

    if "source" not in kwargs:
        raise ValueError(f"Missing required parameters for {metric}: ['source']")


def _validate_distance_sources(sources: list[str], store_ds: xr.Dataset) -> None:
    """Validate that all distance sources exist in store."""
    for src in sources:
        if src not in store_ds.data_vars:
            raise ValueError(
                f"Unknown distance source variable {src!r}. "
                f"Distance sources must exist in active landscape store data vars: {sorted(store_ds.data_vars)!r}"
            )


def _check_distance_conflicts(
    names: list[str],
    if_exists: str,
    staged_overlays: dict,
    store_ds: xr.Dataset,
) -> None:
    """Check for conflicts when if_exists='error'."""
    if if_exists != "error":
        return
    conflicts = [nm for nm in names if nm in staged_overlays or nm in store_ds.data_vars]
    if conflicts:
        raise ValueError(
            f"distance compute would overwrite existing variable(s): {conflicts!r}. "
            "Use if_exists='replace' to overwrite or if_exists='noop' to skip."
        )


def _compute_single_distance(
    src: str,
    nm: str,
    if_exists: str,
    expected_spec: str,
    staged_overlays: dict,
    store_ds: xr.Dataset,
    valid_mask: xr.DataArray,
) -> tuple[xr.DataArray, bool]:
    """Compute distance for a single source, handling noop cases.

    Returns (data, should_stage) tuple. should_stage=False when using existing data.
    """
    staged_exists = nm in staged_overlays
    store_exists = nm in store_ds.data_vars

    if if_exists == "noop":
        if staged_exists:
            staged_da = staged_overlays[nm]
            if not _distance_spec_matches(staged_da, expected_spec):
                raise ValueError(
                    f"Variable {nm!r} already staged with different distance spec. "
                    "Use if_exists='replace' to overwrite."
                )
            return staged_da, False
        if store_exists:
            existing_da = store_ds[nm]
            if not _distance_spec_matches(existing_da, expected_spec):
                raise ValueError(
                    f"Variable {nm!r} already exists in active landscape store with different distance spec. "
                    "Use if_exists='replace' to overwrite."
                )
            return existing_da, False

    dist = distance_to_positive_mask(store_ds[src], valid_mask).rename(nm)
    dist = dist.reset_coords(drop=True)
    dist.attrs["cleo:metric"] = "distance"
    dist.attrs["cleo:distance_source"] = src
    dist.attrs["cleo:distance_spec_json"] = expected_spec
    return dist, True


class WindDomain:
    """
    Domain object for wind data access.

    Provides lazy, cached access to the active wind.zarr store.
    The .data property overlays transient computed metrics staged by compute().

    Turbine selection is persistent on the Atlas instance, not on this domain.
    """

    def __init__(self, atlas):
        """
        Initialize wind domain view for an Atlas instance.

        :param atlas: Owning Atlas instance.
        """
        self._atlas = atlas
        self._data = None
        self._computed_overlays: dict[str, xr.DataArray] = {}

    def _store_data(self) -> xr.Dataset:
        """Open/cache the active wind store dataset without computed overlays."""
        if self._data is not None:
            return self._data

        store_path = self._atlas._active_wind_store_path()

        if not store_path.exists():
            if self._atlas._area_name is not None:
                raise FileNotFoundError(
                    f"Area wind store missing at {store_path}; call atlas.build() after selecting area."
                )
            raise FileNotFoundError(f"Wind store missing at {store_path}; call atlas.build().")

        ds = open_zarr_dataset(store_path, chunk_policy=self._atlas.chunk_policy)
        # Centralized validation via cleo.validation
        try:
            validate_dataset(ds, kind="wind")
        except ValidationError as e:
            raise RuntimeError(f"Wind store validation failed; call atlas.build().\n{e}") from e
        self._data = self._apply_public_turbine_index(ds)
        return self._data

    @property
    def data(self) -> xr.Dataset:
        """
        Lazy access to the active wind zarr store as xr.Dataset.

        Routes to area store when area is selected, otherwise base store.
        Opens the store once and caches it. Validates store_state == "complete".
        Includes staged computed overlays (if any) from :meth:`compute`.

        :returns: Active wind store dataset.
        :raises FileNotFoundError: If wind store does not exist.
        :raises RuntimeError: If ``store_state`` is not ``"complete"``.
        """
        ds = self._store_data()
        if not self._computed_overlays:
            return ds
        return ds.assign(self._computed_overlays)

    @property
    def turbines(self) -> tuple[str, ...]:
        """
        Turbine IDs available in the wind store.

        :returns: Tuple of turbine IDs from ``cleo_turbines_json``.
        :raises RuntimeError: If no turbines are present in wind store metadata.
        """
        ds = self.data
        if "turbine" not in ds.dims:
            raise RuntimeError("No turbines in wind store; call Atlas.build() to add turbines.")
        # Read from cleo_turbines_json attr (avoids string arrays in Zarr v3)
        if "cleo_turbines_json" not in ds.attrs:
            raise RuntimeError("Wind store missing cleo_turbines_json attr; re-run build_canonical().")
        return turbine_ids_from_json(ds.attrs["cleo_turbines_json"])

    @property
    def selected_turbines(self) -> tuple[str, ...] | None:
        """
        Currently selected turbine IDs, or None if no selection (all turbines).

        Selection is persistent on the Atlas instance.

        :returns: Tuple of selected turbine IDs, or ``None`` for all turbines.
        """
        return self._atlas._wind_selected_turbines

    def _apply_public_turbine_index(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Make turbine selection user-facing by name, while keeping internal ids.

        After this:
          - ds.sel(turbine="Enercon.E101.3050") works
          - the old integer ids remain available as coord 'turbine_id'
          - integer-based selection remains possible via .isel(turbine=0)
        """
        if "turbine" not in ds.dims:
            return ds

        meta_json = ds.attrs.get("cleo_turbines_json")
        if not meta_json:
            # Keep current behavior; cannot label turbine axis without mapping
            return ds

        names = list(turbine_ids_from_json(meta_json))
        n = ds.sizes["turbine"]

        if len(names) != n:
            raise RuntimeError(
                f"Wind store turbine mapping mismatch: "
                f"len(cleo_turbines_json)={len(names)} != turbine dim size={n}. "
                f"Re-run atlas.build()."
            )
        if len(set(names)) != len(names):
            raise RuntimeError(
                "Wind store has duplicate turbine ids in cleo_turbines_json; cannot build a unique turbine index."
            )

        # Preserve current integer labels as turbine_id (best effort)
        if "turbine" in ds.coords and ds.coords["turbine"].dims == ("turbine",):
            turbine_id = ds.coords["turbine"].values
        else:
            # fallback: positional ids
            turbine_id = np.arange(n, dtype="int64")

        # Attach both: turbine_id (ints) and turbine (names as index labels)
        # Overwrite turbine coordinate labels to names (in-memory only)
        ds = ds.assign_coords(
            turbine_id=("turbine", turbine_id),
            turbine=("turbine", np.asarray(names, dtype=object)),
        )
        return ds

    def select(
        self,
        *,
        turbines: list[str] | tuple[str, ...] | None = None,
        turbine_indices: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """
        Set persistent turbine selection on the Atlas.

        Selection persists even if atlas.wind creates new WindDomain objects.
        Use clear_selection() to remove selection.

        Exactly one selection mode must be used:
        - ``turbines=[...]`` for turbine IDs.
        - ``turbine_indices=[...]`` for positional indices into ``atlas.wind.turbines``.

        :param turbines: Turbine IDs to select (non-empty list/tuple of strings).
        :param turbine_indices: Positional turbine indices into ``atlas.wind.turbines``
            (non-empty list/tuple of ints).
        :returns: ``None``.
        :raises ValueError: If selection args are invalid, ambiguous, out-of-range,
            duplicate, or unknown.
        """
        if (turbines is None) == (turbine_indices is None):
            raise ValueError("Provide exactly one of turbines=... or turbine_indices=....")

        available = self.turbines

        if turbines is not None:
            items = _validate_sequence_not_scalar(
                turbines,
                "turbines",
                "turbine IDs",
                empty_hint="use clear_selection() to clear",
            )
            validated = _normalize_turbine_ids(items, available=available)
        else:
            items = _validate_sequence_not_scalar(
                turbine_indices,
                "turbine_indices",
                "integers",
                empty_hint="use clear_selection() to clear",
            )
            validated = _validate_turbine_index_list(items, available)

        self._atlas._wind_selected_turbines = validated

    def clear_selection(self) -> None:
        """
        Clear persistent turbine selection.

        :returns: ``None``.
        """
        self._atlas._wind_selected_turbines = None
        return None

    def clear_computed(self) -> None:
        """
        Clear transient computed overlays from ``atlas.wind.data``.

        :returns: ``None``.
        """
        self._computed_overlays.clear()
        return None

    def convert_units(
        self,
        variable: str,
        to_unit: str,
        *,
        from_unit: str | None = None,
        inplace: bool = False,
    ) -> xr.DataArray | None:
        """
        Convert a wind variable to a different unit.

        Uses the canonical unit utilities from :mod:`cleo.units`. Conversion is
        dask-friendly (lazy arrays stay lazy). The result has the canonical
        ``units`` attr set to the target unit.

        :param variable: Variable name in ``atlas.wind.data``.
        :param to_unit: Target unit string (e.g., ``"EUR/kWh"``, ``"km"``).
        :param from_unit: Source unit. If ``None``, reads from the variable's
            ``units`` attr.
        :param inplace: If ``True``, stages the converted DataArray as a computed
            overlay so it appears in ``atlas.wind.data``. If ``False``, returns
            the converted DataArray without modifying state.

        :returns: Converted DataArray if ``inplace=False``, else ``None``.

        :raises ValueError: If variable not found in wind data.
        :raises ValueError: If no unit source available and ``from_unit`` not specified.
        :raises ValueError: If units are not dimensionally compatible.

        Example::

            # Convert LCOE from EUR/MWh to EUR/kWh (in-place)
            atlas.wind.convert_units("lcoe", "EUR/kWh", inplace=True)

            # Get converted DataArray without modifying store
            lcoe_kwh = atlas.wind.convert_units("lcoe", "EUR/kWh")
        """
        from cleo.units import convert_dataarray

        ds = self.data
        if variable not in ds.data_vars:
            raise ValueError(f"Variable {variable!r} not found in wind data. Available: {sorted(ds.data_vars)}")

        da = ds[variable]
        converted = convert_dataarray(da, to_unit, from_unit=from_unit)

        if inplace:
            self._computed_overlays[variable] = converted
            return None
        return converted

    def compute(self, metric: str, **kwargs) -> DomainResult:
        """
        Compute a wind metric from canonical data.

        Returns a DomainResult wrapper supporting .data/.materialize()/.persist() pattern.
        Also stages a lazy normalized overlay so the metric is visible immediately
        in ``atlas.wind.data[resolved_variable_name]`` before materialization.

        :param metric: Metric name (see supported metrics below).
        :param kwargs: Metric-specific parameters. For metrics requiring
            turbines, omitted ``turbines`` uses persistent selection from
            :meth:`select`.

        Supported metrics:
            - "wind_speed": public wind-speed surface.
              - ``method="height_weibull_mean"`` requires ``height``.
              - ``method="rotor_equivalent"`` requires turbines (from select() or kwarg).
            - "capacity_factors": requires turbines (from select() or kwarg).
              Optional: method (``"rotor_node_average"`` [default],
              ``"rotor_moment_matched_weibull"``, ``"hub_height_weibull"``,
              ``"hub_height_weibull_rews_scaled"``), interpolation, rews_n,
              air_density, loss_factor.
            - "lcoe": requires turbines. Uses grouped spec API:
              - ``cf={...}``: CF parameters (method, interpolation, air_density,
                loss_factor, rews_n). Defaults: method="rotor_node_average",
                interpolation="auto", air_density=False, rews_n=12, loss_factor=1.0.
              - ``economics={...}``: Economics parameters (discount_rate, lifetime_a,
                om_fixed_eur_per_kw_a, om_variable_eur_per_kwh, bos_cost_share).
                Required: discount_rate, lifetime_a, om_fixed_eur_per_kw_a, om_variable_eur_per_kwh.
                Optional: bos_cost_share (default 0.0 from _effective_economics()).
                Can be configured at Atlas level via atlas.configure_economics(...).
            - "min_lcoe_turbine": same params as lcoe. Returns the minimum-LCOE
              turbine index at each valid pixel; invalid pixels are masked.
              Turbine ID mapping in attrs["cleo:turbine_ids_json"].
            - "optimal_power": same params as lcoe. Returns rated power (kW)
              of the minimum-LCOE turbine at each pixel.
            - "optimal_energy": same params as lcoe. Returns annual energy (GWh/a)
              of the minimum-LCOE turbine at each pixel.

        Example (LCOE-family metrics)::

            atlas.configure_economics(discount_rate=0.05, lifetime_a=25)
            atlas.wind.compute(
                "lcoe",
                cf={"method": "hub_height_weibull", "air_density": True},
                economics={"om_fixed_eur_per_kw_a": 20, "om_variable_eur_per_kwh": 0.008},
            )

        :returns: :class:`cleo.results.DomainResult` with ``.data``/``.materialize()``/``.persist()``.
        :raises ValueError: If metric is unknown, params are missing, or turbine
            validation fails.
        :raises ValueError: If materialize-only kwargs (``overwrite`` / ``allow_method_change``)
            are passed to ``compute(...)`` instead of ``.materialize(...)``.
        :raises ValueError: If ``inplace`` is passed; ``compute(...)`` does not
            mutate stores directly.
        :raises ValueError: For LCOE-family metrics, if flat CF/economics kwargs
            are passed instead of grouped specs.
        """
        # Validate metric exists
        available_metrics = tuple(sorted(_WIND_METRICS))
        if metric not in _WIND_METRICS:
            raise ValueError(f"Unknown metric {metric!r}. Supported: {list(available_metrics)}")

        spec = _WIND_METRICS[metric]

        # Validate kwargs
        _validate_wind_compute_kwargs(kwargs, metric, spec)

        # Resolve composed metric specs (LCOE-family)
        resolved_cf = _resolve_composed_metric_kwargs(kwargs, metric, spec, self._atlas)

        # Resolve turbines if required
        wind_speed_method = kwargs.get("method")
        requires_turbines = spec["requires_turbines"] or (
            metric == "wind_speed" and wind_speed_method == "rotor_equivalent"
        )
        if requires_turbines:
            _resolve_turbines_for_metric(kwargs, self)

        # Inject economics params and check CF reuse
        _inject_economics_params_and_cf_reuse(kwargs, metric, spec, resolved_cf, self)

        # Prepare canonical inputs and compute
        wind = self._atlas.wind_data
        try:
            land = self._atlas.landscape_data
        except (FileNotFoundError, RuntimeError):
            land = None

        da = spec["fn"](wind, land, **kwargs)
        variable_name = resolved_wind_output_name(metric=metric, params=kwargs, data=da)
        staged = normalize_metric_for_active_wind_store(
            metric=metric,
            variable_name=variable_name,
            da=da,
            existing_ds=self._store_data(),
        )
        self._computed_overlays[variable_name] = staged

        return DomainResult(self, metric, da, dict(kwargs), variable_name=variable_name)


class LandscapeAddResult:
    """Result wrapper for staged landscape additions."""

    def __init__(
        self,
        domain: "LandscapeDomain",
        name: str,
        data: xr.DataArray,
        if_exists: str,
        *,
        noop_existing: bool = False,
    ):
        """
        Initialize staged landscape add result wrapper.

        :param domain: Owning landscape domain.
        :param name: Variable name being staged.
        :param data: Staged data candidate.
        :param if_exists: Conflict policy used for this operation.
        :param noop_existing: True when operation is a no-op over existing data.
        """
        self._domain = domain
        self._name = name
        self._data = data
        self._if_exists = if_exists
        self._noop_existing = noop_existing

    @property
    def data(self) -> xr.DataArray:
        """Staged candidate data."""
        return self._data

    def materialize(self, *, if_exists: str | None = None) -> xr.DataArray:
        """Materialize the staged variable into the active landscape store."""
        effective = self._if_exists if if_exists is None else if_exists
        return self._domain._materialize_staged(
            name=self._name,
            if_exists=effective,
            noop_existing=self._noop_existing,
        )


class LandscapeComputeBatchResult:
    """Result wrapper for staged landscape compute batches."""

    def __init__(
        self,
        domain: "LandscapeDomain",
        *,
        metric: str,
        names: tuple[str, ...],
        data: xr.Dataset,
        if_exists: str,
    ):
        """
        Initialize staged landscape compute batch result wrapper.

        :param domain: Owning landscape domain.
        :param metric: Metric name used for this batch.
        :param names: Output variable names in this batch.
        :param data: Staged compute dataset.
        :param if_exists: Conflict policy used for this operation.
        """
        self._domain = domain
        self._metric = metric
        self._names = names
        self._data = data
        self._if_exists = if_exists

    @property
    def data(self) -> xr.Dataset:
        """Staged compute dataset for this batch."""
        return self._data

    def materialize(self, *, if_exists: str | None = None) -> xr.Dataset:
        """Materialize staged computed variables into the active landscape store."""
        effective = self._if_exists if if_exists is None else if_exists
        return self._domain._materialize_staged_batch(
            names=self._names,
            if_exists=effective,
        )


class LandscapeDomain:
    """
    Domain object for landscape data access.

    Provides lazy, cached access to the active landscape store.
    The .data property overlays staged variables from
    add()/rasterize()/add_clc_category().
    """

    def __init__(self, atlas):
        """
        Initialize landscape domain view for an Atlas instance.

        :param atlas: Owning Atlas instance.
        """
        self._atlas = atlas
        self._data = None
        self._staged_overlays: dict[str, xr.DataArray] = {}
        self._staged_prepared_overlays: dict[str, xr.DataArray] = {}

    @staticmethod
    def _validate_if_exists(if_exists: str) -> None:
        """
        Validate landscape overwrite policy.

        :param if_exists: Conflict policy value.
        :returns: ``None``.
        :raises ValueError: If policy is not ``error``, ``replace``, or ``noop``.
        """
        valid_if_exists = {"error", "replace", "noop"}
        if if_exists not in valid_if_exists:
            raise ValueError(f"if_exists must be one of {sorted(valid_if_exists)!r}; got {if_exists!r}")

    def _store_data(self) -> xr.Dataset:
        """Open/cache the active landscape store dataset without staged overlays."""
        if self._data is not None:
            return self._data

        store_path = resolve_active_landscape_store_path(self._atlas)
        if not store_path.exists():
            if getattr(self._atlas, "_area_name", None) is not None:
                raise FileNotFoundError(
                    f"Area landscape store missing at {store_path}; call atlas.build() after selecting area."
                )
            raise FileNotFoundError(f"Landscape store missing at {store_path}; call atlas.build().")

        ds = open_zarr_dataset(store_path, chunk_policy=self._atlas.chunk_policy)
        # Centralized validation via cleo.validation
        try:
            validate_dataset(ds, kind="landscape")
        except ValidationError as e:
            raise RuntimeError(f"Landscape store validation failed; call atlas.build().\n{e}") from e

        self._data = ds
        return self._data

    @property
    def data(self) -> xr.Dataset:
        """
        Lazy access to the active landscape zarr store as xr.Dataset.

        Includes staged overlays from landscape add/rasterize paths, staged
        distance outputs, and in-place unit conversions.
        """
        ds = self._store_data()
        if not self._staged_overlays:
            return ds
        return ds.assign(self._staged_overlays)

    def _clear_staged_variable(self, name: str) -> None:
        """Clear all staged state for one landscape variable.

        :param name: Variable name to clear from staged state.
        :returns: ``None``.
        """
        self._staged_overlays.pop(name, None)
        self._staged_prepared_overlays.pop(name, None)

    def _stage_visible_overlay(
        self,
        name: str,
        visible: xr.DataArray,
        *,
        prepared: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """Stage visible and optional prepared overlay state together.

        ``atlas.landscape.data`` must stay merge-safe, so the visible staged
        overlay drops scalar coordinates such as ``spatial_ref`` while the
        optional prepared overlay keeps the richer raster metadata needed for
        exact materialization reuse.

        :param name: Variable name being staged.
        :param visible: DataArray surfaced in ``atlas.landscape.data``.
        :param prepared: Optional richer raster used for materialization reuse.
        :returns: The visible staged overlay stored under ``name``.
        :rtype: xarray.DataArray
        """
        self._staged_overlays[name] = visible
        if prepared is None:
            self._staged_prepared_overlays.pop(name, None)
        else:
            self._staged_prepared_overlays[name] = prepared
        return self._staged_overlays[name]

    def clear_staged(self) -> None:
        """
        Clear staged landscape overlays from ``atlas.landscape.data``.

        :returns: ``None``.
        """
        self._staged_overlays.clear()
        self._staged_prepared_overlays.clear()
        return None

    def convert_units(
        self,
        variable: str,
        to_unit: str,
        *,
        from_unit: str | None = None,
        inplace: bool = False,
    ) -> xr.DataArray | None:
        """
        Convert a landscape variable to a different unit.

        Uses the canonical unit utilities from :mod:`cleo.units`. Conversion is
        dask-friendly (lazy arrays stay lazy). The result has the canonical
        ``units`` attr set to the target unit.

        :param variable: Variable name in ``atlas.landscape.data``.
        :param to_unit: Target unit string (e.g., ``"km"``, ``"ft"``).
        :param from_unit: Source unit. If ``None``, reads from the variable's
            ``units`` attr.
        :param inplace: If ``True``, stages the converted DataArray as an
            overlay so it appears in ``atlas.landscape.data``. If ``False``,
            returns the converted DataArray without modifying state.

        :returns: Converted DataArray if ``inplace=False``, else ``None``.

        :raises ValueError: If variable not found in landscape data.
        :raises ValueError: If no unit source available and ``from_unit`` not specified.
        :raises ValueError: If units are not dimensionally compatible.

        Example::

            # Convert distance from m to km (in-place)
            atlas.landscape.convert_units("distance_roads", "km", inplace=True)

            # Get converted DataArray without modifying store
            dist_km = atlas.landscape.convert_units("distance_roads", "km")
        """
        from cleo.units import convert_dataarray

        ds = self.data
        if variable not in ds.data_vars:
            raise ValueError(f"Variable {variable!r} not found in landscape data. Available: {sorted(ds.data_vars)}")

        da = ds[variable]
        converted = convert_dataarray(da, to_unit, from_unit=from_unit)

        if inplace:
            self._stage_visible_overlay(variable, converted)
            return None
        return converted

    @staticmethod
    def _normalize_distance_sources_and_names(
        *,
        source,
        name,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Normalize and validate source/name inputs for distance compute batches.

        :param source: Source variable name or non-empty sequence of names.
        :param name: Optional output name or sequence of output names.
        :returns: Tuple ``(sources, names)`` as normalized non-empty string tuples.
        :raises ValueError: If inputs are empty, duplicated, mismatched, or wrong type.
        """
        if isinstance(source, str):
            sources = (source,)
        elif isinstance(source, (list, tuple)) and source:
            sources = tuple(source)
        else:
            raise ValueError("source must be a non-empty string or non-empty list/tuple of strings.")

        cleaned_sources: list[str] = []
        seen_sources: set[str] = set()
        for raw in sources:
            if not isinstance(raw, str):
                raise ValueError(f"Each source must be a string, got {type(raw).__name__}")
            src = raw.strip()
            if not src:
                raise ValueError("source entries cannot be empty or whitespace-only.")
            if src in seen_sources:
                raise ValueError(f"Duplicate source variable: {src!r}")
            seen_sources.add(src)
            cleaned_sources.append(src)

        if name is None:
            names = tuple(f"distance_{src}" for src in cleaned_sources)
        elif isinstance(name, str):
            if len(cleaned_sources) != 1:
                raise ValueError("name as a string is only allowed when source contains exactly one variable.")
            if not name.strip():
                raise ValueError("name cannot be empty or whitespace-only.")
            names = (name.strip(),)
        elif isinstance(name, (list, tuple)) and name:
            if len(name) != len(cleaned_sources):
                raise ValueError("name list/tuple length must match source length exactly.")
            cleaned_names: list[str] = []
            for raw in name:
                if not isinstance(raw, str):
                    raise ValueError(f"Each name must be a string, got {type(raw).__name__}")
                nm = raw.strip()
                if not nm:
                    raise ValueError("name entries cannot be empty or whitespace-only.")
                cleaned_names.append(nm)
            names = tuple(cleaned_names)
        else:
            raise ValueError("name must be None, a string, or a non-empty list/tuple of strings.")

        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate output variable names are not allowed: {list(names)!r}")

        return tuple(cleaned_sources), tuple(names)

    @staticmethod
    def _distance_spec_json(source_var: str) -> str:
        """
        Build canonical JSON payload for distance metric identity.

        :param source_var: Source variable name.
        :returns: Canonical JSON string payload.
        """
        return _distance_spec_json(source_var)

    @staticmethod
    def _distance_spec_matches(da: xr.DataArray, expected_spec_json: str) -> bool:
        """
        Compare staged/store variable distance identity to expected payload.

        :param da: Candidate distance data variable.
        :param expected_spec_json: Expected canonical JSON payload.
        :returns: ``True`` when payloads match exactly, else ``False``.
        """
        return _distance_spec_matches(da, expected_spec_json)

    def _materialize_staged(
        self,
        *,
        name: str,
        if_exists: str,
        noop_existing: bool = False,
    ) -> xr.DataArray:
        """
        Materialize one staged landscape variable into active store.

        :param name: Variable name to materialize.
        :param if_exists: Conflict policy.
        :param noop_existing: Internal marker for no-op materialization path.
        :returns: Materialized variable from active landscape data.
        """
        self._validate_if_exists(if_exists)
        if noop_existing and if_exists == "noop":
            self._clear_staged_variable(name)
            self._data = None
            return self.data[name]

        atlas = self._atlas
        if not getattr(atlas, "_canonical_ready", False):
            atlas.build_canonical()

        materialize_landscape_variable(
            atlas,
            name,
            chunk_policy=atlas.chunk_policy,
            fingerprint_method=getattr(atlas, "fingerprint_method", "path_mtime_size"),
            if_exists=if_exists,
            prepared_da=self._staged_prepared_overlays.get(name),
        )

        self._clear_staged_variable(name)
        self._data = None
        return self.data[name]

    def _materialize_staged_batch(
        self,
        *,
        names: tuple[str, ...],
        if_exists: str,
    ) -> xr.Dataset:
        """
        Materialize a batch of staged landscape variables into active store.

        :param names: Variable names to materialize.
        :param if_exists: Conflict policy.
        :returns: Dataset containing materialized variables.
        """
        self._validate_if_exists(if_exists)
        if not names:
            return xr.Dataset()

        atlas = self._atlas
        if not getattr(atlas, "_canonical_ready", False):
            atlas.build_canonical()

        staged_to_write = {name: self._staged_overlays[name] for name in names if name in self._staged_overlays}

        if staged_to_write:
            try:
                summary = materialize_landscape_computed_variables(
                    atlas,
                    variables=staged_to_write,
                    chunk_policy=atlas.chunk_policy,
                    if_exists=if_exists,
                )
            except RuntimeError as exc:
                for name in tuple(getattr(exc, "written", ())):
                    self._clear_staged_variable(name)
                self._data = None
                raise

            for name in summary.get("written", []):
                self._clear_staged_variable(name)
            for name in summary.get("skipped", []):
                self._clear_staged_variable(name)

        self._data = None
        return self.data[list(names)]

    def compute(self, metric: str, **kwargs) -> LandscapeComputeBatchResult:
        """Compute supported landscape metrics and stage results in ``atlas.landscape.data``.

        Supported metrics:
        - ``distance``: Euclidean distance in meters to nearest finite positive
          cell in one or more source variables.
        """
        if metric != "distance":
            raise ValueError(f"Unknown metric {metric!r}. Supported: {['distance']}")

        _validate_landscape_compute_kwargs(kwargs, metric)

        source = kwargs["source"]
        name = kwargs.get("name")
        if_exists = kwargs.get("if_exists", "error")
        self._validate_if_exists(if_exists)

        sources, names = self._normalize_distance_sources_and_names(source=source, name=name)
        store_ds = self._store_data()

        _validate_distance_sources(sources, store_ds)
        _check_distance_conflicts(names, if_exists, self._staged_overlays, store_ds)

        result_vars: dict[str, xr.DataArray] = {}
        valid_mask = store_ds["valid_mask"]

        for src, nm in zip(sources, names, strict=True):
            expected_spec = self._distance_spec_json(src)
            dist, should_stage = _compute_single_distance(
                src,
                nm,
                if_exists,
                expected_spec,
                self._staged_overlays,
                store_ds,
                valid_mask,
            )
            if should_stage:
                self._stage_visible_overlay(nm, dist)
            result_vars[nm] = dist

        out_ds = xr.Dataset({nm: result_vars[nm] for nm in names})
        return LandscapeComputeBatchResult(
            self,
            metric=metric,
            names=names,
            data=out_ds,
            if_exists=if_exists,
        )

    def _stage_registered_variable(
        self,
        *,
        name: str,
        if_exists: str,
        store_ds: xr.Dataset,
        staged_exists: bool,
        store_exists: bool,
        prepared_da: xr.DataArray | None = None,
    ) -> LandscapeAddResult:
        """Stage a prepared variable after source registration.

        :param name: Target variable name.
        :param if_exists: Conflict policy used for this operation.
        :param store_ds: Active landscape store dataset.
        :param staged_exists: Whether a staged overlay with the same name exists.
        :param store_exists: Whether the variable already exists in the active store.
        :param prepared_da: Optional prepared raster to stage directly.
        :returns: Operation wrapper for staged addition.
        :rtype: LandscapeAddResult
        """
        if if_exists == "noop":
            if staged_exists:
                return LandscapeAddResult(
                    self,
                    name,
                    self._staged_overlays[name],
                    if_exists,
                )
            if store_exists:
                return LandscapeAddResult(
                    self,
                    name,
                    store_ds[name],
                    if_exists,
                    noop_existing=True,
                )

        prepared = prepared_da
        if prepared is None:
            prepared = prepare_landscape_variable_data(
                self._atlas,
                name,
                chunk_policy=self._atlas.chunk_policy,
            )
        visible = prepared.reset_coords(drop=True)
        return LandscapeAddResult(
            self,
            name,
            self._stage_visible_overlay(name, visible, prepared=prepared),
            if_exists,
        )

    def _prepare_registration_context(
        self,
        *,
        name: str,
        if_exists: str,
    ) -> tuple[xr.Dataset, bool, bool]:
        """Prepare store and conflict context for staged landscape registration.

        :param name: Target variable name.
        :param if_exists: Conflict policy.
        :returns: Tuple ``(store_ds, staged_exists, store_exists)``.
        :rtype: tuple[xarray.Dataset, bool, bool]
        :raises ValueError: If ``if_exists="error"`` and the variable already
            exists in staged overlays or the active store.
        """
        atlas = self._atlas
        if not getattr(atlas, "_canonical_ready", False):
            atlas.build_canonical()

        store_ds = self._store_data()
        staged_exists = name in self._staged_overlays
        store_exists = name in store_ds.data_vars

        if if_exists == "error":
            if staged_exists:
                raise ValueError(
                    f"Variable {name!r} is already staged. "
                    "Use if_exists='replace' to overwrite or if_exists='noop' to keep current staged state."
                )
            if store_exists:
                raise ValueError(
                    f"Variable {name!r} already exists in landscape.zarr.\n"
                    "  Use if_exists='replace' to overwrite or if_exists='noop' to skip."
                )

        return store_ds, staged_exists, store_exists

    def add(
        self,
        name: str,
        source_path: str | Path,
        *,
        kind: str = "raster",
        params: dict | None = None,
        if_exists: str = "error",
    ) -> LandscapeAddResult:
        """
        Stage a raster landscape variable candidate.

        Vector sources are exposed via :meth:`rasterize`.

        :param name: Target variable name.
        :param source_path: Raster source path.
        :param kind: Source kind. Must be ``"raster"``.
        :param params: Optional source registration parameters.
        :param if_exists: Conflict policy (``error``, ``replace``, ``noop``).
        :returns: Operation wrapper for staged addition.
        :raises ValueError: If ``kind`` or ``if_exists`` is invalid, or conflicts are disallowed.
        """
        self._validate_if_exists(if_exists)
        if kind != "raster":
            raise ValueError(
                "add(...) only supports kind='raster'. Use atlas.landscape.rasterize(...) for vector sources."
            )

        atlas = self._atlas
        store_ds, staged_exists, store_exists = self._prepare_registration_context(
            name=name,
            if_exists=if_exists,
        )

        register_landscape_source(
            atlas,
            name=name,
            source_path=Path(source_path),
            kind=kind,
            params=params or {},
            if_exists=if_exists,
        )
        return self._stage_registered_variable(
            name=name,
            if_exists=if_exists,
            store_ds=store_ds,
            staged_exists=staged_exists,
            store_exists=store_exists,
        )

    def rasterize(
        self,
        shape,
        *,
        name: str,
        column: str | None = None,
        all_touched: bool = False,
        if_exists: str = "error",
    ) -> LandscapeAddResult:
        """
        Stage a vector-rasterized landscape variable.

        The input ``shape`` may be a path-like vector source or a GeoDataFrame.
        Rasterization aligns to the atlas wind/landscape grid.

        :param shape: Vector source path or GeoDataFrame.
        :param name: Target variable name.
        :param column: Optional column name used for rasterized values.
        :param all_touched: Rasterization inclusion policy.
        :param if_exists: Conflict policy (``error``, ``replace``, ``noop``).
        :returns: Operation wrapper for staged rasterization.
        :raises ValueError: If ``if_exists`` is invalid or conflicts are disallowed.
        """
        self._validate_if_exists(if_exists)

        atlas = self._atlas
        store_ds, staged_exists, store_exists = self._prepare_registration_context(
            name=name,
            if_exists=if_exists,
        )

        register_landscape_vector_source(
            atlas,
            name=name,
            shape=shape,
            column=column,
            all_touched=all_touched,
            if_exists=if_exists,
        )
        return self._stage_registered_variable(
            name=name,
            if_exists=if_exists,
            store_ds=store_ds,
            staged_exists=staged_exists,
            store_exists=store_exists,
        )

    def add_dataarray(
        self,
        name: str,
        data: xr.DataArray,
        *,
        categorical: bool = False,
        if_exists: str = "error",
    ) -> LandscapeAddResult:
        """
        Stage an in-memory raster candidate for the active landscape store.

        The input must be a single raster represented as an ``xarray.DataArray``.
        CLEO writes a canonical cached raster artifact under
        ``<atlas.path>/intermediates/raster_sources/`` during staging so later
        materialization can reuse the normal manifest-backed raster source path.

        :param name: Target variable name.
        :param data: In-memory raster candidate.
        :param categorical: Whether categorical resampling semantics should be
            used when alignment is required.
        :param if_exists: Conflict policy (``error``, ``replace``, ``noop``).
        :returns: Operation wrapper for staged addition.
        :rtype: LandscapeAddResult
        :raises TypeError: If ``data`` is not an ``xarray.DataArray``.
        :raises ValueError: If ``if_exists`` is invalid or conflicts are disallowed.
        :raises ValueError: If the raster cannot be aligned safely to the active grid.
        :raises RuntimeError: If required spatial metadata are missing.
        """
        self._validate_if_exists(if_exists)
        if not isinstance(data, xr.DataArray):
            raise TypeError(f"data must be an xarray.DataArray, got {type(data).__name__}.")

        store_ds, staged_exists, store_exists = self._prepare_registration_context(
            name=name,
            if_exists=if_exists,
        )
        changed, prepared = register_landscape_dataarray_source(
            self._atlas,
            name=name,
            data=data,
            categorical=categorical,
            if_exists=if_exists,
            chunk_policy=self._atlas.chunk_policy,
        )
        return self._stage_registered_variable(
            name=name,
            if_exists=if_exists,
            store_ds=store_ds,
            staged_exists=staged_exists,
            store_exists=store_exists,
            prepared_da=prepared if changed or if_exists != "noop" else None,
        )

    def add_clc_category(
        self,
        categories: str | int | list[int],
        *,
        name: str | None = None,
        source: str = "clc2018",
        if_exists: str = "error",
    ) -> LandscapeAddResult:
        """Stage a CLC-coded layer or CLC-derived category variable."""
        from cleo.clc import default_category_name

        atlas = self._atlas
        prepared_path = atlas.build_clc(source=source)

        if categories == "all":
            variable_name = name or "land_cover"
            params = {"categorical": True, "clc_source": source}
            return self.add(
                variable_name,
                prepared_path,
                kind="raster",
                params=params,
                if_exists=if_exists,
            )

        if isinstance(categories, int):
            codes = [int(categories)]
        elif isinstance(categories, list) and categories:
            codes = [int(c) for c in categories]
        else:
            raise ValueError("categories must be 'all', an int code, or a non-empty list of int codes.")

        if name is None:
            if len(codes) > 1:
                raise ValueError("name is required when adding multiple CLC codes in one variable.")
            inferred = default_category_name(atlas.path, codes[0])
            if inferred is None:
                raise ValueError(f"No default variable name known for CLC code {codes[0]!r}; pass name=...")
            variable_name = inferred
        else:
            variable_name = name

        params = {
            "categorical": True,
            "clc_source": source,
            "clc_codes": codes,
        }
        return self.add(
            variable_name,
            prepared_path,
            kind="raster",
            params=params,
            if_exists=if_exists,
        )
