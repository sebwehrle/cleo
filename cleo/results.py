# %% imports
from __future__ import annotations
import json
import datetime
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import zarr
import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from cleo.domains import WindDomain

from cleo.dask_utils import compute as dask_compute
from cleo.store import single_writer_lock, zarr_store_lock_dir
from cleo.unification.store_io import active_store_turbine_axis, turbine_ids_from_json

logger = logging.getLogger(__name__)


_STRING_COORDS_ATTR = "cleo_string_coords_json"


def _is_unsafe_zarr_v3_dtype(var: xr.DataArray) -> bool:
    """Return True for dtypes disallowed by the Zarr-v3 storage contract."""
    dtype = var.dtype
    if hasattr(dtype, "kind") and dtype.kind in {"U", "S"}:
        return True
    return np.dtype(dtype).kind == "O"


def _normalize_text_value(value) -> str:
    """Normalize scalar values for JSON-safe text coordinate storage."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _sanitize_result_dataset_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    """Serialize string-like coordinates to attrs and enforce Zarr-v3-safe dtypes."""
    out = ds.copy()
    string_coords: dict[str, list[str]] = {}

    for coord_name, coord in list(out.coords.items()):
        if not _is_unsafe_zarr_v3_dtype(coord):
            continue
        if coord.ndim != 1:
            raise ValueError(
                f"Cannot persist string-like coord {coord_name!r} with ndim={coord.ndim}; "
                "only 1D coordinates are supported for serialization."
            )
        dim = coord.dims[0]
        values = [_normalize_text_value(v) for v in np.asarray(coord.values).tolist()]
        string_coords[coord_name] = values
        out = out.assign_coords({coord_name: (coord.dims, np.arange(coord.sizes[dim], dtype=np.int64))})

    unsafe_after: list[str] = []
    for name, coord in out.coords.items():
        if _is_unsafe_zarr_v3_dtype(coord):
            unsafe_after.append(f"coord:{name}")
    for name, var in out.data_vars.items():
        if _is_unsafe_zarr_v3_dtype(var):
            unsafe_after.append(f"data_var:{name}")

    if unsafe_after:
        raise ValueError(
            "Result dataset contains Zarr-v3-unsafe string/object arrays after sanitization: "
            f"{unsafe_after}. Persist numeric arrays only; store strings in attrs."
        )

    if string_coords:
        attrs = dict(out.attrs)
        attrs[_STRING_COORDS_ATTR] = json.dumps(
            string_coords,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        out = out.assign_attrs(attrs)

    return out


def restore_serialized_string_coords(ds: xr.Dataset) -> xr.Dataset:
    """Restore string coordinates serialized by `_sanitize_result_dataset_for_zarr`."""
    payload = ds.attrs.get(_STRING_COORDS_ATTR)
    if not isinstance(payload, str) or not payload:
        return ds

    try:
        mapping = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning(
            "Invalid %s payload; leaving dataset coordinates as-is.",
            _STRING_COORDS_ATTR,
        )
        return ds

    out = ds
    for coord_name, values in mapping.items():
        if coord_name not in out.coords:
            continue
        coord = out.coords[coord_name]
        if coord.ndim != 1:
            continue
        dim = coord.dims[0]
        if coord.sizes[dim] != len(values):
            logger.warning(
                "Serialized string coord length mismatch for %s: dim=%d, values=%d",
                coord_name,
                coord.sizes[dim],
                len(values),
            )
            continue
        out = out.assign_coords({coord_name: (coord.dims, np.asarray(values, dtype=object))})
    return out


def validate_result_path_token(value: str, *, field: str) -> str:
    """Validate a run/metric token used in result-store filesystem paths."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string, got {type(value).__name__}.")
    if value != value.strip() or not value:
        raise ValueError(f"{field} cannot be empty or have leading/trailing whitespace.")
    if "/" in value or "\\" in value:
        raise ValueError(f"{field} cannot contain path separators: {value!r}")
    if value in {".", ".."}:
        raise ValueError(f"{field} cannot be {value!r}.")
    if Path(value).is_absolute():
        raise ValueError(f"{field} must be a relative token, got absolute path: {value!r}")
    return value


def _resolve_atlas_io_policy(atlas) -> tuple[Any, str, int | None]:
    """Resolve the atlas I/O evaluation policy for lazy xarray objects.

    :param atlas: Atlas-like object exposing optional ``_evaluate_for_io``,
        ``compute_backend``, and ``compute_workers`` attributes.
    :returns: Tuple ``(evaluator, backend, workers)``.
    :rtype: tuple[typing.Any, str, int | None]
    """
    evaluator = getattr(atlas, "_evaluate_for_io", None)
    backend = getattr(atlas, "compute_backend", "serial")
    workers = getattr(atlas, "compute_workers", None)
    return evaluator, backend, workers


def _evaluate_for_io(
    obj: xr.DataArray | xr.Dataset,
    *,
    evaluator: Any,
    backend: str,
    workers: int | None,
) -> xr.DataArray | xr.Dataset:
    """Evaluate an xarray object for I/O using atlas-configured policy.

    :param obj: Xarray object to evaluate.
    :type obj: xarray.DataArray | xarray.Dataset
    :param evaluator: Optional atlas-provided eager-evaluation callable.
    :type evaluator: typing.Any
    :param backend: Compute backend name.
    :type backend: str
    :param workers: Optional worker count for the backend.
    :type workers: int | None
    :returns: Evaluated object.
    :rtype: xarray.DataArray | xarray.Dataset
    """
    if callable(evaluator):
        return evaluator(obj)
    return dask_compute(obj, backend=backend, num_workers=workers)


@dataclass(frozen=True)
class _ActiveWindTurbineAlignment:
    """Canonical turbine-alignment state for active wind-store materialization.

    :param turbine_ids: Ordered turbine IDs from ``cleo_turbines_json``.
    :type turbine_ids: tuple[str, ...]
    :param store_labels: Canonical turbine coordinate labels used by the active
        wind store.
    :type store_labels: tuple[typing.Any, ...]
    :param index_by_id: Mapping from turbine ID to its canonical store index.
    :type index_by_id: dict[str, int]
    """

    turbine_ids: tuple[str, ...]
    store_labels: tuple[Any, ...]
    index_by_id: dict[str, int]

    @classmethod
    def from_store(cls, existing_ds: xr.Dataset) -> "_ActiveWindTurbineAlignment":
        """Build turbine-alignment state from an active wind store.

        :param existing_ds: Active wind-store dataset used as schema source.
        :type existing_ds: xarray.Dataset
        :returns: Canonical alignment state for this store.
        :rtype: _ActiveWindTurbineAlignment
        :raises RuntimeError: If the store lacks ``cleo_turbines_json``.
        """
        axis = active_store_turbine_axis(existing_ds)
        return cls(
            turbine_ids=axis.turbine_ids,
            store_labels=axis.store_labels,
            index_by_id=axis.index_by_id,
        )

    def labels_for_coord(self, coord: xr.DataArray) -> tuple[Any, ...]:
        """Resolve computed turbine coord values onto canonical store labels.

        :param coord: Computed turbine coordinate.
        :type coord: xarray.DataArray
        :returns: Canonical turbine labels matching the active wind store.
        :rtype: tuple[typing.Any, ...]
        :raises ValueError: If a string turbine ID is unknown.
        """
        raw_values = tuple(coord.values.tolist())
        if coord.dtype.kind in ("U", "O", "S"):
            try:
                indices = tuple(self.index_by_id[tid] for tid in raw_values)
            except KeyError as exc:
                raise ValueError(f"Computed metric includes unknown turbine id {exc.args[0]!r}.") from exc
        else:
            indices = raw_values
        return tuple(self.store_labels[i] for i in indices)


def _build_result_store_attrs(
    atlas,
    *,
    run_id: str,
    metric_name: str,
    params: dict | None,
) -> dict[str, Any]:
    """Build root attrs for a persisted result store.

    :param atlas: Atlas owning the result store.
    :type atlas: typing.Any
    :param run_id: Result run identifier.
    :type run_id: str
    :param metric_name: Metric name persisted in the store.
    :type metric_name: str
    :param params: Optional parameter payload for ``params_json``.
    :type params: dict | None
    :returns: Root attrs to write to the result store.
    :rtype: dict[str, typing.Any]
    """
    attrs: dict[str, Any] = {
        "store_state": "complete",
        "run_id": run_id,
        "metric_name": metric_name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if params is not None:
        attrs["params_json"] = json.dumps(params, sort_keys=True, separators=(",", ":"))

    try:
        if getattr(atlas, "_canonical_ready", False):
            wind = atlas.wind_zarr
            land = atlas.landscape_zarr
            attrs["wind_grid_id"] = wind.attrs.get("grid_id", "")
            attrs["landscape_grid_id"] = land.attrs.get("grid_id", "")
    except (AttributeError, OSError, ValueError, TypeError, KeyError):
        logger.debug(
            "Failed to attach canonical provenance attrs to persisted result store.",
            extra={"run_id": run_id, "metric_name": metric_name},
            exc_info=True,
        )

    return attrs


def _align_turbine_axis(
    da: xr.DataArray,
    *,
    alignment: _ActiveWindTurbineAlignment,
) -> xr.DataArray:
    """Align a computed turbine axis onto the active wind-store schema.

    :param da: Computed metric array containing a ``turbine`` dimension.
    :type da: xarray.DataArray
    :param alignment: Canonical turbine-alignment state for the target store.
    :type alignment: _ActiveWindTurbineAlignment
    :returns: Turbine-aligned array reindexed to the full active-store axis.
    :rtype: xarray.DataArray
    """
    aligned_labels = alignment.labels_for_coord(da.coords["turbine"])
    out = da.assign_coords(turbine=list(aligned_labels))
    out = out.reindex(turbine=list(alignment.store_labels), fill_value=np.nan)

    attrs = out.attrs.copy()
    turbine_ids_json = json.dumps(list(alignment.turbine_ids), ensure_ascii=True)
    if "cleo:turbines_json" in attrs:
        attrs.setdefault("cleo:computed_turbines_json", attrs["cleo:turbines_json"])
        attrs["cleo:turbines_json"] = turbine_ids_json
    if "cleo:turbine_ids_json" in attrs:
        attrs["cleo:turbine_ids_json"] = turbine_ids_json
    out.attrs = attrs
    return out


def _remap_min_lcoe_turbine_to_store_axis(
    da: xr.DataArray,
    *,
    existing_ds: xr.Dataset,
) -> xr.DataArray:
    """Remap subset-relative ``min_lcoe_turbine`` indices onto the store axis.

    :param da: Computed ``min_lcoe_turbine`` array with subset-relative indices.
    :type da: xarray.DataArray
    :param existing_ds: Active wind-store dataset providing the canonical turbine
        ordering.
    :type existing_ds: xarray.Dataset
    :returns: Index field aligned to the active-store turbine axis.
    :rtype: xarray.DataArray
    :raises RuntimeError: If the store lacks turbine metadata.
    :raises ValueError: If the computed turbine metadata is missing or invalid.
    """
    meta_json = da.attrs.get("cleo:turbine_ids_json")
    if not meta_json:
        raise ValueError("Computed min_lcoe_turbine is missing cleo:turbine_ids_json.")

    try:
        subset_ids = tuple(str(tid) for tid in json.loads(meta_json))
    except (TypeError, ValueError) as exc:
        raise ValueError("Computed min_lcoe_turbine has invalid cleo:turbine_ids_json.") from exc
    full_meta_json = existing_ds.attrs.get("cleo_turbines_json")
    if not full_meta_json:
        raise RuntimeError("Wind store missing cleo_turbines_json attr; cannot remap min_lcoe_turbine.")

    full_ids = turbine_ids_from_json(full_meta_json)
    full_index_by_id = {tid: i for i, tid in enumerate(full_ids)}
    try:
        index_map = np.asarray([full_index_by_id[tid] for tid in subset_ids], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"Computed min_lcoe_turbine includes unknown turbine id {exc.args[0]!r}.") from exc

    def _remap(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return out

        finite_values = arr[finite]
        idx = finite_values.astype(np.int64)
        if not np.allclose(finite_values, idx.astype(np.float64), rtol=0.0, atol=0.0):
            raise ValueError("Computed min_lcoe_turbine contains non-integer turbine indices.")
        if np.any(idx < 0) or np.any(idx >= len(index_map)):
            raise ValueError("Computed min_lcoe_turbine contains out-of-range turbine indices.")

        out[finite] = index_map[idx]
        return out

    out = xr.apply_ufunc(
        _remap,
        da,
        dask="parallelized",
        output_dtypes=[np.float64],
    ).rename(da.name)
    out.attrs = da.attrs.copy()
    out.attrs["cleo:turbine_ids_json"] = json.dumps(list(full_ids), ensure_ascii=True)
    return out


def result_run_dir_path(*, results_root: Path, run_id: str) -> Path:
    """Return validated run directory path under ``results_root``."""
    run_id_n = validate_result_path_token(run_id, field="run_id")
    root = Path(results_root).resolve()
    run_dir = (root / run_id_n).resolve()
    if root not in run_dir.parents and run_dir != root:
        raise ValueError(f"run_id resolves outside results_root: {run_id!r}")
    return run_dir


def result_store_path(*, results_root: Path, run_id: str, metric_name: str) -> Path:
    """Return validated result store path under ``results_root``."""
    metric_n = validate_result_path_token(metric_name, field="metric_name")
    run_dir = result_run_dir_path(results_root=results_root, run_id=run_id)
    store_path = (run_dir / f"{metric_n}.zarr").resolve()
    root = Path(results_root).resolve()
    if root not in store_path.parents and store_path != root:
        raise ValueError(
            f"result store path resolves outside results_root: run_id={run_id!r}, metric_name={metric_name!r}"
        )
    return store_path


def list_result_stores(
    results_root: Path,
    *,
    run_id: str | None = None,
    metric_name: str | None = None,
) -> list[Path]:
    """List candidate result stores under ``results_root``."""
    base = Path(results_root)
    metric_filter = None
    if metric_name is not None:
        metric_filter = validate_result_path_token(metric_name, field="metric_name")

    if run_id:
        run_dir = result_run_dir_path(results_root=base, run_id=run_id)
        runs = [run_dir] if run_dir.exists() and run_dir.is_dir() else []
    else:
        runs = sorted([p for p in base.iterdir() if p.is_dir()]) if base.exists() else []

    stores: list[Path] = []
    for run_dir in runs:
        for store in sorted(run_dir.glob("*.zarr")):
            if metric_filter is not None and store.name != f"{metric_filter}.zarr":
                continue
            stores.append(store)
    return stores


def read_result_store_datetime(store_path: Path) -> datetime.datetime:
    """Read result-store timestamp from attrs with mtime fallback."""
    store_dt: datetime.datetime | None = None
    try:
        g = zarr.open_group(store_path, mode="r")
        created_at = g.attrs.get("created_at")
        if created_at:
            store_dt = datetime.datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except (OSError, ValueError, TypeError, KeyError):
        logger.debug(
            "Falling back to mtime for result-store age check.",
            extra={"store": str(store_path)},
            exc_info=True,
        )

    if store_dt is None:
        store_dt = datetime.datetime.fromtimestamp(store_path.stat().st_mtime)
    return store_dt


def delete_result_store(path: Path) -> None:
    """Delete a result store directory."""
    shutil.rmtree(path)


def prune_empty_run_dirs(results_root: Path) -> int:
    """Remove empty run directories under ``results_root``."""
    base = Path(results_root)
    if not base.exists():
        return 0
    removed = 0
    for run_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        if not any(run_dir.iterdir()):
            run_dir.rmdir()
            removed += 1
    return removed


def persist_result(
    atlas,
    metric_name: str,
    obj,
    *,
    run_id: str | None = None,
    params: dict | None = None,
) -> Path:
    """Persist a result dataset/dataarray to a Zarr store.

    Creates an atomic Zarr store under ``<results_root>/<run_id>/<metric_name>.zarr``.
    This helper is the canonical persistence path used by fluent result APIs.
    """
    from cleo.store import atomic_dir

    if run_id is None:
        run_id = atlas.new_run_id()

    store_path = result_store_path(
        results_root=Path(atlas.results_root),
        run_id=run_id,
        metric_name=metric_name,
    )
    if store_path.exists():
        raise FileExistsError(f"Result store already exists: {store_path}. Use a different run_id or metric_name.")

    if isinstance(obj, xr.DataArray):
        da = obj
        if da.name is None:
            da = da.rename(metric_name)
        ds = da.to_dataset()
    else:
        ds = obj

    ds = _sanitize_result_dataset_for_zarr(ds)

    evaluator, backend, workers = _resolve_atlas_io_policy(atlas)
    ds = _evaluate_for_io(ds, evaluator=evaluator, backend=backend, workers=workers)

    store_path.parent.mkdir(parents=True, exist_ok=True)

    with atomic_dir(store_path) as tmp:
        ds.to_zarr(tmp, mode="w", consolidated=False)

        g = zarr.open_group(tmp, mode="a")
        for key, value in _build_result_store_attrs(
            atlas,
            run_id=run_id,
            metric_name=metric_name,
            params=params,
        ).items():
            g.attrs[key] = value

    return store_path


def normalize_metric_for_active_wind_store(
    *,
    metric: str,
    variable_name: str | None = None,
    da: xr.DataArray,
    existing_ds: xr.Dataset,
) -> xr.DataArray:
    """Normalize a computed metric to active wind-store schema expectations.

    :param metric: Metric variable name targeted for wind-store materialization.
    :type metric: str
    :param variable_name: Target data-variable name in the active wind store.
        Defaults to ``metric`` when omitted.
    :type variable_name: str | None
    :param da: Computed metric data array.
    :type da: xarray.DataArray
    :param existing_ds: Active wind-store dataset used as the schema reference.
    :type existing_ds: xarray.Dataset
    :returns: Normalized data array aligned to active-store coordinates.
    :rtype: xarray.DataArray
    :raises RuntimeError: If required schema coordinates are missing.
    :raises ValueError: If computed coordinates are incompatible with store
        coordinates.
    """
    target_name = metric if variable_name is None else variable_name
    out = da.copy()

    if "turbine" in out.dims:
        out = _align_turbine_axis(out, alignment=_ActiveWindTurbineAlignment.from_store(existing_ds))
    elif target_name == "min_lcoe_turbine":
        out = _remap_min_lcoe_turbine_to_store_axis(out, existing_ds=existing_ds)

    if target_name == "mean_wind_speed" and "height" in out.dims:
        if "height" not in out.dims:
            raise RuntimeError("mean_wind_speed materialization requires a 'height' dimension in computed output.")
        if "height" not in existing_ds.coords:
            raise RuntimeError("Wind store missing 'height' coordinate; cannot align mean_wind_speed.")
        full_heights = existing_ds.coords["height"].values.tolist()
        out_heights = out.coords["height"].values.tolist()
        missing_heights = [h for h in out_heights if h not in full_heights]
        if missing_heights:
            raise ValueError(
                f"Computed mean_wind_speed includes unsupported height(s) {missing_heights!r}; "
                f"available store heights: {sorted(full_heights)!r}."
            )
        out = out.reindex(height=full_heights, fill_value=np.nan)

    existing_dims = set(existing_ds.sizes.keys())
    coords_to_drop = [
        coord_name for coord_name in out.coords if coord_name in existing_dims and coord_name not in out.dims
    ]
    if coords_to_drop:
        out = out.drop_vars(coords_to_drop)

    out.name = target_name
    return out


# %% Materialize context and helpers


@dataclass
class _HeightAggregatedContext:
    """Context for height-aggregated metric materialization (mean_wind_speed).

    :param target_height_value: Matching height value from store coords.
    :param height_index: Index of target height in store coords.
    """

    target_height_value: Any
    height_index: int


def _validate_height_aggregated_preconditions(
    metric: str,
    params: dict[str, Any],
    existing_ds: xr.Dataset,
) -> _HeightAggregatedContext:
    """
    Validate preconditions for height-aggregated metrics (wind_speed).

    :param metric: Metric name.
    :param params: Compute parameters.
    :param existing_ds: Existing dataset (open).
    :returns: Validated height context.
    :raises RuntimeError: If height param or coord missing.
    :raises ValueError: If requested height not in store.
    """
    requested_height = params.get("height")
    if requested_height is None:
        raise RuntimeError(f"{metric} materialization requires params['height'] from compute(...).")

    if "height" not in existing_ds.coords:
        raise RuntimeError(f"Wind store missing 'height' coordinate; cannot materialize {metric}.")

    store_height_values = existing_ds.coords["height"].values.tolist()

    # Find matching height value
    target_height_value = None
    height_index = -1
    for i, h in enumerate(store_height_values):
        if h == requested_height:
            target_height_value = h
            height_index = i
            break

    if target_height_value is None:
        raise ValueError(
            f"Requested height={requested_height!r} not present in active wind store heights: "
            f"{sorted(store_height_values)!r}"
        )

    return _HeightAggregatedContext(
        target_height_value=target_height_value,
        height_index=height_index,
    )


def _validate_height_slice_overwrite(
    metric: str,
    existing_ds: xr.Dataset,
    height_ctx: _HeightAggregatedContext,
    overwrite: bool,
    evaluator: Any,
    backend: str,
    workers: int | None,
) -> None:
    """
    Validate overwrite for existing height slice data.

    :param metric: Metric name.
    :param existing_ds: Existing dataset with the metric.
    :param height_ctx: Height context.
    :param overwrite: Overwrite flag.
    :param evaluator: Optional evaluator callable.
    :param backend: Compute backend.
    :param workers: Worker count.
    :raises RuntimeError: If existing variable is legacy 2D.
    :raises ValueError: If height slice has data and overwrite=False.
    """
    existing_var = existing_ds[metric]

    if "height" not in existing_var.dims:
        raise RuntimeError(
            f"Existing {metric} in active wind store is legacy 2D (missing 'height'). "
            "Delete the variable or rebuild the store before materializing height-aware version."
        )

    existing_slice_any = existing_var.sel(height=height_ctx.target_height_value).notnull().any()
    existing_slice_any = _evaluate_for_io(
        existing_slice_any,
        evaluator=evaluator,
        backend=backend,
        workers=workers,
    )

    has_existing_height_data = bool(np.asarray(existing_slice_any).item())
    if has_existing_height_data and not overwrite:
        raise ValueError(
            f"{metric} at height={height_ctx.target_height_value!r} already exists in wind.zarr; "
            "use overwrite=True to replace that height slice."
        )


def _validate_cf_method_change(
    existing_ds: xr.Dataset,
    new_data: xr.DataArray,
    allow_method_change: bool,
) -> None:
    """
    Validate capacity_factors method change.

    :param existing_ds: Existing dataset with capacity_factors.
    :param new_data: New data being materialized.
    :param allow_method_change: Whether method change is allowed.
    :raises ValueError: If method would change without permission.
    """
    existing_var = existing_ds["capacity_factors"]
    old_method = existing_var.attrs.get("cleo:cf_method")
    new_method = new_data.attrs.get("cleo:cf_method")

    if old_method is not None and old_method != new_method and not allow_method_change:
        raise ValueError(
            f"capacity_factors already materialized with cleo:cf_method={old_method!r}; "
            f"requested {new_method!r}; pass allow_method_change=True (and overwrite=True) to replace."
        )


def _write_height_slice_to_store(
    store_path: Path,
    metric: str,
    da: xr.DataArray,
    height_ctx: _HeightAggregatedContext,
    store_sizes: dict[str, int],
) -> None:
    """
    Write a single height slice using region-based windowed Zarr write.

    :param store_path: Path to zarr store.
    :param metric: Metric name.
    :param da: DataArray to write.
    :param height_ctx: Height context.
    :param store_sizes: Store dimension sizes.
    """
    ds_to_write = xr.Dataset({metric: da.sel(height=[height_ctx.target_height_value])})
    ds_to_write.to_zarr(
        store_path,
        mode="r+",
        consolidated=False,
        region={
            "height": slice(height_ctx.height_index, height_ctx.height_index + 1),
            "y": slice(0, int(store_sizes["y"])),
            "x": slice(0, int(store_sizes["x"])),
        },
    )


def _write_full_variable_to_store(
    store_path: Path,
    metric: str,
    da: xr.DataArray,
    var_exists: bool,
    overwrite: bool,
) -> None:
    """
    Write full variable, deleting existing if overwriting.

    :param store_path: Path to zarr store.
    :param metric: Metric name.
    :param da: DataArray to write.
    :param var_exists: Whether variable already exists.
    :param overwrite: Whether to overwrite.
    """
    # Delete existing variable first to ensure full replacement
    if var_exists and overwrite:
        root = zarr.open_group(store_path, mode="a")
        if metric in root:
            del root[metric]

    # Write metric to store (append mode to preserve existing vars)
    ds_to_write = xr.Dataset({metric: da})
    ds_to_write.to_zarr(store_path, mode="a", consolidated=False)


# %% DomainResult wrapper for compute(...).materialize()/persist() pattern
class DomainResult:
    """
    Wrapper for computed domain results supporting .materialize()/.persist() pattern.

    Allows chaining, e.g.:
    - ``atlas.wind.compute(...).materialize()``
    - ``atlas.wind.compute(...).persist(run_id=...)``
    """

    def __init__(
        self,
        domain: WindDomain,
        metric: str,
        data: xr.DataArray,
        params: dict,
        *,
        variable_name: str | None = None,
    ):
        """
        Initialize domain result wrapper.

        :param domain: Owning domain instance.
        :param metric: Metric name.
        :param data: Computed data array.
        :param params: Compute parameter payload used for this result.
        """
        self._domain = domain
        self._metric = metric
        self._data = data
        self._params = params
        self._variable_name = metric if variable_name is None else variable_name

    def __repr__(self) -> str:
        """Human-friendly REPL representation with next-step guidance."""
        overlays = getattr(self._domain, "_computed_overlays", None)
        is_staged = isinstance(overlays, dict) and self._variable_name in overlays
        state = "staged" if is_staged else "computed"

        method = None
        if self._metric == "capacity_factors":
            method = self._data.attrs.get("cleo:cf_method")

        target = f'atlas.wind.data["{self._variable_name}"]'
        header = f"DomainResult(metric={self._metric!r}, state={state!r}, target={target!r}"
        if method is not None:
            header += f", method={method!r}"
        header += ")"

        return (
            f"{header}\n"
            "  - Lazy data: .data\n"
            "  - Write to active wind store: .materialize(overwrite=True, allow_method_change=False)\n"
            "  - Persist as run artifact: .persist(run_id=None, metric_name=None)"
        )

    @property
    def data(self) -> xr.DataArray:
        """Access the computed DataArray (lazy)."""
        return self._data

    def persist(
        self,
        *,
        run_id: str | None = None,
        params: dict | None = None,
        metric_name: str | None = None,
    ) -> Path:
        """Persist this result as a standalone run artifact."""
        atlas = self._domain._atlas
        name = metric_name if metric_name is not None else self._variable_name
        payload_params = self._params if params is None else params
        return persist_result(
            atlas,
            name,
            self._data,
            run_id=run_id,
            params=payload_params,
        )

    def materialize(self, *, overwrite: bool = True, allow_method_change: bool = False) -> xr.DataArray:
        """
        Materialize the metric into the active wind store and surface in atlas.wind.data.

        Per contract A8: writes the result into the derived area store for the
        current area selection (or base store if no area), and surfaces it
        immediately as atlas.wind.data[metric_name].

        :param overwrite: If ``True`` (default), overwrite existing variable.
            For height-sliced ``wind_speed``, this applies per requested ``height`` slice.
        :param allow_method_change: If ``True``, allow changing the
            materialized ``capacity_factors`` method.
        :returns: Cached DataArray.
        :raises ValueError: If variable exists and ``overwrite=False``.
            For height-sliced ``wind_speed``, this is evaluated per requested ``height``.
        :raises ValueError: If ``capacity_factors`` method would change without
            ``allow_method_change=True``.
        :raises RuntimeError: If materializing height-sliced ``wind_speed`` into a legacy
            2D variable (missing ``height`` dimension) in an existing store.
        """
        atlas = self._domain._atlas
        store_path = atlas._active_wind_store_path()

        # Open store and capture state
        existing_ds = xr.open_zarr(store_path, consolidated=False)
        var_exists = self._variable_name in existing_ds.data_vars
        is_height_aggregated = self._variable_name == "mean_wind_speed" and "height" in self._data.dims

        # Get compute context from atlas
        evaluator, backend, workers = _resolve_atlas_io_policy(atlas)

        # Validate preconditions (may raise)
        height_ctx: _HeightAggregatedContext | None = None
        try:
            height_ctx = self._validate_preconditions(
                existing_ds,
                var_exists,
                is_height_aggregated,
                overwrite,
                allow_method_change,
                evaluator,
                backend,
                workers,
            )
        except Exception:
            existing_ds.close()
            raise

        # Normalize data for store
        da = normalize_metric_for_active_wind_store(
            metric=self._metric,
            variable_name=self._variable_name,
            da=self._data,
            existing_ds=existing_ds,
        )
        da = _evaluate_for_io(da, evaluator=evaluator, backend=backend, workers=workers)

        # Capture state before closing
        existing_attrs = dict(existing_ds.attrs)
        store_sizes = {k: int(v) for k, v in existing_ds.sizes.items()}
        existing_ds.close()

        # Write to store
        self._write_to_store(
            store_path,
            da,
            var_exists,
            overwrite,
            is_height_aggregated,
            height_ctx,
            existing_attrs,
            store_sizes,
        )

        # Cleanup and return
        self._clear_overlay()
        self._domain._data = None
        return self._domain.data[self._variable_name]

    def _validate_preconditions(
        self,
        existing_ds: xr.Dataset,
        var_exists: bool,
        is_height_aggregated: bool,
        overwrite: bool,
        allow_method_change: bool,
        evaluator: Any,
        backend: str,
        workers: int | None,
    ) -> _HeightAggregatedContext | None:
        """Validate all materialization preconditions."""
        height_ctx: _HeightAggregatedContext | None = None

        if is_height_aggregated:
            height_ctx = _validate_height_aggregated_preconditions(self._metric, self._params, existing_ds)
            if var_exists:
                _validate_height_slice_overwrite(
                    self._variable_name,
                    existing_ds,
                    height_ctx,
                    overwrite,
                    evaluator,
                    backend,
                    workers,
                )
        elif var_exists and not overwrite:
            raise ValueError(
                f"Variable {self._variable_name!r} already exists in wind.zarr; use overwrite=True to replace."
            )

        if self._variable_name == "capacity_factors" and var_exists:
            _validate_cf_method_change(existing_ds, self._data, allow_method_change)

        return height_ctx

    def _write_to_store(
        self,
        store_path: Path,
        da: xr.DataArray,
        var_exists: bool,
        overwrite: bool,
        is_height_aggregated: bool,
        height_ctx: _HeightAggregatedContext | None,
        existing_attrs: dict[str, Any],
        store_sizes: dict[str, int],
    ) -> None:
        """Write metric to store with appropriate strategy."""
        with single_writer_lock(zarr_store_lock_dir(store_path)):
            if is_height_aggregated and var_exists and height_ctx is not None:
                _write_height_slice_to_store(store_path, self._variable_name, da, height_ctx, store_sizes)
            else:
                _write_full_variable_to_store(store_path, self._variable_name, da, var_exists, overwrite)

            # Restore preserved attributes
            root = zarr.open_group(store_path, mode="a")
            for key, val in existing_attrs.items():
                root.attrs[key] = val

    def _clear_overlay(self) -> None:
        """Clear computed overlay for this metric."""
        overlays = getattr(self._domain, "_computed_overlays", None)
        if isinstance(overlays, dict):
            overlays.pop(self._variable_name, None)
