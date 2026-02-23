# %% imports
import json
import datetime
import logging
import shutil
from pathlib import Path

import zarr
import numpy as np
import xarray as xr

from cleo.dask_utils import compute as dask_compute
from cleo.unification.store_io import turbine_ids_from_json

logger = logging.getLogger(__name__)


_STRING_COORDS_ATTR = "cleo_string_coords_json"


def _is_unsafe_zarr_v3_dtype(var: xr.DataArray) -> bool:
    """Return True for dtypes disallowed by the Zarr-v3 storage contract."""
    dtype = var.dtype
    if hasattr(dtype, "kind") and dtype.kind in {"U", "S"}:
        return True
    return dtype == object


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
            store_dt = datetime.datetime.fromisoformat(
                str(created_at).replace("Z", "+00:00")
            )
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
        raise FileExistsError(
            f"Result store already exists: {store_path}. "
            f"Use a different run_id or metric_name."
        )

    if isinstance(obj, xr.DataArray):
        da = obj
        if da.name is None:
            da = da.rename(metric_name)
        ds = da.to_dataset()
    else:
        ds = obj

    ds = _sanitize_result_dataset_for_zarr(ds)

    evaluator = getattr(atlas, "_evaluate_for_io", None)
    if callable(evaluator):
        ds = evaluator(ds)
    else:
        backend = getattr(atlas, "compute_backend", "serial")
        workers = getattr(atlas, "compute_workers", None)
        ds = dask_compute(ds, backend=backend, num_workers=workers)

    store_path.parent.mkdir(parents=True, exist_ok=True)

    with atomic_dir(store_path) as tmp:
        ds.to_zarr(tmp, mode="w", consolidated=False)

        g = zarr.open_group(tmp, mode="a")
        g.attrs["store_state"] = "complete"
        g.attrs["run_id"] = run_id
        g.attrs["metric_name"] = metric_name
        g.attrs["created_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        if params is not None:
            g.attrs["params_json"] = json.dumps(
                params, sort_keys=True, separators=(",", ":")
            )

        try:
            if getattr(atlas, "_canonical_ready", False):
                w = atlas.wind_zarr
                l = atlas.landscape_zarr
                g.attrs["wind_grid_id"] = w.attrs.get("grid_id", "")
                g.attrs["landscape_grid_id"] = l.attrs.get("grid_id", "")
        except (AttributeError, OSError, ValueError, TypeError, KeyError):
            logger.debug(
                "Failed to attach canonical provenance attrs to persisted result store.",
                extra={"run_id": run_id, "metric_name": metric_name},
                exc_info=True,
            )

    return store_path


def normalize_metric_for_active_wind_store(
    *,
    metric: str,
    da: xr.DataArray,
    existing_ds: xr.Dataset,
) -> xr.DataArray:
    """Normalize a computed metric to the active wind-store schema contract."""
    out = da.copy()

    if "turbine" in out.dims:
        meta_json = existing_ds.attrs.get("cleo_turbines_json")
        if not meta_json:
            raise RuntimeError(
                "Wind store missing cleo_turbines_json attr; cannot align turbine coordinates."
            )
        turbine_ids = turbine_ids_from_json(meta_json)
        turbine_id_to_idx = {tid: i for i, tid in enumerate(turbine_ids)}
        full_turbine_indices = list(range(len(turbine_ids)))
        if "turbine" in existing_ds.coords and existing_ds.coords["turbine"].dims == ("turbine",):
            full_turbine_labels = existing_ds.coords["turbine"].values.tolist()
        else:
            full_turbine_labels = full_turbine_indices

        if out.coords["turbine"].dtype.kind in ("U", "O", "S"):
            computed_ids = out.coords["turbine"].values.tolist()
            try:
                computed_indices = [turbine_id_to_idx[tid] for tid in computed_ids]
            except KeyError as exc:
                raise ValueError(
                    f"Computed metric includes unknown turbine id {exc.args[0]!r}."
                ) from exc
        else:
            computed_indices = out.coords["turbine"].values.tolist()

        computed_labels = [full_turbine_labels[i] for i in computed_indices]
        out = out.assign_coords(turbine=computed_labels)
        out = out.reindex(turbine=full_turbine_labels, fill_value=np.nan)

    existing_dims = set(existing_ds.sizes.keys())
    coords_to_drop = [
        coord_name
        for coord_name in out.coords
        if coord_name in existing_dims and coord_name not in out.dims
    ]
    if coords_to_drop:
        out = out.drop_vars(coords_to_drop)

    out.name = metric
    return out


# %% DomainResult wrapper for compute(...).materialize()/persist() pattern
class DomainResult:
    """
    Wrapper for computed domain results supporting .materialize()/.persist() pattern.

    Allows chaining, e.g.:
    - ``atlas.wind.compute(...).materialize()``
    - ``atlas.wind.compute(...).persist(run_id=...)``
    """

    def __init__(self, domain: "WindDomain", metric: str, data: xr.DataArray, params: dict):
        self._domain = domain
        self._metric = metric
        self._data = data
        self._params = params

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
        name = metric_name if metric_name is not None else self._metric
        payload_params = self._params if params is None else params
        return persist_result(
            atlas,
            name,
            self._data,
            run_id=run_id,
            params=payload_params,
        )

    def materialize(self, *, overwrite: bool = True, allow_mode_change: bool = False) -> xr.DataArray:
        """
        Materialize the metric into the active wind store and surface in atlas.wind.data.

        Per contract A8: writes the result into the derived region store for the
        current region selection (or base store if no region), and surfaces it
        immediately as atlas.wind.data[metric_name].

        :param overwrite: If ``True`` (default), overwrite existing variable.
        :param allow_mode_change: If ``True``, allow changing ``capacity_factors`` mode.
        :returns: Cached DataArray.
        :raises ValueError: If variable exists and ``overwrite=False``.
        :raises ValueError: If ``capacity_factors`` mode would change without
            ``allow_mode_change=True``.
        """
        atlas = self._domain._atlas
        # Route to active store (region or base) per contract B1
        store_path = atlas._active_wind_store_path()

        # Open existing store to check and align
        existing_ds = xr.open_zarr(store_path, consolidated=False)

        if self._metric in existing_ds.data_vars and not overwrite:
            existing_ds.close()
            raise ValueError(
                f"Variable {self._metric!r} already exists in wind.zarr; "
                f"use overwrite=True to replace."
            )

        # Mode-guard for capacity_factors: prevent silent mode flips
        if self._metric == "capacity_factors" and self._metric in existing_ds.data_vars:
            existing_var = existing_ds[self._metric]
            old_mode = existing_var.attrs.get("cleo:cf_mode")
            new_mode = self._data.attrs.get("cleo:cf_mode")
            if old_mode is not None and old_mode != new_mode and not allow_mode_change:
                existing_ds.close()
                raise ValueError(
                    f"capacity_factors already materialized with cleo:cf_mode={old_mode!r}; "
                    f"requested {new_mode!r}; pass allow_mode_change=True (and overwrite=True) to replace."
                )

        da = normalize_metric_for_active_wind_store(
            metric=self._metric,
            da=self._data,
            existing_ds=existing_ds,
        )
        evaluator = getattr(atlas, "_evaluate_for_io", None)
        if callable(evaluator):
            da = evaluator(da)
        else:
            backend = getattr(atlas, "compute_backend", "serial")
            workers = getattr(atlas, "compute_workers", None)
            da = dask_compute(da, backend=backend, num_workers=workers)

        # Preserve existing store attributes before writing
        existing_attrs = dict(existing_ds.attrs)
        var_exists = self._metric in existing_ds.data_vars
        existing_ds.close()

        # If overwriting, delete the existing variable first to ensure full replacement
        # (mode="a" with zarr can lead to partial overwrites if shape/coords differ)
        if var_exists and overwrite:
            root = zarr.open_group(store_path, mode="a")
            if self._metric in root:
                del root[self._metric]

        # Write metric to wind.zarr (append mode to preserve existing vars)
        ds_to_write = xr.Dataset({self._metric: da})
        ds_to_write.to_zarr(
            store_path,
            mode="a",  # Append to existing store
            consolidated=False,
        )

        # Restore preserved attributes (to_zarr may overwrite them)
        root = zarr.open_group(store_path, mode="a")
        for key, val in existing_attrs.items():
            root.attrs[key] = val

        overlays = getattr(self._domain, "_computed_overlays", None)
        if isinstance(overlays, dict):
            overlays.pop(self._metric, None)

        # Invalidate cached data so .data reloads with store-backed variable
        self._domain._data = None

        # Return the surfaced store-backed variable so callers receive the exact
        # materialized representation (including any alignment/reindexing done for IO).
        return self._domain.data[self._metric]
