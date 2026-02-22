# %% imports
import json
import datetime
import logging
from pathlib import Path

import zarr
import numpy as np
import xarray as xr

from cleo.dask_utils import compute as dask_compute

logger = logging.getLogger(__name__)


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


# %% DomainResult wrapper for compute(...).cache()/persist() pattern
class DomainResult:
    """
    Wrapper for computed domain results supporting .cache()/.persist() pattern.

    Allows chaining, e.g.:
    - ``atlas.wind.compute(...).cache()``
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

    def cache(self, *, overwrite: bool = True, allow_mode_change: bool = False) -> xr.DataArray:
        """
        Cache the metric into the active wind store and surface in atlas.wind.data.

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
                    f"capacity_factors already cached with cleo:cf_mode={old_mode!r}; "
                    f"requested {new_mode!r}; pass allow_mode_change=True (and overwrite=True) to replace."
                )

        # Get turbine metadata for ID to index mapping
        turbines_meta = json.loads(existing_ds.attrs["cleo_turbines_json"])
        turbine_id_to_idx = {t["id"]: i for i, t in enumerate(turbines_meta)}
        n_turbines = len(turbines_meta)
        full_turbine_indices = list(range(n_turbines))

        da = self._data.copy()
        evaluator = getattr(atlas, "_evaluate_for_io", None)
        if callable(evaluator):
            da = evaluator(da)
        else:
            backend = getattr(atlas, "compute_backend", "serial")
            workers = getattr(atlas, "compute_workers", None)
            da = dask_compute(da, backend=backend, num_workers=workers)

        # Handle turbine dimension: expand to full turbine set with NaN for uncomputed
        if "turbine" in da.dims:
            # Get computed turbine IDs/indices
            if da.coords["turbine"].dtype.kind in ("U", "O", "S"):
                # String turbine IDs - convert to indices
                computed_ids = da.coords["turbine"].values.tolist()
                computed_indices = [turbine_id_to_idx[tid] for tid in computed_ids]
            else:
                # Already integer indices
                computed_indices = da.coords["turbine"].values.tolist()

            # Create full-sized array with NaN for uncomputed turbines
            # and reindex to match existing store's turbine dimension
            da = da.assign_coords(turbine=computed_indices)
            da = da.reindex(turbine=full_turbine_indices, fill_value=np.nan)

        # Drop scalar/non-dimensional coordinates that conflict with existing dims
        # (e.g., capacity_factors may have height=100 as scalar coord, but wind.zarr
        # has height as a dimension with multiple values)
        existing_dims = set(existing_ds.sizes.keys())
        coords_to_drop = []
        for coord_name in da.coords:
            if coord_name in existing_dims and coord_name not in da.dims:
                # This coordinate exists as a dimension in existing store
                # but is a scalar/non-dim coord in da - must drop it
                coords_to_drop.append(coord_name)

        if coords_to_drop:
            da = da.drop_vars(coords_to_drop)

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

        # Invalidate cached data so .data reloads with new variable
        self._domain._data = None

        return self._data
