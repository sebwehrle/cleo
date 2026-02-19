# %% imports
import json
import zarr
import numpy as np
import xarray as xr


# %% MetricResult wrapper for compute(...).cache() pattern
class MetricResult:
    """
    Wrapper for computed metric results supporting .cache() pattern.

    Allows chaining: atlas.wind.compute(...).cache()
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

    def cache(self, *, overwrite: bool = True, allow_mode_change: bool = False) -> xr.DataArray:
        """
        Cache the metric into the active wind store and surface in atlas.wind.data.

        Per contract A8: writes the result into the derived region store for the
        current region selection (or base store if no region), and surfaces it
        immediately as atlas.wind.data[metric_name].

        Args:
            overwrite: If True (default), overwrite existing variable.
            allow_mode_change: If True, allow changing capacity_factors mode.

        Returns:
            The cached DataArray.

        Raises:
            ValueError: If variable exists and overwrite=False.
            ValueError: If capacity_factors mode would change without allow_mode_change.
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
