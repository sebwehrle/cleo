# %% imports
import os
import re
import json
import shutil
# import warnings
# import yaml
import pyproj
import logging
# import zipfile
import datetime
# import rasterio.crs
import zarr
from collections.abc import Sequence
from uuid import uuid4
import numpy as np
import xarray as xr
# import geopandas as gpd
import pycountry as pct
from pathlib import Path
# from xrspatial import proximity
# from rasterio.enums import MergeAlg
# from rasterio.features import rasterize as rio_rasterize

from cleo.class_helpers import (
    deploy_resources,
    set_attributes,
    setup_logging,
)

from cleo.utils import (
    add,
    flatten,
    convert,
    download_file,
)

# Import from cleo.unify
from cleo.unify import (
    Unifier,
    _read_vector_file,
)

from cleo.spatial import (
    clip_to_geometry, reproject, to_crs_if_needed, crs_equal,
)

from cleo.assess import (
    compute_lcoe,
    compute_mean_wind_speed,
    compute_optimal_power_energy,
    compute_wind_shear_coefficient,
    compute_weibull_pdf,
    mean_wind_speed_from_weibull,
    minimum_lcoe,
    simulate_capacity_factors,
    interpolate_weibull_params_to_height,
    weibull_probability_density,
    capacity_factor,
)

logger = logging.getLogger(__name__)

# %% repr helpers (side-effect-free, never raise)
def _safe_basename(path) -> str:
    """Return basename of path, or '?' on any error."""
    try:
        return Path(path).name if path else "?"
    except Exception:
        return "?"


def _fmt_grid(data) -> str:
    """Return 'YxX' grid size, or '?' on any error."""
    try:
        if data is None:
            return "?"
        return f"{data.sizes.get('y', '?')}x{data.sizes.get('x', '?')}"
    except Exception:
        return "?"


def _cap_list(items, max_items: int = 5, max_len: int = 60) -> str:
    """Format list as '[a,b,c,...]' with bounded length."""
    try:
        if not items:
            return "[]"
        items = list(items)[:max_items]
        suffix = ",..." if len(items) == max_items else ""
        s = "[" + ",".join(str(i) for i in items) + suffix + "]"
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s
    except Exception:
        return "[?]"


# %% Wind metric functions (canonical-only, no I/O)
def _wind_metric_mean_wind_speed(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    height: int,
    **_,
) -> xr.DataArray:
    """
    Compute mean wind speed from canonical wind store at specified height.

    Args:
        wind: Canonical wind dataset (must have weibull_A and weibull_k).
        land: Canonical landscape dataset (optional, for valid_mask).
        height: Height level to compute (must exist in wind coords).

    Returns:
        DataArray with mean wind speed (m/s).
    """
    # Get weibull params (canonical store uses weibull_A)
    var_A = "weibull_A" if "weibull_A" in wind else "weibull_a"
    var_k = "weibull_k"

    A = wind[var_A]
    k = wind[var_k]

    # Validate height exists
    if "height" not in A.coords:
        raise ValueError(f"No height dimension in wind store {var_A}")
    available = [int(h) for h in A.coords["height"].values]
    if height not in available:
        raise ValueError(
            f"height={height} not in wind store; available: {sorted(available)}"
        )

    # Compute mean wind speed
    da = mean_wind_speed_from_weibull(A=A.sel(height=height), k=k.sel(height=height))

    # Apply valid_mask if available
    if land is not None and "valid_mask" in land:
        da = da.where(land["valid_mask"])

    return da


def _wind_metric_capacity_factors(
    wind: xr.Dataset,
    land: xr.Dataset | None,
    *,
    turbines: tuple[str, ...],
    height: int = 100,
    air_density: bool = False,
    loss_factor: float = 1.0,
    **_,
) -> xr.DataArray:
    """
    Compute capacity factors from canonical wind store for selected turbines.

    Uses hub-height mode with interpolated Weibull parameters.

    Args:
        wind: Canonical wind dataset (must have weibull_A, weibull_k, power_curve).
        land: Canonical landscape dataset (must have valid_mask).
        turbines: Tuple of turbine IDs to compute.
        height: Reference height for Weibull interpolation (default 100).
        air_density: If True, apply air density correction using rho.
        loss_factor: Loss correction factor (default 1.0).

    Returns:
        DataArray with capacity factors, dims (turbine, y, x).
    """
    # Validate inputs
    if land is None or "valid_mask" not in land:
        raise ValueError("landscape store with valid_mask required for capacity_factors")

    var_A = "weibull_A" if "weibull_A" in wind else "weibull_a"
    var_k = "weibull_k"

    if var_A not in wind or var_k not in wind:
        raise ValueError(f"wind store must have {var_A} and {var_k}")
    if "power_curve" not in wind:
        raise ValueError("wind store must have power_curve variable")
    if "turbine" not in wind.coords:
        raise ValueError("wind store must have turbine coordinate")

    # Get Weibull params and validate height
    A = wind[var_A]
    k = wind[var_k]

    if "height" not in A.dims:
        raise ValueError(f"{var_A} must have height dimension")
    available = [int(h) for h in A.coords["height"].values]
    if height not in available:
        raise ValueError(
            f"height={height} not in wind store; available: {sorted(available)}"
        )

    # Get wind speed grid
    wind_speed = wind.coords["wind_speed"].values

    # Build turbine ID to index mapping from cleo_turbines_json attr
    if "cleo_turbines_json" not in wind.attrs:
        raise ValueError("wind store must have cleo_turbines_json attr")
    turbines_meta = json.loads(wind.attrs["cleo_turbines_json"])
    turbine_id_to_idx = {t["id"]: i for i, t in enumerate(turbines_meta)}

    # Validate turbines exist
    available_turbines = set(turbine_id_to_idx.keys())
    for tid in turbines:
        if tid not in available_turbines:
            raise ValueError(f"turbine {tid!r} not in wind store")

    # Compute capacity factor for each turbine
    cf_list = []
    for turbine_id in turbines:
        turbine_idx = turbine_id_to_idx[turbine_id]
        # Get hub height for this turbine
        hub_height = float(wind["turbine_hub_height"].isel(turbine=turbine_idx).values)

        # Interpolate Weibull params to hub height
        A_hub, k_hub = interpolate_weibull_params_to_height(A, k, hub_height)

        # Compute Weibull PDF at hub height
        pdf_hub = weibull_probability_density(wind_speed, k_hub, A_hub)

        # Get power curve
        p_power_curve = wind["power_curve"].isel(turbine=turbine_idx).values

        # Air density correction (if enabled)
        if air_density and "rho" in wind:
            from cleo.assess import RHO_0, _integrate_cf_with_density_correction

            # Interpolate rho to hub height (linear)
            rho = wind["rho"]
            if "height" in rho.dims:
                rho_hub = rho.interp(height=hub_height, method="linear")
            else:
                rho_hub = rho

            # Compute density correction factor
            c = (rho_hub / RHO_0) ** (1.0 / 3.0)

            cf = _integrate_cf_with_density_correction(
                pdf=pdf_hub,
                u_grid=wind_speed,
                p_curve=p_power_curve,
                c=c,
                loss_factor=loss_factor,
            )
        else:
            # Create dummy shear (zeros, same grid as pdf)
            template = pdf_hub.isel(wind_speed=0)
            dummy_shear = xr.zeros_like(template).rename("wind_shear")

            # Call capacity_factor
            cf = capacity_factor(
                weibull_pdf=pdf_hub,
                wind_shear=dummy_shear,
                u_power_curve=wind_speed,
                p_power_curve=p_power_curve,
                h_turbine=hub_height,
                h_reference=hub_height,  # No shear scaling needed
                correction_factor=loss_factor,
            )

        cf = cf.expand_dims(turbine=[turbine_id])
        cf_list.append(cf)

    # Concatenate along turbine dimension
    result = xr.concat(cf_list, dim="turbine")
    result = result.rename("capacity_factors")

    # Apply valid_mask
    result = result.where(land["valid_mask"])

    return result


# Wind metrics registry
_WIND_METRICS = {
    "mean_wind_speed": {
        "fn": _wind_metric_mean_wind_speed,
        "requires_turbines": False,
        "required": {"height"},
    },
    "capacity_factors": {
        "fn": _wind_metric_capacity_factors,
        "requires_turbines": True,
        "required": set(),  # height has default
    },
}


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

    def cache(self, *, overwrite: bool = True) -> xr.DataArray:
        """
        Cache the metric into the active wind store and surface in atlas.wind.data.

        Per contract A8: writes the result into the derived region store for the
        current region selection (or base store if no region), and surfaces it
        immediately as atlas.wind.data[metric_name].

        Args:
            overwrite: If True (default), overwrite existing variable.

        Returns:
            The cached DataArray.

        Raises:
            ValueError: If variable exists and overwrite=False.
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
                computed_ids = list(da.coords["turbine"].values)
                computed_indices = [turbine_id_to_idx[tid] for tid in computed_ids]
            else:
                # Already integer indices
                computed_indices = list(da.coords["turbine"].values)

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
        existing_ds.close()

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


# %% Domain objects for v1 API
class WindDomain:
    """
    Domain object for wind data access.

    Provides lazy, cached access to the canonical wind.zarr store.
    The .data property opens the store once and caches the result.

    Turbine selection is persistent on the Atlas instance, not on this domain.
    """

    def __init__(self, atlas):
        self._atlas = atlas
        self._data = None

    @property
    def data(self) -> xr.Dataset:
        """
        Lazy access to the active wind zarr store as xr.Dataset.

        Routes to region store when region is selected, otherwise base store.
        Opens the store once and caches it. Validates store_state == "complete".

        Raises:
            FileNotFoundError: If wind.zarr store does not exist.
            RuntimeError: If store_state != "complete".
        """
        if self._data is not None:
            return self._data

        # Route to active store (region or base) per contract B1
        store_path = self._atlas._active_wind_store_path()

        if not store_path.exists():
            # Provide helpful message based on whether region is selected
            if self._atlas._region is not None:
                raise FileNotFoundError(
                    f"Region wind store missing at {store_path}; "
                    f"call atlas.materialize() after selecting region."
                )
            raise FileNotFoundError(
                f"Wind store missing at {store_path}; call atlas.materialize()."
            )

        ds = xr.open_zarr(store_path, consolidated=False)

        state = ds.attrs.get("store_state", "")
        if state != "complete":
            raise RuntimeError(
                f"Wind store incomplete (store_state={state!r}); "
                f"call atlas.materialize()."
            )

        self._data = ds
        return self._data

    @property
    def turbines(self) -> tuple[str, ...]:
        """
        Turbine IDs available in the wind store.

        Returns:
            Tuple of turbine ID strings (from cleo_turbines_json attr).

        Raises:
            RuntimeError: If no turbines in wind store (call Atlas.materialize()).
        """
        ds = self.data
        if "turbine" not in ds.dims:
            raise RuntimeError(
                "No turbines in wind store; call Atlas.materialize() to add turbines."
            )
        # Read from cleo_turbines_json attr (avoids string arrays in Zarr v3)
        if "cleo_turbines_json" not in ds.attrs:
            raise RuntimeError(
                "Wind store missing cleo_turbines_json attr; re-run materialize_canonical()."
            )
        turbines_meta = json.loads(ds.attrs["cleo_turbines_json"])
        return tuple(t["id"] for t in turbines_meta)

    @property
    def selected_turbines(self) -> tuple[str, ...] | None:
        """
        Currently selected turbine IDs, or None if no selection (all turbines).

        Selection is persistent on the Atlas instance.

        Returns:
            Tuple of selected turbine IDs, or None for all turbines.
        """
        return self._atlas._wind_selected_turbines

    def _validate_turbines(self, turbines: list[str]) -> tuple[str, ...]:
        """Validate turbine IDs against available turbines.

        Args:
            turbines: List of turbine IDs to validate.

        Returns:
            Validated tuple of turbine IDs.

        Raises:
            ValueError: If any turbine ID is unknown.
        """
        available = set(self.turbines)
        requested = set(turbines)
        unknown = requested - available
        if unknown:
            raise ValueError(
                f"Unknown turbines: {sorted(unknown)}; see atlas.wind.turbines"
            )
        return tuple(turbines)

    def select(self, *, turbines: list[str] | tuple[str, ...]) -> "WindDomain":
        """
        Set persistent turbine selection on the Atlas.

        Selection persists even if atlas.wind creates new WindDomain objects.
        Use clear_selection() to remove selection.

        Args:
            turbines: Turbine IDs to select (non-empty list/tuple of strings).
                     Each ID is stripped; empty/whitespace-only IDs rejected.
                     Duplicates rejected; order preserved.

        Returns:
            self (for chaining).

        Raises:
            ValueError: If turbines is empty, contains non-strings, empty IDs,
                       duplicates, or unknown turbine IDs.
        """
        if not turbines:
            raise ValueError(
                "turbines must be non-empty; use clear_selection() to clear"
            )

        # Validate: non-empty, strings, strip, reject empty, reject duplicates
        cleaned = []
        seen = set()
        for item in turbines:
            if not isinstance(item, str):
                raise ValueError(f"Each turbine ID must be a string, got {type(item).__name__}")
            stripped = item.strip()
            if not stripped:
                raise ValueError("Turbine ID cannot be empty or whitespace-only")
            if stripped in seen:
                raise ValueError(f"Duplicate turbine ID: {stripped!r}")
            seen.add(stripped)
            cleaned.append(stripped)

        # Validate against available turbines
        validated = self._validate_turbines(cleaned)

        # Persist on Atlas
        self._atlas._wind_selected_turbines = validated
        return self

    def clear_selection(self) -> "WindDomain":
        """
        Clear persistent turbine selection.

        Returns:
            self (for chaining).
        """
        self._atlas._wind_selected_turbines = None
        return self

    def compute(self, metric: str, **kwargs) -> MetricResult:
        """
        Compute a wind metric from canonical data.

        Returns a MetricResult wrapper supporting .data and .cache() pattern.

        Args:
            metric: Metric name (see supported metrics below).
            **kwargs: Metric-specific parameters.
                For metrics requiring turbines: if not provided, uses
                persistent selection from atlas.wind.select().

        Supported metrics:
            - "mean_wind_speed": requires height (int)
            - "capacity_factors": uses turbines from select() or explicit kwarg

        Returns:
            MetricResult with .data (DataArray) and .cache() method.

        Raises:
            ValueError: If metric unknown, required params missing, or turbine validation fails.
        """
        if metric not in _WIND_METRICS:
            supported = sorted(_WIND_METRICS.keys())
            raise ValueError(
                f"Unknown metric {metric!r}. Supported: {supported}"
            )

        spec = _WIND_METRICS[metric]
        fn = spec["fn"]
        required = spec["required"]
        requires_turbines = spec["requires_turbines"]

        # Check required kwargs
        missing = required - kwargs.keys()
        if missing:
            raise ValueError(
                f"Missing required parameters for {metric}: {sorted(missing)}"
            )

        # Turbine enforcement: inject from persistent selection if not provided
        if requires_turbines:
            turbines = kwargs.get("turbines", None)
            if turbines is None:
                # Check if Atlas has persistent selection
                if self.selected_turbines is not None:
                    turbines = self.selected_turbines
                else:
                    raise ValueError(
                        "turbines required; use atlas.wind.select(...) or pass turbines=."
                    )
            else:
                # Validate provided turbines (explicit override for this call only)
                if len(turbines) == 0:
                    raise ValueError(
                        "turbines must be non-empty; see atlas.wind.turbines"
                    )
                turbines = self._validate_turbines(list(turbines))
            kwargs["turbines"] = turbines

        # Prepare canonical inputs
        wind = self._atlas.wind_data
        try:
            land = self._atlas.landscape_data
        except (FileNotFoundError, RuntimeError):
            land = None

        # Compute the metric
        da = fn(wind, land, **kwargs)

        # Build params dict for MetricResult (exclude turbines from kwargs copy)
        params = {k: v for k, v in kwargs.items()}

        return MetricResult(self, metric, da, params)

    def mean_wind_speed(self, height: int, **kwargs) -> MetricResult:
        """
        Compute mean wind speed at specified height.

        Thin wrapper around compute("mean_wind_speed", ...).
        Returns MetricResult supporting .data and .cache() pattern.

        Args:
            height: Height level (must exist in wind store).
            **kwargs: Additional parameters passed to compute.

        Returns:
            MetricResult with .data (DataArray) and .cache() method.
        """
        return self.compute("mean_wind_speed", height=height, **kwargs)

    def capacity_factors(
        self,
        *,
        turbines: list[str] | tuple[str, ...] | None = None,
        height: int = 100,
        air_density: bool = False,
        loss_factor: float = 1.0,
        **kwargs,
    ) -> MetricResult:
        """
        Compute capacity factors for selected turbines.

        Thin wrapper around compute("capacity_factors", ...).
        Returns MetricResult supporting .data and .cache() pattern.

        Args:
            turbines: Turbine IDs (uses persistent selection if None).
            height: Reference height for Weibull interpolation (default 100).
            air_density: If True, apply air density correction.
            loss_factor: Loss correction factor (default 1.0).
            **kwargs: Additional parameters passed to compute.

        Returns:
            MetricResult with .data (DataArray) and .cache() method.
        """
        # Only pass turbines if explicitly provided (let compute() inject from selection)
        compute_kwargs = {
            "height": height,
            "air_density": air_density,
            "loss_factor": loss_factor,
            **kwargs,
        }
        if turbines is not None:
            compute_kwargs["turbines"] = turbines

        return self.compute("capacity_factors", **compute_kwargs)


class LandscapeDomain:
    """
    Domain object for landscape data access.

    Provides lazy, cached access to the canonical landscape.zarr store.
    The .data property opens the store once and caches the result.
    """

    def __init__(self, atlas):
        self._atlas = atlas
        self._data = None

    def add(
        self,
        name: str,
        source_path: str | Path,
        *,
        kind: str = "raster",
        params: dict | None = None,
        materialize: bool = True,
        if_exists: str = "error",
    ) -> None:
        """Add a new variable to the landscape store.

        Registers the source in __manifest__ and optionally materializes
        the variable into landscape.zarr without recomputing existing variables.

        Args:
            name: Variable name for the new layer.
            source_path: Path to the source raster file.
            kind: Source kind (v1 only supports "raster").
            params: Optional parameters dict (e.g., {"categorical": True}).
            materialize: If True, materialize the variable immediately.
            if_exists: Behavior when variable already exists:
                - "error" (default): raise ValueError if variable exists
                - "replace": atomically replace existing variable data
                - "noop": silently skip if variable exists

        Raises:
            ValueError: If kind != "raster", if_exists invalid, or variable exists
                when if_exists="error".
            RuntimeError: If canonical stores not ready.

        Returns:
            None
        """
        from cleo.unify import Unifier

        # Validate if_exists parameter
        valid_if_exists = {"error", "replace", "noop"}
        if if_exists not in valid_if_exists:
            raise ValueError(
                f"if_exists must be one of {sorted(valid_if_exists)!r}; got {if_exists!r}"
            )

        # Ensure canonical stores exist
        atlas = self._atlas
        if not getattr(atlas, "_canonical_ready", False):
            atlas.materialize_canonical()

        u = Unifier(
            chunk_policy=atlas.chunk_policy,
            fingerprint_method=getattr(atlas, "fingerprint_method", "path_mtime_size"),
        )

        # Invalidate cached data since store may change
        self._data = None

        u.register_landscape_source(
            atlas,
            name=name,
            source_path=Path(source_path),
            kind=kind,
            params=params or {},
            if_exists=if_exists,
        )

        if materialize:
            u.materialize_landscape_variable(
                atlas,
                variable_name=name,
                if_exists=if_exists,
            )

    @property
    def data(self) -> xr.Dataset:
        """
        Lazy access to the active landscape zarr store as xr.Dataset.

        Routes to region store when region is selected, otherwise base store.
        Opens the store once and caches it. Validates store_state == "complete"
        and presence of valid_mask variable.

        Raises:
            FileNotFoundError: If landscape.zarr store does not exist.
            RuntimeError: If store_state != "complete" or valid_mask missing.
        """
        if self._data is not None:
            return self._data

        # Route to active store (region or base) per contract B1
        store_path = self._atlas._active_landscape_store_path()

        if not store_path.exists():
            # Provide helpful message based on whether region is selected
            if self._atlas._region is not None:
                raise FileNotFoundError(
                    f"Region landscape store missing at {store_path}; "
                    f"call atlas.materialize() after selecting region."
                )
            raise FileNotFoundError(
                f"Landscape store missing at {store_path}; call atlas.materialize()."
            )

        ds = xr.open_zarr(store_path, consolidated=False)

        state = ds.attrs.get("store_state", "")
        if state != "complete":
            raise RuntimeError(
                f"Landscape store incomplete (store_state={state!r}); "
                f"call atlas.materialize()."
            )

        if "valid_mask" not in ds.data_vars:
            raise RuntimeError(
                "Landscape store missing valid_mask; call atlas.materialize()."
            )

        self._data = ds
        return self._data


# %% classes
class Atlas:
    def __init__(
        self,
        path,
        country,
        crs,
        *,
        chunk_policy: dict[str, int] | None = None,
        region: str | None = None,
        results_root: Path | None = None,
        fingerprint_method: str = "path_mtime_size",
    ):
        self.path = path
        self.country = country
        self._region: str | None = None  # Region selection state (use select() to set)
        self.crs = crs
        self._turbines_configured: tuple[str, ...] | None = None
        self._wind_selected_turbines: tuple[str, ...] | None = None
        self._setup_directories()
        self._setup_logging()
        self._deploy_resources()

        # v1 Domain objects (cached, lazy)
        self._wind_domain = None
        self._landscape_domain = None

        # Canonical store configuration
        self.chunk_policy = chunk_policy if chunk_policy is not None else {"y": 1024, "x": 1024}
        self.fingerprint_method = fingerprint_method

        # Base (country-wide) store paths
        self.wind_store_path = self.path / "wind.zarr"
        self.landscape_store_path = self.path / "landscape.zarr"
        self.results_root = results_root if results_root is not None else (self.path / "results")
        self.results_root.mkdir(parents=True, exist_ok=True)

        # Canonical store readiness flag
        self._canonical_ready = False

        # Apply initial region selection if provided (equivalent to calling select(region=...))
        if region is not None:
            self.select(region=region)

    def __repr__(self) -> str:
        """Audit-safe repr: no IO, no mutation, bounded length."""
        try:
            country = getattr(self, "country", "?")
            region = getattr(self, "_region", None) or ""
            crs = getattr(self, "_crs", "?")
            path = _safe_basename(getattr(self, "_path", None))
            canonical = "ready" if getattr(self, "_canonical_ready", False) else "not_ready"

            region_part = f", region={region!r}" if region else ""
            return f"Atlas(country={country!r}{region_part}, crs={crs!r}, path={path!r}, stores={canonical})"
        except Exception:
            return "Atlas(?)"

    __str__ = __repr__

    def materialize(self):
        """Materialize stores: base (country-wide) and region (if selected).

        Per contract A5:
        - Always creates/updates base stores (wind.zarr, landscape.zarr)
        - If region selected, also creates region stores (contract B1, B4)

        Must be called before accessing wind/landscape data.
        """
        # Always ensure base stores exist
        if not self._canonical_ready:
            self.materialize_canonical()

        # If region selected, ensure region stores exist
        if self._region is not None:
            self._ensure_region_stores()

    @property
    def wind(self) -> WindDomain:
        """Access WindDomain (v1 API).

        Returns a domain object with lazy, cached .data property.
        Use atlas.wind_data for direct dataset access.
        """
        if self._wind_domain is None:
            self._wind_domain = WindDomain(self)
        return self._wind_domain

    @property
    def landscape(self) -> LandscapeDomain:
        """Access LandscapeDomain (v1 API).

        Returns a domain object with lazy, cached .data property.
        Use atlas.landscape_data for direct dataset access.
        """
        if self._landscape_domain is None:
            self._landscape_domain = LandscapeDomain(self)
        return self._landscape_domain

    @property
    def wind_data(self) -> xr.Dataset:
        """Direct access to wind dataset (convenience for atlas.wind.data)."""
        return self.wind.data

    @property
    def landscape_data(self) -> xr.Dataset:
        """Direct access to landscape dataset (convenience for atlas.landscape.data)."""
        return self.landscape.data

    @property
    def wind_zarr(self) -> xr.Dataset:
        """Open canonical wind zarr store as xr.Dataset.

        Requires prior materialize() call to create the skeleton store.
        """
        if not getattr(self, "_canonical_ready", False):
            raise RuntimeError(
                "Canonical stores not ready. Call atlas.materialize() first."
            )
        return xr.open_zarr(self.wind_store_path, consolidated=False, chunks=self.chunk_policy)

    @property
    def landscape_zarr(self) -> xr.Dataset:
        """Open canonical landscape zarr store as xr.Dataset.

        Requires prior materialize() call to create the skeleton store.
        """
        if not getattr(self, "_canonical_ready", False):
            raise RuntimeError(
                "Canonical stores not ready. Call atlas.materialize() first."
            )
        return xr.open_zarr(self.landscape_store_path, consolidated=False, chunks=self.chunk_policy)

    # -------------------------------------------------------------------------
    # Region selection (contract A4, B1)
    # -------------------------------------------------------------------------

    @property
    def region(self) -> str | None:
        """Current region selection (NUTS ID or None for full-country)."""
        return self._region

    @region.setter
    def region(self, value: str | None) -> None:
        """Set region selection (equivalent to select(region=value))."""
        self.select(region=value)

    def select(self, *, region: str | None = None) -> "Atlas":
        """
        Set the active region selection (persistent).

        Per contract A4:
        - Region selection is changeable at any time after construction.
        - Region selection affects what computations operate on and where
          derived outputs are written (region-scoped stores).
        - Region selection does NOT rewrite base stores.

        Args:
            region: NUTS region ID (e.g. "AT13") or None to clear selection.
                   String is stripped; empty/whitespace-only raises ValueError.

        Returns:
            self (for chaining).

        Raises:
            ValueError: If region is empty or whitespace-only string.
        """
        if region is not None:
            if not isinstance(region, str):
                raise ValueError(f"region must be a string or None, got {type(region).__name__}")
            stripped = region.strip()
            if not stripped:
                raise ValueError("region cannot be empty or whitespace-only")
            region = stripped

        # Update region selection
        self._region = region

        # Invalidate domain caches so .data reloads from correct store
        if self._wind_domain is not None:
            self._wind_domain._data = None
        if self._landscape_domain is not None:
            self._landscape_domain._data = None

        return self

    @property
    def _effective_region_id(self) -> str:
        """
        Effective region ID for store paths (contract B1).

        Returns "__all__" when no region selected, otherwise the region ID.
        """
        return self._region if self._region is not None else "__all__"

    def _region_store_root(self) -> Path:
        """
        Root directory for region stores (contract B1).

        Layout: <ROOT>/regions/<region_id>/
        """
        return self.path / "regions" / self._effective_region_id

    def _active_wind_store_path(self) -> Path:
        """
        Path to the active wind store (base or region).

        When region selected: <ROOT>/regions/<region_id>/wind.zarr
        When no region: <ROOT>/wind.zarr (base store)
        """
        if self._region is not None:
            return self._region_store_root() / "wind.zarr"
        return self.wind_store_path

    def _active_landscape_store_path(self) -> Path:
        """
        Path to the active landscape store (base or region).

        When region selected: <ROOT>/regions/<region_id>/landscape.zarr
        When no region: <ROOT>/landscape.zarr (base store)
        """
        if self._region is not None:
            return self._region_store_root() / "landscape.zarr"
        return self.landscape_store_path

    def _ensure_region_stores(self) -> None:
        """
        Ensure region stores exist for current region selection.

        Creates region stores by subsetting from country stores if needed.
        Does nothing if no region selected.
        """
        if self._region is None:
            return

        from cleo.unify import Unifier

        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        u.materialize_region(self, self._region)

    def materialize_canonical(self) -> None:
        """Materialize both wind.zarr and landscape.zarr canonical stores.

        This provides the canonical pathway for creating fully unified stores.

        Creates:
        - wind.zarr: Complete canonical wind store with GWA data
        - landscape.zarr: Complete canonical landscape store with valid_mask and elevation

        Requires that all GWA data files are present.
        """
        from cleo.unify import Unifier

        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        u.materialize_wind(self)
        u.materialize_landscape(self)
        self._canonical_ready = True

    # -------------------------------------------------------------------------
    # Results API v1
    # -------------------------------------------------------------------------

    _RUN_ID_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

    def new_run_id(self, prefix: str | None = None) -> str:
        """Generate a new unique run ID.

        Creates a Windows-safe, sortable run ID with optional prefix.
        Format: [prefix-]YYYYMMDDTHHMMSSz-<8-char-uuid>

        Args:
            prefix: Optional prefix to prepend to the run ID.
                Must match pattern [A-Za-z0-9_-]+ (no colons or slashes).

        Returns:
            A unique run ID string.

        Raises:
            ValueError: If prefix contains invalid characters.
        """
        from datetime import datetime, timezone

        if prefix is not None:
            if not self._RUN_ID_PREFIX_PATTERN.match(prefix):
                raise ValueError(
                    f"prefix must match [A-Za-z0-9_-]+; got {prefix!r}"
                )

        now = datetime.now(timezone.utc)
        base = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]

        if prefix is not None:
            return f"{prefix}-{base}"
        return base

    def persist(
        self,
        metric_name: str,
        obj,
        *,
        run_id: str | None = None,
        params: dict | None = None,
    ) -> Path:
        """Persist a result dataset/dataarray to a Zarr store.

        Creates an atomic, Windows-safe Zarr store
        under <results_root>/<run_id>/<metric_name>.zarr.

        Args:
            metric_name: Name of the metric/result being stored.
            obj: xr.Dataset or xr.DataArray to store.
            run_id: Unique identifier for this run/experiment.
                If None, a new run_id is generated via new_run_id().
            params: Optional parameters dict to store in attrs.

        Returns:
            Path to the created Zarr store.

        Raises:
            FileExistsError: If target store already exists (no silent overwrite).
        """
        from cleo.store import atomic_dir

        if run_id is None:
            run_id = self.new_run_id()

        store_path = Path(self.results_root) / run_id / f"{metric_name}.zarr"

        # Collision check - raise FileExistsError if target exists
        if store_path.exists():
            raise FileExistsError(
                f"Result store already exists: {store_path}. "
                f"Use a different run_id or metric_name."
            )

        # Handle DataArray vs Dataset
        if isinstance(obj, xr.DataArray):
            da = obj
            if da.name is None:
                da = da.rename(metric_name)
            ds = da.to_dataset()
        else:
            ds = obj

        # Ensure parent directory exists
        (Path(self.results_root) / run_id).mkdir(parents=True, exist_ok=True)

        # Atomic write
        with atomic_dir(store_path) as tmp:
            ds.to_zarr(tmp, mode="w", consolidated=False)

            # Add metadata attrs
            g = zarr.open_group(tmp, mode="a")
            g.attrs["store_state"] = "complete"
            g.attrs["run_id"] = run_id
            g.attrs["metric_name"] = metric_name

            # Inline timestamp and JSON serialization (no unify.py helper imports)
            g.attrs["created_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            if params is not None:
                g.attrs["params_json"] = json.dumps(
                    params, sort_keys=True, separators=(",", ":")
                )

            # Optional provenance from canonical stores
            try:
                if getattr(self, "_canonical_ready", False):
                    w = self.wind_zarr
                    l = self.landscape_zarr
                    g.attrs["wind_grid_id"] = w.attrs.get("grid_id", "")
                    g.attrs["landscape_grid_id"] = l.attrs.get("grid_id", "")
            except Exception:
                pass

        return store_path

    def open_result(self, run_id: str, metric_name: str) -> xr.Dataset:
        """Open a persisted result Zarr store.

        Opens the result store lazily.

        Args:
            run_id: Run identifier.
            metric_name: Metric name.

        Returns:
            Lazy xr.Dataset (no compute performed).

        Raises:
            FileNotFoundError: If store doesn't exist.
        """
        store_path = Path(self.results_root) / run_id / f"{metric_name}.zarr"

        if not store_path.exists():
            raise FileNotFoundError(
                f"Result store not found: {store_path}. "
                f"Run persist() first."
            )

        return xr.open_zarr(store_path, consolidated=False)

    def export_result_netcdf(
        self,
        run_id: str,
        metric_name: str,
        out_path: str | Path,
        *,
        encoding: dict | None = None,
    ) -> Path:
        """Export a result Zarr store to a single NetCDF file.

        Reads from the existing Zarr store and writes atomically to NetCDF.

        Args:
            run_id: Run identifier.
            metric_name: Metric name.
            out_path: Output path (must end with ".nc").
            encoding: Optional xarray encoding dict for NetCDF.

        Returns:
            Path to the created NetCDF file.

        Raises:
            ValueError: If out_path doesn't end with ".nc".
            FileNotFoundError: If source Zarr store doesn't exist.
        """
        out_path = Path(out_path)

        if not str(out_path).endswith(".nc"):
            raise ValueError(f"out_path must end with '.nc', got: {out_path}")

        src_store = Path(self.results_root) / run_id / f"{metric_name}.zarr"
        if not src_store.exists():
            raise FileNotFoundError(
                f"Result store not found: {src_store}. "
                f"Run persist() first."
            )

        # Open source store
        ds = xr.open_zarr(src_store, consolidated=False)

        # Build encoding: handle string/object dtype coords and vars
        final_encoding = {}
        for name, var in {**ds.coords, **ds.data_vars}.items():
            if var.dtype == object or (hasattr(var.dtype, "kind") and var.dtype.kind == "U"):
                # Convert object/unicode strings to fixed-length bytes for NetCDF compat
                max_len = max((len(str(v)) for v in var.values.ravel()), default=1)
                final_encoding[name] = {"dtype": f"S{max_len}"}
        # Merge user-provided encoding (takes precedence)
        final_encoding.update(encoding or {})

        # Atomic file write
        tmp = out_path.with_name(out_path.name + f".__tmp__{uuid4().hex}")
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(tmp, encoding=final_encoding)
            os.replace(tmp, out_path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        return out_path

    def clean_results(
        self,
        run_id: str | None = None,
        older_than: str | None = None,
        metric_name: str | None = None,
    ) -> int:
        """Clean up result Zarr stores.

        Deletes Zarr result directories matching the specified criteria.

        Args:
            run_id: If specified, only clean this run's results.
            older_than: If specified, only clean results older than this date.
                Accepts "YYYY-MM-DD" or ISO datetime format.
            metric_name: If specified, only clean this metric's store.

        Returns:
            Number of stores deleted.
        """
        base = Path(self.results_root)
        count = 0

        # Determine which run directories to scan
        if run_id:
            runs = [base / run_id] if (base / run_id).exists() else []
        else:
            runs = sorted([p for p in base.iterdir() if p.is_dir()])

        # Parse older_than threshold if specified
        threshold_dt = None
        if older_than:
            try:
                # Try ISO format first
                threshold_dt = datetime.datetime.fromisoformat(older_than)
            except ValueError:
                # Try date-only format
                threshold_dt = datetime.datetime.strptime(older_than, "%Y-%m-%d")

        for run_dir in runs:
            for store in sorted(run_dir.glob("*.zarr")):
                # Filter by metric_name if specified
                if metric_name and store.name != f"{metric_name}.zarr":
                    continue

                # Filter by age if older_than specified
                if threshold_dt:
                    store_dt = None
                    # Try to get created_at from zarr attrs
                    try:
                        g = zarr.open_group(store, mode="r")
                        created_at = g.attrs.get("created_at")
                        if created_at:
                            store_dt = datetime.datetime.fromisoformat(
                                created_at.replace("Z", "+00:00")
                            )
                    except Exception:
                        pass

                    # Fallback to mtime
                    if store_dt is None:
                        mtime = store.stat().st_mtime
                        store_dt = datetime.datetime.fromtimestamp(mtime)

                    # Skip if not older than threshold
                    if store_dt >= threshold_dt:
                        continue

                # Delete the store
                shutil.rmtree(store)
                count += 1

            # Clean up empty run directories
            if run_dir.exists() and not any(run_dir.iterdir()):
                run_dir.rmdir()

        return count

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        value = Path(value)
        self._path = value

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, value):
        self._region = value

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        """
        Set CRS as a validated string.

        Contract:
        - Invalid CRS raises ValueError with a clear message (no pyproj CRSError leak).
        - Stored value is normalized when possible (EPSG codes become "epsg:<int>").
        """
        try:
            crs_obj = pyproj.CRS(value)
        except pyproj.exceptions.CRSError as e:
            raise ValueError(f"Invalid CRS: {value!r}") from e

        epsg = crs_obj.to_epsg()
        if epsg is not None:
            self._crs = f"epsg:{epsg}"
        else:
            # Fallback: stable string representation
            self._crs = crs_obj.to_string()

    def configure_turbines(self, turbines: Sequence[str]) -> None:
        """Configure turbines for wind materialization.

        Affects wind materialization into wind.zarr; does not change compute
        defaults. Call before Atlas.materialize() or materialize_canonical().

        Changing configured turbines changes the wind inputs_id, triggering
        rebuild on next materialize().

        Args:
            turbines: Non-empty sequence of turbine IDs (e.g., ["Enercon.E40.500"]).

        Raises:
            ValueError: If turbines is empty, contains non-strings, empty/whitespace-only
                IDs, or duplicates.

        Example:
            >>> atlas.configure_turbines(["Enercon.E40.500", "Vestas.V90.2000"])
            >>> atlas.materialize()  # Materializes only configured turbines
        """
        if not turbines:
            raise ValueError("turbines must be a non-empty sequence")

        cleaned = []
        seen = set()
        for item in turbines:
            if not isinstance(item, str):
                raise ValueError(f"Each turbine ID must be a string, got {type(item).__name__}")
            stripped = item.strip()
            if not stripped:
                raise ValueError("Turbine ID cannot be empty or whitespace-only")
            if stripped in seen:
                raise ValueError(f"Duplicate turbine ID: {stripped!r}")
            seen.add(stripped)
            cleaned.append(stripped)

        self._turbines_configured = tuple(cleaned)

    @property
    def turbines_configured(self) -> tuple[str, ...] | None:
        """Turbines configured for wind materialization.

        Returns:
            Tuple of turbine IDs if configure_turbines() was called, else None.
            None means default turbine selection logic applies during materialize().
        """
        return self._turbines_configured

    _setup_logging = setup_logging
    _deploy_resources = deploy_resources

    def _setup_directories(self) -> None:
        """
        Create directories for raw and processed data if they do not exist
        """
        path_raw = self.path / "data" / "raw" / self.country
        path_processed = self.path / "data" / "processed"
        path_logging = self.path / "logs"

        for path in [path_raw, path_processed, path_logging]:
            if not path.is_dir():
                path.mkdir(parents=True)

        # Create the index file in the data directory if it doesn't exist
        index_file_path = self.path / "data" / "index.jsonl"
        if not index_file_path.exists():
            index_file_path.touch()  # Create an empty file
            logger.info(f"Created new index file: {index_file_path}")

    def get_nuts_region(self, region, merged_name=None, to_atlascrs=True):
        nuts_dir = self.path / "data" / "nuts"
        shp_files = sorted(nuts_dir.rglob("*.shp")) if nuts_dir.exists() else []
        if not shp_files:
            raise FileNotFoundError(
                f"NUTS shapefile not found under {nuts_dir}. "
                f"Run NUTS download/extract first (e.g. atlas.landscape._load_nuts() / cleo.loaders.load_nuts)."
            )
        nuts_shape = shp_files[0]
        # Read vector via centralized helper
        nuts = _read_vector_file(nuts_shape)

        # Convert three-digit country code to two-digit country code
        alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2

        # Filter regions by country code
        feasible_regions = nuts[nuts["CNTR_CODE"] == alpha_2]

        if isinstance(region, str):
            region_list = [region]
        elif isinstance(region, list):
            region_list = region
        else:
            raise TypeError("Region must be a string or a list of strings.")

        # Find invalid regions
        invalid_regions = [r for r in region_list if r not in feasible_regions["NAME_LATN"].values]
        if invalid_regions:
            raise ValueError(f"{', '.join(invalid_regions)} are not valid regions in {self.country}.")

        # Select and merge shapes
        selected_shapes = feasible_regions[feasible_regions["NAME_LATN"].isin(region_list)]
        merged_shape = selected_shapes.dissolve()

        # Set the name for the merged region
        merged_shape["NAME_LATN"] = merged_name if merged_name else ", ".join(region_list)
        merged_shape = merged_shape.reset_index(drop=True)

        if to_atlascrs:
            merged_shape = to_crs_if_needed(merged_shape, self.crs)

        return merged_shape

    def get_nuts_country(self):
        nuts_dir = self.path / "data" / "nuts"
        shp_files = sorted(nuts_dir.rglob("*.shp")) if nuts_dir.exists() else []
        if not shp_files:
            raise FileNotFoundError(
                f"NUTS shapefile not found under {nuts_dir}. "
                f"Run NUTS download/extract first (e.g. atlas.landscape._load_nuts() / cleo.loaders.load_nuts)."
            )
        nuts_shape = shp_files[0]
        # Read vector via centralized helper
        nuts = _read_vector_file(nuts_shape)
        alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2
        clip_shape = nuts.loc[(nuts["CNTR_CODE"] == alpha_2) & (nuts["LEVL_CODE"] == 0), :]
        return clip_shape

