# %% imports
import json
import xarray as xr
from pathlib import Path
from cleo.results import MetricResult
from cleo.wind_metrics import _WIND_METRICS


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
            if self._atlas._region_name is not None:
                raise FileNotFoundError(
                    f"Region wind store missing at {store_path}; "
                    f"call atlas.materialize() after selecting region."
                )
            raise FileNotFoundError(
                f"Wind store missing at {store_path}; call atlas.materialize()."
            )

        ds = xr.open_zarr(store_path, consolidated=False, chunks=self._atlas.chunk_policy)

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
            - "mean_wind_speed": requires height (int).
            - "capacity_factors": requires turbines (from select() or kwarg).
              Optional: mode ("hub" or "rews"), rews_n (int, default 9),
              air_density (bool), loss_factor (float).
            - "lcoe": requires turbines + cost params (om_fixed_eur_per_kw_a,
              om_variable_eur_per_kwh, discount_rate, lifetime_a).
              Optional: turbine_cost_share, hours_per_year (default 8766).
            - "min_lcoe_turbine": same params as lcoe. Returns int32 turbine
              index at each pixel. Turbine ID mapping in attrs["cleo:turbine_ids_json"].
            - "optimal_power": same params as lcoe. Returns rated power (kW)
              of the minimum-LCOE turbine at each pixel.
            - "optimal_energy": same params as lcoe. Returns annual energy (GWh/a)
              of the minimum-LCOE turbine at each pixel.

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
            if self._atlas._region_name is not None:
                raise FileNotFoundError(
                    f"Region landscape store missing at {store_path}; "
                    f"call atlas.materialize() after selecting region."
                )
            raise FileNotFoundError(
                f"Landscape store missing at {store_path}; call atlas.materialize()."
            )

        ds = xr.open_zarr(store_path, consolidated=False, chunks=self._atlas.chunk_policy)

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
