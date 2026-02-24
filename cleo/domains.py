# %% imports
import json
import numpy as np
import xarray as xr
from pathlib import Path
from cleo.results import DomainResult, normalize_metric_for_active_wind_store
from cleo.spatial import distance_to_positive_mask
from cleo.wind_metrics import _WIND_METRICS
from cleo.unification.store_io import (
    open_zarr_dataset,
    resolve_active_landscape_store_path,
    turbine_ids_from_json,
)


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


class WindDomain:
    """
    Domain object for wind data access.

    Provides lazy, cached access to the active wind.zarr store.
    The .data property overlays transient computed metrics staged by compute().

    Turbine selection is persistent on the Atlas instance, not on this domain.
    """

    def __init__(self, atlas):
        self._atlas = atlas
        self._data = None
        self._computed_overlays: dict[str, xr.DataArray] = {}

    def _store_data(self) -> xr.Dataset:
        """Open/cache the active wind store dataset without computed overlays."""
        if self._data is not None:
            return self._data

        store_path = self._atlas._active_wind_store_path()

        if not store_path.exists():
            if self._atlas._region_name is not None:
                raise FileNotFoundError(
                    f"Region wind store missing at {store_path}; "
                    f"call atlas.build() after selecting region."
                )
            raise FileNotFoundError(
                f"Wind store missing at {store_path}; call atlas.build()."
            )

        ds = open_zarr_dataset(store_path, chunk_policy=self._atlas.chunk_policy)
        state = ds.attrs.get("store_state", "")
        if state != "complete":
            raise RuntimeError(
                f"Wind store incomplete (store_state={state!r}); "
                f"call atlas.build()."
            )
        self._data = self._apply_public_turbine_index(ds)
        return self._data

    @property
    def data(self) -> xr.Dataset:
        """
        Lazy access to the active wind zarr store as xr.Dataset.

        Routes to region store when region is selected, otherwise base store.
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
            raise RuntimeError(
                "No turbines in wind store; call Atlas.build() to add turbines."
            )
        # Read from cleo_turbines_json attr (avoids string arrays in Zarr v3)
        if "cleo_turbines_json" not in ds.attrs:
            raise RuntimeError(
                "Wind store missing cleo_turbines_json attr; re-run build_canonical()."
            )
        return turbine_ids_from_json(ds.attrs["cleo_turbines_json"])

    @property
    def selected_turbines(self) -> tuple[str, ...] | None:
        """
        Currently selected turbine IDs, or None if no selection (all turbines).

        Selection is persistent on the Atlas instance.

        :returns: Tuple of selected turbine IDs, or ``None`` for all turbines.
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
                "Wind store has duplicate turbine ids in cleo_turbines_json; "
                "cannot build a unique turbine index."
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
            raise ValueError(
                "Provide exactly one of turbines=... or turbine_indices=...."
            )

        if turbines is not None:
            if isinstance(turbines, (str, bytes)):
                raise ValueError(
                    "turbines must be a non-empty list/tuple of turbine IDs; "
                    "got a single string/bytes value. Use turbines=[...]."
                )
            if not isinstance(turbines, (list, tuple)):
                raise ValueError(
                    f"turbines must be a list/tuple of strings, got {type(turbines).__name__}"
                )
            if not turbines:
                raise ValueError(
                    "turbines must be non-empty; use clear_selection() to clear"
                )

            # Validate: strings, strip, reject empty, reject duplicates
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
            self._atlas._wind_selected_turbines = validated
            return None

        # turbine_indices path
        if isinstance(turbine_indices, (str, bytes)):
            raise ValueError(
                "turbine_indices must be a non-empty list/tuple of integers; "
                "got a string/bytes value."
            )
        if not isinstance(turbine_indices, (list, tuple)):
            raise ValueError(
                f"turbine_indices must be a list/tuple of integers, got {type(turbine_indices).__name__}"
            )
        if not turbine_indices:
            raise ValueError(
                "turbine_indices must be non-empty; use clear_selection() to clear"
            )

        available = self.turbines
        n_available = len(available)
        selected: list[str] = []
        seen_indices: set[int] = set()
        for idx in turbine_indices:
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise ValueError(
                    f"Each turbine index must be an integer, got {type(idx).__name__}"
                )
            if idx < 0 or idx >= n_available:
                raise ValueError(
                    f"turbine index out of range: {idx}. Valid range is [0, {n_available - 1}]"
                )
            if idx in seen_indices:
                raise ValueError(f"Duplicate turbine index: {idx}")
            seen_indices.add(idx)
            selected.append(available[idx])

        self._atlas._wind_selected_turbines = tuple(selected)
        return None

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

    def compute(self, metric: str, **kwargs) -> DomainResult:
        """
        Compute a wind metric from canonical data.

        Returns a DomainResult wrapper supporting .data/.materialize()/.persist() pattern.
        Also stages a lazy normalized overlay so the metric is visible immediately
        in ``atlas.wind.data[metric]`` before materialization.

        :param metric: Metric name (see supported metrics below).
        :param kwargs: Metric-specific parameters. For metrics requiring
            turbines, omitted ``turbines`` uses persistent selection from
            :meth:`select`.

        Supported metrics:
            - "mean_wind_speed": requires height (int).
            - "capacity_factors": requires turbines (from select() or kwarg).
              Optional: mode ("direct_cf_quadrature" [default], "momentmatch_weibull",
              "hub", "rews"), rews_n (int, default 12), air_density (bool),
              loss_factor (float).
            - "rews_mps": requires turbines (from select() or kwarg).
              Optional: rews_n (int, default 12), air_density (bool).
            - "lcoe": requires turbines + cost params (om_fixed_eur_per_kw_a,
              om_variable_eur_per_kwh, discount_rate, lifetime_a).
              Optional: turbine_cost_share, hours_per_year (default 8766).
            - "min_lcoe_turbine": same params as lcoe. Returns int32 turbine
              index at each pixel. Turbine ID mapping in attrs["cleo:turbine_ids_json"].
            - "optimal_power": same params as lcoe. Returns rated power (kW)
              of the minimum-LCOE turbine at each pixel.
            - "optimal_energy": same params as lcoe. Returns annual energy (GWh/a)
              of the minimum-LCOE turbine at each pixel.

        :returns: :class:`cleo.results.DomainResult` with ``.data``/``.materialize()``/``.persist()``.
        :raises ValueError: If metric is unknown, params are missing, or turbine
            validation fails.
        :raises ValueError: If materialize-only kwargs (``overwrite`` / ``allow_mode_change``)
            are passed to ``compute(...)`` instead of ``.materialize(...)``.
        :raises ValueError: If ``inplace`` is passed; ``compute(...)`` does not
            mutate stores directly.
        """
        if "inplace" in kwargs:
            raise ValueError(
                "compute(...) does not accept inplace. "
                "Use atlas.wind.compute(...).materialize(...) to write into the active wind store."
            )

        materialize_only_kwargs = tuple(
            key for key in ("overwrite", "allow_mode_change") if key in kwargs
        )
        if materialize_only_kwargs:
            keys_text = ", ".join(repr(key) for key in materialize_only_kwargs)
            raise ValueError(
                f"materialize-only parameter(s) {keys_text} were passed to compute(...); "
                "pass them to .materialize(...), e.g. "
                "atlas.wind.compute(...).materialize(overwrite=True, allow_mode_change=True)."
            )

        # Reject timebase kwargs - must use atlas.configure_timebase()
        timebase_kwargs = tuple(
            key for key in ("hours_per_year",) if key in kwargs
        )
        if timebase_kwargs:
            keys_text = ", ".join(repr(key) for key in timebase_kwargs)
            raise ValueError(
                f"Timebase parameter(s) {keys_text} cannot be passed to compute(...). "
                f"Use atlas.configure_timebase(hours_per_year=...) to set timebase assumptions."
            )

        if metric not in _WIND_METRICS:
            supported = sorted(_WIND_METRICS.keys())
            raise ValueError(
                f"Unknown metric {metric!r}. Supported: {supported}"
            )

        spec = _WIND_METRICS[metric]
        fn = spec["fn"]
        required = spec["required"]
        requires_turbines = spec["requires_turbines"]
        allowed = spec.get("allowed")

        # Check required kwargs
        missing = required - kwargs.keys()
        if missing:
            raise ValueError(
                f"Missing required parameters for {metric}: {sorted(missing)}"
            )

        # Check for unknown kwargs (strict enforcement)
        if allowed is not None:
            unknown = set(kwargs.keys()) - allowed
            if unknown:
                # Filter out internal params for user-facing error message
                internal_params = {"hours_per_year"}
                user_allowed = sorted(allowed - internal_params)
                raise ValueError(
                    f"Unknown parameter(s) for metric {metric!r}: {sorted(unknown)}. "
                    f"Allowed: {user_allowed}"
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

        # Inject resolved timebase for economics metrics
        _ECONOMICS_METRICS = {"lcoe", "min_lcoe_turbine", "optimal_power", "optimal_energy"}
        if metric in _ECONOMICS_METRICS:
            kwargs["hours_per_year"] = self._atlas._effective_hours_per_year()

        # Prepare canonical inputs
        wind = self._atlas.wind_data
        try:
            land = self._atlas.landscape_data
        except (FileNotFoundError, RuntimeError):
            land = None

        # Compute the metric
        da = fn(wind, land, **kwargs)
        staged = normalize_metric_for_active_wind_store(
            metric=metric,
            da=da,
            existing_ds=self._store_data(),
        )
        self._computed_overlays[metric] = staged

        # Build params dict for DomainResult (exclude turbines from kwargs copy)
        params = {k: v for k, v in kwargs.items()}

        return DomainResult(self, metric, da, params)

    def mean_wind_speed(self, height: int, **kwargs) -> DomainResult:
        """
        Compute mean wind speed at specified height.

        Thin wrapper around compute("mean_wind_speed", ...).
        Returns DomainResult supporting .data/.materialize()/.persist() pattern.

        :param height: Height level that must exist in the wind store.
        :param kwargs: Additional parameters forwarded to :meth:`compute`.
        :returns: :class:`cleo.results.DomainResult`.
        """
        return self.compute("mean_wind_speed", height=height, **kwargs)

    def capacity_factors(
        self,
        *,
        turbines: list[str] | tuple[str, ...] | None = None,
        air_density: bool = False,
        loss_factor: float = 1.0,
        mode: str = "direct_cf_quadrature",
        rews_n: int = 12,
        **kwargs,
    ) -> DomainResult:
        """
        Compute capacity factors for selected turbines.

        Thin wrapper around compute("capacity_factors", ...).
        Returns DomainResult supporting .data/.materialize()/.persist() pattern.

        :param turbines: Turbine IDs; if ``None``, uses persistent selection.
        :param air_density: If ``True``, apply air-density correction.
        :param loss_factor: Multiplicative loss factor.
        :param mode: CF mode (default ``"direct_cf_quadrature"``).
        :param rews_n: Rotor quadrature nodes for rotor-aware modes.
        :param kwargs: Additional parameters forwarded to :meth:`compute`.
        :returns: :class:`cleo.results.DomainResult`.
        """
        if "height" in kwargs:
            raise ValueError(
                "capacity_factors() does not accept a free 'height' argument. "
                "Hub height is derived from each turbine definition."
            )

        # Only pass turbines if explicitly provided (let compute() inject from selection)
        compute_kwargs = {
            "air_density": air_density,
            "loss_factor": loss_factor,
            "mode": mode,
            "rews_n": rews_n,
            **kwargs,
        }
        if turbines is not None:
            compute_kwargs["turbines"] = turbines

        return self.compute("capacity_factors", **compute_kwargs)

    def rews_mps(
        self,
        *,
        turbines: list[str] | tuple[str, ...] | None = None,
        air_density: bool = False,
        rews_n: int = 12,
        **kwargs,
    ) -> DomainResult:
        """Compute rotor-equivalent wind speed (m/s) as first-class metric."""
        compute_kwargs = {
            "air_density": air_density,
            "rews_n": rews_n,
            **kwargs,
        }
        if turbines is not None:
            compute_kwargs["turbines"] = turbines
        return self.compute("rews_mps", **compute_kwargs)


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
        self._atlas = atlas
        self._data = None
        self._staged_overlays: dict[str, xr.DataArray] = {}

    @staticmethod
    def _validate_if_exists(if_exists: str) -> None:
        valid_if_exists = {"error", "replace", "noop"}
        if if_exists not in valid_if_exists:
            raise ValueError(
                f"if_exists must be one of {sorted(valid_if_exists)!r}; got {if_exists!r}"
            )

    def _build_unifier(self):
        from cleo.unification import Unifier

        atlas = self._atlas
        return Unifier(
            chunk_policy=atlas.chunk_policy,
            fingerprint_method=getattr(atlas, "fingerprint_method", "path_mtime_size"),
        )

    def _store_data(self) -> xr.Dataset:
        """Open/cache the active landscape store dataset without staged overlays."""
        if self._data is not None:
            return self._data

        store_path = resolve_active_landscape_store_path(self._atlas)
        if not store_path.exists():
            if getattr(self._atlas, "_region_name", None) is not None:
                raise FileNotFoundError(
                    f"Region landscape store missing at {store_path}; "
                    f"call atlas.build() after selecting region."
                )
            raise FileNotFoundError(
                f"Landscape store missing at {store_path}; call atlas.build()."
            )

        ds = open_zarr_dataset(store_path, chunk_policy=self._atlas.chunk_policy)
        state = ds.attrs.get("store_state", "")
        if state != "complete":
            raise RuntimeError(
                f"Landscape store incomplete (store_state={state!r}); "
                f"call atlas.build()."
            )
        if "valid_mask" not in ds.data_vars:
            raise RuntimeError(
                "Landscape store missing valid_mask; call atlas.build()."
            )

        self._data = ds
        return self._data

    @property
    def data(self) -> xr.Dataset:
        """
        Lazy access to the active landscape zarr store as xr.Dataset.

        Includes staged overlays from :meth:`add` and :meth:`add_clc_category`.
        """
        ds = self._store_data()
        if not self._staged_overlays:
            return ds
        return ds.assign(self._staged_overlays)

    def clear_staged(self) -> None:
        """
        Clear staged landscape overlays from ``atlas.landscape.data``.

        :returns: ``None``.
        """
        self._staged_overlays.clear()
        return None

    @staticmethod
    def _normalize_distance_sources_and_names(
        *,
        source,
        name,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
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
        return _distance_spec_json(source_var)

    @staticmethod
    def _distance_spec_matches(da: xr.DataArray, expected_spec_json: str) -> bool:
        return _distance_spec_matches(da, expected_spec_json)

    def _materialize_staged(
        self,
        *,
        name: str,
        if_exists: str,
        noop_existing: bool = False,
    ) -> xr.DataArray:
        self._validate_if_exists(if_exists)
        if noop_existing and if_exists == "noop":
            self._staged_overlays.pop(name, None)
            self._data = None
            return self.data[name]

        atlas = self._atlas
        if not getattr(atlas, "_canonical_ready", False):
            atlas.build_canonical()

        u = self._build_unifier()
        u.materialize_landscape_variable(
            atlas,
            variable_name=name,
            if_exists=if_exists,
        )

        self._staged_overlays.pop(name, None)
        self._data = None
        return self.data[name]

    def _materialize_staged_batch(
        self,
        *,
        names: tuple[str, ...],
        if_exists: str,
    ) -> xr.Dataset:
        self._validate_if_exists(if_exists)
        if not names:
            return xr.Dataset()

        atlas = self._atlas
        if not getattr(atlas, "_canonical_ready", False):
            atlas.build_canonical()

        staged_to_write = {
            name: self._staged_overlays[name]
            for name in names
            if name in self._staged_overlays
        }

        if staged_to_write:
            u = self._build_unifier()
            try:
                summary = u.materialize_landscape_computed_variables(
                    atlas,
                    variables=staged_to_write,
                    if_exists=if_exists,
                )
            except RuntimeError as exc:
                for name in tuple(getattr(exc, "written", ())):
                    self._staged_overlays.pop(name, None)
                self._data = None
                raise

            for name in summary.get("written", []):
                self._staged_overlays.pop(name, None)
            for name in summary.get("skipped", []):
                self._staged_overlays.pop(name, None)

        self._data = None
        return self.data[list(names)]

    def compute(self, metric: str, **kwargs) -> LandscapeComputeBatchResult:
        """Compute supported landscape metrics and stage results in ``atlas.landscape.data``.

        Supported metrics:
        - ``distance``: Euclidean distance in meters to nearest finite positive
          cell in one or more source variables.
        """
        if metric != "distance":
            raise ValueError(
                f"Unknown metric {metric!r}. Supported: {['distance']}"
            )

        if "inplace" in kwargs:
            raise ValueError(
                "compute(...) does not accept inplace. "
                "Use atlas.landscape.compute(...).materialize(...) to write into the active landscape store."
            )

        allowed = {"source", "name", "if_exists"}
        unknown = sorted(set(kwargs.keys()) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) for landscape metric 'distance': {unknown!r}. "
                f"Supported keys: {sorted(allowed)!r}"
            )
        if "source" not in kwargs:
            raise ValueError("Missing required parameters for distance: ['source']")

        source = kwargs["source"]
        name = kwargs.get("name")
        if_exists = kwargs.get("if_exists", "error")
        self._validate_if_exists(if_exists)

        sources, names = self._normalize_distance_sources_and_names(
            source=source,
            name=name,
        )

        store_ds = self._store_data()
        for src in sources:
            if src not in store_ds.data_vars:
                raise ValueError(
                    f"Unknown distance source variable {src!r}. "
                    f"Distance sources must exist in active landscape store data vars: {sorted(store_ds.data_vars)!r}"
                )

        if if_exists == "error":
            conflicts = [
                nm for nm in names
                if nm in self._staged_overlays or nm in store_ds.data_vars
            ]
            if conflicts:
                raise ValueError(
                    f"distance compute would overwrite existing variable(s): {conflicts!r}. "
                    "Use if_exists='replace' to overwrite or if_exists='noop' to skip."
                )

        result_vars: dict[str, xr.DataArray] = {}
        valid_mask = store_ds["valid_mask"]

        for src, nm in zip(sources, names, strict=True):
            expected_spec = self._distance_spec_json(src)
            staged_exists = nm in self._staged_overlays
            store_exists = nm in store_ds.data_vars

            if if_exists == "noop":
                if staged_exists:
                    staged_da = self._staged_overlays[nm]
                    if not self._distance_spec_matches(staged_da, expected_spec):
                        raise ValueError(
                            f"Variable {nm!r} already staged with different distance spec. "
                            "Use if_exists='replace' to overwrite."
                        )
                    result_vars[nm] = staged_da
                    continue
                if store_exists:
                    existing_da = store_ds[nm]
                    if not self._distance_spec_matches(existing_da, expected_spec):
                        raise ValueError(
                            f"Variable {nm!r} already exists in active landscape store with different distance spec. "
                            "Use if_exists='replace' to overwrite."
                        )
                    result_vars[nm] = existing_da
                    continue

            dist = distance_to_positive_mask(store_ds[src], valid_mask).rename(nm)
            dist = dist.reset_coords(drop=True)
            dist.attrs["cleo:metric"] = "distance"
            dist.attrs["cleo:distance_source"] = src
            dist.attrs["cleo:distance_spec_json"] = expected_spec
            self._staged_overlays[nm] = dist
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
        u,
        store_ds: xr.Dataset,
        staged_exists: bool,
        store_exists: bool,
    ) -> LandscapeAddResult:
        """Stage a prepared variable after source registration."""
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

        staged = u.prepare_landscape_variable_data(
            self._atlas,
            variable_name=name,
        )
        staged = staged.reset_coords(drop=True)
        self._staged_overlays[name] = staged
        return LandscapeAddResult(
            self,
            name,
            staged,
            if_exists,
        )

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
        Stage a raster landscape variable candidate and return an operation object.

        Vector sources are exposed via :meth:`rasterize`.
        """
        self._validate_if_exists(if_exists)
        if kind != "raster":
            raise ValueError(
                "add(...) only supports kind='raster'. "
                "Use atlas.landscape.rasterize(...) for vector sources."
            )

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
                    f"  Use if_exists='replace' to overwrite or if_exists='noop' to skip."
                )

        u = self._build_unifier()
        u.register_landscape_source(
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
            u=u,
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
        Stage a vector-rasterized landscape variable and return an operation object.

        The input ``shape`` may be a path-like vector source or a GeoDataFrame.
        Rasterization aligns to the atlas wind/landscape grid.
        """
        self._validate_if_exists(if_exists)

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

        u = self._build_unifier()
        u.register_landscape_vector_source(
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
            u=u,
            store_ds=store_ds,
            staged_exists=staged_exists,
            store_exists=store_exists,
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
            raise ValueError(
                "categories must be 'all', an int code, or a non-empty list of int codes."
            )

        if name is None:
            if len(codes) > 1:
                raise ValueError(
                    "name is required when adding multiple CLC codes in one variable."
                )
            inferred = default_category_name(atlas.path, codes[0])
            if inferred is None:
                raise ValueError(
                    f"No default variable name known for CLC code {codes[0]!r}; pass name=..."
                )
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
