"""Atlas orchestration and lifecycle management.

This module provides the ``Atlas`` class, the single public entry point for CLEO.
Atlas coordinates wind resource assessment workflows including:

- Workspace initialization and resource deployment
- Base store materialization (wind.zarr, landscape.zarr)
- Area selection and derived store management
- Domain access (wind, landscape) for computation and data retrieval
- Results persistence, export, and cleanup

The Atlas class follows a progressive disclosure API pattern where simple workflows
require minimal configuration while advanced features remain accessible through
optional parameters and method calls.

Example usage::

    from cleo import Atlas

    atlas = Atlas(path="./my_atlas", country="AUT", crs="EPSG:3035")
    atlas.build()

    # Compute capacity factors
    result = atlas.wind.compute("capacity_factors")
    result.materialize()

    # Access data
    cf = atlas.wind.data["capacity_factors"]

See Also
--------
cleo.domains : Domain facades for wind and landscape data access
docs/CONTRACT_UNIFIED_ATLAS.md : Normative API contract
"""

# %% imports
import math
import re
import pyproj
import logging
import xarray as xr
import pycountry as pct
from uuid import uuid4
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import pandas as pd

from cleo.domains import WindDomain, LandscapeDomain
from cleo.atlas_policies.nuts_catalog import load_nuts_area_catalog as load_nuts_area_catalog_policy
from cleo.atlas_policies.region_selection import (
    normalize_area_name as normalize_area_name_policy,
    nuts_areas_level_names as nuts_areas_level_names_policy,
    resolve_area_name as resolve_area_name_policy,
    select_area_decision as select_area_decision_policy,
    validate_nuts_level as validate_nuts_level_policy,
)
from cleo.atlas_policies.cleanup import (
    parse_older_than as parse_older_than_policy,
    resolve_area_cleanup_id as resolve_area_cleanup_id_policy,
    select_area_dirs_for_cleanup as select_area_dirs_for_cleanup_policy,
    select_result_stores_for_cleanup as select_result_stores_for_cleanup_policy,
)
from cleo.results import (
    delete_result_store,
    list_result_stores,
    prune_empty_run_dirs,
    read_result_store_datetime,
    validate_result_path_token,
)
from cleo.unification.nuts_io import _read_vector_file, _read_nuts_area_catalog
from cleo.unification.store_io import (
    DEFAULT_CHUNK_POLICY,
    delete_area_dir,
    list_area_dirs,
    open_zarr_dataset,
    read_area_store_meta,
    read_zarr_group_attrs,
    write_netcdf_atomic,
)
from cleo.spatial import to_crs_if_needed
from cleo.dask_utils import normalize_compute_backend, normalize_compute_workers

logger = logging.getLogger(__name__)


def _safe_basename(path) -> str:
    """Return basename of path, or '?' on any error."""
    try:
        return Path(path).name if path else "?"
    except (TypeError, ValueError, OSError):
        return "?"


def _validate_non_negative_numeric(name: str, value: object) -> float:
    """Validate that value is numeric, finite, and >= 0.

    :param name: Parameter name for error messages.
    :param value: Value to validate.
    :returns: Validated value as float.
    :raises TypeError: If value is not numeric.
    :raises ValueError: If value is not finite or is negative.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    result = float(value)
    if not (result >= 0.0 and math.isfinite(result)):
        raise ValueError(f"{name} must be finite and >= 0, got {result}")
    return result


def _validate_unit_range(name: str, value: object, *, include_max: bool = True) -> float:
    """Validate that value is numeric, finite, and in unit range [0, 1] or [0, 1).

    :param name: Parameter name for error messages.
    :param value: Value to validate.
    :param include_max: If True, range is [0, 1]; if False, range is [0, 1).
    :returns: Validated value as float.
    :raises TypeError: If value is not numeric.
    :raises ValueError: If value is not finite or out of range.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    result = float(value)
    upper_ok = result <= 1.0 if include_max else result < 1.0
    if not (0.0 <= result and upper_ok and math.isfinite(result)):
        right = "]" if include_max else ")"
        raise ValueError(f"{name} must be finite and in range [0, 1{right}, got {result}")
    return result


def _validate_positive_int(name: str, value: object) -> int:
    """Validate that value is a positive integer.

    :param name: Parameter name for error messages.
    :param value: Value to validate.
    :returns: Validated value as int.
    :raises TypeError: If value is not an int.
    :raises ValueError: If value is not positive.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class NutsAreaName(str):
    """String-like NUTS area name carrying its NUTS level metadata."""

    _nuts_level: int

    def __new__(cls, value: str, level: int):
        """
        Create area-name wrapper with attached NUTS level metadata.

        :param value: Region display name.
        :param level: NUTS level.
        :returns: New ``NutsAreaName`` instance.
        """
        obj = str.__new__(cls, value)
        obj._nuts_level = int(level)
        return obj

    @property
    def level(self) -> int:
        """
        Return NUTS level metadata attached to this area name.

        :returns: NUTS level.
        """
        return int(self._nuts_level)


class Atlas:
    DEFAULT_NUTS_LEVEL = 2
    _VALID_NUTS_LEVELS = (0, 1, 2, 3)
    DEFAULT_HOURS_PER_YEAR = 8766.0

    def __init__(
        self,
        path: str | Path,
        country: str,
        crs: str,
        *,
        chunk_policy: dict[str, int] | None = None,
        compute_backend: str = "serial",
        compute_workers: int | None = None,
        area: str | None = None,
        results_root: Path | None = None,
        fingerprint_method: str = "path_mtime_size",
    ) -> None:
        """
        Initialize Atlas workspace and runtime configuration.

        :param path: Atlas workspace root directory.
        :param country: ISO3 country code.
        :param crs: Canonical projected CRS.
        :param chunk_policy: Optional dataset chunk policy.
        :param compute_backend: Compute backend for eager I/O materialization.
        :param compute_workers: Optional local worker cap.
        :param area: Optional initial area selection (applied after build).
        :param results_root: Optional override for results artifact root.
        :param fingerprint_method: Unification fingerprint strategy.
        """
        self.path = path
        self.country = country
        # Area selection state (use select() to set after build())
        # _area_name: public human-readable name (or None)
        # _area_id: internal stable id ("__all__" when no area selected)
        self._area_name: str | None = None
        self._area_id: str = "__all__"
        # _pending_area: area name passed to constructor, applied in build()
        self._pending_area: str | None = area
        self.crs = crs
        self._turbines_configured: tuple[str, ...] | None = None
        self._wind_selected_turbines: tuple[str, ...] | None = None
        self._timebase_configured: dict[str, float] | None = None
        self._economics_configured: dict[str, float | int] | None = None
        self._setup_directories()
        self._setup_logging()
        self._deploy_resources()

        # v1 Domain objects (cached, lazy)
        self._wind_domain: WindDomain | None = None
        self._landscape_domain: LandscapeDomain | None = None

        # Canonical store configuration
        self.chunk_policy = chunk_policy if chunk_policy is not None else dict(DEFAULT_CHUNK_POLICY)
        self.compute_backend = normalize_compute_backend(compute_backend)
        self.compute_workers = normalize_compute_workers(
            compute_workers,
            backend=self.compute_backend,
        )
        self.fingerprint_method = fingerprint_method

        # Base (country-wide) store paths
        self.wind_store_path = self.path / "wind.zarr"
        self.landscape_store_path = self.path / "landscape.zarr"
        self.results_root = results_root if results_root is not None else (self.path / "results")
        self.results_root.mkdir(parents=True, exist_ok=True)

        # Canonical store readiness flag
        self._canonical_ready = False
        self._nuts_area_catalog_cache: tuple[dict, ...] | None = None
        self._nuts_area_catalog_cache_country: str | None = None

    def __repr__(self) -> str:
        """Audit-safe repr: no IO, no mutation, bounded length."""
        try:
            country = getattr(self, "country", "?")
            area_name = getattr(self, "_area_name", None) or ""
            crs = getattr(self, "_crs", "?")
            path = _safe_basename(getattr(self, "_path", None))
            canonical = "ready" if getattr(self, "_canonical_ready", False) else "not_ready"

            area_part = f", area={area_name!r}" if area_name else ""
            return f"Atlas(country={country!r}{area_part}, crs={crs!r}, path={path!r}, stores={canonical})"
        except (AttributeError, TypeError, ValueError):
            return "Atlas(?)"

    __str__ = __repr__

    def build(self) -> None:
        """Materialize stores: base (country-wide) and area (if selected).

        - Always creates/updates base stores (wind.zarr, landscape.zarr)
        - If area passed to constructor or select(), also creates area stores

        Must be called before accessing wind/landscape data.
        """
        # Area-aware build paths require NUTS boundaries.
        if self.area is not None:
            self._ensure_nuts_shapefile(auto_download=True)

        # Always ensure base stores exist
        if not self._canonical_ready:
            self.build_canonical()

        # Apply pending area from constructor (now that stores exist)
        if self._pending_area is not None:
            self.select(area=self._pending_area, inplace=True)
            self._pending_area = None  # Clear after applying

        # If area selected, ensure area stores exist
        if self._area_name is not None:
            self._ensure_area_stores()

        # Build may alter active-store routing/state; clear transient overlays.
        self._invalidate_domain_views(clear_wind=True, clear_landscape=True)

    @property
    def wind(self) -> WindDomain:
        """
        Access wind domain facade.

        :returns: Wind domain object with lazy cached data access.
        """
        if self._wind_domain is None:
            self._wind_domain = WindDomain(self)
        return self._wind_domain

    @property
    def landscape(self) -> LandscapeDomain:
        """
        Access landscape domain facade.

        :returns: Landscape domain object with lazy cached data access.
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

    def flatten(
        self,
        *,
        domain: str = "wind",
        digits: int = 5,
        exclude_template: bool = True,
        include_domain_prefix: bool = True,
        cast_binary_to_int: bool = False,
        include_only: Sequence[str] | None = None,
    ) -> "pd.DataFrame":
        """Flatten atlas domain data to a pandas DataFrame for downstream models.

        The resulting frame uses a rounded `(y, x)` MultiIndex and one column per
        supported data variable / non-spatial coordinate slice, matching the
        semantics of :func:`cleo.utils.flatten`.

        :param domain: Dataset source to flatten: ``"wind"``, ``"landscape"``, or ``"both"``.
        :param digits: Number of decimal digits used to round ``x``/``y`` index.
        :param exclude_template: If ``True``, skip the ``template`` data variable.
        :param include_domain_prefix: When ``domain="both"``, prefix columns with
            ``"wind__"`` / ``"landscape__"`` to avoid collisions.
        :param cast_binary_to_int: If ``True``, cast binary columns (bool or
            numeric ``{0,1}``) to nullable integer ``Int8``.
        :param include_only: Optional list of output columns to keep. Raises if
            any requested column is missing.
        :returns: Flattened :class:`pandas.DataFrame`.
        :raises ValueError: If ``domain`` is unsupported.
        """
        from types import SimpleNamespace
        from cleo.utils import flatten as flatten_data

        if domain == "wind":
            data = self.wind_data
        elif domain == "landscape":
            data = self.landscape_data
        elif domain == "both":
            wind_proxy = SimpleNamespace(data=self.wind_data)
            land_proxy = SimpleNamespace(data=self.landscape_data)
            wind_df = flatten_data(
                wind_proxy,
                digits=digits,
                exclude_template=exclude_template,
                cast_binary_to_int=cast_binary_to_int,
                include_only=None,
            )
            land_df = flatten_data(
                land_proxy,
                digits=digits,
                exclude_template=exclude_template,
                cast_binary_to_int=cast_binary_to_int,
                include_only=None,
            )

            if include_domain_prefix:
                wind_df = wind_df.rename(columns={c: f"wind__{c}" for c in wind_df.columns})
                land_df = land_df.rename(columns={c: f"landscape__{c}" for c in land_df.columns})
            else:
                overlap = set(wind_df.columns) & set(land_df.columns)
                if overlap:
                    raise ValueError(
                        "Column name collision when flattening both domains: "
                        f"{sorted(overlap)}. Set include_domain_prefix=True."
                    )
            out = wind_df.join(land_df, how="outer")
            if include_only is not None:
                include_only = list(include_only)
                missing = [c for c in include_only if c not in out.columns]
                if missing:
                    raise ValueError(
                        f"include_only contains unknown columns: {sorted(missing)!r}. "
                        f"Available columns: {sorted(map(str, out.columns))!r}"
                    )
                out = out.loc[:, include_only]
            return out
        else:
            raise ValueError(f"Unsupported domain {domain!r}; expected 'wind', 'landscape', or 'both'.")

        proxy = SimpleNamespace(data=data)
        return flatten_data(
            proxy,
            digits=digits,
            exclude_template=exclude_template,
            cast_binary_to_int=cast_binary_to_int,
            include_only=include_only,
        )

    @staticmethod
    def validate_flatten_schema(df, required_columns: Sequence[str]) -> None:
        """Validate flattened DataFrame contains required model columns.

        This helper performs schema checks only. It does not filter rows,
        cast values, or modify the DataFrame.

        :param df: Flattened DataFrame to validate.
        :param required_columns: Columns required by downstream modeling.
        :returns: ``None``
        :raises ValueError: If required columns are missing.
        """
        required = list(required_columns)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required flatten columns: {sorted(missing)!r}. "
                f"Available columns: {sorted(map(str, df.columns))!r}"
            )

    @property
    def wind_zarr(self) -> xr.Dataset:
        """Open canonical wind zarr store as xr.Dataset.

        Requires prior build() call to create the skeleton store.
        """
        if not getattr(self, "_canonical_ready", False):
            raise RuntimeError("Canonical stores not ready. Call atlas.build() first.")
        return open_zarr_dataset(self.wind_store_path, chunk_policy=self.chunk_policy)

    @property
    def landscape_zarr(self) -> xr.Dataset:
        """Open canonical landscape zarr store as xr.Dataset.

        Requires prior build() call to create the skeleton store.
        """
        if not getattr(self, "_canonical_ready", False):
            raise RuntimeError("Canonical stores not ready. Call atlas.build() first.")
        return open_zarr_dataset(self.landscape_store_path, chunk_policy=self.chunk_policy)

    # -------------------------------------------------------------------------
    # Area selection (contract A4, B1)
    # -------------------------------------------------------------------------

    @property
    def area(self) -> str | None:
        """
        Current area selection (public name or ``None`` for full-country).

        Returns pending area if passed to constructor but not yet applied
        (before build()), otherwise returns the resolved area name.

        :returns: Active or pending area name, or ``None``.
        """
        if self._area_name is not None:
            return self._area_name
        # Return pending area if set but not yet applied (before build())
        return self._pending_area

    @area.setter
    def area(self, value: str | None) -> None:
        """Set area selection (equivalent to select(area=value))."""
        self.select(area=value, inplace=True)

    def _normalize_area_name(self, name: str) -> str:
        """Normalize area name: strip, collapse whitespace, casefold."""
        return normalize_area_name_policy(name)

    def _validate_nuts_level(self, level: int) -> int:
        """Validate and normalize NUTS level."""
        return validate_nuts_level_policy(level, valid_levels=self._VALID_NUTS_LEVELS)

    def _load_nuts_area_catalog(self) -> list[dict]:
        """Load NUTS area catalog from attrs/raw sources for the current country.

        :returns: Catalog rows for ``self.country`` only.
        :rtype: list[dict]
        """

        def _log_debug(msg: str) -> None:
            logger.debug(
                msg,
                extra={"landscape_store_path": str(self.landscape_store_path)},
                exc_info=True,
            )

        def _read_raw_catalog() -> list[dict]:
            self._ensure_nuts_shapefile(auto_download=True)
            return _read_nuts_area_catalog(self.path, self.country)

        expected_country_iso3 = str(self.country).strip().upper()

        catalog, cache, cache_country_iso3 = load_nuts_area_catalog_policy(
            cached_rows=self._nuts_area_catalog_cache,
            cached_country_iso3=self._nuts_area_catalog_cache_country,
            expected_country_iso3=expected_country_iso3,
            landscape_store_path=self.landscape_store_path,
            valid_levels=self._VALID_NUTS_LEVELS,
            read_store_attrs=read_zarr_group_attrs,
            read_raw_catalog=_read_raw_catalog,
            log_debug=_log_debug,
        )
        self._nuts_area_catalog_cache = cache
        self._nuts_area_catalog_cache_country = cache_country_iso3
        return catalog

    @property
    def nuts_areas(self) -> tuple[NutsAreaName, ...]:
        """List selectable area names for default NUTS level (level 2)."""
        return self.nuts_areas_level(self.DEFAULT_NUTS_LEVEL)

    def nuts_areas_level(self, level: int) -> tuple[NutsAreaName, ...]:
        """List selectable area names for a specific NUTS level.

        :param level: NUTS level (1, 2, or 3).
        :returns: Tuple of area names tagged with level metadata.
        :raises ValueError: If level is invalid or no areas are available.
        """
        catalog = self._load_nuts_area_catalog()
        return nuts_areas_level_names_policy(
            level=level,
            country=self.country,
            catalog_rows=catalog,
            validate_level=self._validate_nuts_level,
            make_area_name=lambda name, level_i: NutsAreaName(name, level_i),
        )

    def _resolve_area_name(self, name: str, *, nuts_level: int | None = None) -> tuple[str, str, int]:
        """
        Resolve area name to ``(normalized_name, area_id, level)``.
        """
        catalog = self._load_nuts_area_catalog()
        return resolve_area_name_policy(
            name=name,
            nuts_level=nuts_level,
            default_level=self.DEFAULT_NUTS_LEVEL,
            country=self.country,
            catalog_rows=catalog,
            normalize_name=self._normalize_area_name,
            validate_level=self._validate_nuts_level,
            default_areas_supplier=lambda: self.nuts_areas_level(self.DEFAULT_NUTS_LEVEL),
        )

    def _clone_for_selection(self) -> "Atlas":
        """
        Create a new Atlas instance that shares the same on-disk stores.

        This is intentionally a fresh instance (no domain caches), so the returned
        object behaves like a normal Atlas created from scratch.

        :returns: Cloned Atlas instance with copied runtime configuration.
        """

        clone = Atlas(
            self.path,
            country=self.country,
            crs=self.crs,
            chunk_policy=dict(self.chunk_policy) if self.chunk_policy is not None else None,
            compute_backend=self.compute_backend,
            compute_workers=self.compute_workers,
            area=None,
            results_root=self.results_root,
            fingerprint_method=self.fingerprint_method,
        )

        clone._canonical_ready = bool(getattr(self, "_canonical_ready", False))
        clone._turbines_configured = getattr(self, "_turbines_configured", None)
        clone._wind_selected_turbines = getattr(self, "_wind_selected_turbines", None)
        clone._timebase_configured = dict(self._timebase_configured) if self._timebase_configured is not None else None
        clone._economics_configured = (
            dict(self._economics_configured) if self._economics_configured is not None else None
        )

        return clone

    def select(
        self,
        *,
        area: str | NutsAreaName | None = None,
        nuts_level: int | None = None,
        inplace: bool = False,
    ) -> "Atlas | None":
        """
        Select a subregion within the atlas-country.

        Semantics mirror pandas' inplace=:
            - inplace=False (default): return a new constrained Atlas; leave self unchanged
            - inplace=True: constrain self in-place and return None

        :param area: Human-readable area name (for example ``"Niederösterreich"``),
            a ``NutsAreaName`` value, or ``None`` to clear selection.
        :param nuts_level: Optional NUTS level override for resolving plain strings.
        :param inplace: If ``True``, mutate ``self`` and return ``None``.
        :returns: A new constrained :class:`Atlas` when ``inplace=False``, else ``None``.
        :raises ValueError: If ``area`` is empty, whitespace-only, wrong type, or unknown.
        """
        if not inplace:
            clone = self._clone_for_selection()
            clone.select(area=area, nuts_level=nuts_level, inplace=True)
            return clone

        decision = select_area_decision_policy(
            area=area,
            nuts_level=nuts_level,
            nuts_area_name_type=NutsAreaName,
            validate_level=self._validate_nuts_level,
            resolve_name=lambda name, level: self._resolve_area_name(name, nuts_level=level),
        )
        self._area_name = decision.area_name
        self._area_id = decision.area_id

        # Area routing changed: invalidate caches and staged overlays.
        self._invalidate_domain_views(clear_wind=True, clear_landscape=True)

        return None

    @property
    def _effective_area_id(self) -> str:
        """
        Effective area ID for store paths (contract B1).

        Returns "__all__" when no area selected, otherwise the internal area_id.

        :returns: Effective area ID used in path routing.
        """
        return self._area_id

    def _area_store_root(self) -> Path:
        """
        Root directory for area stores (contract B1).

        Layout: <ROOT>/areas/<area_id>/
        """
        return self.path / "areas" / self._effective_area_id

    def _active_wind_store_path(self) -> Path:
        """
        Path to the active wind store (base or area).

        When area selected: <ROOT>/areas/<area_id>/wind.zarr
        When no area: <ROOT>/wind.zarr (base store)
        """
        if self._area_name is not None:
            return self._area_store_root() / "wind.zarr"
        return self.wind_store_path

    def _active_landscape_store_path(self) -> Path:
        """
        Path to the active landscape store (base or area).

        When area selected: <ROOT>/areas/<area_id>/landscape.zarr
        When no area: <ROOT>/landscape.zarr (base store)
        """
        if self._area_name is not None:
            return self._area_store_root() / "landscape.zarr"
        return self.landscape_store_path

    def _evaluate_for_io(self, obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        """Materialize potentially lazy xarray objects using configured backend."""
        from cleo.dask_utils import compute as dask_compute, is_dask_backed

        if not is_dask_backed(obj):
            return obj
        return dask_compute(
            obj,
            backend=self.compute_backend,
            num_workers=self.compute_workers,
        )

    def _ensure_area_stores(self) -> None:
        """
        Ensure area stores exist for current area selection.

        Creates area stores by subsetting from country stores if needed.
        Does nothing if no area selected.
        """
        if self._area_name is None:
            return

        from cleo.unification import Unifier

        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        u.ensure_area_stores(
            self,
            self._area_id,
            logger=logger,
        )

    def build_canonical(self) -> None:
        """Materialize both wind.zarr and landscape.zarr canonical stores.

        This provides the canonical pathway for creating fully unified stores.

        Creates:
        - wind.zarr: Complete canonical wind store with GWA data
        - landscape.zarr: Complete canonical landscape store with valid_mask and elevation

        Missing required GWA rasters are auto-downloaded before failing.
        """
        from cleo.unification import Unifier

        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        u.materialize_wind(self)
        u.materialize_landscape(self)
        self._canonical_ready = True
        self._invalidate_domain_views(clear_wind=True, clear_landscape=True)

    def build_clc(
        self,
        *,
        source: str = "clc2018",
        url: str | None = None,
        force_download: bool = False,
        force_prepare: bool = False,
    ) -> Path:
        """Prepare country CLC cache aligned to wind/GWA grid.

        Downloads CLC source data if missing, then creates a prepared GeoTIFF
        clipped/cropped to country coverage and aligned to the canonical wind grid.

        :param source: CLC source identifier (currently ``"clc2018"``).
        :param url: Optional source URL override. If ``None``, CLEO resolves a
            default CLMS download URL for the source and requires CLMS auth
            via supported environment variables.
        :param force_download: If ``True``, re-download source raster.
        :param force_prepare: If ``True``, rebuild prepared country cache.
        :returns: Path to prepared CLC GeoTIFF.
        """
        from cleo.clc import materialize_clc

        out = materialize_clc(
            self,
            source=source,
            url=url,
            force_download=force_download,
            force_prepare=force_prepare,
        )
        self._invalidate_domain_views(clear_wind=True, clear_landscape=True)
        return out

    def _invalidate_domain_views(
        self,
        *,
        clear_wind: bool = False,
        clear_landscape: bool = False,
    ) -> None:
        """Invalidate cached domain datasets and optional transient overlays."""
        if self._wind_domain is not None:
            self._wind_domain._data = None
            if clear_wind:
                clear_wind_fn = getattr(self._wind_domain, "clear_computed", None)
                if callable(clear_wind_fn):
                    clear_wind_fn()
                elif hasattr(self._wind_domain, "_computed_overlays"):
                    self._wind_domain._computed_overlays.clear()

        if self._landscape_domain is not None:
            self._landscape_domain._data = None
            if clear_landscape:
                clear_land_fn = getattr(self._landscape_domain, "clear_staged", None)
                if callable(clear_land_fn):
                    clear_land_fn()
                elif hasattr(self._landscape_domain, "_staged_overlays"):
                    self._landscape_domain._staged_overlays.clear()

    # -------------------------------------------------------------------------
    # Results API v1
    # -------------------------------------------------------------------------

    _RUN_ID_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

    def new_run_id(self, prefix: str | None = None) -> str:
        """Generate a new unique run ID.

        Creates a Windows-safe, sortable run ID with optional prefix.
        Format: [prefix-]YYYYMMDDTHHMMSSz-<8-char-uuid>

        :param prefix: Optional prefix prepended to the run ID. Must match
            ``[A-Za-z0-9_-]+`` (no colons or slashes).
        :returns: A unique run ID string.
        :raises ValueError: If ``prefix`` contains invalid characters.
        """
        from datetime import datetime, timezone

        if prefix is not None:
            if not self._RUN_ID_PREFIX_PATTERN.match(prefix):
                raise ValueError(f"prefix must match [A-Za-z0-9_-]+; got {prefix!r}")

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

        Transitional low-level API. Prefer fluent ``<domain-result>.persist(...)``.
        """
        from cleo.results import persist_result

        return persist_result(
            self,
            metric_name,
            obj,
            run_id=run_id,
            params=params,
        )

    def open_result(self, run_id: str, metric_name: str) -> xr.Dataset:
        """Open a persisted result Zarr store.

        Opens the result store lazily.

        :param run_id: Run identifier.
        :param metric_name: Metric name.
        :returns: Lazy :class:`xarray.Dataset` (no compute performed).
        :raises FileNotFoundError: If the store does not exist.
        """
        from cleo.results import result_store_path

        store_path = result_store_path(
            results_root=Path(self.results_root),
            run_id=run_id,
            metric_name=metric_name,
        )

        if not store_path.exists():
            raise FileNotFoundError(f"Result store not found: {store_path}. Run persist() first.")

        return open_zarr_dataset(store_path)

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

        :param run_id: Run identifier.
        :param metric_name: Metric name.
        :param out_path: Output path. Must end with ``.nc``.
        :param encoding: Optional xarray encoding dictionary for NetCDF.
        :returns: Path to the created NetCDF file.
        :raises ValueError: If ``out_path`` does not end with ``.nc``.
        :raises FileNotFoundError: If source Zarr store does not exist.
        """
        out_path = Path(out_path)

        if not str(out_path).endswith(".nc"):
            raise ValueError(f"out_path must end with '.nc', got: {out_path}")

        from cleo.results import result_store_path, restore_serialized_string_coords

        src_store = result_store_path(
            results_root=Path(self.results_root),
            run_id=run_id,
            metric_name=metric_name,
        )
        if not src_store.exists():
            raise FileNotFoundError(f"Result store not found: {src_store}. Run persist() first.")

        # Open source store
        ds = open_zarr_dataset(src_store)
        ds = restore_serialized_string_coords(ds)
        ds = self._evaluate_for_io(ds)

        # Build encoding: handle string/object dtype coords and vars
        final_encoding = {}
        for name, var in {**ds.coords, **ds.data_vars}.items():
            if var.dtype == object or (hasattr(var.dtype, "kind") and var.dtype.kind == "U"):
                # Convert object/unicode strings to fixed-length bytes for NetCDF compat
                max_len = max((len(str(v)) for v in var.to_numpy().ravel()), default=1)
                final_encoding[name] = {"dtype": f"S{max_len}"}
        # Merge user-provided encoding (takes precedence)
        final_encoding.update(encoding or {})

        try:
            write_netcdf_atomic(ds, out_path, encoding=final_encoding)
        except (OSError, ValueError, RuntimeError, TypeError):
            logger.error(
                "Failed to export result to NetCDF.",
                extra={
                    "run_id": run_id,
                    "metric_name": metric_name,
                    "out_path": str(out_path),
                },
                exc_info=True,
            )
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

        :param run_id: If provided, only clean this run's results.
        :param older_than: If provided, only clean results older than this date.
            Accepts ``YYYY-MM-DD`` or ISO datetime format.
        :param metric_name: If provided, only clean this metric's store.
        :returns: Number of stores deleted.
        """
        base = Path(self.results_root)

        metric_name_n = metric_name
        if metric_name_n is not None:
            metric_name_n = validate_result_path_token(metric_name_n, field="metric_name")

        threshold_dt = parse_older_than_policy(older_than)

        stores = list_result_stores(base, run_id=run_id, metric_name=None)
        selected, scanned = select_result_stores_for_cleanup_policy(
            stores=stores,
            metric_name=metric_name_n,
            threshold_dt=threshold_dt,
            read_store_datetime=read_result_store_datetime,
        )

        for store in selected:
            delete_result_store(store)
        count = len(selected)
        prune_empty_run_dirs(base)

        if count == 0:
            logger.info(
                "clean_results: deleted=0 (scanned=%d, run_id=%r, metric_name=%r, older_than=%r, results_root=%s). "
                "No matching persisted result stores were found. "
                "Note: atlas.wind.compute(...).materialize() writes to wind.zarr, not results_root.",
                scanned,
                run_id,
                metric_name_n,
                older_than,
                base,
            )
        else:
            logger.info(
                "clean_results: deleted=%d (scanned=%d, run_id=%r, metric_name=%r, older_than=%r, results_root=%s).",
                count,
                scanned,
                run_id,
                metric_name_n,
                older_than,
                base,
            )

        return count

    def clean_areas(
        self,
        area: str | None = None,
        older_than: str | None = None,
        *,
        include_incomplete: bool = True,
    ) -> int:
        """Clean up materialized area stores under ``<atlas.path>/areas``.

        Deletes area directories that match the given filters. A area directory
        is expected to contain ``wind.zarr`` and/or ``landscape.zarr``.

        :param area: Optional human-readable area name. When provided, it is
            resolved with the same area-resolution logic as :meth:`select`, and
            only that resolved area directory is considered.
        :param older_than: Optional age threshold. Only area directories older
            than this timestamp are deleted. Accepts ``YYYY-MM-DD`` or ISO datetime.
            Age is determined from ``created_at`` in area store attrs when available,
            otherwise filesystem mtime.
        :param include_incomplete: If ``True`` (default), incomplete/partial area
            directories are eligible for deletion. If ``False``, only complete area
            directories (both stores present and ``store_state == "complete"``) are
            eligible.
        :returns: Number of area directories deleted.
        :raises ValueError: If ``area`` is not a string/None, is empty, or if
            ``older_than`` has invalid format.
        """
        areas_root = self.path / "areas"

        if not areas_root.exists():
            logger.info(
                "clean_areas: deleted=0 (scanned=0, area=%r, older_than=%r, include_incomplete=%r, areas_root=%s). "
                "No area stores directory exists.",
                area,
                older_than,
                include_incomplete,
                areas_root,
            )
            return 0

        area_id_filter = resolve_area_cleanup_id_policy(
            area=area,
            resolve_area_name=lambda value: self._resolve_area_name(value),
        )
        area_dirs = list_area_dirs(areas_root)
        if area_id_filter is not None:
            area_dirs = [p for p in area_dirs if p.name == area_id_filter]

        threshold_dt = parse_older_than_policy(older_than)
        selected, scanned = select_area_dirs_for_cleanup_policy(
            area_dirs=area_dirs,
            include_incomplete=include_incomplete,
            threshold_dt=threshold_dt,
            read_area_meta=read_area_store_meta,
        )
        for area_dir in selected:
            delete_area_dir(area_dir)
        deleted = len(selected)

        if deleted == 0:
            logger.info(
                "clean_areas: deleted=0 (scanned=%d, area=%r, older_than=%r, include_incomplete=%r, areas_root=%s). "
                "No matching area stores were found.",
                scanned,
                area,
                older_than,
                include_incomplete,
                areas_root,
            )
        else:
            logger.info(
                "clean_areas: deleted=%d (scanned=%d, area=%r, older_than=%r, include_incomplete=%r, areas_root=%s).",
                deleted,
                scanned,
                area,
                older_than,
                include_incomplete,
                areas_root,
            )

        return deleted

    # -------------------------------------------------------------------------
    # Consolidated Analysis Export API (PR7)
    # -------------------------------------------------------------------------

    def export_analysis_dataset_zarr(
        self,
        path: str | Path,
        *,
        domain: str = "both",
        include_only: Sequence[str] | None = None,
        prefix: bool = True,
        exclude_template: bool = True,
        compute: bool = True,
    ) -> Path:
        """Export consolidated analysis dataset to a Zarr store.

        Creates a schema-versioned Zarr store with provenance tracking.
        The export includes variables from wind and/or landscape domains
        with optional domain prefixing for collision avoidance.

        :param path: Output path for the Zarr store. Must end with '.zarr'.
        :param domain: Which domain(s) to include: ``"wind"``, ``"landscape"``,
            or ``"both"`` (default).
        :param include_only: Optional list of variables to export. For
            ``domain="both"`` with ``prefix=True``, use prefixed names
            (e.g., ``"wind__capacity_factors"``, ``"landscape__valid_mask"``).
        :param prefix: When ``domain="both"``, prefix variables with
            ``"wind__"`` / ``"landscape__"`` to avoid collisions.
            Ignored for single-domain exports. Default ``True``.
        :param exclude_template: If ``True`` (default), exclude template
            variables from the export.
        :param compute: If ``True`` (default), compute dask arrays before
            writing. If ``False``, write lazily.

        :returns: Path to the created Zarr store.
        :rtype: pathlib.Path

        :raises ValueError: If ``path`` doesn't end with '.zarr', if stores
            are not ready, or if variable selection fails.
        :raises FileExistsError: If the export store already exists.

        Example::

            # Export both domains with prefixing
            atlas.export_analysis_dataset_zarr(
                "output/analysis_export.zarr",
                domain="both",
                prefix=True,
            )

            # Export only wind domain, specific variables
            atlas.export_analysis_dataset_zarr(
                "output/wind_export.zarr",
                domain="wind",
                include_only=["capacity_factors", "mean_wind_speed"],
            )
        """
        from cleo.exports import export_analysis_dataset_zarr

        return export_analysis_dataset_zarr(
            self,
            path,
            domain=domain,
            include_only=include_only,
            prefix=prefix,
            exclude_template=exclude_template,
            compute=compute,
        )

    @property
    def path(self):
        """Atlas workspace root path.

        :returns: Current atlas root path.
        :rtype: pathlib.Path
        """
        return self._path

    @path.setter
    def path(self, value):
        """Set atlas workspace root path.

        :param value: Filesystem path for the atlas root.
        :type value: str | pathlib.Path
        """
        value = Path(value)
        self._path = value

    @property
    def crs(self):
        """Atlas CRS string.

        :returns: Canonical CRS string used for atlas operations.
        :rtype: str
        """
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
        defaults. Call before Atlas.build() or build_canonical().

        Changing configured turbines changes the wind inputs_id, triggering
        rebuild on next build().

        :param turbines: Non-empty sequence of turbine IDs
            (for example ``["Enercon.E40.500"]``).
        :raises ValueError: If ``turbines`` is empty, contains non-strings,
            contains empty/whitespace-only IDs, or contains duplicates.

        Example:
            >>> Atlas.configure_turbines(["Enercon.E40.500", "Vestas.V90.2000"])
            >>> Atlas.build()  # Materializes only configured turbines
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

        :returns: Tuple of turbine IDs if :meth:`configure_turbines` was called,
            otherwise ``None``. ``None`` means default turbine selection logic
            is used during materialization.
        """
        return self._turbines_configured

    def configure_timebase(self, *, hours_per_year: float) -> None:
        """Configure timebase assumptions for annualized metrics.

        Affects LCOE-family metrics (lcoe, min_lcoe_turbine, optimal_power,
        optimal_energy). Physics metrics (capacity_factors, mean_wind_speed,
        rews_mps) are timebase-independent.

        :param hours_per_year: Hours per year for energy calculations.
            Must be finite and > 0. Default (if not configured) is 8766.0.
        :raises TypeError: If hours_per_year is not numeric.
        :raises ValueError: If hours_per_year is not finite or <= 0.
        """
        if not isinstance(hours_per_year, (int, float)):
            raise TypeError(f"hours_per_year must be numeric, got {type(hours_per_year).__name__}")
        hours_per_year = float(hours_per_year)
        if not (hours_per_year > 0 and math.isfinite(hours_per_year)):
            raise ValueError(f"hours_per_year must be finite and > 0, got {hours_per_year}")
        self._timebase_configured = {"hours_per_year": hours_per_year}

    @property
    def timebase_configured(self) -> dict[str, float] | None:
        """Configured timebase assumptions, or None for defaults.

        :returns: Dict with 'hours_per_year' if configured, else None.
        """
        return self._timebase_configured

    def _effective_hours_per_year(self) -> float:
        """Resolve effective hours_per_year for economics computations."""
        if self._timebase_configured is not None:
            return float(self._timebase_configured["hours_per_year"])
        return float(self.DEFAULT_HOURS_PER_YEAR)

    def configure_economics(
        self,
        *,
        discount_rate: float | None = None,
        lifetime_a: int | None = None,
        om_fixed_eur_per_kw_a: float | None = None,
        om_variable_eur_per_kwh: float | None = None,
        bos_cost_share: float | None = None,
        grid_connect_cost_eur_per_kw: float | None = None,
    ) -> None:
        """Configure baseline economics assumptions for LCOE-family metrics.

        Configured values serve as defaults for LCOE-family metric computations.
        Per-call overrides via ``economics={...}`` in ``compute(...)`` take
        precedence over these baseline values.

        :param discount_rate: Discount rate for NPV calculations (e.g., 0.05 for 5%).
            Must be in range [0, 1).
        :param lifetime_a: Project lifetime in years. Must be positive integer.
        :param om_fixed_eur_per_kw_a: Fixed O&M cost in EUR/kW/year.
            Must be non-negative.
        :param om_variable_eur_per_kwh: Variable O&M cost in EUR/kWh.
            Must be non-negative.
        :param bos_cost_share: Balance-of-system CAPEX share (location-dependent).
            Must be in range [0, 1]. Default when not configured is 0.0.
        :param grid_connect_cost_eur_per_kw: Grid connection cost rate in EUR/kW.
            Must be non-negative. Default when not configured is 50.0 (based on
            Austrian regulation §54 ElWOG). Set to 0.0 to exclude grid connection
            costs (e.g., for paper qLCOE without grid costs).

        :raises TypeError: If any parameter has wrong type.
        :raises ValueError: If any parameter is out of valid range.

        Example:
            >>> atlas.configure_economics(
            ...     discount_rate=0.05,
            ...     lifetime_a=25,
            ...     om_fixed_eur_per_kw_a=20.0,
            ...     om_variable_eur_per_kwh=0.008,
            ...     bos_cost_share=0.30,
            ...     grid_connect_cost_eur_per_kw=50.0,
            ... )
        """
        config: dict[str, float | int] = {}

        if discount_rate is not None:
            config["discount_rate"] = _validate_unit_range("discount_rate", discount_rate, include_max=False)

        if lifetime_a is not None:
            config["lifetime_a"] = _validate_positive_int("lifetime_a", lifetime_a)

        if om_fixed_eur_per_kw_a is not None:
            config["om_fixed_eur_per_kw_a"] = _validate_non_negative_numeric(
                "om_fixed_eur_per_kw_a", om_fixed_eur_per_kw_a
            )

        if om_variable_eur_per_kwh is not None:
            config["om_variable_eur_per_kwh"] = _validate_non_negative_numeric(
                "om_variable_eur_per_kwh", om_variable_eur_per_kwh
            )

        if bos_cost_share is not None:
            config["bos_cost_share"] = _validate_unit_range("bos_cost_share", bos_cost_share, include_max=True)

        if grid_connect_cost_eur_per_kw is not None:
            config["grid_connect_cost_eur_per_kw"] = _validate_non_negative_numeric(
                "grid_connect_cost_eur_per_kw", grid_connect_cost_eur_per_kw
            )

        if config:
            if self._economics_configured is None:
                self._economics_configured = config
            else:
                self._economics_configured.update(config)

    @property
    def economics_configured(self) -> dict[str, float | int] | None:
        """Configured economics assumptions, or None if not configured.

        :returns: Dict with configured economics fields, or None.
        """
        return self._economics_configured

    def _effective_economics(self, overrides: dict | None = None) -> dict[str, float | int]:
        """Resolve effective economics by merging baseline + per-call overrides.

        :param overrides: Per-call override dict from ``economics={...}``.
        :returns: Merged effective economics dict.
        """
        # Start with defaults
        effective: dict[str, float | int] = {
            "bos_cost_share": 0.0,  # Default: all CAPEX is location-independent
            "grid_connect_cost_eur_per_kw": 50.0,  # Default: 50 EUR/kW (Austrian §54 ElWOG)
        }

        # Apply configured baseline
        if self._economics_configured is not None:
            effective.update(self._economics_configured)

        # Apply per-call overrides
        if overrides is not None:
            effective.update(overrides)

        return effective

    def _deploy_resources(self) -> None:
        """Ensure packaged YAML resources exist in ``<atlas.path>/resources``."""
        import shutil
        from importlib import resources as importlib_resources

        dest_dir = Path(self.path) / "resources"
        dest_dir.mkdir(parents=True, exist_ok=True)

        pkg_root = importlib_resources.files("cleo").joinpath("resources")
        if not pkg_root.is_dir():
            raise FileNotFoundError(
                "Cleo packaged resources are missing (expected package dir `cleo/resources`). "
                "This indicates a broken installation/build. "
                "Reinstall from a proper wheel/sdist and ensure project dependencies are installed."
            )

        packaged = [p for p in pkg_root.iterdir() if p.is_file() and p.name.lower().endswith(".yml")]
        if not packaged:
            raise FileNotFoundError(
                "Cleo packaged resources directory exists but contains no *.yml files. "
                "This indicates a broken installation/build."
            )

        copied = 0
        skipped = 0
        for p in packaged:
            dest = dest_dir / p.name
            if dest.exists():
                skipped += 1
                continue
            with importlib_resources.as_file(p) as src_path:
                shutil.copy(src_path, dest)
            copied += 1

        logger.info(f"Resource files ensured in {dest_dir} (copied={copied}, skipped_existing={skipped}).")

    def _setup_logging(self, console_level: str = "INFO", file_level: str = "DEBUG") -> None:
        """Configure the ``cleo`` logger namespace without mutating root logger."""
        log_dir = Path(self.path) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"cleo_{self.country}.log"

        cleo_logger = logging.getLogger("cleo")
        cleo_logger.setLevel(logging.DEBUG)
        cleo_logger.propagate = False

        for handler in list(cleo_logger.handlers):
            cleo_logger.removeHandler(handler)

        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, str(console_level).upper(), logging.INFO))
        ch.setFormatter(fmt)

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(getattr(logging, str(file_level).upper(), logging.DEBUG))
        fh.setFormatter(fmt)

        cleo_logger.addHandler(ch)
        cleo_logger.addHandler(fh)

    def _setup_directories(self) -> None:
        """
        Create required workspace directories if they do not exist.
        """
        path_raw = self.path / "data" / "raw" / self.country
        path_logging = self.path / "logs"

        for path in [path_raw, path_logging]:
            if not path.is_dir():
                path.mkdir(parents=True)

    def _ensure_nuts_shapefile(self, *, auto_download: bool = True) -> Path:
        """Ensure a NUTS shapefile is available under ``<atlas.path>/data/nuts``.

        :param auto_download: If ``True`` (default), attempt NUTS
            download/extract when no local shapefile exists.
        :type auto_download: bool
        :returns: Path to a discovered NUTS shapefile.
        :rtype: pathlib.Path
        :raises FileNotFoundError: If no shapefile is available after optional
            auto-download.
        """

        def _first_nuts_shapefile() -> Path | None:
            nuts_dir_local = self.path / "data" / "nuts"
            shp_files = sorted(nuts_dir_local.rglob("*.shp")) if nuts_dir_local.exists() else []
            return shp_files[0] if shp_files else None

        nuts_dir = self.path / "data" / "nuts"
        existing = _first_nuts_shapefile()
        if existing is not None:
            return existing

        download_error: Exception | None = None
        if auto_download:
            logger.info(
                "NUTS shapefile missing; attempting automatic download/extract.",
                extra={"country": self.country, "nuts_dir": str(nuts_dir)},
            )
            from cleo import loaders as loaders_module

            try:
                loaders_module.load_nuts(self)
            except Exception as e:  # pragma: no cover - exact exception type depends on network/fs failures
                download_error = e
                logger.warning(
                    "Automatic NUTS download/extract failed.",
                    extra={"country": self.country, "nuts_dir": str(nuts_dir)},
                    exc_info=True,
                )

        existing = _first_nuts_shapefile()
        if existing is not None:
            return existing

        message = f"NUTS shapefile not found under {nuts_dir}."
        if auto_download:
            message += " Automatic NUTS download/extract was attempted but no shapefile is available."
            if download_error is not None:
                message += f" Last error: {download_error}"
            message += " Check network access and retry, or run cleo.loaders.load_nuts manually."
        else:
            message += " Run NUTS download/extract first (e.g. via cleo.loaders.load_nuts)."
        raise FileNotFoundError(message)

    def get_nuts_area(self, area, merged_name=None, to_atlascrs=True):
        """Return dissolved NUTS geometry for one or more areas.

        Area inputs may be NUTS names (``NAME_LATN``) or NUTS IDs.

        :param area: Area name/ID or a list of names/IDs.
        :type area: str | list[str]
        :param merged_name: Optional output value for ``NAME_LATN`` in the dissolved
            GeoDataFrame.
        :type merged_name: str | None
        :param to_atlascrs: If ``True``, reproject output to ``self.crs``.
        :returns: Single-row dissolved area geometry.
        :rtype: geopandas.GeoDataFrame
        :raises FileNotFoundError: If no NUTS shapefile is available.
        :raises TypeError: If ``area`` is neither ``str`` nor ``list[str]``.
        :raises ValueError: If any requested area is not valid for the atlas country.
        """
        nuts_shape = self._ensure_nuts_shapefile(auto_download=True)
        # Read vector via centralized helper
        nuts = _read_vector_file(nuts_shape)

        # Convert three-digit country code to two-digit country code
        alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2

        # Filter areas by country code
        feasible_areas = nuts[nuts["CNTR_CODE"] == alpha_2]

        if isinstance(area, str):
            area_list = [area]
        elif isinstance(area, list):
            area_list = area
        else:
            raise TypeError("Area must be a string or a list of strings.")

        # Accept either human names (NAME_LATN) or NUTS IDs (typically NUTS_ID)
        name_col = "NAME_LATN"
        id_col = "NUTS_ID" if "NUTS_ID" in feasible_areas.columns else None

        valid_names = set(feasible_areas[name_col].astype(str).to_numpy())
        valid_ids = set(feasible_areas[id_col].astype(str).to_numpy()) if id_col else set()

        invalid_areas = [r for r in area_list if (r not in valid_names) and (r not in valid_ids)]
        if invalid_areas:
            hint = "NAME_LATN" if id_col is None else "NAME_LATN or NUTS_ID"
            raise ValueError(f"{', '.join(invalid_areas)} are not valid areas in {self.country} (expected {hint}).")

        if id_col:
            selected_shapes = feasible_areas[
                feasible_areas[name_col].isin(area_list) | feasible_areas[id_col].isin(area_list)
            ]
        else:
            selected_shapes = feasible_areas[feasible_areas[name_col].isin(area_list)]

        merged_shape = selected_shapes.dissolve()

        # Set the name for the merged area
        merged_shape["NAME_LATN"] = merged_name if merged_name else ", ".join(area_list)
        merged_shape = merged_shape.reset_index(drop=True)

        if to_atlascrs:
            merged_shape = to_crs_if_needed(merged_shape, self.crs)

        return merged_shape

    def get_nuts_area_country(self):
        """Return country-level NUTS geometry for the configured atlas country.

        :returns: NUTS level-0 geometry rows matching atlas country.
        :rtype: geopandas.GeoDataFrame
        :raises FileNotFoundError: If no NUTS shapefile is available.
        """
        nuts_shape = self._ensure_nuts_shapefile(auto_download=True)
        # Read vector via centralized helper
        nuts = _read_vector_file(nuts_shape)
        alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2
        clip_shape = nuts.loc[(nuts["CNTR_CODE"] == alpha_2) & (nuts["LEVL_CODE"] == 0), :]
        return clip_shape
