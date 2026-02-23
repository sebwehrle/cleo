# %% imports
import re
import shutil
import pyproj
import logging
import xarray as xr
import pycountry as pct
from uuid import uuid4
from pathlib import Path
from typing import Sequence

from cleo.domains import WindDomain, LandscapeDomain
from cleo.atlas_policies.nuts_catalog import load_nuts_region_catalog as load_nuts_region_catalog_policy
from cleo.atlas_policies.region_selection import (
    normalize_region_name as normalize_region_name_policy,
    nuts_regions_level_names as nuts_regions_level_names_policy,
    resolve_region_name as resolve_region_name_policy,
    select_region_decision as select_region_decision_policy,
    validate_nuts_level as validate_nuts_level_policy,
)
from cleo.atlas_policies.cleanup import (
    parse_older_than as parse_older_than_policy,
    resolve_region_cleanup_id as resolve_region_cleanup_id_policy,
    select_region_dirs_for_cleanup as select_region_dirs_for_cleanup_policy,
    select_result_stores_for_cleanup as select_result_stores_for_cleanup_policy,
)
from cleo.results import (
    delete_result_store,
    list_result_stores,
    prune_empty_run_dirs,
    read_result_store_datetime,
    validate_result_path_token,
)
from cleo.unification.nuts_io import _read_vector_file, _read_nuts_region_catalog
from cleo.unification.store_io import (
    delete_region_dir,
    list_region_dirs,
    open_zarr_dataset,
    read_region_store_meta,
    read_zarr_group_attrs,
    write_netcdf_atomic,
)
from cleo.spatial import to_crs_if_needed
from cleo.class_helpers import deploy_resources, setup_logging
from cleo.dask_utils import normalize_compute_backend, normalize_compute_workers

logger = logging.getLogger(__name__)


def _safe_basename(path) -> str:
    """Return basename of path, or '?' on any error."""
    try:
        return Path(path).name if path else "?"
    except (TypeError, ValueError, OSError):
        return "?"


def _slugify_region_dir(name: str) -> str:
    """Best-effort filesystem-safe region directory name (ASCII-ish)."""
    import unicodedata as _ud
    import re as _re

    # Normalize + drop diacritics
    norm = _ud.normalize("NFKD", name)
    asciiish = "".join(ch for ch in norm if not _ud.combining(ch))
    # Keep reasonably portable characters
    slug = _re.sub(r"[^A-Za-z0-9._-]+", "_", asciiish).strip("_")
    return slug or "region"


def _region_dir_candidates(region_name: str) -> list[str]:
    """Candidate directory names a region might have been stored under."""
    cand: list[str] = []
    s = region_name.strip()
    if not s:
        return cand
    cand.append(s)
    cand.append(re.sub(r"\s+", " ", s).casefold())
    cand.append(_slugify_region_dir(s))
    cand.append(_slugify_region_dir(re.sub(r"\s+", " ", s).casefold()))
    # Unique, stable order
    out: list[str] = []
    seen = set()
    for c in cand:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


class NutsRegionName(str):
    """String-like NUTS region name carrying its NUTS level metadata."""

    def __new__(cls, value: str, level: int):
        obj = str.__new__(cls, value)
        obj._nuts_level = int(level)
        return obj

    @property
    def level(self) -> int:
        """Return NUTS level metadata attached to this region name."""
        return int(self._nuts_level)


class Atlas:
    DEFAULT_NUTS_LEVEL = 2
    _VALID_NUTS_LEVELS = (1, 2, 3)

    def __init__(
        self,
        path,
        country,
        crs,
        *,
        chunk_policy: dict[str, int] | None = None,
        compute_backend: str = "serial",
        compute_workers: int | None = None,
        region: str | None = None,
        results_root: Path | None = None,
        fingerprint_method: str = "path_mtime_size",
    ):
        self.path = path
        self.country = country
        # Region selection state (use select() to set after materialize())
        # _region_name: public human-readable name (or None)
        # _region_id: internal stable id ("__all__" when no region selected)
        self._region_name: str | None = None
        self._region_id: str = "__all__"
        # _pending_region: region name passed to constructor, applied in materialize()
        self._pending_region: str | None = region
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
        self._nuts_region_catalog_cache: tuple[dict, ...] | None = None

    def __repr__(self) -> str:
        """Audit-safe repr: no IO, no mutation, bounded length."""
        try:
            country = getattr(self, "country", "?")
            region_name = getattr(self, "_region_name", None) or ""
            crs = getattr(self, "_crs", "?")
            path = _safe_basename(getattr(self, "_path", None))
            canonical = "ready" if getattr(self, "_canonical_ready", False) else "not_ready"

            region_part = f", region={region_name!r}" if region_name else ""
            return f"Atlas(country={country!r}{region_part}, crs={crs!r}, path={path!r}, stores={canonical})"
        except (AttributeError, TypeError, ValueError):
            return "Atlas(?)"

    __str__ = __repr__

    def materialize(self):
        """Materialize stores: base (country-wide) and region (if selected).

        - Always creates/updates base stores (wind.zarr, landscape.zarr)
        - If region passed to constructor or select(), also creates region stores

        Must be called before accessing wind/landscape data.
        """
        # Always ensure base stores exist
        if not self._canonical_ready:
            self.materialize_canonical()

        # Apply pending region from constructor (now that stores exist)
        if self._pending_region is not None:
            self.select(region=self._pending_region, inplace=True)
            self._pending_region = None  # Clear after applying

        # If region selected, ensure region stores exist
        if self._region_name is not None:
            self._ensure_region_stores()

    @property
    def wind(self) -> WindDomain:
        """Access WindDomain

        Returns a domain object with lazy, cached .data property.
        Use atlas.wind_data for direct dataset access.
        """
        if self._wind_domain is None:
            self._wind_domain = WindDomain(self)
        return self._wind_domain

    @property
    def landscape(self) -> LandscapeDomain:
        """Access LandscapeDomain

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

    def flatten(
        self,
        *,
        domain: str = "wind",
        digits: int = 5,
        exclude_template: bool = True,
        include_domain_prefix: bool = True,
        cast_binary_to_int: bool = False,
        include_only: Sequence[str] | None = None,
    ):
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

        Requires prior materialize() call to create the skeleton store.
        """
        if not getattr(self, "_canonical_ready", False):
            raise RuntimeError(
                "Canonical stores not ready. Call atlas.materialize() first."
            )
        return open_zarr_dataset(self.wind_store_path, chunk_policy=self.chunk_policy)

    @property
    def landscape_zarr(self) -> xr.Dataset:
        """Open canonical landscape zarr store as xr.Dataset.

        Requires prior materialize() call to create the skeleton store.
        """
        if not getattr(self, "_canonical_ready", False):
            raise RuntimeError(
                "Canonical stores not ready. Call atlas.materialize() first."
            )
        return open_zarr_dataset(self.landscape_store_path, chunk_policy=self.chunk_policy)

    # -------------------------------------------------------------------------
    # Region selection (contract A4, B1)
    # -------------------------------------------------------------------------

    @property
    def region(self) -> str | None:
        """Current region selection (public name or None for full-country).

        Returns pending region if passed to constructor but not yet applied
        (before materialize()), otherwise returns the resolved region name.
        """
        if self._region_name is not None:
            return self._region_name
        # Return pending region if set but not yet applied (before materialize())
        return self._pending_region

    @region.setter
    def region(self, value: str | None) -> None:
        """Set region selection (equivalent to select(region=value))."""
        self.select(region=value, inplace=True)

    def _normalize_region_name(self, name: str) -> str:
        """Normalize region name: strip, collapse whitespace, casefold."""
        return normalize_region_name_policy(name)

    def _validate_nuts_level(self, level: int) -> int:
        """Validate and normalize NUTS level."""
        return validate_nuts_level_policy(level, valid_levels=self._VALID_NUTS_LEVELS)

    def _load_nuts_region_catalog(self) -> list[dict]:
        """Load NUTS region catalog from store attrs (fast) or raw NUTS files (fallback)."""
        def _log_debug(msg: str) -> None:
            logger.debug(
                msg,
                extra={"landscape_store_path": str(self.landscape_store_path)},
                exc_info=True,
            )

        catalog, cache = load_nuts_region_catalog_policy(
            cached_rows=self._nuts_region_catalog_cache,
            landscape_store_path=self.landscape_store_path,
            valid_levels=self._VALID_NUTS_LEVELS,
            read_store_attrs=read_zarr_group_attrs,
            read_raw_catalog=lambda: _read_nuts_region_catalog(self.path, self.country),
            log_debug=_log_debug,
        )
        self._nuts_region_catalog_cache = cache
        return catalog

    @property
    def nuts_regions(self) -> tuple[NutsRegionName, ...]:
        """List selectable region names for default NUTS level (level 2)."""
        return self.nuts_regions_level(self.DEFAULT_NUTS_LEVEL)

    def nuts_regions_level(self, level: int) -> tuple[NutsRegionName, ...]:
        """List selectable region names for a specific NUTS level.

        :param level: NUTS level (1, 2, or 3).
        :returns: Tuple of region names tagged with level metadata.
        :raises ValueError: If level is invalid or no regions are available.
        """
        catalog = self._load_nuts_region_catalog()
        return nuts_regions_level_names_policy(
            level=level,
            country=self.country,
            catalog_rows=catalog,
            validate_level=self._validate_nuts_level,
            make_region_name=lambda name, level_i: NutsRegionName(name, level_i),
        )

    def _resolve_region_name(self, name: str, *, region_level: int | None = None) -> tuple[str, str, int]:
        """
        Resolve region name to ``(normalized_name, region_id, level)``.
        """
        catalog = self._load_nuts_region_catalog()
        return resolve_region_name_policy(
            name=name,
            region_level=region_level,
            default_level=self.DEFAULT_NUTS_LEVEL,
            country=self.country,
            catalog_rows=catalog,
            normalize_name=self._normalize_region_name,
            validate_level=self._validate_nuts_level,
            default_regions_supplier=lambda: self.nuts_regions_level(self.DEFAULT_NUTS_LEVEL),
        )

    def _clone_for_selection(self) -> "Atlas":
        """
        Create a new Atlas instance that shares the same on-disk stores.

        This is intentionally a fresh instance (no domain caches), so the returned
        object behaves like a normal Atlas created from scratch.
        """

        clone = Atlas(
            self.path,
            country = self.country,
            crs = self.crs,
            chunk_policy = dict(self.chunk_policy) if self.chunk_policy is not None else None,
            compute_backend = self.compute_backend,
            compute_workers = self.compute_workers,
            region = None,
            results_root = self.results_root,
            fingerprint_method = self.fingerprint_method
        )

        clone._canonical_ready = bool(getattr(self, "_canonical_ready", False))
        clone._turbines_configured = getattr(self, "_turbines_configured", None)
        clone._wind_selected_turbines = getattr(self, "_wind_selected_turbines", None)

        return clone

    def select(
        self,
        *,
        region: str | NutsRegionName | None = None,
        region_level: int | None = None,
        inplace: bool = False,
    ) -> "Atlas | None":
        """
        Select a subregion within the atlas-country.

        Semantics mirror pandas' inplace=:
            - inplace=False (default): return a new constrained Atlas; leave self unchanged
            - inplace=True: constrain self in-place and return None

        :param region: Human-readable region name (for example ``"Niederösterreich"``),
            a ``NutsRegionName`` value, or ``None`` to clear selection.
        :param region_level: Optional NUTS level override for resolving plain strings.
        :param inplace: If ``True``, mutate ``self`` and return ``None``.
        :returns: A new constrained :class:`Atlas` when ``inplace=False``, else ``None``.
        :raises ValueError: If ``region`` is empty, whitespace-only, wrong type, or unknown.
        """
        if not inplace:
            clone = self._clone_for_selection()
            clone.select(region=region, region_level=region_level, inplace=True)
            return clone

        decision = select_region_decision_policy(
            region=region,
            region_level=region_level,
            nuts_region_name_type=NutsRegionName,
            validate_level=self._validate_nuts_level,
            resolve_name=lambda name, level: self._resolve_region_name(name, region_level=level),
        )
        self._region_name = decision.region_name
        self._region_id = decision.region_id

        # Invalidate domain caches so .data reloads from correct store
        if self._wind_domain is not None:
            self._wind_domain._data = None
        if self._landscape_domain is not None:
            self._landscape_domain._data = None

        return None

    @property
    def _effective_region_id(self) -> str:
        """
        Effective region ID for store paths (contract B1).

        Returns "__all__" when no region selected, otherwise the internal region_id.
        """
        return self._region_id

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
        if self._region_name is not None:
            return self._region_store_root() / "wind.zarr"
        return self.wind_store_path

    def _active_landscape_store_path(self) -> Path:
        """
        Path to the active landscape store (base or region).

        When region selected: <ROOT>/regions/<region_id>/landscape.zarr
        When no region: <ROOT>/landscape.zarr (base store)
        """
        if self._region_name is not None:
            return self._region_store_root() / "landscape.zarr"
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

    def _ensure_region_stores(self) -> None:
        """
        Ensure region stores exist for current region selection.

        Creates region stores by subsetting from country stores if needed.
        Does nothing if no region selected.

        Robustness:
        - materialize_region is called with region_id (NUTS) first to match store layout
        - if a legacy directory exists under a different name, it is migrated to region_id
        - if a stale / partial region directory exists, it is removed and rebuilt
        """
        if self._region_name is None:
            return

        expected_root = self.path / "regions" / self._region_id
        expected_wind = expected_root / "wind.zarr"
        expected_land = expected_root / "landscape.zarr"


        if expected_wind.exists() and expected_land.exists():
            return

        # If we have a stale/partial directory, remove it so Unifier can't mis-detect completeness.

        if expected_root.exists() and (not expected_wind.exists() or not expected_land.exists()):
            logger.warning(
                f"Region store directory exists but is incomplete: "
                f"{expected_root} (wind={expected_wind.exists()}, landscape={expected_land.exists()}). "
                "Removing and rebuilding."
            )
            shutil.rmtree(expected_root)

        from cleo.unification import Unifier

        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        # u.materialize_region(self, self._region_name)
        # Prefer region_id to avoid name-vs-id directory mismatches.
        err_id: Exception | None = None
        try:
            u.materialize_region(self, self._region_id)
        except (RuntimeError, ValueError, TypeError, FileNotFoundError, OSError) as e:
            err_id = e
            logger.warning(
                "materialize_region(region_id) failed; retrying with region name for compatibility.",
                extra={"region_id": self._region_id, "region_name": self._region_name},
                exc_info=True,
            )
            # Backwards-compat: older Unifier versions may expect region name.
            u.materialize_region(self, self._region_name)

        # If Unifier wrote into a legacy directory name, migrate to the canonical region_id layout.
        if not (expected_wind.exists() and expected_land.exists()):
            for cand in _region_dir_candidates(self._region_name):
                alt_root = self.path / "regions" / cand
                if alt_root == expected_root:
                    continue
                alt_wind = alt_root / "wind.zarr"
                alt_land = alt_root / "landscape.zarr"
                if alt_wind.exists() and alt_land.exists():
                    logger.warning(f"Found region stores under legacy directory {alt_root}; moving to {expected_root}.")
                    if expected_root.exists():
                        shutil.rmtree(expected_root)
                    shutil.move(str(alt_root), str(expected_root))
                    break

        if not (expected_wind.exists() and expected_land.exists()):
            details = {
                "expected_root": str(expected_root),
                "expected_wind_exists": expected_wind.exists(),
                "expected_landscape_exists": expected_land.exists(),
            }
            cand_roots = [str(self.path / "regions" / c) for c in _region_dir_candidates(self._region_name)[:8]]
            details["candidate_roots"] = cand_roots
            msg = (f"Region stores are still missing after materialize_region(). Details: {details}")
            if err_id is not None:
                msg += f" (materialize_region(region_id) failed with: {type(err_id).__name__}: {err_id})"
            raise RuntimeError(msg)

    def materialize_canonical(self) -> None:
        """Materialize both wind.zarr and landscape.zarr canonical stores.

        This provides the canonical pathway for creating fully unified stores.

        Creates:
        - wind.zarr: Complete canonical wind store with GWA data
        - landscape.zarr: Complete canonical landscape store with valid_mask and elevation

        Requires that all GWA data files are present.
        """
        from cleo.unification import Unifier

        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        u.materialize_wind(self)
        u.materialize_landscape(self)
        self._canonical_ready = True

    def materialize_clc(
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
        :param url: Optional source URL override.
        :param force_download: If ``True``, re-download source raster.
        :param force_prepare: If ``True``, rebuild prepared country cache.
        :returns: Path to prepared CLC GeoTIFF.
        """
        from cleo.clc import materialize_clc

        return materialize_clc(
            self,
            source=source,
            url=url,
            force_download=force_download,
            force_prepare=force_prepare,
        )

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
            raise FileNotFoundError(
                f"Result store not found: {store_path}. "
                f"Run persist() first."
            )

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
            raise FileNotFoundError(
                f"Result store not found: {src_store}. "
                f"Run persist() first."
            )

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
                "Note: atlas.wind.compute(...).cache() writes to wind.zarr, not results_root.",
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

    def clean_regions(
        self,
        region: str | None = None,
        older_than: str | None = None,
        *,
        include_incomplete: bool = True,
    ) -> int:
        """Clean up materialized region stores under ``<atlas.path>/regions``.

        Deletes region directories that match the given filters. A region directory
        is expected to contain ``wind.zarr`` and/or ``landscape.zarr``.

        :param region: Optional human-readable region name. When provided, it is
            resolved with the same region-resolution logic as :meth:`select`, and
            only that resolved region directory is considered.
        :param older_than: Optional age threshold. Only region directories older
            than this timestamp are deleted. Accepts ``YYYY-MM-DD`` or ISO datetime.
            Age is determined from ``created_at`` in region store attrs when available,
            otherwise filesystem mtime.
        :param include_incomplete: If ``True`` (default), incomplete/partial region
            directories are eligible for deletion. If ``False``, only complete region
            directories (both stores present and ``store_state == "complete"``) are
            eligible.
        :returns: Number of region directories deleted.
        :raises ValueError: If ``region`` is not a string/None, is empty, or if
            ``older_than`` has invalid format.
        """
        regions_root = self.path / "regions"

        if not regions_root.exists():
            logger.info(
                "clean_regions: deleted=0 (scanned=0, region=%r, older_than=%r, include_incomplete=%r, regions_root=%s). "
                "No region stores directory exists.",
                region,
                older_than,
                include_incomplete,
                regions_root,
            )
            return 0

        region_id_filter = resolve_region_cleanup_id_policy(
            region=region,
            resolve_region_name=lambda value: self._resolve_region_name(value),
        )
        region_dirs = list_region_dirs(regions_root)
        if region_id_filter is not None:
            region_dirs = [p for p in region_dirs if p.name == region_id_filter]

        threshold_dt = parse_older_than_policy(older_than)
        selected, scanned = select_region_dirs_for_cleanup_policy(
            region_dirs=region_dirs,
            include_incomplete=include_incomplete,
            threshold_dt=threshold_dt,
            read_region_meta=read_region_store_meta,
        )
        for region_dir in selected:
            delete_region_dir(region_dir)
        deleted = len(selected)

        if deleted == 0:
            logger.info(
                "clean_regions: deleted=0 (scanned=%d, region=%r, older_than=%r, include_incomplete=%r, regions_root=%s). "
                "No matching region stores were found.",
                scanned,
                region,
                older_than,
                include_incomplete,
                regions_root,
            )
        else:
            logger.info(
                "clean_regions: deleted=%d (scanned=%d, region=%r, older_than=%r, include_incomplete=%r, regions_root=%s).",
                deleted,
                scanned,
                region,
                older_than,
                include_incomplete,
                regions_root,
            )

        return deleted

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
        defaults. Call before Atlas.materialize() or materialize_canonical().

        Changing configured turbines changes the wind inputs_id, triggering
        rebuild on next materialize().

        :param turbines: Non-empty sequence of turbine IDs
            (for example ``["Enercon.E40.500"]``).
        :raises ValueError: If ``turbines`` is empty, contains non-strings,
            contains empty/whitespace-only IDs, or contains duplicates.

        Example:
            >>> Atlas.configure_turbines(["Enercon.E40.500", "Vestas.V90.2000"])
            >>> Atlas.materialize()  # Materializes only configured turbines
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
        """Return dissolved NUTS geometry for one or more regions.

        Region inputs may be NUTS names (``NAME_LATN``) or NUTS IDs.

        :param region: Region name/ID or a list of names/IDs.
        :type region: str | list[str]
        :param merged_name: Optional output value for ``NAME_LATN`` in the dissolved
            GeoDataFrame.
        :type merged_name: str | None
        :param to_atlascrs: If ``True``, reproject output to ``self.crs``.
        :returns: Single-row dissolved region geometry.
        :rtype: geopandas.GeoDataFrame
        :raises FileNotFoundError: If no NUTS shapefile is available.
        :raises TypeError: If ``region`` is neither ``str`` nor ``list[str]``.
        :raises ValueError: If any requested region is not valid for the atlas country.
        """
        nuts_dir = self.path / "data" / "nuts"
        shp_files = sorted(nuts_dir.rglob("*.shp")) if nuts_dir.exists() else []
        if not shp_files:
            raise FileNotFoundError(
                f"NUTS shapefile not found under {nuts_dir}. "
                "Run NUTS download/extract first (e.g. via cleo.loaders.load_nuts)."
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

        # Accept either human names (NAME_LATN) or NUTS IDs (typically NUTS_ID)
        name_col = "NAME_LATN"
        id_col = "NUTS_ID" if "NUTS_ID" in feasible_regions.columns else None

        valid_names = set(feasible_regions[name_col].astype(str).to_numpy())
        valid_ids = set(feasible_regions[id_col].astype(str).to_numpy()) if id_col else set()

        invalid_regions = [
            r for r in region_list
            if (r not in valid_names) and (r not in valid_ids)
        ]
        if invalid_regions:
            hint = "NAME_LATN" if id_col is None else "NAME_LATN or NUTS_ID"
            raise ValueError(
                f"{', '.join(invalid_regions)} are not valid regions in {self.country} "
                f"(expected {hint})."
            )

        if id_col:
            selected_shapes = feasible_regions[
                feasible_regions[name_col].isin(region_list)
                | feasible_regions[id_col].isin(region_list)
                ]
        else:
            selected_shapes = feasible_regions[feasible_regions[name_col].isin(region_list)]

        merged_shape = selected_shapes.dissolve()

        # Set the name for the merged region
        merged_shape["NAME_LATN"] = merged_name if merged_name else ", ".join(region_list)
        merged_shape = merged_shape.reset_index(drop=True)

        if to_atlascrs:
            merged_shape = to_crs_if_needed(merged_shape, self.crs)

        return merged_shape

    def get_nuts_country(self):
        """Return country-level NUTS geometry for the configured atlas country.

        :returns: NUTS level-0 geometry rows matching atlas country.
        :rtype: geopandas.GeoDataFrame
        :raises FileNotFoundError: If no NUTS shapefile is available.
        """
        nuts_dir = self.path / "data" / "nuts"
        shp_files = sorted(nuts_dir.rglob("*.shp")) if nuts_dir.exists() else []
        if not shp_files:
            raise FileNotFoundError(
                f"NUTS shapefile not found under {nuts_dir}. "
                "Run NUTS download/extract first (e.g. via cleo.loaders.load_nuts)."
            )
        nuts_shape = shp_files[0]
        # Read vector via centralized helper
        nuts = _read_vector_file(nuts_shape)
        alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2
        clip_shape = nuts.loc[(nuts["CNTR_CODE"] == alpha_2) & (nuts["LEVL_CODE"] == 0), :]
        return clip_shape
