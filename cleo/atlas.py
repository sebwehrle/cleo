# %% imports
import os
import re
import json
import zarr
import shutil
import pyproj
import logging
import datetime
import xarray as xr
import pycountry as pct
from uuid import uuid4
from pathlib import Path
from typing import Sequence

from xarray.util.generate_ops import inplace

from cleo.domains import WindDomain, LandscapeDomain
from cleo.unify import _read_vector_file
from cleo.spatial import to_crs_if_needed
from cleo.class_helpers import deploy_resources, setup_logging

logger = logging.getLogger(__name__)


def _safe_basename(path) -> str:
    """Return basename of path, or '?' on any error."""
    try:
        return Path(path).name if path else "?"
    except Exception:
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
        self.fingerprint_method = fingerprint_method

        # Base (country-wide) store paths
        self.wind_store_path = self.path / "wind.zarr"
        self.landscape_store_path = self.path / "landscape.zarr"
        self.results_root = results_root if results_root is not None else (self.path / "results")
        self.results_root.mkdir(parents=True, exist_ok=True)

        # Canonical store readiness flag
        self._canonical_ready = False

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
        except Exception:
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
        import re as _re
        return _re.sub(r"\s+", " ", name.strip()).casefold()

    def _resolve_region_name(self, name: str) -> tuple[str, str]:
        """
        Resolve region name to (normalized_name, region_id).

        Uses the region-name index stored in landscape.zarr attrs (NO heavy I/O).
        Raises if index is missing - must call materialize() first.

        Args:
            name: Human-readable region name (e.g. "Niederösterreich").

        Returns:
            Tuple of (normalized_name, region_id) where region_id is NUTS ID.

        Raises:
            ValueError: If region index missing or region name not found.
        """
        name_norm = self._normalize_region_name(name)

        # Load region-name index from landscape.zarr base store attrs
        if not self.landscape_store_path.exists():
            raise ValueError(
                "Region index missing. Run atlas.materialize() to build region index."
            )

        try:
            g = zarr.open_group(self.landscape_store_path, mode="r")
            index_json = g.attrs.get("cleo_region_name_to_id_json")
        except Exception as e:
            raise ValueError(
                f"Region index unreadable. Run atlas.materialize() to build region index. "
                f"Error: {e}"
            )

        if not index_json:
            raise ValueError(
                "Region index missing from landscape store. "
                "Run atlas.materialize() to build region index."
            )

        try:
            index = json.loads(index_json)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Region index corrupted (invalid JSON). "
                f"Re-run atlas.materialize() to rebuild. Error: {e}"
            )

        if name_norm in index:
            return name_norm, index[name_norm]

        # Provide helpful error with available regions (capped at 20)
        available = list(index.keys())[:20]
        suffix = "..." if len(index) > 20 else ""
        raise ValueError(
            f"Region '{name}' (normalized: '{name_norm}') not found in region index. "
            f"Available regions: {available}{suffix}"
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
            region = None,
            results_root = self.results_root,
            fingerprint_method = self.fingerprint_method
        )

        clone._canonical_ready = bool(getattr(self, "_canonical_ready", False))
        clone._turbines_configured = getattr(self, "_turbines_configured", None)
        clone._wind_selected_turbines = getattr(self, "_wind_selected_turbines", None)

        return clone

    def select(self, *, region: str | None = None, inplace: bool = False) -> "Atlas | None":
        """
        Select a subregion within the atlas-country.

        Semantics mirror pandas' inplace=:
            - inplace=False (default): return a new constrained Atlas; leave self unchanged
            - inplace=True: constrain self in-place and return None

        :param region: Human-readable region name (for example ``"Niederösterreich"``),
            or ``None`` to clear the selection. Strings are stripped.
        :param inplace: If ``True``, mutate ``self`` and return ``None``.
        :returns: A new constrained :class:`Atlas` when ``inplace=False``, else ``None``.
        :raises ValueError: If ``region`` is empty, whitespace-only, wrong type, or unknown.
        """
        if not inplace:
            clone = self._clone_for_selection()
            clone.select(region=region, inplace=True)
            return clone

        if region is not None:
            if not isinstance(region, str):
                raise ValueError(f"region must be a string or None, got {type(region).__name__}")
            stripped = region.strip()
            if not stripped:
                raise ValueError("region cannot be empty or whitespace-only")

            # Resolve region name to (name_norm, region_id)
            _name_norm, region_id = self._resolve_region_name(stripped)
            self._region_name = stripped  # Store original (stripped) name
            self._region_id = region_id
        else:
            self._region_name = None
            self._region_id = "__all__"

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

        from cleo.unify import Unifier

        u = Unifier(
            chunk_policy=self.chunk_policy,
            fingerprint_method=self.fingerprint_method,
        )
        # u.materialize_region(self, self._region_name)
        # Prefer region_id to avoid name-vs-id directory mismatches.
        err_id: Exception | None = None
        try:
            u.materialize_region(self, self._region_id)
        except Exception as e:
            err_id = e
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

        Creates an atomic, Windows-safe Zarr store
        under <results_root>/<run_id>/<metric_name>.zarr.

        :param metric_name: Name of the metric/result being stored.
        :param obj: :class:`xarray.Dataset` or :class:`xarray.DataArray` to store.
        :param run_id: Unique identifier for this run/experiment. If ``None``,
            :meth:`new_run_id` is used.
        :param params: Optional parameters dictionary to persist in store attrs.
        :returns: Path to the created Zarr store.
        :raises FileExistsError: If the target store already exists.
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

        :param run_id: Run identifier.
        :param metric_name: Metric name.
        :returns: Lazy :class:`xarray.Dataset` (no compute performed).
        :raises FileNotFoundError: If the store does not exist.
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

        :param run_id: If provided, only clean this run's results.
        :param older_than: If provided, only clean results older than this date.
            Accepts ``YYYY-MM-DD`` or ISO datetime format.
        :param metric_name: If provided, only clean this metric's store.
        :returns: Number of stores deleted.
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

        # Accept either human names (NAME_LATN) or NUTS IDs (typically NUTS_ID)
        name_col = "NAME_LATN"
        id_col = "NUTS_ID" if "NUTS_ID" in feasible_regions.columns else None

        valid_names = set(feasible_regions[name_col].astype(str).values)
        valid_ids = set(feasible_regions[id_col].astype(str).values) if id_col else set()

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
                f"Run NUTS download/extract first (e.g. atlas.landscape._load_nuts() / cleo.loaders.load_nuts)."
            )
        nuts_shape = shp_files[0]
        # Read vector via centralized helper
        nuts = _read_vector_file(nuts_shape)
        alpha_2 = pct.countries.get(alpha_3=self.country).alpha_2
        clip_shape = nuts.loc[(nuts["CNTR_CODE"] == alpha_2) & (nuts["LEVL_CODE"] == 0), :]
        return clip_shape
