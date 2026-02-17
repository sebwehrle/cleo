"""
Unified Atlas foundations: Unifier and __manifest__ schema helpers.

This module provides the infrastructure for creating and managing unified
atlas stores with proper versioning, identity tracking, and metadata.
All raw I/O (GeoTIFF/YAML/network) is centralized here.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import xarray as xr
import yaml
import zarr
from rasterio.crs import CRS
from rasterio.enums import Resampling

from cleo.store import atomic_dir

# NOTE: cleo.loaders is imported only inside methods that need it,
# keeping the import boundary explicit. No re-exports for external use.

# Constants
GWA_HEIGHTS = [10, 50, 100, 150, 200]


# =============================================================================
# Git/Version Helpers
# =============================================================================


def get_git_info(repo_root: Path) -> dict[str, Any]:
    """Get git repository information for versioning.

    Args:
        repo_root: Path to the repository root.

    Returns:
        Dictionary with keys:
        - unify_version: git hash if available else f"unknown+{package_version}"
        - code_dirty: bool (False if git unavailable)
        - git_diff_hash: optional string (only if dirty; hash of `git diff` bytes)
        - package_version: package version if available else "unknown"
    """
    # Get package version
    try:
        from importlib.metadata import version

        package_version = version("cleo")
    except Exception:
        package_version = "unknown"

    result: dict[str, Any] = {
        "package_version": package_version,
        "code_dirty": False,
        "unify_version": f"unknown+{package_version}",
    }

    try:
        # Get git hash
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if git_hash.returncode == 0:
            result["unify_version"] = git_hash.stdout.strip()

            # Check if dirty
            git_status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if git_status.returncode == 0 and git_status.stdout.strip():
                result["code_dirty"] = True

                # Get diff hash if dirty
                git_diff = subprocess.run(
                    ["git", "diff"],
                    cwd=repo_root,
                    capture_output=True,
                    timeout=10,
                )

                if git_diff.returncode == 0:
                    diff_hash = hashlib.sha256(git_diff.stdout).hexdigest()[:16]
                    result["git_diff_hash"] = diff_hash

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # Git not available or not a repo - use defaults
        pass

    return result


# =============================================================================
# Hash Helpers
# =============================================================================


def hash_grid_id(
    crs_wkt: str,
    transform: tuple[float, ...],
    shape: tuple[int, int],
    y: np.ndarray,
    x: np.ndarray,
    mask_policy: str,
) -> str:
    """Compute a stable hash identifying a grid configuration.

    Args:
        crs_wkt: CRS in WKT format.
        transform: Affine transform tuple.
        shape: Grid shape (height, width).
        y: Y coordinate array.
        x: X coordinate array.
        mask_policy: Mask handling policy string.

    Returns:
        SHA256 hash string (first 16 chars).
    """
    h = hashlib.sha256()

    # Hash scalars as JSON for stability
    scalars = {
        "crs_wkt": crs_wkt,
        "transform": list(transform),
        "shape": list(shape),
        "mask_policy": mask_policy,
    }
    h.update(json.dumps(scalars, sort_keys=True).encode("utf-8"))

    # Hash coordinate arrays as bytes
    h.update(np.asarray(y, dtype=np.float64).tobytes())
    h.update(np.asarray(x, dtype=np.float64).tobytes())

    return h.hexdigest()[:16]


def hash_inputs_id(items: list[tuple[str, str]], method: str) -> str:
    """Compute a stable hash identifying input sources.

    Args:
        items: List of (name, fingerprint) pairs.
        method: Fingerprinting method used.

    Returns:
        SHA256 hash string (first 16 chars).
    """
    h = hashlib.sha256()

    # Include method in hash
    h.update(f"method={method}\n".encode("utf-8"))

    # Sort items for stability and hash
    for name, fingerprint in sorted(items):
        h.update(f"{name}:{fingerprint}\n".encode("utf-8"))

    return h.hexdigest()[:16]


# =============================================================================
# Fingerprinting
# =============================================================================


def fingerprint_path_mtime_size(path: Path) -> str:
    """Compute fingerprint from path, mtime, and size.

    Args:
        path: Path to file.

    Returns:
        Fingerprint string.
    """
    stat = path.stat()
    return f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}"


def fingerprint_file(path: Path, method: str = "path_mtime_size") -> str:
    """Compute fingerprint for a file using specified method.

    Args:
        path: Path to file.
        method: Fingerprinting method (currently only "path_mtime_size" supported).

    Returns:
        Fingerprint string.
    """
    if method == "path_mtime_size":
        return fingerprint_path_mtime_size(path)
    else:
        raise ValueError(f"Unknown fingerprint method: {method}")


# =============================================================================
# Manifest Helpers (stored in zarr root attrs as JSON strings)
# =============================================================================

# Manifest is stored in zarr root attrs:
#   cleo_manifest_sources_json: JSON array of source entries
#   cleo_manifest_variables_json: JSON array of variable entries


def _read_manifest(store_path: Path) -> dict:
    """Read manifest from zarr root attrs.

    Returns:
        Manifest dict with keys: version, sources, variables.
        Returns empty manifest if attrs not present.
    """
    try:
        root = zarr.open_group(store_path, mode="r")
        sources_json = root.attrs.get("cleo_manifest_sources_json", "[]")
        variables_json = root.attrs.get("cleo_manifest_variables_json", "[]")
        return {
            "version": 1,
            "sources": json.loads(sources_json),
            "variables": json.loads(variables_json),
        }
    except Exception:
        return {"version": 1, "sources": [], "variables": []}


def _write_manifest_atomic(store_path: Path, manifest: dict) -> None:
    """Write manifest to zarr root attrs."""
    root = zarr.open_group(store_path, mode="a")
    root.attrs["cleo_manifest_sources_json"] = json.dumps(
        manifest.get("sources", []), separators=(",", ":"), ensure_ascii=False
    )
    root.attrs["cleo_manifest_variables_json"] = json.dumps(
        manifest.get("variables", []), separators=(",", ":"), ensure_ascii=False
    )


def init_manifest(store_path: Path) -> None:
    """Initialize empty manifest in zarr root attrs if not present."""
    try:
        root = zarr.open_group(store_path, mode="a")
        if "cleo_manifest_sources_json" not in root.attrs:
            root.attrs["cleo_manifest_sources_json"] = "[]"
        if "cleo_manifest_variables_json" not in root.attrs:
            root.attrs["cleo_manifest_variables_json"] = "[]"
    except Exception:
        pass  # Store may not exist yet; will be initialized on first write


def write_manifest_sources(store_path: Path, sources: list[dict]) -> None:
    """Write/replace sources in the manifest.

    Args:
        store_path: Path to the zarr store root.
        sources: List of source dicts with keys:
            source_id, name, kind, path, params_json, fingerprint
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = _read_manifest(store_path)

    # Add created_at timestamp to each source
    sources_with_ts = []
    for s in sources:
        s_copy = dict(s)
        s_copy["created_at"] = now
        sources_with_ts.append(s_copy)

    manifest["sources"] = sources_with_ts
    _write_manifest_atomic(store_path, manifest)


def write_manifest_variables(store_path: Path, variables: list[dict]) -> None:
    """Write/replace variables in the manifest.

    Args:
        store_path: Path to the zarr store root.
        variables: List of variable dicts with keys:
            variable_name, source_id, resampling_method, nodata_policy, dtype
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = _read_manifest(store_path)

    # Add materialized_at timestamp to each variable
    vars_with_ts = []
    for v in variables:
        v_copy = dict(v)
        v_copy["materialized_at"] = now
        vars_with_ts.append(v_copy)

    manifest["variables"] = vars_with_ts
    _write_manifest_atomic(store_path, manifest)


# =============================================================================
# CRS Cache Helpers
# =============================================================================


def _crs_cache_path(atlas, iso3: str) -> Path:
    """Get the path for CRS cache file.

    Args:
        atlas: Atlas instance.
        iso3: ISO3 country code.

    Returns:
        Path to the CRS cache file.
    """
    return Path(atlas.path) / "intermediates" / "crs_cache" / f"{iso3}.wkt"


def _load_or_fetch_gwa_crs(atlas, iso3: str) -> CRS:
    """Load CRS from cache or fetch from GWA API.

    Fetch-once semantics: fetches from network only if cache is missing,
    then persists to cache. Subsequent calls read from cache.

    Args:
        atlas: Atlas instance.
        iso3: ISO3 country code.

    Returns:
        rasterio CRS object.

    Raises:
        RuntimeError: If fetch fails or cache is empty/corrupt.
    """
    import cleo.loaders

    cache = _crs_cache_path(atlas, iso3)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        wkt = cache.read_text(encoding="utf-8").strip()
        if not wkt:
            raise RuntimeError(f"Empty CRS cache file: {cache}")
        return CRS.from_wkt(wkt)

    # Fetch from network
    try:
        crs_str = cleo.loaders.fetch_gwa_crs(iso3)
        crs = CRS.from_string(crs_str)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch GWA CRS for {iso3}; cache missing at {cache}. Error: {e}"
        ) from e

    # Persist to cache
    try:
        cache.write_text(crs.to_wkt(), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(
            f"Fetched CRS but failed to persist cache at {cache}: {e}"
        ) from e

    return crs


# =============================================================================
# Required GWA Files
# =============================================================================


def _required_gwa_files(atlas) -> list[tuple[str, Path]]:
    """Get list of required GWA files for wind unification.

    Args:
        atlas: Atlas instance.

    Returns:
        List of (source_id, path) tuples for all required files.
    """
    raw_dir = Path(atlas.path) / "data" / "raw" / atlas.country
    required = []

    for h in GWA_HEIGHTS:
        required.extend([
            (f"gwa:file:weibull_A:{h}", raw_dir / f"{atlas.country}_combined-Weibull-A_{h}.tif"),
            (f"gwa:file:weibull_k:{h}", raw_dir / f"{atlas.country}_combined-Weibull-k_{h}.tif"),
            (f"gwa:file:rho:{h}", raw_dir / f"{atlas.country}_air-density_{h}.tif"),
        ])

    return required


def _assert_all_required_gwa_present(atlas) -> list[tuple[str, Path]]:
    """Assert all required GWA files exist. Fail fast with all missing paths.

    Args:
        atlas: Atlas instance.

    Returns:
        List of (source_id, path) tuples for all required files.

    Raises:
        FileNotFoundError: If any required files are missing (lists ALL missing paths).
    """
    req = _required_gwa_files(atlas)
    missing = [str(p) for (_sid, p) in req if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required GWA files:\n" + "\n".join(missing)
        )

    return req


# =============================================================================
# Raster Loading with CRS Cache
# =============================================================================


def _open_gwa_raster(
    atlas,
    path: Path,
    *,
    iso3: str,
    target_crs,
    ref_da: xr.DataArray | None = None,
    clip_geom=None,
    resampling: str = "bilinear",
) -> xr.DataArray:
    """Open a GWA raster with CRS cache support.

    Args:
        atlas: Atlas instance.
        path: Path to the raster file.
        iso3: ISO3 country code for CRS lookup.
        target_crs: Target CRS to reproject to.
        ref_da: Reference DataArray to match grid to (optional).
        clip_geom: Geometry to clip to (optional).
        resampling: Resampling method (default "bilinear").

    Returns:
        DataArray with CRS set, reprojected/clipped as specified.
    """
    da = rxr.open_rasterio(path, chunks=None).squeeze(drop=True)

    # Ensure CRS is set (use cache if needed)
    if da.rio.crs is None:
        crs = _load_or_fetch_gwa_crs(atlas, iso3)
        da = da.rio.write_crs(crs)

    # Map resampling string to enum
    resampling_enum = getattr(Resampling, resampling, Resampling.bilinear)

    # Reproject to target CRS if needed
    from cleo.spatial import crs_equal

    if not crs_equal(da.rio.crs, target_crs):
        da = da.rio.reproject(target_crs, nodata=np.nan, resampling=resampling_enum)

    # Clip to geometry if provided
    if clip_geom is not None:
        from cleo.spatial import to_crs_if_needed

        if hasattr(clip_geom, "geometry"):
            # GeoDataFrame
            clip_geom = to_crs_if_needed(clip_geom, target_crs)
            da = da.rio.clip(clip_geom.geometry, drop=True)
        else:
            # Assume iterable of geometries
            da = da.rio.clip(clip_geom, drop=True)

    # Match to reference grid if provided
    if ref_da is not None:
        da = da.rio.reproject_match(ref_da, nodata=np.nan, resampling=resampling_enum)

    # Convert nodata to NaN
    nodata = da.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        da = da.where(da != nodata, np.nan)

    return da


# =============================================================================
# NUTS Region Geometry
# =============================================================================


def _get_clip_geometry(atlas):
    """Get clipping geometry from NUTS region if specified.

    Args:
        atlas: Atlas instance.

    Returns:
        GeoDataFrame for clipping, or None if no region specified.
    """
    if atlas.region is None:
        return None

    return atlas.get_nuts_region(atlas.region)


# =============================================================================
# Turbine and Cost Ingestion
# =============================================================================


def _load_turbine_yaml(yaml_path: Path) -> dict:
    """Load turbine YAML file.

    Args:
        yaml_path: Path to turbine YAML file.

    Returns:
        Dict with turbine data.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Known non-turbine resource file stems to exclude from default turbine discovery
_NON_TURBINE_RESOURCE_STEMS = {"clc_codes", "cost_assumptions"}


def _default_turbines_from_resources(resources_dir: Path) -> list[str]:
    """Discover default turbines from resources directory.

    Globs for *.yml and *.yaml files, excludes known non-turbine resources,
    and returns sorted list of turbine IDs (file stems).

    Args:
        resources_dir: Path to atlas resources directory.

    Returns:
        Sorted list of turbine IDs (e.g. ["Enercon.E40.500", "Vestas.V90.2000"]).
    """
    if not resources_dir.exists():
        return []

    turbine_ids = []
    for pattern in ("*.yml", "*.yaml"):
        for yaml_path in resources_dir.glob(pattern):
            stem = yaml_path.stem
            if stem not in _NON_TURBINE_RESOURCE_STEMS:
                turbine_ids.append(stem)

    # Remove duplicates (in case both .yml and .yaml exist) and sort
    turbine_ids = sorted(set(turbine_ids))
    return turbine_ids


def _ingest_turbines_and_costs(
    atlas,
    fingerprint_method: str = "path_mtime_size",
) -> tuple[xr.Dataset, list[dict], list[dict]]:
    """Ingest turbine power curves and cost assumptions.

    Args:
        atlas: Atlas instance.
        fingerprint_method: Method for fingerprinting source files.

    Returns:
        Tuple of (dataset, sources, variables) where:
        - dataset: xr.Dataset with power_curve and turbine metadata
        - sources: List of source dicts for manifest
        - variables: List of variable dicts for manifest
    """
    resources_dir = Path(atlas.path) / "resources"

    # Determine turbines: use configured list or discover from resources
    turbines = atlas.turbines_configured
    if turbines is not None:
        turbine_names = list(turbines)
        turbines_mode = "configured"
    else:
        turbine_names = _default_turbines_from_resources(resources_dir)
        turbines_mode = "default"
        if not turbine_names:
            raise RuntimeError(
                "No turbines configured and none found in <atlas>/resources. "
                "Either call atlas.configure_turbines([...]) or add turbine YAMLs to resources/."
            )

    sources = []
    variables = []

    # Canonical wind_speed grid
    wind_speed = np.arange(0.0, 40.0 + 0.5, 0.5)

    power_curves = []
    meta_manufacturer = []
    meta_model = []
    meta_capacity = []
    meta_hub_height = []
    meta_rotor_diameter = []
    meta_commissioning_year = []
    meta_model_key = []
    turbine_ids = []

    for turbine_id in turbine_names:
        yaml_path = resources_dir / f"{turbine_id}.yml"
        if not yaml_path.exists():
            continue

        data = _load_turbine_yaml(yaml_path)

        # Register source
        sources.append({
            "source_id": f"turbine:{turbine_id}",
            "name": turbine_id,
            "kind": "yaml",
            "path": str(yaml_path),
            "params_json": json.dumps({"turbine_id": turbine_id}),
            "fingerprint": fingerprint_file(yaml_path, fingerprint_method),
        })

        # Extract data
        manufacturer = str(data["manufacturer"])
        model = str(data["model"])
        capacity = float(data["capacity"])
        hub_height = float(data["hub_height"])
        rotor_diameter = float(data["rotor_diameter"])
        commissioning_year = int(data["commissioning_year"])
        model_key = f"{manufacturer}.{model}.{capacity}"

        # Resample power curve to canonical wind_speed grid
        old_u = np.array(list(map(float, data["V"])))
        old_p = np.array(list(map(float, data["cf"])))
        new_p = np.interp(wind_speed, old_u, old_p, left=0.0, right=0.0)

        power_curves.append(new_p)
        meta_manufacturer.append(manufacturer)
        meta_model.append(model)
        meta_capacity.append(capacity)
        meta_hub_height.append(hub_height)
        meta_rotor_diameter.append(rotor_diameter)
        meta_commissioning_year.append(commissioning_year)
        meta_model_key.append(model_key)
        turbine_ids.append(turbine_id)

    # Build dataset if we have turbines
    if turbine_ids:
        # Create turbines bundle source
        turbine_source_ids = [f"turbine:{tid}" for tid in turbine_ids]
        bundle_fingerprint = hashlib.sha256(
            json.dumps(turbine_source_ids, sort_keys=True).encode()
        ).hexdigest()[:16]

        sources.append({
            "source_id": "turbines:bundle",
            "name": "turbines_bundle",
            "kind": "bundle",
            "path": "",
            "params_json": json.dumps({"source_ids": turbine_source_ids}),
            "fingerprint": bundle_fingerprint,
        })

        # Power curve array: (turbine, wind_speed)
        # Use integer indices for turbine dim; store string metadata in attrs JSON
        n_turbines = len(turbine_ids)
        turbine_indices = np.arange(n_turbines, dtype=np.int64)

        pc_data = np.array(power_curves)
        power_curve_da = xr.DataArray(
            pc_data,
            dims=["turbine", "wind_speed"],
            coords={"turbine": turbine_indices, "wind_speed": wind_speed},
            name="power_curve",
        )

        # Build turbine metadata JSON (stored in attrs, NOT as arrays)
        # This avoids Zarr v3 string dtype warnings
        turbines_meta = []
        for i, tid in enumerate(turbine_ids):
            turbines_meta.append({
                "id": tid,
                "manufacturer": meta_manufacturer[i],
                "model": meta_model[i],
                "model_key": meta_model_key[i],
            })

        # Only numeric arrays stored in dataset (no string dtype)
        ds = xr.Dataset({
            "power_curve": power_curve_da,
            "turbine_capacity": ("turbine", np.array(meta_capacity, dtype=np.float64)),
            "turbine_hub_height": ("turbine", np.array(meta_hub_height, dtype=np.float64)),
            "turbine_rotor_diameter": ("turbine", np.array(meta_rotor_diameter, dtype=np.float64)),
            "turbine_commissioning_year": ("turbine", np.array(meta_commissioning_year, dtype=np.int64)),
        })
        ds = ds.assign_coords(turbine=turbine_indices)

        # Store turbine metadata as JSON in attrs (avoids string arrays)
        ds.attrs["cleo_turbines_json"] = json.dumps(
            turbines_meta, separators=(",", ":"), ensure_ascii=False
        )

        # Register variables (only the numeric ones we're storing)
        for var_name in ds.data_vars:
            variables.append({
                "variable_name": var_name,
                "source_id": "turbines:bundle",
                "resampling_method": "none",
                "nodata_policy": "none",
                "dtype": str(ds[var_name].dtype),
            })
    else:
        ds = xr.Dataset()
        ds = ds.assign_coords(wind_speed=wind_speed)

    # Load cost assumptions if available
    cost_path = resources_dir / "cost_assumptions.yml"
    if cost_path.exists():
        sources.append({
            "source_id": "costs:default",
            "name": "cost_assumptions",
            "kind": "yaml",
            "path": str(cost_path),
            "params_json": "{}",
            "fingerprint": fingerprint_file(cost_path, fingerprint_method),
        })

    return ds, sources, variables


# =============================================================================
# Unifier Class
# =============================================================================


class Unifier:
    """Unified atlas store management.

    The Unifier handles creation and validation of unified atlas stores
    with proper versioning, identity tracking, and metadata.
    """

    def __init__(
        self,
        *,
        chunk_policy: dict[str, int] | None = None,
        fingerprint_method: str = "path_mtime_size",
    ) -> None:
        """Initialize the Unifier.

        Args:
            chunk_policy: Default chunking policy (e.g., {"x": 512, "y": 512}).
            fingerprint_method: Method for computing source fingerprints.
                Default is "path_mtime_size".
        """
        self.chunk_policy = chunk_policy or {}
        self.fingerprint_method = fingerprint_method

    def ensure_store_skeleton(
        self,
        store_path: Path,
        *,
        chunk_policy: dict[str, int],
    ) -> None:
        """Ensure a store skeleton exists at the given path.

        If the store exists, ensures __manifest__ group is present.
        If missing, creates a new skeleton store atomically.

        The skeleton store has:
        - store_state="skeleton" attribute (marker for incomplete store)
        - Placeholder grid_id and inputs_id
        - Version and code state information
        - __manifest__ group with source/variable tables

        Args:
            store_path: Path to the zarr store.
            chunk_policy: Chunking policy for the store.
        """
        if store_path.exists():
            # Ensure manifest JSON exists in existing store
            init_manifest(store_path)
            return

        # Get git/version info
        git_info = get_git_info(Path.cwd())

        # Create new skeleton store atomically
        with atomic_dir(store_path) as tmp_path:
            # Create zarr root with attrs
            root = zarr.open_group(tmp_path, mode="w")

            root.attrs["store_state"] = "skeleton"
            root.attrs["grid_id"] = ""
            root.attrs["inputs_id"] = ""
            root.attrs["unify_version"] = git_info["unify_version"]
            root.attrs["code_dirty"] = git_info["code_dirty"]
            if "git_diff_hash" in git_info:
                root.attrs["git_diff_hash"] = git_info["git_diff_hash"]
            root.attrs["chunk_policy"] = json.dumps(chunk_policy)
            root.attrs["fingerprint_method"] = self.fingerprint_method

            # Create __manifest__ group
            init_manifest(tmp_path)

    def materialize_wind(self, atlas) -> None:
        """Materialize wind.zarr as a complete canonical store.

        Creates a unified wind dataset from GWA GeoTIFFs, turbine configs,
        and cost assumptions. The store is marked as store_state="complete"
        only after successful atomic write.

        Args:
            atlas: Atlas instance with path, country, crs, and region attributes.

        Raises:
            FileNotFoundError: If any required GWA files are missing.
            RuntimeError: If CRS fetch fails.
        """
        store_path = Path(atlas.path) / "wind.zarr"
        iso3 = atlas.country
        target_crs = atlas.crs
        chunk_policy = getattr(atlas, "chunk_policy", self.chunk_policy) or {"y": 1024, "x": 1024}
        mask_policy = "nan+valid_mask_in_landscape"

        # 1. Validate all required GWA files exist (fail fast)
        req_files = _assert_all_required_gwa_present(atlas)

        # 2. Get clipping geometry if region specified
        clip_geom = _get_clip_geometry(atlas)

        # 3. Open reference raster (A at h=100) to establish grid
        ref_height = 100
        ref_path = None
        for sid, p in req_files:
            if sid == f"gwa:file:weibull_A:{ref_height}":
                ref_path = p
                break

        ref_da = _open_gwa_raster(
            atlas, ref_path,
            iso3=iso3,
            target_crs=target_crs,
            ref_da=None,
            clip_geom=clip_geom,
            resampling="bilinear",
        )

        # 4. Build sources list and load all rasters
        sources = []
        weibull_A_arrays = []
        weibull_k_arrays = []
        rho_arrays = []

        weibull_A_source_ids = []
        weibull_k_source_ids = []
        rho_source_ids = []

        for sid, path in req_files:
            # Register source
            sources.append({
                "source_id": sid,
                "name": path.name,
                "kind": "raster",
                "path": str(path),
                "params_json": json.dumps({"layer": sid.split(":")[2], "height": int(sid.split(":")[-1])}),
                "fingerprint": fingerprint_file(path, self.fingerprint_method),
            })

            # Load raster matched to reference grid
            da = _open_gwa_raster(
                atlas, path,
                iso3=iso3,
                target_crs=target_crs,
                ref_da=ref_da,
                clip_geom=clip_geom,
                resampling="bilinear",
            )

            # Categorize by layer type
            height = int(sid.split(":")[-1])
            if "weibull_A" in sid:
                weibull_A_arrays.append((height, da))
                weibull_A_source_ids.append(sid)
            elif "weibull_k" in sid:
                weibull_k_arrays.append((height, da))
                weibull_k_source_ids.append(sid)
            elif "rho" in sid:
                rho_arrays.append((height, da))
                rho_source_ids.append(sid)

        # 5. Create bundle sources
        for bundle_name, source_ids in [
            ("gwa:bundle:weibull_A", weibull_A_source_ids),
            ("gwa:bundle:weibull_k", weibull_k_source_ids),
            ("gwa:bundle:rho", rho_source_ids),
        ]:
            bundle_fingerprint = hashlib.sha256(
                json.dumps(sorted(source_ids)).encode()
            ).hexdigest()[:16]
            sources.append({
                "source_id": bundle_name,
                "name": bundle_name.split(":")[-1],
                "kind": "bundle",
                "path": "",
                "params_json": json.dumps({"source_ids": sorted(source_ids)}),
                "fingerprint": bundle_fingerprint,
            })

        # 6. Stack arrays into 3D (height, y, x)
        def stack_by_height(arrays_list):
            sorted_arrays = sorted(arrays_list, key=lambda x: x[0])
            heights = [h for h, _ in sorted_arrays]
            stacked = xr.concat([da for _, da in sorted_arrays], dim="height")
            stacked = stacked.assign_coords(height=heights)
            return stacked

        weibull_A = stack_by_height(weibull_A_arrays)
        weibull_A.name = "weibull_A"

        weibull_k = stack_by_height(weibull_k_arrays)
        weibull_k.name = "weibull_k"

        rho = stack_by_height(rho_arrays)
        rho.name = "rho"

        # 7. Create wind dataset
        ds_wind = xr.Dataset({
            "weibull_A": weibull_A,
            "weibull_k": weibull_k,
            "rho": rho,
            "template": ref_da,
        })

        # 8. Add turbines and costs
        ds_tech, tech_sources, tech_variables = _ingest_turbines_and_costs(
            atlas, self.fingerprint_method
        )
        sources.extend(tech_sources)

        # 9. Merge datasets
        ds = xr.merge([ds_wind, ds_tech])

        # Preserve turbine metadata JSON attr (xr.merge doesn't preserve all attrs)
        if "cleo_turbines_json" in ds_tech.attrs:
            ds.attrs["cleo_turbines_json"] = ds_tech.attrs["cleo_turbines_json"]

        # 10. Add wind-specific variables to manifest
        variables = []
        for var_name, bundle_source in [
            ("weibull_A", "gwa:bundle:weibull_A"),
            ("weibull_k", "gwa:bundle:weibull_k"),
            ("rho", "gwa:bundle:rho"),
        ]:
            variables.append({
                "variable_name": var_name,
                "source_id": bundle_source,
                "resampling_method": "bilinear",
                "nodata_policy": "nan",
                "dtype": str(ds[var_name].dtype),
            })

        variables.append({
            "variable_name": "template",
            "source_id": f"gwa:file:weibull_A:{ref_height}",
            "resampling_method": "bilinear",
            "nodata_policy": "nan",
            "dtype": str(ds["template"].dtype),
        })

        variables.extend(tech_variables)

        # 11. Apply chunking (y/x only, other dims use full extent)
        encoding = {}
        for var_name in ds.data_vars:
            var = ds[var_name]
            var_dims = var.dims
            if "y" in var_dims and "x" in var_dims:
                var_chunks = []
                for dim in var_dims:
                    if dim in chunk_policy:
                        var_chunks.append(chunk_policy[dim])
                    elif dim in ("y", "x"):
                        var_chunks.append(chunk_policy.get(dim, var.sizes[dim]))
                    else:
                        # Use full extent for non-spatial dims (unchunked)
                        var_chunks.append(var.sizes[dim])
                encoding[var_name] = {"chunks": tuple(var_chunks)}

        # 12. Compute grid_id
        transform = ref_da.rio.transform()
        transform_tuple = (
            transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f,
        )
        crs_wkt = str(ref_da.rio.crs.to_wkt()) if ref_da.rio.crs else ""
        shape = (ref_da.sizes["y"], ref_da.sizes["x"])

        grid_id = hash_grid_id(
            crs_wkt=crs_wkt,
            transform=transform_tuple,
            shape=shape,
            y=ref_da["y"].values,
            x=ref_da["x"].values,
            mask_policy=mask_policy,
        )

        # 13. Compute inputs_id
        input_items = []
        for s in sources:
            input_items.append((s["source_id"], s["fingerprint"]))

        # Add unify params to inputs_id
        params_str = json.dumps({
            "country": atlas.country,
            "target_crs": str(target_crs),
            "region": atlas.region,
            "chunk_policy": chunk_policy,
            "mask_policy": mask_policy,
            "fingerprint_method": self.fingerprint_method,
        }, sort_keys=True)
        params_fingerprint = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        input_items.append(("unify:params", params_fingerprint))

        # Add turbines to inputs_id (stable order, no timestamps)
        turbines_cfg = atlas.turbines_configured
        turbines_part = _stable_json(list(turbines_cfg)) if turbines_cfg is not None else "default"
        input_items.append(("turbines_configured", turbines_part))

        # Add effective turbines list (always explicit, for deterministic inputs_id)
        if turbines_cfg is not None:
            effective_turbines = sorted(turbines_cfg)
        else:
            resources_dir = Path(atlas.path) / "resources"
            effective_turbines = _default_turbines_from_resources(resources_dir)
        input_items.append(("turbines_effective", _stable_json(effective_turbines)))

        # Add SHA256 content fingerprints for each turbine YAML (deterministic, no timestamps)
        resources_dir = Path(atlas.path) / "resources"
        turbine_sha256_list = []
        for tid in effective_turbines:
            yaml_path = resources_dir / f"{tid}.yml"
            if yaml_path.exists():
                content_hash = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
                turbine_sha256_list.append({"turbine_id": tid, "sha256": content_hash})
        input_items.append(("turbines_sha256", _stable_json(turbine_sha256_list)))

        inputs_id = hash_inputs_id(input_items, self.fingerprint_method)

        # 14. Check idempotency
        if store_path.exists():
            try:
                existing_root = zarr.open_group(store_path, mode="r")
                if (
                    existing_root.attrs.get("store_state") == "complete"
                    and existing_root.attrs.get("inputs_id") == inputs_id
                ):
                    # Already up-to-date
                    return
            except Exception:
                pass

        # 15. Get git info
        git_info = get_git_info(Path.cwd())

        # 16. Atomic write
        with atomic_dir(store_path) as tmp_path:
            # Write dataset
            ds.to_zarr(tmp_path, mode="w", encoding=encoding, consolidated=False)

            # Write root attrs
            root = zarr.open_group(tmp_path, mode="a")
            root.attrs["store_state"] = "complete"
            root.attrs["grid_id"] = grid_id
            root.attrs["inputs_id"] = inputs_id
            root.attrs["mask_policy"] = mask_policy
            root.attrs["requires_landscape_valid_mask"] = True
            root.attrs["unify_version"] = git_info["unify_version"]
            root.attrs["code_dirty"] = git_info["code_dirty"]
            if "git_diff_hash" in git_info:
                root.attrs["git_diff_hash"] = git_info["git_diff_hash"]
            root.attrs["chunk_policy"] = json.dumps(chunk_policy)
            root.attrs["fingerprint_method"] = self.fingerprint_method

            # Write manifest
            init_manifest(tmp_path)
            write_manifest_sources(tmp_path, sources)
            write_manifest_variables(tmp_path, variables)

    def materialize_landscape(self, atlas) -> None:
        """Materialize landscape.zarr as a complete canonical store.

        Creates a unified landscape dataset with valid_mask (derived from wind)
        and elevation data. The store is aligned exactly to the wind grid.

        Args:
            atlas: Atlas instance with path, country, crs, and region attributes.

        Raises:
            RuntimeError: If wind.zarr is not complete or missing required attrs.
            FileNotFoundError: If no elevation source is found.
        """
        store_path = Path(atlas.path) / "landscape.zarr"
        wind_path = Path(atlas.path) / "wind.zarr"

        # Open wind canonical store (must be complete)
        wind = xr.open_zarr(wind_path, consolidated=False, chunks=self.chunk_policy)

        if wind.attrs.get("store_state") != "complete":
            raise RuntimeError(
                "wind.zarr is not complete; run Unifier.materialize_wind(atlas) first."
            )

        wind_grid_id = wind.attrs.get("grid_id") or ""
        wind_inputs_id = wind.attrs.get("inputs_id") or ""
        if not wind_grid_id or not wind_inputs_id:
            raise RuntimeError(
                "wind.zarr missing grid_id/inputs_id; cannot materialize landscape."
            )

        # Choose wind_ref for alignment
        if "weibull_A" not in wind:
            raise RuntimeError(
                "wind.zarr missing weibull_A; cannot define canonical grid."
            )

        wind_ref = wind["weibull_A"].isel(height=0)
        if "height" in wind["weibull_A"].dims:
            height_values = wind["weibull_A"]["height"].values
            if 100 in height_values:
                wind_ref = wind["weibull_A"].sel(height=100)

        # Ensure wind_ref has CRS set (zarr may not preserve it)
        if wind_ref.rio.crs is None:
            # Try to get CRS from template or atlas.crs
            if "template" in wind and wind["template"].rio.crs is not None:
                wind_ref = wind_ref.rio.write_crs(wind["template"].rio.crs)
            else:
                wind_ref = wind_ref.rio.write_crs(atlas.crs)

        # Ensure wind_ref has a valid transform
        if wind_ref.rio.transform() is None:
            wind_ref = wind_ref.rio.write_transform(
                wind_ref.rio.transform(recalc=True)
            )

        # AOI geometry: same definition as wind
        aoi_gdf = _aoi_geom_or_none(atlas)

        # valid_mask derived from wind_ref (True where wind data is valid)
        valid_mask = wind_ref.notnull().rename("valid_mask")

        # Elevation: prefer local GeoTIFF (for offline use); else CopDEM path
        raw_country = Path(atlas.path) / "data" / "raw" / atlas.country
        local_elev = raw_country / f"{atlas.country}_elevation_w_bathymetry.tif"

        elev_meta: dict[str, Any] | None = None
        if local_elev.exists():
            elevation = _open_local_elevation(
                atlas, local_elev, wind_ref, aoi_gdf
            ).rename("elevation")
            elev_kind = "local"
        else:
            elevation, elev_meta = _build_copdem_elevation(atlas, wind_ref, aoi_gdf)
            elevation = elevation.rename("elevation")
            elev_kind = "copdem"

        # Assemble dataset with EXACT wind y/x coords, and explicit CRS/transform
        ds_land = xr.Dataset(
            coords={"y": wind["y"], "x": wind["x"]},
            data_vars={"valid_mask": valid_mask, "elevation": elevation},
        )
        ds_land = ds_land.rio.write_crs(wind_ref.rio.crs)
        ds_land = ds_land.rio.write_transform(wind_ref.rio.transform())

        # Apply chunking
        chunk_y = self.chunk_policy.get("y", 1024)
        chunk_x = self.chunk_policy.get("x", 1024)
        ds_land = ds_land.chunk({"y": chunk_y, "x": chunk_x})

        # Ensure elevation NaN where valid_mask False
        ds_land["elevation"] = ds_land["elevation"].where(ds_land["valid_mask"], np.nan)

        # Deterministic inputs_id
        items: list[tuple[str, str]] = []
        items.append(("wind:grid_id", wind_grid_id))
        items.append(("wind:inputs_id", wind_inputs_id))
        items.append(("mask_policy", "nan+valid_mask_in_landscape"))
        items.append(("region", _stable_json(getattr(atlas, "region", None))))
        items.append(("chunk_policy", _stable_json(self.chunk_policy)))

        if elev_kind == "local":
            items.append(("elevation:kind", "legacy_tif"))
            items.append(("elevation:path", str(local_elev)))
            items.append(("elevation:fingerprint", fingerprint_path_mtime_size(local_elev)))
            items.append(("elevation:clip", "aoi" if aoi_gdf is not None else "none"))
            landscape_fingerprint_method = self.fingerprint_method
        else:
            # copdem tiles mode must be deterministic
            items.append(("elevation:kind", "copdem"))
            items.append(("elevation:provider", elev_meta["provider"]))
            items.append(("elevation:version", elev_meta["version"]))
            items.append(("elevation:bbox_4326", _stable_json(elev_meta["bbox_4326"])))
            items.append(("elevation:tile_ids", _stable_json(elev_meta["tile_ids"])))
            items.append(("elevation:clip", elev_meta["clip"]))
            landscape_fingerprint_method = "copdem_tiles"

        inputs_id = hash_inputs_id(items, method=landscape_fingerprint_method)

        # Idempotency check
        if store_path.exists():
            try:
                g = zarr.open_group(store_path, mode="r")
                if (
                    g.attrs.get("store_state") == "complete"
                    and g.attrs.get("inputs_id") == inputs_id
                    and g.attrs.get("grid_id") == wind_grid_id
                ):
                    return
            except Exception:
                pass

        # Atomic write full landscape store
        git = get_git_info(repo_root=Path(__file__).resolve().parents[1])
        with atomic_dir(store_path) as tmp:
            ds_land.to_zarr(tmp, mode="w", consolidated=False)
            g = zarr.open_group(tmp, mode="a")
            g.attrs.update(
                store_state="complete",
                grid_id=wind_grid_id,
                inputs_id=inputs_id,
                unify_version=git["unify_version"],
                code_dirty=git["code_dirty"],
                chunk_policy=_stable_json(self.chunk_policy),
                fingerprint_method=landscape_fingerprint_method,
            )
            if git.get("git_diff_hash"):
                g.attrs["git_diff_hash"] = git["git_diff_hash"]

            init_manifest(tmp)

            # Build sources list
            sources: list[dict] = []
            sources.append(
                dict(
                    source_id="mask:derived_from_wind",
                    name="valid_mask derived from wind weibull_A",
                    kind="derived",
                    path=str(wind_path),
                    params_json=_stable_json({
                        "ref": "weibull_A",
                        "height": int(wind_ref["height"].values)
                        if "height" in wind_ref.coords
                        else None,
                    }),
                    fingerprint=hashlib.sha256(
                        f"{wind_grid_id}:{wind_inputs_id}".encode("utf-8")
                    ).hexdigest(),
                    created_at=_now_iso(),
                )
            )

            if elev_kind == "local":
                sources.append(
                    dict(
                        source_id="elevation:local",
                        name="local elevation GeoTIFF",
                        kind="raster",
                        path=str(local_elev),
                        params_json=_stable_json({
                            "clip": "aoi" if aoi_gdf is not None else "none"
                        }),
                        fingerprint=fingerprint_path_mtime_size(local_elev),
                        created_at=_now_iso(),
                    )
                )
            else:
                sources.append(
                    dict(
                        source_id="elevation:copdem",
                        name="copdem elevation",
                        kind="network+raster",
                        path="copdem://",
                        params_json=_stable_json(elev_meta),
                        fingerprint=hashlib.sha256(
                            _stable_json(elev_meta).encode("utf-8")
                        ).hexdigest(),
                        created_at=_now_iso(),
                    )
                )
            write_manifest_sources(tmp, sources)

            # Build variables list
            vars_: list[dict] = []
            vars_.append(
                dict(
                    variable_name="valid_mask",
                    source_id="mask:derived_from_wind",
                    materialized_at=_now_iso(),
                    resampling_method="derived",
                    nodata_policy="nan",
                    dtype=str(ds_land["valid_mask"].dtype),
                )
            )
            vars_.append(
                dict(
                    variable_name="elevation",
                    source_id=(
                        "elevation:local" if elev_kind == "local" else "elevation:copdem"
                    ),
                    materialized_at=_now_iso(),
                    resampling_method="bilinear",
                    nodata_policy="nan",
                    dtype=str(ds_land["elevation"].dtype),
                )
            )
            write_manifest_variables(tmp, vars_)

    def register_landscape_source(
        self,
        atlas,
        *,
        name: str,
        source_path: Path,
        kind: str = "raster",
        params: dict | None = None,
        if_exists: str = "error",
    ) -> bool:
        """Register a new landscape source in __manifest__/sources.

        Does NOT materialize the variable - only records metadata about the source.

        Args:
            atlas: Atlas instance.
            name: Variable name (will be source_id suffix).
            source_path: Path to the source raster file.
            kind: Source kind (v1 only supports "raster").
            params: Optional parameters dict (e.g., {"categorical": True}).
            if_exists: Behavior when source already exists:
                - "error" (default): raise ValueError if source exists with different config
                - "replace": update existing source registration with new config
                - "noop": skip only if existing registration exactly matches
                  (kind, path, params_json, fingerprint); otherwise raise ValueError

        Returns:
            True if registration was performed, False if skipped (exact match with noop).

        Raises:
            ValueError: If kind != "raster", if_exists invalid, or source exists
                with different config when if_exists="error" or "noop".
            RuntimeError: If landscape.zarr is not complete.
        """
        if kind != "raster":
            raise ValueError(f"Only kind='raster' supported in v1; got {kind!r}")

        valid_if_exists = {"error", "replace", "noop"}
        if if_exists not in valid_if_exists:
            raise ValueError(
                f"if_exists must be one of {sorted(valid_if_exists)!r}; got {if_exists!r}"
            )

        store_path = Path(atlas.path) / "landscape.zarr"

        # Require complete landscape store
        root = zarr.open_group(store_path, mode="r")
        if root.attrs.get("store_state") != "complete":
            raise RuntimeError(
                "landscape.zarr is not complete; run Unifier.materialize_landscape(atlas) first."
            )

        source_id = f"land:raster:{name}"
        params = params or {}
        params_json = _stable_json(params)
        fingerprint = fingerprint_path_mtime_size(source_path)

        # Ensure manifest JSON exists
        init_manifest(store_path)

        # Read existing manifest
        manifest = _read_manifest(store_path)
        existing_sources = manifest.get("sources", [])
        source_by_id = {s["source_id"]: s for s in existing_sources}

        # Check if source already exists
        if source_id in source_by_id:
            existing = source_by_id[source_id]
            existing_kind = existing.get("kind", "")
            existing_path = existing.get("path", "")
            existing_params = existing.get("params_json", "")
            existing_fingerprint = existing.get("fingerprint", "")

            # Check for exact match on all fields (kind, path, params_json, fingerprint)
            exact_match = (
                existing_kind == kind
                and existing_path == str(source_path)
                and existing_params == params_json
                and existing_fingerprint == fingerprint
            )

            if if_exists == "noop":
                if exact_match:
                    # Exact match - skip without any changes
                    return False
                else:
                    raise ValueError(
                        f"Source {source_id!r} already registered with different configuration.\n"
                        f"  Existing: path={existing_path!r}, params={existing_params!r}, "
                        f"fingerprint={existing_fingerprint!r}\n"
                        f"  New: path={str(source_path)!r}, params={params_json!r}, "
                        f"fingerprint={fingerprint!r}\n"
                        f"  Use atlas.landscape.add(..., if_exists='replace') to overwrite."
                    )

            if if_exists == "error":
                # For error mode, we check path+params (fingerprint may differ if file changed)
                config_matches = (
                    existing_path == str(source_path) and existing_params == params_json
                )
                if not config_matches:
                    raise ValueError(
                        f"Source {source_id!r} already registered with different configuration.\n"
                        f"  Existing: path={existing_path!r}, params={existing_params!r}\n"
                        f"  New: path={str(source_path)!r}, params={params_json!r}\n"
                        f"  Use if_exists='replace' to overwrite."
                    )
                # Config matches - idempotent no-op for error mode
                return False

            # if_exists == "replace" - fall through to update the source registration

        # Create new/updated source entry
        new_source = {
            "source_id": source_id,
            "name": name,
            "kind": kind,
            "path": str(source_path),
            "params_json": params_json,
            "fingerprint": fingerprint,
        }

        # Update or add source
        updated_sources = []
        source_updated = False
        for src in existing_sources:
            if src["source_id"] == source_id:
                updated_sources.append(new_source)
                source_updated = True
            else:
                updated_sources.append(src)

        if not source_updated:
            updated_sources.append(new_source)

        # Write manifest with updated sources
        manifest["sources"] = updated_sources
        _write_manifest_atomic(store_path, manifest)
        return True

    def materialize_landscape_variable(
        self,
        atlas,
        variable_name: str,
        *,
        if_exists: str = "error",
    ) -> bool:
        """Materialize a single landscape variable from a registered source.

        Reads the registered source, aligns to wind grid, enforces valid_mask
        semantics, and appends to landscape.zarr. Updates inputs_id deterministically.

        Args:
            atlas: Atlas instance.
            variable_name: Name of the variable to materialize.
            if_exists: Behavior when variable already exists in the store:
                - "error" (default): raise ValueError if variable exists
                - "replace": atomically replace existing variable data
                - "noop": skip only if existing materialization exactly matches
                  the current source registration; otherwise raise ValueError

        Returns:
            True if materialization was performed, False if skipped (exact match with noop).

        Raises:
            ValueError: If if_exists invalid, variable exists when if_exists="error",
                or variable exists with different config when if_exists="noop".
            KeyError: If source not registered in __manifest__/sources.
            RuntimeError: If wind.zarr or landscape.zarr not complete, CRS missing,
                or y/x coords do not exactly match wind reference after materialize.
        """
        valid_if_exists = {"error", "replace", "noop"}
        if if_exists not in valid_if_exists:
            raise ValueError(
                f"if_exists must be one of {sorted(valid_if_exists)!r}; got {if_exists!r}"
            )
        store_path = Path(atlas.path) / "landscape.zarr"
        wind_path = Path(atlas.path) / "wind.zarr"

        # 1) Open wind canonical store to get wind_ref
        wind = xr.open_zarr(wind_path, consolidated=False, chunks=self.chunk_policy)

        if wind.attrs.get("store_state") != "complete":
            raise RuntimeError(
                "wind.zarr is not complete; run Unifier.materialize_wind(atlas) first."
            )

        wind_grid_id = wind.attrs.get("grid_id") or ""
        wind_inputs_id = wind.attrs.get("inputs_id") or ""

        # Get wind_ref for alignment
        wind_ref = wind["weibull_A"].isel(height=0)
        if "height" in wind["weibull_A"].dims:
            height_values = wind["weibull_A"]["height"].values
            if 100 in height_values:
                wind_ref = wind["weibull_A"].sel(height=100)

        # Ensure wind_ref has CRS
        if wind_ref.rio.crs is None:
            if "template" in wind and wind["template"].rio.crs is not None:
                wind_ref = wind_ref.rio.write_crs(wind["template"].rio.crs)
            else:
                wind_ref = wind_ref.rio.write_crs(atlas.crs)

        if wind_ref.rio.transform() is None:
            wind_ref = wind_ref.rio.write_transform(wind_ref.rio.transform(recalc=True))

        # 2) Open landscape canonical store
        land = xr.open_zarr(store_path, consolidated=False, chunks=self.chunk_policy)

        land_root = zarr.open_group(store_path, mode="r")
        if land_root.attrs.get("store_state") != "complete":
            raise RuntimeError(
                "landscape.zarr is not complete; run Unifier.materialize_landscape(atlas) first."
            )

        land_grid_id = land_root.attrs.get("grid_id") or ""
        if land_grid_id != wind_grid_id:
            raise RuntimeError(
                f"landscape.zarr grid_id mismatch: {land_grid_id!r} != wind grid_id {wind_grid_id!r}"
            )

        if "valid_mask" not in land:
            raise RuntimeError("landscape.zarr missing valid_mask")

        # 3) Ensure source is registered (do this BEFORE if_exists check for noop validation)
        source_id = f"land:raster:{variable_name}"
        manifest = _read_manifest(store_path)
        sources = manifest.get("sources", [])
        source_by_id = {s["source_id"]: s for s in sources}

        if source_id not in source_by_id:
            raise KeyError(
                f"Source {source_id!r} not registered. "
                f"Call register_landscape_source first."
            )

        source_entry = source_by_id[source_id]
        source_path = Path(source_entry.get("path", ""))
        params_json = source_entry.get("params_json", "")
        stored_fingerprint = source_entry.get("fingerprint", "")
        stored_kind = source_entry.get("kind", "")
        params = json.loads(params_json) if params_json else {}

        # Compute current fingerprint for noop validation
        current_fingerprint = fingerprint_path_mtime_size(source_path)

        # 2b) Check if variable already exists - apply if_exists semantics
        if variable_name in land.data_vars:
            if if_exists == "noop":
                # Verify existing materialization exactly matches current source registration
                # Check variables table has expected linkage
                variables = manifest.get("variables", [])
                var_by_name = {v["variable_name"]: v for v in variables}

                if variable_name not in var_by_name:
                    raise ValueError(
                        f"Variable {variable_name!r} exists in store but not in manifest.\n"
                        f"  Use atlas.landscape.add(..., if_exists='replace') to fix."
                    )

                stored_source_id = var_by_name[variable_name].get("source_id", "")

                if stored_source_id != source_id:
                    raise ValueError(
                        f"Variable {variable_name!r} exists with different source_id.\n"
                        f"  Existing: source_id={stored_source_id!r}\n"
                        f"  Expected: source_id={source_id!r}\n"
                        f"  Use atlas.landscape.add(..., if_exists='replace') to overwrite."
                    )

                # Verify fingerprint matches (source file unchanged since registration)
                if stored_fingerprint != current_fingerprint:
                    raise ValueError(
                        f"Variable {variable_name!r} exists but source file has changed.\n"
                        f"  Stored fingerprint: {stored_fingerprint!r}\n"
                        f"  Current fingerprint: {current_fingerprint!r}\n"
                        f"  Use atlas.landscape.add(..., if_exists='replace') to re-materialize."
                    )

                # All checks passed - exact match, skip without changes
                return False

            elif if_exists == "error":
                raise ValueError(
                    f"Variable {variable_name!r} already exists in landscape.zarr.\n"
                    f"  Use if_exists='replace' to overwrite or if_exists='noop' to skip."
                )
            elif if_exists == "replace":
                # Remove existing variable directory for atomic replacement
                _atomic_replace_variable_dir(store_path, variable_name)
                # Re-open landscape store after modification
                land = xr.open_zarr(store_path, consolidated=False, chunks=self.chunk_policy)

        # Use stored fingerprint for inputs_id (consistent with registration)
        fingerprint = stored_fingerprint

        # 4) AOI
        aoi = _aoi_geom_or_none(atlas)

        # 5) Read + unify raster (raw I/O)
        da = rxr.open_rasterio(source_path, parse_coordinates=True).squeeze(drop=True)

        if da.rio.crs is None:
            raise RuntimeError(
                f"Source raster {source_path} has no CRS; cannot materialize."
            )

        # Clip to AOI if specified
        if aoi is not None:
            from cleo.spatial import to_crs_if_needed
            aoi_in_da_crs = to_crs_if_needed(aoi, da.rio.crs)
            da = da.rio.clip(aoi_in_da_crs.geometry, drop=False)

        # Determine resampling based on categorical flag
        categorical = bool(params.get("categorical", False))
        resampling = Resampling.nearest if categorical else Resampling.bilinear

        # Reproject to match wind grid
        da = da.rio.reproject_match(wind_ref, resampling=resampling, nodata=np.nan)
        da = da.rename(variable_name)

        # Enforce y/x coords EXACT match to wind reference coords
        wind_y = wind_ref.coords["y"].values
        wind_x = wind_ref.coords["x"].values
        da_y = da.coords["y"].values
        da_x = da.coords["x"].values

        if not (np.array_equal(da_y, wind_y) and np.array_equal(da_x, wind_x)):
            raise ValueError(
                f"Materialized variable {variable_name!r} has y/x coords that do not match "
                f"wind reference grid exactly.\n"
                f"  wind_ref y: shape={wind_y.shape}, range=[{wind_y.min()}, {wind_y.max()}]\n"
                f"  da y: shape={da_y.shape}, range=[{da_y.min()}, {da_y.max()}]\n"
                f"  wind_ref x: shape={wind_x.shape}, range=[{wind_x.min()}, {wind_x.max()}]\n"
                f"  da x: shape={da_x.shape}, range=[{da_x.min()}, {da_x.max()}]"
            )

        # Enforce valid_mask semantics: NaN where valid_mask is False
        valid_mask = land["valid_mask"].load()
        da = da.where(valid_mask, np.nan)

        # Apply chunking
        chunk_y = self.chunk_policy.get("y", 1024)
        chunk_x = self.chunk_policy.get("x", 1024)
        da = da.chunk({"y": chunk_y, "x": chunk_x})

        # 6) Preserve existing attrs before appending (to_zarr may overwrite)
        preserve_attrs = dict(land_root.attrs)

        # 7) Append variable into store
        da.to_dataset().to_zarr(store_path, mode="a", consolidated=False)

        # 8) Restore preserved attrs after append
        root = zarr.open_group(store_path, mode="a")
        for k, v in preserve_attrs.items():
            root.attrs[k] = v

        # 9) Update manifest JSON with new variable
        manifest = _read_manifest(store_path)
        existing_vars = manifest.get("variables", [])
        existing_var_names = [v["variable_name"] for v in existing_vars]

        # Update or add the new variable
        new_var = {
            "variable_name": variable_name,
            "source_id": source_id,
            "resampling_method": "nearest" if categorical else "bilinear",
            "nodata_policy": "nan",
            "dtype": str(da.dtype),
        }

        if variable_name in existing_var_names:
            # Update existing entry
            for i, v in enumerate(existing_vars):
                if v["variable_name"] == variable_name:
                    existing_vars[i] = new_var
                    break
        else:
            existing_vars.append(new_var)

        manifest["variables"] = existing_vars
        _write_manifest_atomic(store_path, manifest)

        # 11) Update inputs_id deterministically
        items: list[tuple[str, str]] = []
        items.append(("wind:grid_id", wind_grid_id))
        items.append(("wind:inputs_id", wind_inputs_id))
        items.append(("mask_policy", "nan+valid_mask_in_landscape"))
        items.append(("region", _stable_json(getattr(atlas, "region", None))))
        items.append(("chunk_policy", _stable_json(self.chunk_policy)))
        items.append(("incremental_add", "landscape_add_v1"))
        items.append((f"layer:{variable_name}:source_id", source_id))
        items.append((f"layer:{variable_name}:fingerprint", fingerprint))
        items.append((f"layer:{variable_name}:params_json", params_json))

        new_inputs_id = hash_inputs_id(items, method=self.fingerprint_method)

        # Update store attrs (reopen to ensure we have latest state)
        root = zarr.open_group(store_path, mode="a")
        root.attrs["inputs_id"] = new_inputs_id
        # grid_id remains unchanged (preserved from step 8)

        return True

    def compute_air_density_correction(
        self,
        atlas,
        *,
        chunk_size=None,
        force: bool = False,
    ) -> xr.DataArray:
        """
        Compute air density correction from canonical stores (wind + landscape).

        This method reads elevation from the landscape store and computes the
        air density correction factor using the pure compute function from
        cleo.assess. No raw I/O is performed.

        Contract:
        - Canonical stores (wind.zarr, landscape.zarr) MUST exist and be complete.
        - Elevation and wind template MUST be aligned on the same grid.
        - Returns lazy DataArray (no .compute()/.load()).
        - Does NOT write to wind.zarr; caller handles persistence/caching.

        Args:
            atlas: Atlas instance with canonical stores.
            chunk_size: Unused (kept for signature compatibility).
            force: Unused (kept for signature compatibility).

        Returns:
            DataArray "air_density_correction" on canonical grid.

        Raises:
            FileNotFoundError: If canonical stores do not exist.
            RuntimeError: If stores are not complete or grids are misaligned.
        """
        from cleo.assess import compute_air_density_correction_core

        atlas_root = Path(atlas.path)
        wind_path = atlas_root / "wind.zarr"
        landscape_path = atlas_root / "landscape.zarr"

        # 1) Require canonical stores exist
        if not wind_path.exists():
            raise FileNotFoundError(
                f"wind.zarr not found at {wind_path}. "
                "Run atlas.materialize_canonical() first."
            )
        if not landscape_path.exists():
            raise FileNotFoundError(
                f"landscape.zarr not found at {landscape_path}. "
                "Run atlas.materialize_canonical() first."
            )

        # 2) Open canonical stores
        chunk_y = self.chunk_policy.get("y", 1024)
        chunk_x = self.chunk_policy.get("x", 1024)
        chunks = {"y": chunk_y, "x": chunk_x}

        wind = xr.open_zarr(wind_path, consolidated=False, chunks=chunks)
        land = xr.open_zarr(landscape_path, consolidated=False, chunks=chunks)

        # 3) Validate store_state is "complete"
        wind_state = wind.attrs.get("store_state", None)
        land_state = land.attrs.get("store_state", None)

        if wind_state != "complete":
            raise RuntimeError(
                f"wind.zarr store_state={wind_state!r}, expected 'complete'. "
                "Run atlas.materialize_canonical() to complete unification."
            )
        if land_state != "complete":
            raise RuntimeError(
                f"landscape.zarr store_state={land_state!r}, expected 'complete'. "
                "Run atlas.materialize_canonical() to complete unification."
            )

        # 4) Get template from wind store
        if "weibull_A" in wind.data_vars:
            weibull_a = wind["weibull_A"]
            if "height" in weibull_a.dims:
                # Select height=100 if available, else first height
                if 100 in weibull_a.coords.get("height", xr.DataArray([])).values:
                    template = weibull_a.sel(height=100)
                else:
                    template = weibull_a.isel(height=0)
            else:
                template = weibull_a
        else:
            raise RuntimeError(
                "wind.zarr missing 'weibull_A' variable for template grid."
            )

        # 5) Get elevation from landscape store
        if "elevation" not in land.data_vars:
            raise RuntimeError(
                "landscape.zarr missing 'elevation' variable. "
                "Ensure landscape store was materialized with elevation."
            )
        elevation = land["elevation"]

        # 6) Validate alignment (same y/x coords)
        if not (
            np.array_equal(template.coords["y"].values, elevation.coords["y"].values)
            and np.array_equal(template.coords["x"].values, elevation.coords["x"].values)
        ):
            raise RuntimeError(
                "Elevation not aligned to wind grid; re-run unification. "
                f"template y: {len(template.y)}, elevation y: {len(elevation.y)}; "
                f"template x: {len(template.x)}, elevation x: {len(elevation.x)}"
            )

        # 7) Call pure compute function
        result = compute_air_density_correction_core(
            elevation=elevation,
            template=template,
        )

        return result

    def materialize_region(self, atlas, region_id: str) -> None:
        """Materialize region stores by subsetting from country stores.

        Per contract S2/B1/B4:
        - Region stores are derived from base country stores
        - Stored at <ROOT>/regions/<region_id>/wind.zarr and .../landscape.zarr
        - Region stores have reduced y/x dims (true subsetting)

        Args:
            atlas: Atlas instance with completed base stores.
            region_id: NUTS region ID (e.g., "AT13", "Niederösterreich").

        Raises:
            RuntimeError: If base stores are not complete.
            FileNotFoundError: If region geometry cannot be found.
        """
        atlas_root = Path(atlas.path)
        wind_base_path = atlas_root / "wind.zarr"
        land_base_path = atlas_root / "landscape.zarr"
        region_root = atlas_root / "regions" / region_id
        wind_region_path = region_root / "wind.zarr"
        land_region_path = region_root / "landscape.zarr"

        # Check if region stores already exist and are complete
        if wind_region_path.exists() and land_region_path.exists():
            try:
                wind_region = xr.open_zarr(wind_region_path, consolidated=False)
                land_region = xr.open_zarr(land_region_path, consolidated=False)
                if (wind_region.attrs.get("store_state") == "complete" and
                    land_region.attrs.get("store_state") == "complete" and
                    wind_region.attrs.get("region_id") == region_id and
                    land_region.attrs.get("region_id") == region_id):
                    # Region stores already complete
                    logger.info(f"Region stores for {region_id!r} already complete, skipping.")
                    return
            except Exception:
                pass  # Need to recreate

        # 1) Open base stores (must be complete)
        wind_base = xr.open_zarr(wind_base_path, consolidated=False, chunks=self.chunk_policy)
        land_base = xr.open_zarr(land_base_path, consolidated=False, chunks=self.chunk_policy)

        if wind_base.attrs.get("store_state") != "complete":
            raise RuntimeError(
                "wind.zarr is not complete; run atlas.materialize() first."
            )
        if land_base.attrs.get("store_state") != "complete":
            raise RuntimeError(
                "landscape.zarr is not complete; run atlas.materialize() first."
            )

        base_wind_inputs_id = wind_base.attrs.get("inputs_id", "")
        base_land_inputs_id = land_base.attrs.get("inputs_id", "")
        base_grid_id = wind_base.attrs.get("grid_id", "")

        # 2) Get region geometry
        region_gdf = atlas.get_nuts_region(region_id)
        if region_gdf is None or region_gdf.empty:
            raise FileNotFoundError(
                f"Could not find region geometry for {region_id!r}. "
                f"Ensure NUTS shapefile is available."
            )

        # 3) Get bounding box of region in store CRS
        region_bounds = region_gdf.total_bounds  # [minx, miny, maxx, maxy]

        # 4) Subset wind store to region bbox
        # Find indices within the bbox
        wind_y = wind_base.coords["y"].values
        wind_x = wind_base.coords["x"].values

        # Handle y coordinate (may be decreasing)
        if wind_y[0] > wind_y[-1]:  # Decreasing y
            y_mask = (wind_y >= region_bounds[1]) & (wind_y <= region_bounds[3])
        else:  # Increasing y
            y_mask = (wind_y >= region_bounds[1]) & (wind_y <= region_bounds[3])

        x_mask = (wind_x >= region_bounds[0]) & (wind_x <= region_bounds[2])

        # Get indices
        y_indices = np.where(y_mask)[0]
        x_indices = np.where(x_mask)[0]

        if len(y_indices) == 0 or len(x_indices) == 0:
            raise RuntimeError(
                f"Region {region_id!r} does not overlap with data extent. "
                f"Region bounds: {region_bounds}, Data y: [{wind_y.min()}, {wind_y.max()}], "
                f"Data x: [{wind_x.min()}, {wind_x.max()}]"
            )

        # Subset using isel for efficiency
        y_slice = slice(y_indices.min(), y_indices.max() + 1)
        x_slice = slice(x_indices.min(), x_indices.max() + 1)

        wind_region_ds = wind_base.isel(y=y_slice, x=x_slice)
        land_region_ds = land_base.isel(y=y_slice, x=x_slice)

        # 5) Compute region inputs_id deterministically
        region_inputs_items = [
            ("region_id", region_id),
            ("base_wind_inputs_id", base_wind_inputs_id),
            ("base_land_inputs_id", base_land_inputs_id),
            ("base_grid_id", base_grid_id),
            ("region_bounds", _stable_json(list(region_bounds))),
            ("y_slice", f"{y_slice.start}:{y_slice.stop}"),
            ("x_slice", f"{x_slice.start}:{x_slice.stop}"),
        ]
        region_inputs_id = hash_inputs_id(region_inputs_items, method=self.fingerprint_method)

        # Compute region grid_id
        region_grid_items = [
            ("base_grid_id", base_grid_id),
            ("y_slice", f"{y_slice.start}:{y_slice.stop}"),
            ("x_slice", f"{x_slice.start}:{x_slice.stop}"),
        ]
        region_grid_id = hash_inputs_id(region_grid_items, method=self.fingerprint_method)

        # 6) Write region wind store
        region_root.mkdir(parents=True, exist_ok=True)

        # Copy attrs from base store and add region-specific attrs
        wind_region_attrs = dict(wind_base.attrs)
        wind_region_attrs["store_state"] = "complete"
        wind_region_attrs["region_id"] = region_id
        wind_region_attrs["inputs_id"] = region_inputs_id
        wind_region_attrs["grid_id"] = region_grid_id
        wind_region_attrs["base_wind_inputs_id"] = base_wind_inputs_id

        # Remove store_path if temp, add region-specific path
        wind_region_ds = wind_region_ds.assign_attrs(wind_region_attrs)

        # Write wind region store (compute to materialize)
        if wind_region_path.exists():
            import shutil
            shutil.rmtree(wind_region_path)

        wind_region_ds.to_zarr(
            wind_region_path,
            mode="w",
            consolidated=False,
        )

        logger.info(
            f"Created region wind store: {wind_region_path} "
            f"(y: {wind_region_ds.sizes['y']}, x: {wind_region_ds.sizes['x']})"
        )

        # 7) Write region landscape store
        land_region_attrs = dict(land_base.attrs)
        land_region_attrs["store_state"] = "complete"
        land_region_attrs["region_id"] = region_id
        land_region_attrs["inputs_id"] = region_inputs_id
        land_region_attrs["grid_id"] = region_grid_id
        land_region_attrs["base_land_inputs_id"] = base_land_inputs_id

        land_region_ds = land_region_ds.assign_attrs(land_region_attrs)

        if land_region_path.exists():
            import shutil
            shutil.rmtree(land_region_path)

        land_region_ds.to_zarr(
            land_region_path,
            mode="w",
            consolidated=False,
        )

        logger.info(
            f"Created region landscape store: {land_region_path} "
            f"(y: {land_region_ds.sizes['y']}, x: {land_region_ds.sizes['x']})"
        )

    # =========================================================================
# =============================================================================
# Helper Functions for Landscape Materialization
# =============================================================================


def _stable_json(obj: Any) -> str:
    """Convert object to stable JSON string for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _nuts_region_geom(atlas):
    """Get NUTS region geometry if atlas.region is set.

    Args:
        atlas: Atlas instance.

    Returns:
        GeoDataFrame for the NUTS region, or None if atlas.region is None.
    """
    if atlas.region is None:
        return None
    return atlas.get_nuts_region(atlas.region)


def _aoi_geom_or_none(atlas):
    """Get AOI geometry from NUTS region if specified.

    Reuses the same logic as wind clipping - returns the NUTS region
    geometry if atlas.region is set, None otherwise.

    Args:
        atlas: Atlas instance.

    Returns:
        GeoDataFrame for clipping, or None if no region specified.
    """
    return _nuts_region_geom(atlas)


def _open_local_elevation(
    atlas,
    elev_path: Path,
    wind_ref: xr.DataArray,
    aoi_gdf,
) -> xr.DataArray:
    """Open local elevation GeoTIFF and align to wind grid.

    This function loads an existing elevation raster and reprojects/clips
    it to match the wind reference grid. Does NOT fetch GWA CRS.

    Args:
        atlas: Atlas instance.
        elev_path: Path to the elevation GeoTIFF.
        wind_ref: Reference DataArray from wind for grid alignment.
        aoi_gdf: GeoDataFrame for AOI clipping (or None).

    Returns:
        DataArray with elevation values aligned to wind grid.
    """
    import warnings

    da = rxr.open_rasterio(elev_path, parse_coordinates=True).squeeze(drop=True)

    if da.rio.crs is None:
        # Safer default: assume wind CRS but warn clearly
        warnings.warn(
            f"Elevation raster {elev_path} has no CRS; assuming wind CRS {wind_ref.rio.crs}.",
            RuntimeWarning,
        )
        da = da.rio.write_crs(wind_ref.rio.crs)

    # Clip/mask to AOI BEFORE matching grid
    if aoi_gdf is not None:
        from cleo.spatial import to_crs_if_needed

        aoi_in_da_crs = to_crs_if_needed(aoi_gdf, da.rio.crs)
        da = da.rio.clip(aoi_in_da_crs.geometry, drop=False)

    # Match to wind grid
    da = da.rio.reproject_match(wind_ref, resampling=Resampling.bilinear, nodata=np.nan)

    # Convert nodata to NaN
    nodata = da.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        da = da.where(da != nodata, np.nan)

    return da


def _build_copdem_elevation(
    atlas,
    wind_ref: xr.DataArray,
    aoi_gdf,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Build CopDEM elevation data aligned to wind grid.

    Downloads CopDEM tiles, mosaics them, and reprojects to match the wind
    reference grid.

    Args:
        atlas: Atlas instance.
        wind_ref: Reference DataArray from wind for grid alignment.
        aoi_gdf: GeoDataFrame for AOI clipping (or None).

    Returns:
        Tuple of (elevation DataArray, metadata dict).

    Raises:
        FileNotFoundError: If CopDEM tiles cannot be downloaded.
    """
    from rasterio.warp import transform_bounds

    from cleo.copdem import (
        tiles_for_bbox,
        download_copdem_tiles_for_bbox,
        build_copdem_elevation_like,
    )

    # Determine bbox in EPSG:4326
    bounds = wind_ref.rio.bounds()
    wind_crs = wind_ref.rio.crs
    if str(wind_crs) != "EPSG:4326":
        bbox_4326 = transform_bounds(wind_crs, "EPSG:4326", *bounds, densify_pts=21)
    else:
        bbox_4326 = bounds

    min_lon, min_lat, max_lon, max_lat = bbox_4326

    # Get tile IDs for deterministic fingerprinting (sorted lexicographically)
    tile_ids = tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)

    # Download tiles (uses cache; deterministic order)
    tile_paths = download_copdem_tiles_for_bbox(
        atlas.path,
        atlas.country,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
    )

    # Build mosaic aligned to wind_ref
    elevation = build_copdem_elevation_like(wind_ref, tile_paths)

    # Build metadata for deterministic inputs_id
    meta = {
        "provider": "copernicus",
        "version": "GLO-30",
        "bbox_4326": list(bbox_4326),
        "tile_ids": tile_ids,
        "clip": "aoi" if aoi_gdf is not None else "none",
    }

    return elevation, meta


# =============================================================================
# Raw I/O Helpers (centralized; other modules must delegate here)
# =============================================================================


def _open_raster(
    path: Path | str,
    *,
    parse_coordinates: bool = True,
    chunks: dict | None = None,
) -> xr.DataArray:
    """
    Open a raster file (GeoTIFF etc.) as an xarray DataArray.

    This is the centralized raw I/O entry point for raster reads.
    Other modules must delegate here instead of calling rxr.open_rasterio directly.

    Args:
        path: Path to the raster file.
        parse_coordinates: Whether to parse coordinates from the raster.
        chunks: Chunking specification for dask (None for eager).

    Returns:
        DataArray with raster data.
    """
    da = rxr.open_rasterio(path, parse_coordinates=parse_coordinates, chunks=chunks)
    return da.squeeze(drop=True)


def _open_dataset(path: Path | str, **kwargs) -> xr.Dataset:
    """
    Open a NetCDF/HDF5 dataset as an xarray Dataset.

    This is the centralized raw I/O entry point for dataset reads.
    Other modules must delegate here instead of calling xr.open_dataset directly.

    Args:
        path: Path to the dataset file.
        **kwargs: Additional arguments passed to xr.open_dataset.

    Returns:
        Dataset with the loaded data.
    """
    return xr.open_dataset(path, **kwargs)


def _ensure_crs_from_gwa(da: xr.DataArray, iso3: str) -> xr.DataArray:
    """
    Ensure a raster DataArray has a CRS set, fetching from GWA if needed.

    Delegates to cleo.loaders.ensure_crs_from_gwa.

    Args:
        da: Raster DataArray.
        iso3: ISO 3166-1 alpha-3 country code.

    Returns:
        DataArray with CRS set.
    """
    from cleo.loaders import ensure_crs_from_gwa
    return ensure_crs_from_gwa(da, iso3)


def _build_copdem_mosaic(
    tile_paths: list[Path],
    reference_da: xr.DataArray,
) -> xr.DataArray:
    """
    Mosaic Copernicus DEM tiles and reproject to match a reference raster.

    This function performs raw I/O (rasterio.open) and is centralized here.
    Other modules must delegate here for CopDEM mosaicing.

    Contracts:
    - tile_paths must be non-empty
    - CRS must be present in both tiles and reference
    - tile nodata (if defined) is masked to NaN before reprojection
    - elevation is continuous -> bilinear resampling

    Args:
        tile_paths: List of Paths to Copernicus DEM tile GeoTIFFs.
        reference_da: Reference xarray DataArray with rioxarray metadata.

    Returns:
        DataArray with elevation data matching reference grid.

    Raises:
        ValueError: If tile_paths is empty or CRS is missing.
        RuntimeError: If mosaicking fails.
    """
    import rasterio
    from rasterio.merge import merge

    if not tile_paths:
        raise ValueError("tile_paths cannot be empty")

    tile_datasets = []
    nodata = None
    try:
        for path in tile_paths:
            ds = rasterio.open(path)
            if ds.crs is None:
                raise ValueError(f"CRS missing in tile: {path}")
            tile_datasets.append(ds)

        nodata = tile_datasets[0].nodata
        mosaic_arr, mosaic_transform = merge(tile_datasets)
        mosaic_crs = tile_datasets[0].crs

    finally:
        for ds in tile_datasets:
            try:
                ds.close()
            except Exception:
                pass

    mosaic_2d = mosaic_arr[0]

    # Mask nodata / masked arrays to NaN
    if np.ma.isMaskedArray(mosaic_2d):
        mosaic_2d = mosaic_2d.filled(np.nan).astype("float32")
    else:
        mosaic_2d = mosaic_2d.astype("float32", copy=False)

    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        mosaic_2d = np.where(mosaic_2d == float(nodata), np.nan, mosaic_2d)

    height, width = mosaic_2d.shape
    cols = np.arange(width)
    rows = np.arange(height)
    xs = mosaic_transform.c + mosaic_transform.a * (cols + 0.5)
    ys = mosaic_transform.f + mosaic_transform.e * (rows + 0.5)

    mosaic_da = xr.DataArray(
        mosaic_2d,
        dims=["y", "x"],
        coords={"y": ys, "x": xs},
        name="elevation",
    )
    mosaic_da = mosaic_da.rio.write_crs(mosaic_crs)
    mosaic_da = mosaic_da.rio.write_transform(mosaic_transform)
    mosaic_da = mosaic_da.rio.write_nodata(np.nan)

    if reference_da.rio.crs is None:
        raise ValueError("CRS missing in reference DataArray")

    result = mosaic_da.rio.reproject_match(
        reference_da,
        resampling=Resampling.bilinear,
        nodata=np.nan,
    )
    result.name = "elevation"
    return result


def _atomic_replace_variable_dir(store_path: Path, variable_name: str) -> None:
    """Atomically replace a variable directory in a zarr store.

    Removes the existing variable directory if present. This is used when
    if_exists="replace" to ensure clean replacement of variable data.

    The operation is atomic: the directory is removed in one operation.

    Args:
        store_path: Path to the zarr store root.
        variable_name: Name of the variable directory to remove.
    """
    import shutil

    var_dir = store_path / variable_name
    if var_dir.exists():
        shutil.rmtree(var_dir)


# -----------------------------------------------------------------------------
# Raw I/O Helpers
# These helpers centralize raw I/O that is forbidden in classes.py.
# -----------------------------------------------------------------------------

def _read_vector_file(path: Path | str) -> gpd.GeoDataFrame:
    """Read a vector file (shapefile, GeoJSON, etc.).

    Args:
        path: Path to vector file.

    Returns:
        GeoDataFrame with loaded features.
    """
    return gpd.read_file(path)
