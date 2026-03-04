"""Region materialization and recovery policies."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import mapping

from cleo.store import single_writer_lock, zarr_store_lock_dir
from cleo.unification.store_io import open_zarr_dataset
from cleo.unification.fingerprint import hash_inputs_id
from cleo.unification.materializers.shared import _stable_json
from cleo.unification.materializers._helpers import read_store_attrs_safe

logger = logging.getLogger(__name__)


@dataclass
class _RegionPaths:
    """Paths for region materialization."""

    atlas_root: Path
    wind_base: Path
    land_base: Path
    region_root: Path
    wind_region: Path
    land_region: Path


@dataclass
class _BboxSlices:
    """Bounding box slices and bounds."""

    y_slice: slice
    x_slice: slice
    bounds: np.ndarray


def _infer_axis_step(subset_coords: np.ndarray, base_coords: np.ndarray, axis_name: str) -> float:
    """Infer coordinate step for an axis, handling single-cell subsets safely."""
    subset = np.asarray(subset_coords, dtype=np.float64)
    base = np.asarray(base_coords, dtype=np.float64)

    if subset.size >= 2:
        step = float(subset[1] - subset[0])
    elif base.size >= 2:
        step = float(base[1] - base[0])
    else:
        raise RuntimeError(f"Cannot infer {axis_name}-axis spacing from a single-cell grid.")

    if not np.isfinite(step) or step == 0.0:
        raise RuntimeError(f"Invalid {axis_name}-axis spacing {step!r}; expected a non-zero finite value.")
    return step


def _build_region_paths(atlas, region_id: str) -> _RegionPaths:
    """Build all paths needed for region materialization."""
    atlas_root = Path(atlas.path)
    return _RegionPaths(
        atlas_root=atlas_root,
        wind_base=atlas_root / "wind.zarr",
        land_base=atlas_root / "landscape.zarr",
        region_root=atlas_root / "regions" / region_id,
        wind_region=atlas_root / "regions" / region_id / "wind.zarr",
        land_region=atlas_root / "regions" / region_id / "landscape.zarr",
    )


def _read_base_inputs_ids(paths: _RegionPaths, region_id: str) -> tuple[str | None, str | None]:
    """Read current base-store inputs IDs for stale-region detection."""
    wind_attrs = read_store_attrs_safe(paths.wind_base, "inputs_id", default=None)
    land_attrs = read_store_attrs_safe(paths.land_base, "inputs_id", default=None)
    return wind_attrs.get("inputs_id"), land_attrs.get("inputs_id")


def _check_region_stores_fresh(
    paths: _RegionPaths,
    region_id: str,
    base_wind_inputs_id: str | None,
    base_land_inputs_id: str | None,
) -> bool:
    """Check if region stores already exist, are complete, and match base inputs.

    Returns True if stores are fresh and can be skipped.
    """
    if not (paths.wind_region.exists() and paths.land_region.exists()):
        return False

    wind_region = None
    land_region = None
    try:
        wind_region = open_zarr_dataset(paths.wind_region)
        land_region = open_zarr_dataset(paths.land_region)

        stores_complete = (
            wind_region.attrs.get("store_state") == "complete"
            and land_region.attrs.get("store_state") == "complete"
            and wind_region.attrs.get("region_id") == region_id
            and land_region.attrs.get("region_id") == region_id
        )
        if not stores_complete:
            return False

        base_ids_available = bool(base_wind_inputs_id) and bool(base_land_inputs_id)
        base_link_matches = (
            base_ids_available
            and wind_region.attrs.get("base_wind_inputs_id") == base_wind_inputs_id
            and land_region.attrs.get("base_land_inputs_id") == base_land_inputs_id
        )

        if base_link_matches:
            logger.info(f"Region stores for {region_id!r} already complete, skipping.")
            return True

        logger.info(
            "Region stores for %r are stale versus current base inputs_id; rebuilding.",
            region_id,
        )
        return False

    except (OSError, ValueError, TypeError, KeyError):
        logger.debug(
            "Failed to verify existing region stores; recreating.",
            extra={"region_id": region_id},
            exc_info=True,
        )
        return False
    finally:
        if wind_region is not None:
            wind_region.close()
        if land_region is not None:
            land_region.close()


def _open_base_stores(
    paths: _RegionPaths,
    chunk_policy: dict[str, int] | None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Open and validate base stores."""
    wind_base = open_zarr_dataset(paths.wind_base, chunk_policy=chunk_policy)
    land_base = open_zarr_dataset(paths.land_base, chunk_policy=chunk_policy)

    if wind_base.attrs.get("store_state") != "complete":
        wind_base.close()
        land_base.close()
        raise RuntimeError("wind.zarr is not complete; run atlas.build() first.")
    if land_base.attrs.get("store_state") != "complete":
        wind_base.close()
        land_base.close()
        raise RuntimeError("landscape.zarr is not complete; run atlas.build() first.")

    return wind_base, land_base


def _get_region_geometry(atlas, region_id: str):
    """Get and validate region geometry."""
    region_gdf = atlas.get_nuts_region(region_id)
    if region_gdf is None or region_gdf.empty:
        raise FileNotFoundError(
            f"Could not find region geometry for {region_id!r}. Ensure NUTS shapefile is available."
        )
    return region_gdf


def _compute_bbox_slices(
    wind_base: xr.Dataset,
    region_bounds: np.ndarray,
    region_id: str,
) -> _BboxSlices:
    """Compute y and x slices from region bounding box."""
    wind_y = wind_base.coords["y"].values
    wind_x = wind_base.coords["x"].values

    y_mask = (wind_y >= region_bounds[1]) & (wind_y <= region_bounds[3])
    x_mask = (wind_x >= region_bounds[0]) & (wind_x <= region_bounds[2])

    y_indices = np.where(y_mask)[0]
    x_indices = np.where(x_mask)[0]

    if len(y_indices) == 0 or len(x_indices) == 0:
        raise RuntimeError(
            f"Region {region_id!r} does not overlap with data extent. "
            f"Region bounds: {region_bounds}, Data y: [{wind_y.min()}, {wind_y.max()}], "
            f"Data x: [{wind_x.min()}, {wind_x.max()}]"
        )

    return _BboxSlices(
        y_slice=slice(y_indices.min(), y_indices.max() + 1),
        x_slice=slice(x_indices.min(), x_indices.max() + 1),
        bounds=region_bounds,
    )


def _create_geometry_mask(
    wind_region_ds: xr.Dataset,
    wind_base: xr.Dataset,
    geom,
) -> xr.DataArray:
    """Create geometry mask for region subsetting."""
    x = wind_region_ds.coords["x"].values
    y = wind_region_ds.coords["y"].values

    dx = _infer_axis_step(x, wind_base.coords["x"].values, "x")
    dy = _infer_axis_step(y, wind_base.coords["y"].values, "y")

    x0 = float(x[0]) - dx / 2.0
    y0 = float(y[0]) - dy / 2.0
    transform = Affine(dx, 0.0, x0, 0.0, dy, y0)

    mask = features.geometry_mask(
        geometries=[mapping(geom)],
        out_shape=(wind_region_ds.sizes["y"], wind_region_ds.sizes["x"]),
        transform=transform,
        invert=True,
    )

    return xr.DataArray(
        mask,
        coords={"y": wind_region_ds["y"], "x": wind_region_ds["x"]},
        dims=("y", "x"),
    )


def _apply_mask_to_spatial_vars(ds: xr.Dataset, mask_da: xr.DataArray) -> xr.Dataset:
    """Apply mask only to spatial variables."""
    spatial_vars = [name for name, var in ds.data_vars.items() if "y" in var.dims and "x" in var.dims]
    if spatial_vars:
        ds[spatial_vars] = ds[spatial_vars].where(mask_da)
    return ds


def _compute_region_ids(
    region_id: str,
    base_wind_inputs_id: str,
    base_land_inputs_id: str,
    base_grid_id: str,
    bbox_slices: _BboxSlices,
    fingerprint_method: str,
) -> tuple[str, str]:
    """Compute region inputs_id and grid_id."""
    region_inputs_items = [
        ("region_id", region_id),
        ("base_wind_inputs_id", base_wind_inputs_id),
        ("base_land_inputs_id", base_land_inputs_id),
        ("base_grid_id", base_grid_id),
        ("region_bounds", _stable_json(list(bbox_slices.bounds))),
        ("y_slice", f"{bbox_slices.y_slice.start}:{bbox_slices.y_slice.stop}"),
        ("x_slice", f"{bbox_slices.x_slice.start}:{bbox_slices.x_slice.stop}"),
    ]
    region_inputs_id = hash_inputs_id(region_inputs_items, method=fingerprint_method)

    region_grid_items = [
        ("base_grid_id", base_grid_id),
        ("y_slice", f"{bbox_slices.y_slice.start}:{bbox_slices.y_slice.stop}"),
        ("x_slice", f"{bbox_slices.x_slice.start}:{bbox_slices.x_slice.stop}"),
    ]
    region_grid_id = hash_inputs_id(region_grid_items, method=fingerprint_method)

    return region_inputs_id, region_grid_id


def _prepare_region_dataset(
    ds: xr.Dataset,
    base_attrs: dict[str, Any],
    region_id: str,
    region_inputs_id: str,
    region_grid_id: str,
    base_inputs_id_key: str,
    base_inputs_id_value: str,
    chunk_policy: dict[str, int],
) -> xr.Dataset:
    """Prepare region dataset with attrs and chunking."""
    region_attrs = dict(base_attrs)
    region_attrs["store_state"] = "complete"
    region_attrs["region_id"] = region_id
    region_attrs["inputs_id"] = region_inputs_id
    region_attrs["grid_id"] = region_grid_id
    region_attrs[base_inputs_id_key] = base_inputs_id_value

    ds = ds.assign_attrs(region_attrs)
    chunk_y = chunk_policy.get("y", 1024)
    chunk_x = chunk_policy.get("x", 1024)
    ds = ds.chunk({"y": chunk_y, "x": chunk_x})

    for var in ds.variables.values():
        var.encoding.pop("chunks", None)

    return ds


def _write_region_store(store_path: Path, ds: xr.Dataset, store_type: str) -> None:
    """Write region store with single-writer lock."""
    with single_writer_lock(zarr_store_lock_dir(store_path)):
        if store_path.exists():
            shutil.rmtree(store_path)

        ds.to_zarr(
            store_path,
            mode="w",
            consolidated=False,
            align_chunks=True,
        )

    logger.info(f"Created region {store_type} store: {store_path} (y: {ds.sizes['y']}, x: {ds.sizes['x']})")


def _ensure_region_stores_ready(
    *,
    atlas,
    unifier,
    region_id: str,
    logger,
) -> None:
    """Ensure region stores exist for current region selection."""
    expected_root = atlas.path / "regions" / region_id
    expected_wind = expected_root / "wind.zarr"
    expected_land = expected_root / "landscape.zarr"

    # If we have a stale/partial directory, remove it so Unifier can't mis-detect completeness.
    if expected_root.exists() and (not expected_wind.exists() or not expected_land.exists()):
        logger.warning(
            f"Region store directory exists but is incomplete: "
            f"{expected_root} (wind={expected_wind.exists()}, landscape={expected_land.exists()}). "
            "Removing and rebuilding."
        )
        shutil.rmtree(expected_root)

    # Always delegate freshness/completeness decision to materialize_region().
    # It performs the full compatibility check (including base inputs linkage)
    # and can no-op when region stores are already up-to-date.
    unifier.materialize_region(atlas, region_id)

    if not (expected_wind.exists() and expected_land.exists()):
        details = {
            "expected_root": str(expected_root),
            "expected_wind_exists": expected_wind.exists(),
            "expected_landscape_exists": expected_land.exists(),
        }
        raise RuntimeError(
            f"Region stores are still missing after materialize_region({region_id!r}). Details: {details}"
        )


def materialize_region(unifier, atlas, region_id: str) -> None:
    """Materialize region stores by subsetting from country stores."""
    paths = _build_region_paths(atlas, region_id)

    # Check freshness before doing any work
    base_wind_id, base_land_id = _read_base_inputs_ids(paths, region_id)
    if _check_region_stores_fresh(paths, region_id, base_wind_id, base_land_id):
        return

    # Open and validate base stores
    wind_base, land_base = _open_base_stores(paths, unifier.chunk_policy)

    try:
        # Get region geometry and compute bbox slices
        region_gdf = _get_region_geometry(atlas, region_id)
        bbox_slices = _compute_bbox_slices(wind_base, region_gdf.total_bounds, region_id)

        # Subset datasets
        wind_region_ds = wind_base.isel(y=bbox_slices.y_slice, x=bbox_slices.x_slice)
        land_region_ds = land_base.isel(y=bbox_slices.y_slice, x=bbox_slices.x_slice)

        # Create and apply geometry mask
        geom = region_gdf.geometry.iloc[0]
        mask_da = _create_geometry_mask(wind_region_ds, wind_base, geom)
        wind_region_ds = _apply_mask_to_spatial_vars(wind_region_ds, mask_da)
        land_region_ds = _apply_mask_to_spatial_vars(land_region_ds, mask_da)

        # Compute region IDs
        base_wind_inputs_id = wind_base.attrs.get("inputs_id", "")
        base_land_inputs_id = land_base.attrs.get("inputs_id", "")
        base_grid_id = wind_base.attrs.get("grid_id", "")

        region_inputs_id, region_grid_id = _compute_region_ids(
            region_id,
            base_wind_inputs_id,
            base_land_inputs_id,
            base_grid_id,
            bbox_slices,
            unifier.fingerprint_method,
        )

        # Prepare and write region stores
        paths.region_root.mkdir(parents=True, exist_ok=True)

        wind_region_ds = _prepare_region_dataset(
            wind_region_ds,
            dict(wind_base.attrs),
            region_id,
            region_inputs_id,
            region_grid_id,
            "base_wind_inputs_id",
            base_wind_inputs_id,
            unifier.chunk_policy,
        )
        _write_region_store(paths.wind_region, wind_region_ds, "wind")

        land_region_ds = _prepare_region_dataset(
            land_region_ds,
            dict(land_base.attrs),
            region_id,
            region_inputs_id,
            region_grid_id,
            "base_land_inputs_id",
            base_land_inputs_id,
            unifier.chunk_policy,
        )
        _write_region_store(paths.land_region, land_region_ds, "landscape")

    finally:
        wind_base.close()
        land_base.close()
