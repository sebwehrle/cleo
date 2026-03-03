"""Region materialization and recovery policies."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import mapping

from cleo.store import single_writer_lock, zarr_store_lock_dir
from cleo.unification.store_io import open_zarr_dataset
from cleo.unification.fingerprint import hash_inputs_id
from cleo.unification.materializers.shared import _stable_json

logger = logging.getLogger(__name__)


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
    atlas_root = Path(atlas.path)
    wind_base_path = atlas_root / "wind.zarr"
    land_base_path = atlas_root / "landscape.zarr"
    region_root = atlas_root / "regions" / region_id
    wind_region_path = region_root / "wind.zarr"
    land_region_path = region_root / "landscape.zarr"

    # Read current base-store inputs IDs for stale-region detection.
    base_wind_inputs_id_current: str | None = None
    base_land_inputs_id_current: str | None = None
    try:
        if wind_base_path.exists():
            base_wind_inputs_id_current = zarr.open_group(wind_base_path, mode="r").attrs.get("inputs_id")
        if land_base_path.exists():
            base_land_inputs_id_current = zarr.open_group(land_base_path, mode="r").attrs.get("inputs_id")
    except (OSError, ValueError, TypeError, KeyError):
        logger.debug(
            "Failed to read base store attrs for region freshness check; forcing region rebuild.",
            extra={
                "region_id": region_id,
                "wind_base_path": str(wind_base_path),
                "land_base_path": str(land_base_path),
            },
            exc_info=True,
        )
        base_wind_inputs_id_current = None
        base_land_inputs_id_current = None

    # Check if region stores already exist and are complete
    if wind_region_path.exists() and land_region_path.exists():
        wind_region = None
        land_region = None
        try:
            wind_region = open_zarr_dataset(wind_region_path)
            land_region = open_zarr_dataset(land_region_path)
            stores_complete = (
                wind_region.attrs.get("store_state") == "complete"
                and land_region.attrs.get("store_state") == "complete"
                and wind_region.attrs.get("region_id") == region_id
                and land_region.attrs.get("region_id") == region_id
            )
            base_ids_available = bool(base_wind_inputs_id_current) and bool(base_land_inputs_id_current)
            base_link_matches = (
                base_ids_available
                and wind_region.attrs.get("base_wind_inputs_id") == base_wind_inputs_id_current
                and land_region.attrs.get("base_land_inputs_id") == base_land_inputs_id_current
            )
            if stores_complete and base_link_matches:
                # Region stores already complete
                logger.info(f"Region stores for {region_id!r} already complete, skipping.")
                return
            if stores_complete and not base_link_matches:
                logger.info(
                    "Region stores for %r are stale versus current base inputs_id; rebuilding.",
                    region_id,
                    extra={
                        "region_id": region_id,
                        "region_base_wind_inputs_id": wind_region.attrs.get("base_wind_inputs_id"),
                        "region_base_land_inputs_id": land_region.attrs.get("base_land_inputs_id"),
                        "current_base_wind_inputs_id": base_wind_inputs_id_current,
                        "current_base_land_inputs_id": base_land_inputs_id_current,
                    },
                )
        except (OSError, ValueError, TypeError, KeyError):
            logger.debug(
                "Failed to verify existing region stores; recreating.",
                extra={
                    "region_id": region_id,
                    "wind_region_path": str(wind_region_path),
                    "land_region_path": str(land_region_path),
                },
                exc_info=True,
            )
        finally:
            if wind_region is not None:
                wind_region.close()
            if land_region is not None:
                land_region.close()

    # 1) Open base stores (must be complete)
    wind_base = open_zarr_dataset(wind_base_path, chunk_policy=unifier.chunk_policy)
    land_base = open_zarr_dataset(land_base_path, chunk_policy=unifier.chunk_policy)

    if wind_base.attrs.get("store_state") != "complete":
        raise RuntimeError("wind.zarr is not complete; run atlas.build() first.")
    if land_base.attrs.get("store_state") != "complete":
        raise RuntimeError("landscape.zarr is not complete; run atlas.build() first.")

    base_wind_inputs_id = wind_base.attrs.get("inputs_id", "")
    base_land_inputs_id = land_base.attrs.get("inputs_id", "")
    base_grid_id = wind_base.attrs.get("grid_id", "")

    # 2) Get region geometry
    region_gdf = atlas.get_nuts_region(region_id)
    if region_gdf is None or region_gdf.empty:
        raise FileNotFoundError(
            f"Could not find region geometry for {region_id!r}. Ensure NUTS shapefile is available."
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

    geom = region_gdf.geometry.iloc[0]  # dissolved polygon/multipolygon

    # x/y are cell centers; build an affine transform for that grid
    x = wind_region_ds.coords["x"].values
    y = wind_region_ds.coords["y"].values

    dx = _infer_axis_step(x, wind_x, "x")
    dy = _infer_axis_step(y, wind_y, "y")  # can be negative if y decreases

    x0 = float(x[0]) - dx / 2.0
    y0 = float(y[0]) - dy / 2.0
    transform = Affine(dx, 0.0, x0, 0.0, dy, y0)

    mask = features.geometry_mask(
        geometries=[mapping(geom)],
        out_shape=(wind_region_ds.sizes["y"], wind_region_ds.sizes["x"]),
        transform=transform,
        invert=True,  # True inside geometry
        all_touched=False,  # set True if you want edge pixels included
    )

    mask_da = xr.DataArray(mask, coords={"y": wind_region_ds["y"], "x": wind_region_ds["x"]}, dims=("y", "x"))

    # Apply mask only to spatial variables to avoid broadcasting
    # non-spatial turbine metadata onto y/x.
    wind_spatial_vars = [name for name, var in wind_region_ds.data_vars.items() if "y" in var.dims and "x" in var.dims]
    if wind_spatial_vars:
        wind_region_ds[wind_spatial_vars] = wind_region_ds[wind_spatial_vars].where(mask_da)

    land_spatial_vars = [name for name, var in land_region_ds.data_vars.items() if "y" in var.dims and "x" in var.dims]
    if land_spatial_vars:
        land_region_ds[land_spatial_vars] = land_region_ds[land_spatial_vars].where(mask_da)

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
    region_inputs_id = hash_inputs_id(region_inputs_items, method=unifier.fingerprint_method)

    # Compute region grid_id
    region_grid_items = [
        ("base_grid_id", base_grid_id),
        ("y_slice", f"{y_slice.start}:{y_slice.stop}"),
        ("x_slice", f"{x_slice.start}:{x_slice.stop}"),
    ]
    region_grid_id = hash_inputs_id(region_grid_items, method=unifier.fingerprint_method)

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
    chunk_y = unifier.chunk_policy.get("y", 1024)
    chunk_x = unifier.chunk_policy.get("x", 1024)
    wind_region_ds = wind_region_ds.chunk({"y": chunk_y, "x": chunk_x})
    for var in wind_region_ds.variables.values():
        var.encoding.pop("chunks", None)

    # Write wind region store (compute to materialize) with single-writer lock
    with single_writer_lock(zarr_store_lock_dir(wind_region_path)):
        if wind_region_path.exists():
            shutil.rmtree(wind_region_path)

        wind_region_ds.to_zarr(
            wind_region_path,
            mode="w",
            consolidated=False,
            align_chunks=True,
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
    land_region_ds = land_region_ds.chunk({"y": chunk_y, "x": chunk_x})
    for var in land_region_ds.variables.values():
        var.encoding.pop("chunks", None)

    # Write landscape region store with single-writer lock
    with single_writer_lock(zarr_store_lock_dir(land_region_path)):
        if land_region_path.exists():
            shutil.rmtree(land_region_path)

        land_region_ds.to_zarr(
            land_region_path,
            mode="w",
            consolidated=False,
            align_chunks=True,
        )

    logger.info(
        f"Created region landscape store: {land_region_path} "
        f"(y: {land_region_ds.sizes['y']}, x: {land_region_ds.sizes['x']})"
    )
