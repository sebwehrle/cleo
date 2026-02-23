"""Region materialization and recovery policies."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import mapping

from cleo.unification.fingerprint import hash_inputs_id
from cleo.unification.materializers.shared import _stable_json

logger = logging.getLogger(__name__)


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


def _ensure_region_stores_ready(
    *,
    atlas,
    unifier,
    region_id: str,
    region_name: str,
    logger,
) -> None:
    """Ensure region stores exist for current region selection."""
    expected_root = atlas.path / "regions" / region_id
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

    # Prefer region_id to avoid name-vs-id directory mismatches.
    err_id: Exception | None = None
    try:
        unifier.materialize_region(atlas, region_id)
    except (RuntimeError, ValueError, TypeError, FileNotFoundError, OSError) as e:
        err_id = e
        logger.warning(
            "materialize_region(region_id) failed; retrying with region name for compatibility.",
            extra={"region_id": region_id, "region_name": region_name},
            exc_info=True,
        )
        # Backwards-compat: older Unifier versions may expect region name.
        unifier.materialize_region(atlas, region_name)

    # If Unifier wrote into a legacy directory name, migrate to the canonical region_id layout.
    if not (expected_wind.exists() and expected_land.exists()):
        for cand in _region_dir_candidates(region_name):
            alt_root = atlas.path / "regions" / cand
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
        cand_roots = [str(atlas.path / "regions" / c) for c in _region_dir_candidates(region_name)[:8]]
        details["candidate_roots"] = cand_roots
        msg = (f"Region stores are still missing after materialize_region(). Details: {details}")
        if err_id is not None:
            msg += f" (materialize_region(region_id) failed with: {type(err_id).__name__}: {err_id})"
        raise RuntimeError(msg)


def materialize_region(unifier, atlas, region_id: str) -> None:
    """Materialize region stores by subsetting from country stores."""
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

    # 1) Open base stores (must be complete)
    wind_base = xr.open_zarr(wind_base_path, consolidated=False, chunks=unifier.chunk_policy)
    land_base = xr.open_zarr(land_base_path, consolidated=False, chunks=unifier.chunk_policy)

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

    geom = region_gdf.geometry.iloc[0]  # dissolved polygon/multipolygon

    # x/y are cell centers; build an affine transform for that grid
    x = wind_region_ds.coords["x"].values
    y = wind_region_ds.coords["y"].values

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])  # can be negative if y decreases

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
    wind_spatial_vars = [
        name for name, var in wind_region_ds.data_vars.items()
        if "y" in var.dims and "x" in var.dims
    ]
    if wind_spatial_vars:
        wind_region_ds[wind_spatial_vars] = wind_region_ds[wind_spatial_vars].where(mask_da)

    land_spatial_vars = [
        name for name, var in land_region_ds.data_vars.items()
        if "y" in var.dims and "x" in var.dims
    ]
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

    # Write wind region store (compute to materialize)
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
