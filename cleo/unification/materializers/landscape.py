"""Landscape materialization policies."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rioxarray as rxr
import xarray as xr
import zarr
from rasterio.enums import Resampling

from cleo.store import atomic_dir
from cleo.unification.fingerprint import (
    fingerprint_path_mtime_size,
    get_git_info,
    hash_inputs_id,
)
from cleo.unification.manifest import (
    _read_manifest,
    _write_manifest_atomic,
    init_manifest,
    write_manifest_sources,
    write_manifest_variables,
)
from cleo.unification.nuts_io import _read_nuts_region_catalog
from cleo.unification.raster_io import (
    _atomic_replace_variable_dir,
    _build_copdem_elevation,
    _open_local_elevation,
)
from cleo.unification.materializers.shared import _aoi_geom_or_none, _now_iso, _stable_json
from cleo.unification.vertical_policy import HASH_ALGORITHM, HASH_SCHEMA_VERSION

logger = logging.getLogger(__name__)


def materialize_landscape(unifier, atlas) -> None:
    """Materialize landscape.zarr as a complete canonical store."""
    store_path = Path(atlas.path) / "landscape.zarr"
    wind_path = Path(atlas.path) / "wind.zarr"

    # Open wind canonical store (must be complete)
    wind = xr.open_zarr(wind_path, consolidated=False, chunks=unifier.chunk_policy)

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
    chunk_y = unifier.chunk_policy.get("y", 1024)
    chunk_x = unifier.chunk_policy.get("x", 1024)
    ds_land = ds_land.chunk({"y": chunk_y, "x": chunk_x})

    # Ensure elevation NaN where valid_mask False
    ds_land["elevation"] = ds_land["elevation"].where(ds_land["valid_mask"], np.nan)

    # Deterministic inputs_id
    items: list[tuple[str, str]] = []
    items.append(("wind:grid_id", wind_grid_id))
    items.append(("wind:inputs_id", wind_inputs_id))
    items.append(("mask_policy", "nan+valid_mask_in_landscape"))
    items.append(("region", _stable_json(getattr(atlas, "region", None))))
    items.append(("chunk_policy", _stable_json(unifier.chunk_policy)))
    wind_vertical_policy_checksum = str(wind.attrs.get("cleo_vertical_policy_checksum", ""))
    if wind_vertical_policy_checksum:
        items.append(("wind:vertical_policy_checksum", wind_vertical_policy_checksum))

    if elev_kind == "local":
        items.append(("elevation:kind", "legacy_tif"))
        items.append(("elevation:path", str(local_elev)))
        items.append(("elevation:fingerprint", fingerprint_path_mtime_size(local_elev)))
        items.append(("elevation:clip", "aoi" if aoi_gdf is not None else "none"))
        landscape_fingerprint_method = unifier.fingerprint_method
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
        except (OSError, ValueError, TypeError, KeyError):
            logger.debug(
                "Failed to read existing landscape store attrs for idempotency check; rebuilding.",
                extra={"store_path": str(store_path)},
                exc_info=True,
            )

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
            chunk_policy=_stable_json(unifier.chunk_policy),
            fingerprint_method=landscape_fingerprint_method,
        )
        if wind_vertical_policy_checksum:
            g.attrs["cleo_vertical_policy_checksum"] = wind_vertical_policy_checksum
        wind_vertical_policy_json = wind.attrs.get("cleo_vertical_policy_json")
        if isinstance(wind_vertical_policy_json, str) and wind_vertical_policy_json:
            g.attrs["cleo_vertical_policy_json"] = wind_vertical_policy_json
        wind_speed_grid_len = wind.attrs.get("wind_speed_grid_len")
        if wind_speed_grid_len is not None:
            g.attrs["wind_speed_grid_len"] = wind_speed_grid_len
        wind_speed_grid_checksum = wind.attrs.get("wind_speed_grid_checksum")
        if isinstance(wind_speed_grid_checksum, str) and wind_speed_grid_checksum:
            g.attrs["wind_speed_grid_checksum"] = wind_speed_grid_checksum
        wind_speed_coord_source = wind.attrs.get("wind_speed_coord_source")
        if isinstance(wind_speed_coord_source, str) and wind_speed_coord_source:
            g.attrs["wind_speed_coord_source"] = wind_speed_coord_source
        g.attrs["hash_algorithm"] = HASH_ALGORITHM
        g.attrs["hash_schema_version"] = HASH_SCHEMA_VERSION
        if git.get("git_diff_hash"):
            g.attrs["git_diff_hash"] = git["git_diff_hash"]

        # Build and store region-name metadata for selection/discovery.
        try:
            region_catalog = _read_nuts_region_catalog(atlas)
        except (FileNotFoundError, OSError, ValueError, TypeError, KeyError):
            logger.debug(
                "Failed to load NUTS region catalog for landscape attrs; continuing without catalog attrs.",
                extra={"atlas_path": str(atlas.path), "country": getattr(atlas, "country", None)},
                exc_info=True,
            )
            region_catalog = []

        if region_catalog:
            g.attrs["cleo_region_catalog_json"] = _stable_json(region_catalog)

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
    unifier,
    atlas,
    *,
    name: str,
    source_path: Path,
    kind: str = "raster",
    params: dict | None = None,
    if_exists: str = "error",
) -> bool:
    """Register a new landscape source in __manifest__/sources."""
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


def prepare_landscape_variable_data(
    unifier,
    atlas,
    variable_name: str,
) -> xr.DataArray:
    """Prepare a landscape variable DataArray from a registered source without writing."""
    store_path = Path(atlas.path) / "landscape.zarr"
    wind_path = Path(atlas.path) / "wind.zarr"

    wind = xr.open_zarr(wind_path, consolidated=False, chunks=unifier.chunk_policy)
    if wind.attrs.get("store_state") != "complete":
        raise RuntimeError(
            "wind.zarr is not complete; run Unifier.materialize_wind(atlas) first."
        )
    wind_grid_id = wind.attrs.get("grid_id") or ""

    wind_ref = wind["weibull_A"].isel(height=0)
    if "height" in wind["weibull_A"].dims:
        height_values = wind["weibull_A"]["height"].values
        if 100 in height_values:
            wind_ref = wind["weibull_A"].sel(height=100)

    if wind_ref.rio.crs is None:
        if "template" in wind and wind["template"].rio.crs is not None:
            wind_ref = wind_ref.rio.write_crs(wind["template"].rio.crs)
        else:
            wind_ref = wind_ref.rio.write_crs(atlas.crs)

    if wind_ref.rio.transform() is None:
        wind_ref = wind_ref.rio.write_transform(wind_ref.rio.transform(recalc=True))

    land = xr.open_zarr(store_path, consolidated=False, chunks=unifier.chunk_policy)
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
    params = json.loads(params_json) if params_json else {}

    aoi = _aoi_geom_or_none(atlas)

    da = rxr.open_rasterio(source_path, parse_coordinates=True).squeeze(drop=True)
    if da.rio.crs is None:
        raise RuntimeError(
            f"Source raster {source_path} has no CRS; cannot materialize."
        )

    if aoi is not None:
        from cleo.spatial import to_crs_if_needed

        aoi_in_da_crs = to_crs_if_needed(aoi, da.rio.crs)
        da = da.rio.clip(aoi_in_da_crs.geometry, drop=False)

    categorical = bool(params.get("categorical", False))
    resampling = Resampling.nearest if categorical else Resampling.bilinear

    da = da.rio.reproject_match(wind_ref, resampling=resampling, nodata=np.nan)
    da = da.rename(variable_name)

    clc_codes_raw = params.get("clc_codes")
    if clc_codes_raw is not None:
        if not isinstance(clc_codes_raw, list) or not clc_codes_raw:
            raise ValueError(
                "params['clc_codes'] must be a non-empty list of integer CLC codes."
            )
        clc_codes = [int(code) for code in clc_codes_raw]
        da = xr.where(da.isin(clc_codes), 1.0, 0.0).astype(np.float32)

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

    valid_mask = land["valid_mask"].load()
    da = da.where(valid_mask, np.nan)

    chunk_y = unifier.chunk_policy.get("y", 1024)
    chunk_x = unifier.chunk_policy.get("x", 1024)
    da = da.chunk({"y": chunk_y, "x": chunk_x})
    return da


def materialize_landscape_variable(
    unifier,
    atlas,
    variable_name: str,
    *,
    if_exists: str = "error",
) -> bool:
    """Materialize a single landscape variable from a registered source."""
    valid_if_exists = {"error", "replace", "noop"}
    if if_exists not in valid_if_exists:
        raise ValueError(
            f"if_exists must be one of {sorted(valid_if_exists)!r}; got {if_exists!r}"
        )
    store_path = Path(atlas.path) / "landscape.zarr"
    wind_path = Path(atlas.path) / "wind.zarr"

    # 1) Open wind canonical store to get wind_ref
    wind = xr.open_zarr(wind_path, consolidated=False, chunks=unifier.chunk_policy)

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
    land = xr.open_zarr(store_path, consolidated=False, chunks=unifier.chunk_policy)

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
            land = xr.open_zarr(store_path, consolidated=False, chunks=unifier.chunk_policy)

    # Use stored fingerprint for inputs_id (consistent with registration)
    fingerprint = stored_fingerprint

    # Reuse the same preparation path as staged overlays for parity.
    categorical = bool(params.get("categorical", False))
    da = prepare_landscape_variable_data(unifier, atlas, variable_name)

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
    items.append(("chunk_policy", _stable_json(unifier.chunk_policy)))
    items.append(("incremental_add", "landscape_add_v1"))
    items.append((f"layer:{variable_name}:source_id", source_id))
    items.append((f"layer:{variable_name}:fingerprint", fingerprint))
    items.append((f"layer:{variable_name}:params_json", params_json))

    new_inputs_id = hash_inputs_id(items, method=unifier.fingerprint_method)

    # Update store attrs (reopen to ensure we have latest state)
    root = zarr.open_group(store_path, mode="a")
    root.attrs["inputs_id"] = new_inputs_id
    # grid_id remains unchanged (preserved from step 8)

    return True


def compute_air_density_correction(
    unifier,
    atlas,
    *,
    chunk_size=None,
    force: bool = False,
) -> xr.DataArray:
    """Compute air density correction from canonical stores."""
    from cleo.assess import compute_air_density_correction_core

    atlas_root = Path(atlas.path)
    wind_path = atlas_root / "wind.zarr"
    landscape_path = atlas_root / "landscape.zarr"

    # 1) Require canonical stores exist
    if not wind_path.exists():
        raise FileNotFoundError(
            f"wind.zarr not found at {wind_path}. "
            "Run atlas.build_canonical() first."
        )
    if not landscape_path.exists():
        raise FileNotFoundError(
            f"landscape.zarr not found at {landscape_path}. "
            "Run atlas.build_canonical() first."
        )

    # 2) Open canonical stores
    chunk_y = unifier.chunk_policy.get("y", 1024)
    chunk_x = unifier.chunk_policy.get("x", 1024)
    chunks = {"y": chunk_y, "x": chunk_x}

    wind = xr.open_zarr(wind_path, consolidated=False, chunks=chunks)
    land = xr.open_zarr(landscape_path, consolidated=False, chunks=chunks)

    # 3) Validate store_state is "complete"
    wind_state = wind.attrs.get("store_state", None)
    land_state = land.attrs.get("store_state", None)

    if wind_state != "complete":
        raise RuntimeError(
            f"wind.zarr store_state={wind_state!r}, expected 'complete'. "
            "Run atlas.build_canonical() to complete unification."
        )
    if land_state != "complete":
        raise RuntimeError(
            f"landscape.zarr store_state={land_state!r}, expected 'complete'. "
            "Run atlas.build_canonical() to complete unification."
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
