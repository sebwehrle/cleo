"""Public API functions for landscape materialization.

This private module contains the user-facing functions for registering sources
and materializing landscape variables. These are re-exported from the main
landscape.py facade.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rioxarray as rxr
import xarray as xr
import zarr
from rasterio.enums import Resampling

from cleo.spatial import to_crs_if_needed
from cleo.store import single_writer_lock, zarr_store_lock_dir
from cleo.unification.fingerprint import fingerprint_path_mtime_size, hash_inputs_id
from cleo.unification.manifest import _read_manifest, _write_manifest_atomic
from cleo.unification.raster_io import _atomic_replace_variable_dir
from cleo.unification.store_io import (
    open_zarr_dataset,
    resolve_active_landscape_store_path,
    resolve_active_wind_store_path,
)
from cleo.unification.materializers.shared import _aoi_geom_or_none, _stable_json
from cleo.unification.materializers._landscape_vector import _canonical_vector_source_artifact
from cleo.unification.materializers._landscape_sources import (
    _current_landscape_source_fingerprint,
    _prepare_vector_landscape_variable_data,
    _register_landscape_source_entry,
    _resolve_landscape_source_for_variable,
)
from cleo.unification.materializers._helpers import (
    get_wind_reference,
    validate_if_exists_param,
)

logger = logging.getLogger(__name__)


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
    params = params or {}
    return _register_landscape_source_entry(
        atlas=atlas,
        name=name,
        kind="raster",
        source_path=source_path,
        params=params,
        fingerprint=fingerprint_path_mtime_size(source_path),
        if_exists=if_exists,
    )


def register_landscape_vector_source(
    unifier,
    atlas,
    *,
    name: str,
    shape,
    column: str | None = None,
    all_touched: bool = False,
    if_exists: str = "error",
) -> bool:
    """Register a vector source in manifest for later rasterization/materialization."""
    del unifier
    source_path, semantic_hash = _canonical_vector_source_artifact(
        atlas,
        shape=shape,
        column=column,
        all_touched=all_touched,
    )
    params = {"column": column, "all_touched": bool(all_touched)}
    return _register_landscape_source_entry(
        atlas=atlas,
        name=name,
        kind="vector",
        source_path=source_path,
        params=params,
        fingerprint=semantic_hash,
        if_exists=if_exists,
    )


def prepare_landscape_variable_data(
    unifier,
    atlas,
    variable_name: str,
) -> xr.DataArray:
    """Prepare a landscape variable DataArray from a registered source.

    Reads/writes are scoped to the active store routing (base or selected region).

    :param unifier: Unifier instance providing chunk policy and helpers.
    :param atlas: Atlas-like object with store routing context.
    :param variable_name: Target landscape variable name.
    :returns: Prepared DataArray aligned to active wind/landscape grids.
    :rtype: xarray.DataArray
    """
    store_path = resolve_active_landscape_store_path(atlas)
    wind_path = resolve_active_wind_store_path(atlas)

    wind = open_zarr_dataset(wind_path, chunk_policy=unifier.chunk_policy)
    if wind.attrs.get("store_state") != "complete":
        raise RuntimeError("wind.zarr is not complete; run Unifier.materialize_wind(atlas) first.")
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

    land = open_zarr_dataset(store_path, chunk_policy=unifier.chunk_policy)
    land_root = zarr.open_group(store_path, mode="r")
    if land_root.attrs.get("store_state") != "complete":
        raise RuntimeError("landscape.zarr is not complete; run Unifier.materialize_landscape(atlas) first.")
    land_grid_id = land_root.attrs.get("grid_id") or ""
    if land_grid_id != wind_grid_id:
        raise RuntimeError(f"landscape.zarr grid_id mismatch: {land_grid_id!r} != wind grid_id {wind_grid_id!r}")
    if "valid_mask" not in land:
        raise RuntimeError("landscape.zarr missing valid_mask")

    manifest = _read_manifest(store_path)
    source_id, source_entry = _resolve_landscape_source_for_variable(
        manifest=manifest,
        variable_name=variable_name,
    )
    source_path = Path(source_entry.get("path", ""))
    source_kind = str(source_entry.get("kind", ""))
    params_json = source_entry.get("params_json", "")
    params = json.loads(params_json) if params_json else {}

    if source_kind == "vector":
        da = _prepare_vector_landscape_variable_data(
            atlas=atlas,
            variable_name=variable_name,
            source_path=source_path,
            params=params,
            wind_ref=wind_ref,
            valid_mask=land["valid_mask"].load(),
            chunk_policy=unifier.chunk_policy,
        )
    else:
        aoi = _aoi_geom_or_none(atlas)

        da = rxr.open_rasterio(source_path, parse_coordinates=True).squeeze(drop=True)
        if da.rio.crs is None:
            raise RuntimeError(f"Source raster {source_path} has no CRS; cannot materialize.")

        if aoi is not None:
            aoi_in_da_crs = to_crs_if_needed(aoi, da.rio.crs)
            da = da.rio.clip(aoi_in_da_crs.geometry, drop=False)

        categorical = bool(params.get("categorical", False))
        resampling = Resampling.nearest if categorical else Resampling.bilinear

        da = da.rio.reproject_match(wind_ref, resampling=resampling, nodata=np.nan)
        da = da.rename(variable_name)

        clc_codes_raw = params.get("clc_codes")
        if clc_codes_raw is not None:
            if not isinstance(clc_codes_raw, list) or not clc_codes_raw:
                raise ValueError("params['clc_codes'] must be a non-empty list of integer CLC codes.")
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

    if source_kind != "vector":
        valid_mask = land["valid_mask"].load()
        da = da.where(valid_mask, np.nan)

        chunk_y = unifier.chunk_policy.get("y", 1024)
        chunk_x = unifier.chunk_policy.get("x", 1024)
        da = da.chunk({"y": chunk_y, "x": chunk_x})
    return da


def materialize_landscape_computed_variables(
    unifier,
    atlas,
    *,
    variables: dict[str, xr.DataArray],
    if_exists: str = "error",
) -> dict[str, list[str]]:
    """Materialize precomputed landscape variables into active landscape store."""
    valid_if_exists = {"error", "replace", "noop"}
    if if_exists not in valid_if_exists:
        raise ValueError(f"if_exists must be one of {sorted(valid_if_exists)!r}; got {if_exists!r}")
    if not variables:
        return {"written": [], "skipped": []}

    store_path = resolve_active_landscape_store_path(atlas)
    if not store_path.exists():
        raise FileNotFoundError(f"Active landscape store does not exist at {store_path}; call atlas.build().")

    root = zarr.open_group(store_path, mode="r")
    if root.attrs.get("store_state") != "complete":
        raise RuntimeError(
            f"Landscape store incomplete (store_state={root.attrs.get('store_state')!r}); call atlas.build()."
        )

    land = open_zarr_dataset(store_path, chunk_policy=unifier.chunk_policy)
    if "valid_mask" not in land:
        raise RuntimeError("Active landscape store missing valid_mask.")

    existing_vars = set(land.data_vars)
    names = list(variables.keys())

    if if_exists == "error":
        conflicts = [name for name in names if name in existing_vars]
        if conflicts:
            raise ValueError(
                f"Variable(s) already exist in active landscape store: {conflicts!r}. "
                "Use if_exists='replace' to overwrite or if_exists='noop' to skip."
            )

    preserve_attrs = dict(root.attrs)
    y_ref = land.coords["y"].values
    x_ref = land.coords["x"].values

    written: list[str] = []
    skipped: list[str] = []

    # Acquire single-writer lock for the duration of all write operations
    with single_writer_lock(zarr_store_lock_dir(store_path)):
        for name in names:
            da = variables[name]
            exists = name in existing_vars
            try:
                if exists and if_exists == "noop":
                    skipped.append(name)
                    continue

                if "x" in da.dims and "y" in da.dims:
                    if not np.array_equal(da.coords["y"].values, y_ref):
                        raise ValueError(f"Computed variable {name!r} has y coords not matching active landscape grid.")
                    if not np.array_equal(da.coords["x"].values, x_ref):
                        raise ValueError(f"Computed variable {name!r} has x coords not matching active landscape grid.")

                if exists and if_exists == "replace":
                    _atomic_replace_variable_dir(store_path, name)
                    existing_vars.discard(name)

                da.rename(name).to_dataset().to_zarr(store_path, mode="a", consolidated=False)

                root_w = zarr.open_group(store_path, mode="a")
                for key, value in preserve_attrs.items():
                    root_w.attrs[key] = value

                written.append(name)
                existing_vars.add(name)

            except Exception as exc:
                remaining = [n for n in names if n not in written and n not in skipped and n != name]
                err = RuntimeError(
                    "Failed to materialize computed landscape variables. "
                    f"written={written!r}, skipped={skipped!r}, failed={[name] + remaining!r}"
                )
                setattr(err, "written", tuple(written))
                setattr(err, "skipped", tuple(skipped))
                setattr(err, "failed", tuple([name] + remaining))
                raise err from exc

    return {"written": written, "skipped": skipped}


def materialize_landscape_variable(
    unifier,
    atlas,
    variable_name: str,
    *,
    if_exists: str = "error",
) -> bool:
    """Materialize a single landscape variable from a registered source.

    Reads/writes are scoped to the active store routing (base or selected region).

    :param unifier: Unifier instance providing chunk policy and helpers.
    :param atlas: Atlas-like object with store routing context.
    :param variable_name: Target landscape variable name.
    :param if_exists: Conflict policy (``error``, ``replace``, ``noop``).
    :returns: ``True`` when written, ``False`` when a validated noop is applied.
    :rtype: bool
    """
    validate_if_exists_param(if_exists)

    store_path = resolve_active_landscape_store_path(atlas)
    wind_path = resolve_active_wind_store_path(atlas)

    # Get wind reference for grid alignment
    wind_ref = get_wind_reference(wind_path, atlas.crs, chunk_policy=unifier.chunk_policy)

    # Open and validate landscape store
    land = open_zarr_dataset(store_path, chunk_policy=unifier.chunk_policy)
    land_root = zarr.open_group(store_path, mode="r")

    _validate_landscape_store_state(land_root, wind_ref.grid_id)

    if "valid_mask" not in land:
        raise RuntimeError("landscape.zarr missing valid_mask")

    # Resolve source and check if_exists
    source_ctx = _resolve_source_context(store_path, atlas, variable_name)

    # Apply if_exists semantics
    do_replace, should_skip = _check_variable_exists(variable_name, land, source_ctx, if_exists)
    if should_skip:
        return False

    # Prepare data
    params = json.loads(source_ctx["params_json"]) if source_ctx["params_json"] else {}
    categorical = bool(params.get("categorical", False))
    da = prepare_landscape_variable_data(unifier, atlas, variable_name)

    # Write to store
    _write_landscape_variable(
        store_path=store_path,
        variable_name=variable_name,
        da=da,
        source_ctx=source_ctx,
        wind_ref=wind_ref,
        unifier=unifier,
        atlas=atlas,
        do_replace=do_replace,
        categorical=categorical,
        preserve_attrs=dict(land_root.attrs),
    )

    return True


def _validate_landscape_store_state(land_root: zarr.Group, wind_grid_id: str) -> None:
    """Validate landscape store is complete and grid matches wind."""
    if land_root.attrs.get("store_state") != "complete":
        raise RuntimeError("landscape.zarr is not complete; run Unifier.materialize_landscape(atlas) first.")

    land_grid_id = land_root.attrs.get("grid_id") or ""
    if land_grid_id != wind_grid_id:
        raise RuntimeError(f"landscape.zarr grid_id mismatch: {land_grid_id!r} != wind grid_id {wind_grid_id!r}")


def _resolve_source_context(store_path: Path, atlas, variable_name: str) -> dict[str, Any]:
    """Resolve source from manifest and compute current fingerprint."""
    manifest = _read_manifest(store_path)
    source_id, source_entry = _resolve_landscape_source_for_variable(
        manifest=manifest,
        variable_name=variable_name,
    )

    source_path = Path(source_entry.get("path", ""))
    params_json = source_entry.get("params_json", "")
    stored_fingerprint = source_entry.get("fingerprint", "")
    stored_kind = source_entry.get("kind", "")
    params = json.loads(params_json) if params_json else {}

    current_fingerprint = _current_landscape_source_fingerprint(
        atlas=atlas,
        kind=stored_kind,
        source_path=source_path,
        params=params,
    )

    return {
        "source_id": source_id,
        "source_path": source_path,
        "params_json": params_json,
        "stored_fingerprint": stored_fingerprint,
        "current_fingerprint": current_fingerprint,
        "manifest": manifest,
    }


def _check_variable_exists(
    variable_name: str,
    land: xr.Dataset,
    source_ctx: dict[str, Any],
    if_exists: str,
) -> tuple[bool, bool]:
    """Check if variable exists and apply if_exists semantics.

    :returns: (do_replace, should_skip) tuple.
    """
    if variable_name not in land.data_vars:
        return False, False

    if if_exists == "noop":
        _validate_noop_match(variable_name, source_ctx)
        return False, True  # Skip without changes

    if if_exists == "error":
        raise ValueError(
            f"Variable {variable_name!r} already exists in landscape.zarr.\n"
            f"  Use if_exists='replace' to overwrite or if_exists='noop' to skip."
        )

    # if_exists == "replace"
    return True, False


def _validate_noop_match(variable_name: str, source_ctx: dict[str, Any]) -> None:
    """Validate that existing variable matches current source for noop."""
    manifest = source_ctx["manifest"]
    variables = manifest.get("variables", [])
    var_by_name = {v["variable_name"]: v for v in variables}

    if variable_name not in var_by_name:
        raise ValueError(
            f"Variable {variable_name!r} exists in store but not in manifest.\n  Use if_exists='replace' to fix."
        )

    stored_source_id = var_by_name[variable_name].get("source_id", "")
    if stored_source_id != source_ctx["source_id"]:
        raise ValueError(
            f"Variable {variable_name!r} exists with different source_id.\n"
            f"  Existing: source_id={stored_source_id!r}\n"
            f"  Expected: source_id={source_ctx['source_id']!r}\n"
            "  Use if_exists='replace' to overwrite."
        )

    if source_ctx["stored_fingerprint"] != source_ctx["current_fingerprint"]:
        raise ValueError(
            f"Variable {variable_name!r} exists but source file has changed.\n"
            f"  Stored fingerprint: {source_ctx['stored_fingerprint']!r}\n"
            f"  Current fingerprint: {source_ctx['current_fingerprint']!r}\n"
            "  Use if_exists='replace' to re-materialize."
        )


def _write_landscape_variable(
    *,
    store_path: Path,
    variable_name: str,
    da: xr.DataArray,
    source_ctx: dict[str, Any],
    wind_ref,
    unifier,
    atlas,
    do_replace: bool,
    categorical: bool,
    preserve_attrs: dict[str, Any],
) -> None:
    """Write landscape variable to store with locking."""
    with single_writer_lock(zarr_store_lock_dir(store_path)):
        if do_replace:
            _atomic_replace_variable_dir(store_path, variable_name)

        # Write variable
        da.to_dataset().to_zarr(store_path, mode="a", consolidated=False)

        # Restore preserved attrs
        root = zarr.open_group(store_path, mode="a")
        for k, v in preserve_attrs.items():
            root.attrs[k] = v

        # Update manifest
        _update_manifest_for_variable(store_path, variable_name, source_ctx["source_id"], categorical, da.dtype)

        # Update inputs_id
        _update_inputs_id(store_path, variable_name, source_ctx, wind_ref, unifier, atlas)


def _update_manifest_for_variable(
    store_path: Path,
    variable_name: str,
    source_id: str,
    categorical: bool,
    dtype,
) -> None:
    """Update manifest with new or updated variable entry."""
    manifest = _read_manifest(store_path)
    existing_vars = manifest.get("variables", [])
    existing_var_names = [v["variable_name"] for v in existing_vars]

    new_var = {
        "variable_name": variable_name,
        "source_id": source_id,
        "resampling_method": "nearest" if categorical else "bilinear",
        "nodata_policy": "nan",
        "dtype": str(dtype),
    }

    if variable_name in existing_var_names:
        for i, v in enumerate(existing_vars):
            if v["variable_name"] == variable_name:
                existing_vars[i] = new_var
                break
    else:
        existing_vars.append(new_var)

    manifest["variables"] = existing_vars
    _write_manifest_atomic(store_path, manifest)


def _update_inputs_id(
    store_path: Path,
    variable_name: str,
    source_ctx: dict[str, Any],
    wind_ref,
    unifier,
    atlas,
) -> None:
    """Update store inputs_id after variable write."""
    items: list[tuple[str, str]] = [
        ("wind:grid_id", wind_ref.grid_id),
        ("wind:inputs_id", wind_ref.inputs_id),
        ("mask_policy", "nan+valid_mask_in_landscape"),
        ("region", _stable_json(getattr(atlas, "region", None))),
        ("chunk_policy", _stable_json(unifier.chunk_policy)),
        ("incremental_add", "landscape_add_v1"),
        (f"layer:{variable_name}:source_id", source_ctx["source_id"]),
        (f"layer:{variable_name}:fingerprint", source_ctx["stored_fingerprint"]),
        (f"layer:{variable_name}:params_json", source_ctx["params_json"]),
    ]

    new_inputs_id = hash_inputs_id(items, method=unifier.fingerprint_method)

    root = zarr.open_group(store_path, mode="a")
    root.attrs["inputs_id"] = new_inputs_id


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
        raise FileNotFoundError(f"wind.zarr not found at {wind_path}. Run atlas.build_canonical() first.")
    if not landscape_path.exists():
        raise FileNotFoundError(f"landscape.zarr not found at {landscape_path}. Run atlas.build_canonical() first.")

    # 2) Open canonical stores
    chunk_y = unifier.chunk_policy.get("y", 1024)
    chunk_x = unifier.chunk_policy.get("x", 1024)
    chunks = {"y": chunk_y, "x": chunk_x}

    wind = open_zarr_dataset(wind_path, chunk_policy=chunks)
    land = open_zarr_dataset(landscape_path, chunk_policy=chunks)

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
        raise RuntimeError("wind.zarr missing 'weibull_A' variable for template grid.")

    # 5) Get elevation from landscape store
    if "elevation" not in land.data_vars:
        raise RuntimeError(
            "landscape.zarr missing 'elevation' variable. Ensure landscape store was materialized with elevation."
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
