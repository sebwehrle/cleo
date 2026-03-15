"""Landscape registration and materialization helpers.

This private module owns the active-store registration/materialization flow for
incremental landscape variables.
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

from cleo.spatial import canonical_crs_str, to_crs_if_needed
from cleo.store import single_writer_lock, zarr_store_lock_dir
from cleo.unification.fingerprint import (
    fingerprint_path_mtime_size,
    hash_inputs_id,
    semantic_raster_fingerprint,
)
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
    atlas,
    *,
    name: str,
    source_path: Path,
    kind: str = "raster",
    params: dict | None = None,
    fingerprint: str | None = None,
    if_exists: str = "error",
) -> bool:
    """Register a raster landscape source in the active manifest.

    :param atlas: Atlas instance with active store routing context.
    :param name: Target variable name.
    :param source_path: Path to the raster source artifact.
    :param kind: Source kind. Must be ``"raster"``.
    :param params: Optional source-parameter payload stored in the manifest.
    :param fingerprint: Optional explicit source fingerprint. When omitted,
        the path/mtime/size fingerprint of ``source_path`` is used.
    :param if_exists: Conflict policy (``error``, ``replace``, ``noop``).
    :returns: ``True`` when the manifest entry changes, else ``False``.
    :rtype: bool
    :raises ValueError: If ``kind`` is unsupported.
    """
    if kind != "raster":
        raise ValueError(f"Only kind='raster' is currently supported; got {kind!r}")
    params = params or {}
    return _register_landscape_source_entry(
        atlas=atlas,
        name=name,
        kind="raster",
        source_path=source_path,
        params=params,
        fingerprint=fingerprint_path_mtime_size(source_path) if fingerprint is None else fingerprint,
        if_exists=if_exists,
    )


def register_landscape_vector_source(
    atlas,
    *,
    name: str,
    shape,
    column: str | None = None,
    all_touched: bool = False,
    if_exists: str = "error",
) -> bool:
    """Register a vector source in the active manifest.

    :param atlas: Atlas instance with active store routing context.
    :param name: Target variable name.
    :param shape: Path-like vector source or GeoDataFrame.
    :param column: Optional value column for rasterization.
    :param all_touched: Rasterization inclusion policy.
    :param if_exists: Conflict policy (``error``, ``replace``, ``noop``).
    :returns: ``True`` when the manifest entry changes, else ``False``.
    :rtype: bool
    """
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


def register_landscape_dataarray_source(
    atlas,
    *,
    name: str,
    data: xr.DataArray,
    categorical: bool = False,
    if_exists: str = "error",
    chunk_policy: dict[str, int] | None = None,
) -> tuple[bool, xr.DataArray]:
    """Register a canonical cached raster source derived from an in-memory DataArray.

    :param atlas: Atlas instance with active store routing context.
    :param name: Target variable name.
    :param data: In-memory raster candidate.
    :param categorical: Whether categorical resampling semantics should be used.
    :param if_exists: Conflict policy (``error``, ``replace``, ``noop``).
    :param chunk_policy: Chunking policy used for normalization.
    :returns: Tuple ``(changed, prepared)`` with manifest change status and
        prepared exact-grid raster ready for staging.
    :rtype: tuple[bool, xarray.DataArray]
    """
    source_path, semantic_hash, prepared = _canonical_raster_source_artifact(
        atlas,
        name=name,
        data=data,
        categorical=categorical,
        chunk_policy=chunk_policy,
    )
    changed = _register_landscape_source_entry(
        atlas=atlas,
        name=name,
        kind="raster_cached",
        source_path=source_path,
        params={"categorical": bool(categorical)},
        fingerprint=semantic_hash,
        if_exists=if_exists,
    )
    return changed, prepared


def _load_active_landscape_prepare_context(
    atlas,
    *,
    chunk_policy: dict[str, int],
) -> dict[str, Any]:
    """Load the active landscape/wind context used by registered materialization.

    :param atlas: Atlas instance with active store routing context.
    :param chunk_policy: Chunking policy used for store opens.
    :returns: Mapping with store path, wind reference, and valid mask.
    :rtype: dict[str, typing.Any]
    :raises RuntimeError: If the active stores are incomplete or inconsistent.
    """
    store_path = resolve_active_landscape_store_path(atlas)
    wind_path = resolve_active_wind_store_path(atlas)

    wind = open_zarr_dataset(wind_path, chunk_policy=chunk_policy)
    if wind.attrs.get("store_state") != "complete":
        raise RuntimeError("wind.zarr is not complete; run atlas.build() first.")
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

    land = open_zarr_dataset(store_path, chunk_policy=chunk_policy)
    land_root = zarr.open_group(store_path, mode="r")
    _validate_landscape_store_state(land_root, wind_grid_id)
    if "valid_mask" not in land:
        raise RuntimeError("landscape.zarr missing valid_mask")

    return {
        "store_path": store_path,
        "wind_ref": wind_ref,
        "valid_mask": land["valid_mask"].load(),
    }


def _resolve_registered_source_entry(
    *,
    store_path: Path,
    variable_name: str,
) -> dict[str, Any]:
    """Resolve registered source metadata for one landscape variable.

    :param store_path: Active landscape store path.
    :param variable_name: Target variable name.
    :returns: Mapping with resolved source kind, path, and params.
    :rtype: dict[str, typing.Any]
    """
    manifest = _read_manifest(store_path)
    source_id, source_entry = _resolve_landscape_source_for_variable(
        manifest=manifest,
        variable_name=variable_name,
    )
    params_json = source_entry.get("params_json", "")
    params = json.loads(params_json) if params_json else {}
    return {
        "source_kind": str(source_entry.get("kind", "")),
        "source_path": Path(source_entry.get("path", "")),
        "params": params,
    }


def _prepare_registered_vector_data(
    *,
    atlas,
    variable_name: str,
    source_path: Path,
    params: dict[str, Any],
    wind_ref: xr.DataArray,
    valid_mask: xr.DataArray,
    chunk_policy: dict[str, int],
) -> xr.DataArray:
    """Prepare one registered vector-backed landscape variable.

    :param atlas: Atlas instance with active store routing context.
    :param variable_name: Target variable name.
    :param source_path: Canonical vector artifact path.
    :param params: Vector rasterization parameters.
    :param wind_ref: Wind reference raster used for grid alignment.
    :param valid_mask: Boolean active landscape validity mask.
    :param chunk_policy: Chunking policy used for staged output.
    :returns: Prepared variable on the active wind/landscape grid.
    :rtype: xarray.DataArray
    """
    return _prepare_vector_landscape_variable_data(
        atlas=atlas,
        variable_name=variable_name,
        source_path=source_path,
        params=params,
        wind_ref=wind_ref,
        valid_mask=valid_mask,
        chunk_policy=chunk_policy,
    )


def _prepare_registered_raster_data(
    *,
    atlas,
    variable_name: str,
    source_path: Path,
    params: dict[str, Any],
    wind_ref: xr.DataArray,
    valid_mask: xr.DataArray,
    chunk_policy: dict[str, int],
) -> xr.DataArray:
    """Prepare one registered raster-backed landscape variable.

    :param atlas: Atlas instance with active store routing context.
    :param variable_name: Target variable name.
    :param source_path: Source raster path.
    :param params: Raster preparation parameters.
    :param wind_ref: Wind reference raster used for grid alignment.
    :param valid_mask: Boolean active landscape validity mask.
    :param chunk_policy: Chunking policy used for staged output.
    :returns: Prepared raster on the active grid.
    :rtype: xarray.DataArray
    """
    da_raw = rxr.open_rasterio(source_path, parse_coordinates=True)
    da = da_raw.squeeze(drop=True)
    categorical = bool(params.get("categorical", False))
    da = _normalize_raster_candidate_to_active_grid(
        atlas=atlas,
        variable_name=variable_name,
        da=da,
        wind_ref=wind_ref,
        valid_mask=valid_mask,
        chunk_policy=chunk_policy,
        categorical=categorical,
        allow_missing_crs_exact_match=False,
    )
    clc_codes_raw = params.get("clc_codes")
    if clc_codes_raw is not None:
        if not isinstance(clc_codes_raw, list) or not clc_codes_raw:
            raise ValueError("params['clc_codes'] must be a non-empty list of integer CLC codes.")
        clc_codes = [int(code) for code in clc_codes_raw]
        da = xr.where(da.isnull(), np.nan, xr.where(da.isin(clc_codes), 1.0, 0.0)).astype(np.float32)
    return da


def _normalize_raster_candidate_to_active_grid(
    *,
    atlas,
    variable_name: str,
    da: xr.DataArray,
    wind_ref: xr.DataArray,
    valid_mask: xr.DataArray,
    chunk_policy: dict[str, int],
    categorical: bool,
    allow_missing_crs_exact_match: bool,
) -> xr.DataArray:
    """Normalize a raster candidate onto the active wind/landscape grid.

    :param atlas: Atlas instance with active store routing context.
    :param variable_name: Target variable name.
    :param da: Raster candidate.
    :param wind_ref: Active wind reference raster.
    :param valid_mask: Active landscape validity mask.
    :param chunk_policy: Chunking policy for staged output.
    :param categorical: Whether categorical resampling semantics should be used.
    :param allow_missing_crs_exact_match: Whether missing CRS may be accepted
        when the candidate already matches the active grid exactly.
    :returns: Prepared raster on the exact active grid.
    :rtype: xarray.DataArray
    :raises ValueError: If the candidate is ambiguous or cannot be aligned safely.
    :raises RuntimeError: If CRS metadata are required but missing.
    """
    da = da.squeeze(drop=True)
    if "x" not in da.dims or "y" not in da.dims:
        raise ValueError("data must include spatial dims 'y' and 'x'.")

    extra_dims = [dim for dim in da.dims if dim not in {"y", "x"}]
    if extra_dims:
        raise ValueError(
            f"data must represent exactly one raster with dims ('y', 'x'); unexpected extra dims: {extra_dims!r}"
        )

    da = da.transpose("y", "x").rename(variable_name)
    exact_grid_match = np.array_equal(da.coords["y"].values, wind_ref.coords["y"].values) and np.array_equal(
        da.coords["x"].values, wind_ref.coords["x"].values
    )

    if exact_grid_match:
        if da.rio.crs is None:
            if not allow_missing_crs_exact_match:
                raise RuntimeError("data has no CRS; cannot materialize safely.")
            da = da.rio.write_crs(wind_ref.rio.crs)
        else:
            da_crs = canonical_crs_str(da.rio.crs)
            wind_crs = canonical_crs_str(wind_ref.rio.crs)
            if da_crs != wind_crs:
                raise ValueError(
                    f"data CRS {da_crs!r} does not match active atlas CRS {wind_crs!r} for exact-grid ingestion."
                )
        da = da.rio.write_transform(wind_ref.rio.transform(recalc=True))
    else:
        if da.rio.crs is None:
            raise RuntimeError("data has no CRS and does not match the active atlas grid exactly.")
        aoi = _aoi_geom_or_none(atlas)
        if aoi is not None:
            aoi_in_da_crs = to_crs_if_needed(aoi, da.rio.crs)
            da = da.rio.clip(aoi_in_da_crs.geometry, drop=False)
        resampling = Resampling.nearest if categorical else Resampling.bilinear
        da = da.rio.reproject_match(wind_ref, resampling=resampling, nodata=np.nan).rename(variable_name)

    da = _mask_and_chunk_raster_variable(
        da,
        valid_mask=valid_mask,
        chunk_policy=chunk_policy,
    )
    _validate_materialized_variable_grid(variable_name, da, wind_ref)
    return da


def _canonical_raster_source_artifact(
    atlas,
    *,
    name: str,
    data: xr.DataArray,
    categorical: bool,
    chunk_policy: dict[str, int] | None = None,
) -> tuple[Path, str, xr.DataArray]:
    """Normalize an in-memory raster and persist a canonical cache artifact.

    :param atlas: Atlas instance with active store routing context.
    :param name: Target variable name.
    :param data: In-memory raster candidate.
    :param categorical: Whether categorical resampling semantics should be used.
    :param chunk_policy: Chunking policy used for normalization.
    The normalized raster is realized once during staging so semantic hashing,
    cache writes, and later materialization reuse do not recompute the same
    graph independently.
    :returns: Tuple ``(path, semantic_hash, prepared)``.
    :rtype: tuple[pathlib.Path, str, xarray.DataArray]
    """
    chunk_policy = chunk_policy or {}
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"data must be an xarray.DataArray, got {type(data).__name__}.")

    context = _load_active_landscape_prepare_context(atlas, chunk_policy=chunk_policy)
    prepared = _normalize_raster_candidate_to_active_grid(
        atlas=atlas,
        variable_name=name,
        da=data,
        wind_ref=context["wind_ref"],
        valid_mask=context["valid_mask"],
        chunk_policy=chunk_policy,
        categorical=categorical,
        allow_missing_crs_exact_match=True,
    )
    prepared = prepared.load()
    crs_wkt = canonical_crs_str(prepared.rio.crs) if prepared.rio.crs is not None else ""
    semantic_hash = semantic_raster_fingerprint(
        values=np.asarray(prepared.values),
        y=prepared.coords["y"].values,
        x=prepared.coords["x"].values,
        dtype=str(prepared.dtype),
        crs_wkt=crs_wkt,
        categorical=categorical,
    )

    out_dir = Path(atlas.path) / "intermediates" / "raster_sources"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{semantic_hash}.tif"
    if not out_path.exists():
        prepared.rio.to_raster(out_path)

    return out_path, semantic_hash, prepared


def _validate_materialized_variable_grid(
    variable_name: str,
    da: xr.DataArray,
    wind_ref: xr.DataArray,
) -> None:
    """Validate that prepared data matches the active wind reference grid.

    :param variable_name: Target variable name.
    :param da: Prepared landscape variable.
    :param wind_ref: Wind reference raster used for alignment.
    :raises ValueError: If the prepared variable does not match the wind grid exactly.
    """
    wind_y = wind_ref.coords["y"].values
    wind_x = wind_ref.coords["x"].values
    da_y = da.coords["y"].values
    da_x = da.coords["x"].values
    if np.array_equal(da_y, wind_y) and np.array_equal(da_x, wind_x):
        return
    raise ValueError(
        f"Materialized variable {variable_name!r} has y/x coords that do not match "
        f"wind reference grid exactly.\n"
        f"  wind_ref y: shape={wind_y.shape}, range=[{wind_y.min()}, {wind_y.max()}]\n"
        f"  da y: shape={da_y.shape}, range=[{da_y.min()}, {da_y.max()}]\n"
        f"  wind_ref x: shape={wind_x.shape}, range=[{wind_x.min()}, {wind_x.max()}]\n"
        f"  da x: shape={da_x.shape}, range=[{da_x.min()}, {da_x.max()}]"
    )


def _mask_and_chunk_raster_variable(
    da: xr.DataArray,
    *,
    valid_mask: xr.DataArray,
    chunk_policy: dict[str, int],
) -> xr.DataArray:
    """Mask a raster-backed variable to the active validity mask and rechunk.

    :param da: Prepared raster-backed landscape variable.
    :param valid_mask: Boolean active landscape validity mask.
    :param chunk_policy: Chunking policy used for staged output.
    :returns: Masked and rechunked variable.
    :rtype: xarray.DataArray
    """
    chunk_y = chunk_policy.get("y", 1024)
    chunk_x = chunk_policy.get("x", 1024)
    return da.where(valid_mask, np.nan).chunk({"y": chunk_y, "x": chunk_x})


def _dataset_for_landscape_write(name: str, da: xr.DataArray) -> xr.Dataset:
    """Build the canonical single-variable dataset used for landscape writes.

    :param name: Output variable name.
    :param da: Prepared DataArray to serialize.
    :returns: Dataset ready for Zarr write.
    :rtype: xarray.Dataset
    """
    return da.rename(name).to_dataset()


def prepare_landscape_variable_data(
    atlas,
    variable_name: str,
    *,
    chunk_policy: dict[str, int] | None = None,
) -> xr.DataArray:
    """Prepare a landscape variable DataArray from a registered source.

    Reads/writes are scoped to the active store routing (base or selected area).

    :param atlas: Atlas instance with active store routing context.
    :param variable_name: Target landscape variable name.
    :param chunk_policy: Chunking policy used when opening or chunking stores.
    :returns: Prepared DataArray aligned to active wind/landscape grids.
    :rtype: xarray.DataArray
    """
    chunk_policy = chunk_policy or {}
    context = _load_active_landscape_prepare_context(atlas, chunk_policy=chunk_policy)
    source = _resolve_registered_source_entry(
        store_path=context["store_path"],
        variable_name=variable_name,
    )

    if source["source_kind"] == "vector":
        da = _prepare_registered_vector_data(
            atlas=atlas,
            variable_name=variable_name,
            source_path=source["source_path"],
            params=source["params"],
            wind_ref=context["wind_ref"],
            valid_mask=context["valid_mask"],
            chunk_policy=chunk_policy,
        )
    else:
        da = _prepare_registered_raster_data(
            atlas=atlas,
            variable_name=variable_name,
            source_path=source["source_path"],
            params=source["params"],
            wind_ref=context["wind_ref"],
            valid_mask=context["valid_mask"],
            chunk_policy=chunk_policy,
        )

    _validate_materialized_variable_grid(variable_name, da, context["wind_ref"])
    return da


def materialize_landscape_computed_variables(
    atlas,
    *,
    variables: dict[str, xr.DataArray],
    chunk_policy: dict[str, int] | None = None,
    if_exists: str = "error",
) -> dict[str, list[str]]:
    """Materialize precomputed landscape variables into the active store.

    :param atlas: Atlas instance with active store routing context.
    :param variables: Mapping of variable names to prepared DataArrays.
    :param chunk_policy: Chunking policy used when opening stores.
    :param if_exists: Conflict policy (``error``, ``replace``, ``noop``).
    :returns: Mapping containing written and skipped variable-name lists.
    :rtype: dict[str, list[str]]
    """
    chunk_policy = chunk_policy or {}
    validate_if_exists_param(if_exists)
    if not variables:
        return {"written": [], "skipped": []}

    store_state = _load_active_computed_landscape_store(atlas, chunk_policy=chunk_policy)
    store_path = store_state["store_path"]
    existing_vars = store_state["existing_vars"]
    names = list(variables.keys())

    if if_exists == "error":
        conflicts = [name for name in names if name in existing_vars]
        if conflicts:
            raise ValueError(
                f"Variable(s) already exist in active landscape store: {conflicts!r}. "
                "Use if_exists='replace' to overwrite or if_exists='noop' to skip."
            )

    written: list[str] = []
    skipped: list[str] = []

    with single_writer_lock(zarr_store_lock_dir(store_path)):
        for name in names:
            try:
                if _write_computed_landscape_variable(
                    store_path=store_path,
                    name=name,
                    da=variables[name],
                    if_exists=if_exists,
                    existing_vars=existing_vars,
                    y_ref=store_state["y_ref"],
                    x_ref=store_state["x_ref"],
                    preserve_attrs=store_state["preserve_attrs"],
                ):
                    written.append(name)
                    existing_vars.add(name)
                else:
                    skipped.append(name)
            except Exception as exc:
                raise _materialize_computed_variables_error(
                    exc,
                    failed_name=name,
                    names=names,
                    written=written,
                    skipped=skipped,
                ) from exc

    return {"written": written, "skipped": skipped}


def _load_active_computed_landscape_store(
    atlas,
    *,
    chunk_policy: dict[str, int],
) -> dict[str, Any]:
    """Load the active landscape store state for computed-variable writes.

    :param atlas: Atlas instance with active store routing context.
    :param chunk_policy: Chunking policy used for store opens.
    :returns: Mapping with store path, attrs, existing variables, and reference coordinates.
    :rtype: dict[str, typing.Any]
    :raises FileNotFoundError: If the active landscape store is missing.
    :raises RuntimeError: If the active landscape store is incomplete.
    """
    store_path = resolve_active_landscape_store_path(atlas)
    if not store_path.exists():
        raise FileNotFoundError(f"Active landscape store does not exist at {store_path}; call atlas.build().")

    root = zarr.open_group(store_path, mode="r")
    if root.attrs.get("store_state") != "complete":
        raise RuntimeError(
            f"Landscape store incomplete (store_state={root.attrs.get('store_state')!r}); call atlas.build()."
        )

    land = open_zarr_dataset(store_path, chunk_policy=chunk_policy)
    if "valid_mask" not in land:
        raise RuntimeError("Active landscape store missing valid_mask.")

    return {
        "store_path": store_path,
        "existing_vars": set(land.data_vars),
        "preserve_attrs": dict(root.attrs),
        "y_ref": land.coords["y"].values,
        "x_ref": land.coords["x"].values,
    }


def _validate_computed_landscape_variable_grid(
    *,
    name: str,
    da: xr.DataArray,
    y_ref: np.ndarray,
    x_ref: np.ndarray,
) -> None:
    """Validate that a computed variable matches the active landscape grid.

    :param name: Variable name being written.
    :param da: Computed DataArray.
    :param y_ref: Active landscape y coordinates.
    :param x_ref: Active landscape x coordinates.
    :raises ValueError: If x/y coordinates do not match the active grid.
    """
    if "x" not in da.dims or "y" not in da.dims:
        return
    if not np.array_equal(da.coords["y"].values, y_ref):
        raise ValueError(f"Computed variable {name!r} has y coords not matching active landscape grid.")
    if not np.array_equal(da.coords["x"].values, x_ref):
        raise ValueError(f"Computed variable {name!r} has x coords not matching active landscape grid.")


def _restore_store_attrs(store_path: Path, preserve_attrs: dict[str, Any]) -> None:
    """Restore preserved root attributes after an incremental variable write.

    :param store_path: Target landscape store path.
    :param preserve_attrs: Root attrs that must survive the write.
    """
    root_w = zarr.open_group(store_path, mode="a")
    for key, value in preserve_attrs.items():
        root_w.attrs[key] = value


def _write_computed_landscape_variable(
    *,
    store_path: Path,
    name: str,
    da: xr.DataArray,
    if_exists: str,
    existing_vars: set[str],
    y_ref: np.ndarray,
    x_ref: np.ndarray,
    preserve_attrs: dict[str, Any],
) -> bool:
    """Write one computed landscape variable to the active store.

    :param store_path: Active landscape store path.
    :param name: Variable name to write.
    :param da: Computed DataArray.
    :param if_exists: Conflict policy.
    :param existing_vars: Mutable set of current variable names.
    :param y_ref: Active landscape y coordinates.
    :param x_ref: Active landscape x coordinates.
    :param preserve_attrs: Root attrs that must survive the write.
    :returns: ``True`` when written, ``False`` when skipped via noop.
    :rtype: bool
    """
    exists = name in existing_vars
    if exists and if_exists == "noop":
        return False

    _validate_computed_landscape_variable_grid(name=name, da=da, y_ref=y_ref, x_ref=x_ref)

    if exists and if_exists == "replace":
        _atomic_replace_variable_dir(store_path, name)
        existing_vars.discard(name)

    _dataset_for_landscape_write(name, da).to_zarr(store_path, mode="a", consolidated=False)
    _restore_store_attrs(store_path, preserve_attrs)
    return True


def _materialize_computed_variables_error(
    exc: Exception,
    *,
    failed_name: str,
    names: list[str],
    written: list[str],
    skipped: list[str],
) -> RuntimeError:
    """Build a partial-progress error for computed landscape writes.

    :param exc: Original exception.
    :param failed_name: Variable that failed to write.
    :param names: Requested write order.
    :param written: Variables already written.
    :param skipped: Variables already skipped.
    :returns: RuntimeError with progress attributes attached.
    :rtype: RuntimeError
    """
    del exc
    remaining = [n for n in names if n not in written and n not in skipped and n != failed_name]
    err = RuntimeError(
        "Failed to materialize computed landscape variables. "
        f"written={written!r}, skipped={skipped!r}, failed={[failed_name] + remaining!r}"
    )
    setattr(err, "written", tuple(written))
    setattr(err, "skipped", tuple(skipped))
    setattr(err, "failed", tuple([failed_name] + remaining))
    return err


def materialize_landscape_variable(
    atlas,
    variable_name: str,
    *,
    chunk_policy: dict[str, int] | None = None,
    fingerprint_method: str = "path_mtime_size",
    if_exists: str = "error",
    prepared_da: xr.DataArray | None = None,
) -> bool:
    """Materialize a single landscape variable from a registered source.

    Reads/writes are scoped to the active store routing (base or selected area).

    :param atlas: Atlas instance with store routing context.
    :param variable_name: Target landscape variable name.
    :param chunk_policy: Chunking policy used for store opens and writes.
    :param fingerprint_method: Fingerprint method used for inputs identity updates.
    :param if_exists: Conflict policy (``error``, ``replace``, ``noop``).
    :param prepared_da: Optional already-prepared exact-grid raster to reuse for
        materialization.
    :returns: ``True`` when written, ``False`` when a validated noop is applied.
    :rtype: bool
    """
    chunk_policy = chunk_policy or {}
    validate_if_exists_param(if_exists)

    store_path = resolve_active_landscape_store_path(atlas)
    wind_path = resolve_active_wind_store_path(atlas)

    # Get wind reference for grid alignment
    wind_ref = get_wind_reference(wind_path, atlas.crs, chunk_policy=chunk_policy)

    # Open and validate landscape store
    land = open_zarr_dataset(store_path, chunk_policy=chunk_policy)
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
    if prepared_da is None:
        da = prepare_landscape_variable_data(atlas, variable_name, chunk_policy=chunk_policy)
    else:
        da = prepared_da.rename(variable_name)
        _validate_materialized_variable_grid(variable_name, da, wind_ref.ref_da)

    # Write to store
    _write_landscape_variable(
        store_path=store_path,
        variable_name=variable_name,
        da=da,
        source_ctx=source_ctx,
        wind_ref=wind_ref,
        atlas=atlas,
        chunk_policy=chunk_policy,
        fingerprint_method=fingerprint_method,
        do_replace=do_replace,
        categorical=categorical,
        preserve_attrs=dict(land_root.attrs),
    )

    return True


def _validate_landscape_store_state(land_root: zarr.Group, wind_grid_id: str) -> None:
    """Validate active landscape store completeness and grid identity.

    :param land_root: Open active landscape store root.
    :param wind_grid_id: Expected active wind grid identifier.
    :raises RuntimeError: If the store is incomplete or uses a different grid.
    """
    if land_root.attrs.get("store_state") != "complete":
        raise RuntimeError("landscape.zarr is not complete; run atlas.build() first.")

    land_grid_id = land_root.attrs.get("grid_id") or ""
    if land_grid_id != wind_grid_id:
        raise RuntimeError(f"landscape.zarr grid_id mismatch: {land_grid_id!r} != wind grid_id {wind_grid_id!r}")


def _resolve_source_context(store_path: Path, atlas, variable_name: str) -> dict[str, Any]:
    """Resolve manifest source context for one landscape variable.

    :param store_path: Active landscape store path.
    :param atlas: Atlas instance with active store routing context.
    :param variable_name: Target landscape variable name.
    :returns: Mapping with source identity, params, manifest, and fingerprints.
    :rtype: dict[str, typing.Any]
    """
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
        stored_fingerprint=stored_fingerprint,
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
    """Validate that ``if_exists="noop"`` can safely skip a write.

    :param variable_name: Target landscape variable name.
    :param source_ctx: Resolved manifest/source fingerprint context.
    :raises ValueError: If manifest/source identity does not match the existing variable.
    """
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
    atlas,
    chunk_policy: dict[str, int],
    fingerprint_method: str,
    do_replace: bool,
    categorical: bool,
    preserve_attrs: dict[str, Any],
) -> None:
    """Write one prepared landscape variable and update manifest state.

    :param store_path: Active landscape store path.
    :param variable_name: Target variable name.
    :param da: Prepared exact-grid DataArray to write.
    :param source_ctx: Resolved manifest/source fingerprint context.
    :param wind_ref: Active wind reference wrapper.
    :param atlas: Atlas instance with active store routing context.
    :param chunk_policy: Chunking policy recorded in ``inputs_id``.
    :param fingerprint_method: Inputs-id fingerprint method label.
    :param do_replace: Whether the existing variable directory should be replaced.
    :param categorical: Whether categorical resampling semantics apply.
    :param preserve_attrs: Root attrs that must survive the write.
    """
    with single_writer_lock(zarr_store_lock_dir(store_path)):
        if do_replace:
            _atomic_replace_variable_dir(store_path, variable_name)

        # Write variable
        _dataset_for_landscape_write(variable_name, da).to_zarr(store_path, mode="a", consolidated=False)

        # Restore preserved attrs
        root = zarr.open_group(store_path, mode="a")
        for k, v in preserve_attrs.items():
            root.attrs[k] = v

        # Update manifest
        _update_manifest_for_variable(store_path, variable_name, source_ctx["source_id"], categorical, da.dtype)

        # Update inputs_id
        _update_inputs_id(
            store_path,
            variable_name,
            source_ctx,
            wind_ref,
            atlas,
            chunk_policy=chunk_policy,
            fingerprint_method=fingerprint_method,
        )


def _update_manifest_for_variable(
    store_path: Path,
    variable_name: str,
    source_id: str,
    categorical: bool,
    dtype,
) -> None:
    """Update manifest variable metadata after a successful write.

    :param store_path: Active landscape store path.
    :param variable_name: Target variable name.
    :param source_id: Registered source identifier for the variable.
    :param categorical: Whether categorical resampling semantics apply.
    :param dtype: Stored raster dtype.
    """
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
    atlas,
    *,
    chunk_policy: dict[str, int],
    fingerprint_method: str,
) -> None:
    """Update active-store ``inputs_id`` after one landscape write.

    :param store_path: Active landscape store path.
    :param variable_name: Target variable name.
    :param source_ctx: Resolved manifest/source fingerprint context.
    :param wind_ref: Active wind reference wrapper.
    :param atlas: Atlas instance with active store routing context.
    :param chunk_policy: Chunking policy recorded in inputs provenance.
    :param fingerprint_method: Inputs-id fingerprint method label.
    """
    items: list[tuple[str, str]] = [
        ("wind:grid_id", wind_ref.grid_id),
        ("wind:inputs_id", wind_ref.inputs_id),
        ("mask_policy", "nan+valid_mask_in_landscape"),
        ("area", _stable_json(getattr(atlas, "area", None))),
        ("chunk_policy", _stable_json(chunk_policy)),
        ("incremental_add", "landscape_add"),
        (f"layer:{variable_name}:source_id", source_ctx["source_id"]),
        (f"layer:{variable_name}:fingerprint", source_ctx["stored_fingerprint"]),
        (f"layer:{variable_name}:params_json", source_ctx["params_json"]),
    ]

    new_inputs_id = hash_inputs_id(items, method=fingerprint_method)

    root = zarr.open_group(store_path, mode="a")
    root.attrs["inputs_id"] = new_inputs_id


def compute_air_density_correction(
    atlas,
    *,
    chunk_policy: dict[str, int] | None = None,
    chunk_size=None,
    force: bool = False,
) -> xr.DataArray:
    """Compute air density correction from canonical stores."""
    from cleo.assess import compute_air_density_correction_core

    del chunk_size, force
    chunk_policy = chunk_policy or {}

    atlas_root = Path(atlas.path)
    wind_path = atlas_root / "wind.zarr"
    landscape_path = atlas_root / "landscape.zarr"

    # 1) Require canonical stores exist
    if not wind_path.exists():
        raise FileNotFoundError(f"wind.zarr not found at {wind_path}. Run atlas.build_canonical() first.")
    if not landscape_path.exists():
        raise FileNotFoundError(f"landscape.zarr not found at {landscape_path}. Run atlas.build_canonical() first.")

    # 2) Open canonical stores
    chunk_y = chunk_policy.get("y", 1024)
    chunk_x = chunk_policy.get("x", 1024)
    chunks = {"y": chunk_y, "x": chunk_x}

    wind = open_zarr_dataset(wind_path, chunk_policy=chunks)
    land = open_zarr_dataset(landscape_path, chunk_policy=chunks)

    # 3) Validate store_state is "complete"
    wind_state = wind.attrs.get("store_state", None)
    land_state = land.attrs.get("store_state", None)

    if wind_state != "complete":
        raise RuntimeError(
            f"wind.zarr store_state={wind_state!r}, expected 'complete'. Run atlas.build() to complete unification."
        )
    if land_state != "complete":
        raise RuntimeError(
            f"landscape.zarr store_state={land_state!r}, expected 'complete'. "
            "Run atlas.build() to complete unification."
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
    )

    return result
