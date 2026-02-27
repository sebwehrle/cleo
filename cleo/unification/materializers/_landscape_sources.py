"""Source registration and resolution helpers for landscape materialization.

This private module contains functions for registering landscape sources
in the manifest and resolving them for variable materialization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from rasterio.enums import MergeAlg
from rasterio.features import rasterize as rio_rasterize

from cleo.spatial import to_crs_if_needed
from cleo.unification.fingerprint import fingerprint_path_mtime_size
from cleo.unification.manifest import (
    _read_manifest,
    _write_manifest_atomic,
    init_manifest,
)
from cleo.unification.materializers.shared import _stable_json
from cleo.unification.materializers._landscape_vector import (
    _load_vector_shape,
    _vector_semantic_hash,
    _vector_values_for_column,
)


def _register_landscape_source_entry(
    *,
    atlas,
    name: str,
    kind: str,
    source_path: Path,
    params: dict,
    fingerprint: str,
    if_exists: str,
) -> bool:
    """Register/update one landscape source entry in manifest."""
    if kind not in {"raster", "vector"}:
        raise ValueError(f"Unsupported landscape source kind: {kind!r}")

    valid_if_exists = {"error", "replace", "noop"}
    if if_exists not in valid_if_exists:
        raise ValueError(f"if_exists must be one of {sorted(valid_if_exists)!r}; got {if_exists!r}")

    store_path = Path(atlas.path) / "landscape.zarr"
    root = zarr.open_group(store_path, mode="r")
    if root.attrs.get("store_state") != "complete":
        raise RuntimeError("landscape.zarr is not complete; run Unifier.materialize_landscape(atlas) first.")

    source_id = f"land:{kind}:{name}"
    other_kind = "vector" if kind == "raster" else "raster"
    other_source_id = f"land:{other_kind}:{name}"
    params_json = _stable_json(params)

    init_manifest(store_path)
    manifest = _read_manifest(store_path)
    existing_sources = manifest.get("sources", [])
    source_by_id = {s["source_id"]: s for s in existing_sources}

    if other_source_id in source_by_id:
        if if_exists != "replace":
            raise ValueError(
                f"Variable {name!r} is already registered as {other_kind!r} source.\n"
                f"  Existing source_id: {other_source_id!r}\n"
                "  Use if_exists='replace' to replace with new source kind."
            )
        existing_sources = [src for src in existing_sources if src["source_id"] != other_source_id]
        source_by_id.pop(other_source_id, None)

    if source_id in source_by_id:
        existing = source_by_id[source_id]
        existing_kind = existing.get("kind", "")
        existing_path = existing.get("path", "")
        existing_params = existing.get("params_json", "")
        existing_fingerprint = existing.get("fingerprint", "")

        exact_match = (
            existing_kind == kind
            and existing_path == str(source_path)
            and existing_params == params_json
            and existing_fingerprint == fingerprint
        )

        if if_exists == "noop":
            if exact_match:
                return False
            raise ValueError(
                f"Source {source_id!r} already registered with different configuration.\n"
                f"  Existing: path={existing_path!r}, params={existing_params!r}, "
                f"fingerprint={existing_fingerprint!r}\n"
                f"  New: path={str(source_path)!r}, params={params_json!r}, "
                f"fingerprint={fingerprint!r}\n"
                "  Use if_exists='replace' to overwrite."
            )

        if if_exists == "error":
            config_matches = existing_path == str(source_path) and existing_params == params_json
            if not config_matches:
                raise ValueError(
                    f"Source {source_id!r} already registered with different configuration.\n"
                    f"  Existing: path={existing_path!r}, params={existing_params!r}\n"
                    f"  New: path={str(source_path)!r}, params={params_json!r}\n"
                    f"  Use if_exists='replace' to overwrite."
                )
            return False

    new_source = {
        "source_id": source_id,
        "name": name,
        "kind": kind,
        "path": str(source_path),
        "params_json": params_json,
        "fingerprint": fingerprint,
    }

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

    manifest["sources"] = updated_sources
    _write_manifest_atomic(store_path, manifest)
    return True


def _resolve_landscape_source_for_variable(
    *,
    manifest: dict,
    variable_name: str,
) -> tuple[str, dict]:
    """Resolve the registered source for a variable name across source kinds."""
    sources = manifest.get("sources", [])
    source_by_id = {s["source_id"]: s for s in sources}
    candidates = [
        sid
        for sid in (
            f"land:raster:{variable_name}",
            f"land:vector:{variable_name}",
        )
        if sid in source_by_id
    ]

    if not candidates:
        raise KeyError(
            f"No registered source for variable {variable_name!r}. "
            "Call register_landscape_source/register_landscape_vector_source first."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple sources registered for variable {variable_name!r}: {candidates!r}. "
            "Use if_exists='replace' to keep a single source kind."
        )

    source_id = candidates[0]
    return source_id, source_by_id[source_id]


def _current_landscape_source_fingerprint(
    *,
    atlas,
    kind: str,
    source_path: Path,
    params: dict,
) -> str:
    """Compute current source fingerprint for noop/exact-match checks."""
    if kind == "raster":
        return fingerprint_path_mtime_size(source_path)
    if kind != "vector":
        raise ValueError(f"Unsupported source kind {kind!r}")

    column = params.get("column")
    if column is not None and not isinstance(column, str):
        raise TypeError(f"Vector source params['column'] must be str|None, got {type(column).__name__}.")
    all_touched = bool(params.get("all_touched", False))
    gdf = _load_vector_shape(source_path)
    if gdf.crs is None:
        raise ValueError(f"Vector source {source_path} has no CRS; cannot materialize.")
    gdf = to_crs_if_needed(gdf, atlas.crs)
    gdf = gdf.reset_index(drop=True)
    return _vector_semantic_hash(
        gdf,
        column=column,
        all_touched=all_touched,
    )


def _prepare_vector_landscape_variable_data(
    *,
    atlas,
    variable_name: str,
    source_path: Path,
    params: dict,
    wind_ref: xr.DataArray,
    valid_mask: xr.DataArray,
    chunk_policy: dict,
) -> xr.DataArray:
    """Prepare a vector-sourced landscape variable without writing to store."""
    gdf = _load_vector_shape(source_path)
    if gdf.crs is None:
        raise RuntimeError(f"Source vector {source_path} has no CRS; cannot materialize.")

    column = params.get("column")
    if column is not None and not isinstance(column, str):
        raise TypeError(f"Vector source params['column'] must be str|None, got {type(column).__name__}.")
    all_touched = bool(params.get("all_touched", False))

    gdf = to_crs_if_needed(gdf, wind_ref.rio.crs)
    burn_values = _vector_values_for_column(gdf, column=column)
    shapes = [(geom, value) for geom, value in zip(gdf.geometry.tolist(), burn_values, strict=True)]

    transform = wind_ref.rio.transform(recalc=True)
    out_shape = (int(wind_ref.sizes["y"]), int(wind_ref.sizes["x"]))
    burned = rio_rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0.0,
        all_touched=all_touched,
        merge_alg=MergeAlg.replace,
        dtype="float32",
    )

    da = xr.DataArray(
        burned,
        dims=("y", "x"),
        coords={"y": wind_ref["y"].values, "x": wind_ref["x"].values},
        name=variable_name,
    )
    da = da.astype(np.float32)
    da = da.rio.write_transform(transform).rio.write_crs(wind_ref.rio.crs)
    da = da.where(valid_mask, np.nan)

    chunk_y = chunk_policy.get("y", 1024)
    chunk_x = chunk_policy.get("x", 1024)
    return da.chunk({"y": chunk_y, "x": chunk_x})
