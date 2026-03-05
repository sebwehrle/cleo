"""Wind materialization policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import xarray as xr
import zarr

from cleo.store import atomic_dir
from cleo.unification.fingerprint import (
    fingerprint_file,
    get_git_info,
    hash_grid_id,
    hash_inputs_id,
)
from cleo.unification.gwa_io import ensure_required_gwa_files, _open_gwa_raster
from cleo.unification.manifest import init_manifest, write_manifest_sources, write_manifest_variables
from cleo.unification.materializers.shared import _get_clip_geometry, _stable_json
from cleo.unification.materializers._helpers import check_store_idempotent
from cleo.unification.turbines import _default_turbines_from_resources, _ingest_turbines_and_costs
from cleo.policies.vertical_policy import (
    HASH_ALGORITHM,
    HASH_SCHEMA_VERSION,
    checksum_float64_le,
    policy_snapshot,
)


@dataclass
class _GWALoadResult:
    """Result of loading GWA files."""

    ref_da: xr.DataArray
    weibull_A: xr.DataArray
    weibull_k: xr.DataArray
    rho: xr.DataArray
    sources: list[dict[str, Any]]
    weibull_A_source_ids: list[str] = field(default_factory=list)
    weibull_k_source_ids: list[str] = field(default_factory=list)
    rho_source_ids: list[str] = field(default_factory=list)


def materialize_wind(unifier, atlas) -> None:
    """Materialize wind.zarr as a complete canonical store."""
    store_path = Path(atlas.path) / "wind.zarr"
    chunk_policy = getattr(atlas, "chunk_policy", unifier.chunk_policy) or {"y": 1024, "x": 1024}
    mask_policy = "nan+valid_mask_in_landscape"
    vp_snapshot = policy_snapshot(getattr(atlas, "vertical_policy", None))

    # Load GWA data
    gwa = _load_gwa_data(atlas, unifier)

    # Merge with turbine tech data
    ds, sources, variables = _build_wind_dataset(atlas, unifier, gwa)

    # Compute grid_id and inputs_id
    grid_id = _compute_grid_id(gwa.ref_da, mask_policy)
    inputs_id = _compute_wind_inputs_id(atlas, unifier, sources, ds, chunk_policy, mask_policy, vp_snapshot)

    # Idempotency check
    if check_store_idempotent(store_path, inputs_id=inputs_id):
        return

    # Atomic write
    encoding = _compute_encoding(ds, chunk_policy)
    _write_wind_store(
        store_path,
        ds,
        sources,
        variables,
        encoding,
        grid_id,
        inputs_id,
        mask_policy,
        vp_snapshot,
        chunk_policy,
        unifier,
    )


def _load_gwa_data(atlas, unifier) -> _GWALoadResult:
    """Load and process GWA raster files."""
    req_files = ensure_required_gwa_files(atlas, auto_download=True)
    clip_geom = _get_clip_geometry(atlas)

    # Find reference raster
    ref_height = 100
    ref_path = None
    for sid, p in req_files:
        if sid == f"gwa:file:weibull_A:{ref_height}":
            ref_path = p
            break

    ref_da = _open_gwa_raster(
        atlas,
        ref_path,
        iso3=atlas.country,
        target_crs=atlas.crs,
        ref_da=None,
        clip_geom=clip_geom,
        resampling="bilinear",
    )

    # Load all GWA files
    sources: list[dict[str, Any]] = []
    arrays: dict[str, list[tuple[int, xr.DataArray]]] = {"weibull_A": [], "weibull_k": [], "rho": []}
    source_ids: dict[str, list[str]] = {"weibull_A": [], "weibull_k": [], "rho": []}

    for sid, path in req_files:
        sources.append(
            {
                "source_id": sid,
                "name": path.name,
                "kind": "raster",
                "path": str(path),
                "params_json": json.dumps({"layer": sid.split(":")[2], "height": int(sid.split(":")[-1])}),
                "fingerprint": fingerprint_file(path, unifier.fingerprint_method),
            }
        )

        da = _open_gwa_raster(
            atlas,
            path,
            iso3=atlas.country,
            target_crs=atlas.crs,
            ref_da=ref_da,
            clip_geom=clip_geom,
            resampling="bilinear",
        )

        height = int(sid.split(":")[-1])
        for var_name in arrays:
            if var_name in sid:
                arrays[var_name].append((height, da))
                source_ids[var_name].append(sid)
                break

    # Add bundle sources
    for var_name, ids in source_ids.items():
        bundle_name = f"gwa:bundle:{var_name}"
        bundle_fp = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()[:16]
        sources.append(
            {
                "source_id": bundle_name,
                "name": var_name,
                "kind": "bundle",
                "path": "",
                "params_json": json.dumps({"source_ids": sorted(ids)}),
                "fingerprint": bundle_fp,
            }
        )

    return _GWALoadResult(
        ref_da=ref_da,
        weibull_A=_stack_by_height(arrays["weibull_A"], "weibull_A"),
        weibull_k=_stack_by_height(arrays["weibull_k"], "weibull_k"),
        rho=_stack_by_height(arrays["rho"], "rho"),
        sources=sources,
        weibull_A_source_ids=source_ids["weibull_A"],
        weibull_k_source_ids=source_ids["weibull_k"],
        rho_source_ids=source_ids["rho"],
    )


def _stack_by_height(arrays_list: list[tuple[int, xr.DataArray]], name: str) -> xr.DataArray:
    """Stack arrays by height coordinate."""
    sorted_arrays = sorted(arrays_list, key=lambda x: x[0])
    heights = [h for h, _ in sorted_arrays]
    stacked = xr.concat([da for _, da in sorted_arrays], dim="height")
    stacked = stacked.assign_coords(height=heights)
    stacked.name = name
    return stacked


def _build_wind_dataset(atlas, unifier, gwa: _GWALoadResult) -> tuple[xr.Dataset, list[dict], list[dict]]:
    """Build wind dataset with GWA and turbine data."""
    ds_wind = xr.Dataset(
        {
            "weibull_A": gwa.weibull_A,
            "weibull_k": gwa.weibull_k,
            "rho": gwa.rho,
            "template": gwa.ref_da,
        }
    )

    ds_tech, tech_sources, tech_variables = _ingest_turbines_and_costs(atlas, unifier.fingerprint_method)

    sources = gwa.sources + tech_sources
    ds = xr.merge([ds_wind, ds_tech])
    if "cleo_turbines_json" in ds_tech.attrs:
        ds.attrs["cleo_turbines_json"] = ds_tech.attrs["cleo_turbines_json"]

    # Build variables list
    variables = [
        {
            "variable_name": name,
            "source_id": f"gwa:bundle:{name}",
            "resampling_method": "bilinear",
            "nodata_policy": "nan",
            "dtype": str(ds[name].dtype),
        }
        for name in ["weibull_A", "weibull_k", "rho"]
    ]
    variables.append(
        {
            "variable_name": "template",
            "source_id": "gwa:file:weibull_A:100",
            "resampling_method": "bilinear",
            "nodata_policy": "nan",
            "dtype": str(ds["template"].dtype),
        }
    )
    variables.extend(tech_variables)

    return ds, sources, variables


def _compute_grid_id(ref_da: xr.DataArray, mask_policy: str) -> str:
    """Compute grid_id from reference DataArray."""
    transform = ref_da.rio.transform()
    transform_tuple = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
    crs_wkt = str(ref_da.rio.crs.to_wkt()) if ref_da.rio.crs else ""
    shape = (ref_da.sizes["y"], ref_da.sizes["x"])

    return hash_grid_id(
        crs_wkt=crs_wkt,
        transform=transform_tuple,
        shape=shape,
        y=ref_da["y"].values,
        x=ref_da["x"].values,
        mask_policy=mask_policy,
    )


def _compute_wind_inputs_id(
    atlas,
    unifier,
    sources: list[dict],
    ds: xr.Dataset,
    chunk_policy: dict,
    mask_policy: str,
    vp_snapshot: dict,
) -> str:
    """Compute inputs_id for wind store."""
    items = [(s["source_id"], s["fingerprint"]) for s in sources]

    # Add params fingerprint
    params_str = json.dumps(
        {
            "country": atlas.country,
            "target_crs": str(atlas.crs),
            "area": atlas.area,
            "chunk_policy": chunk_policy,
            "mask_policy": mask_policy,
            "fingerprint_method": unifier.fingerprint_method,
        },
        sort_keys=True,
    )
    items.append(("unify:params", hashlib.sha256(params_str.encode()).hexdigest()[:16]))

    # Add turbine configuration
    items.extend(_turbine_config_items(atlas))

    # Add policy and grid info
    items.extend(
        [
            ("vertical_policy_sha256", vp_snapshot["sha256"]),
            ("wind_speed_grid_len", str(int(ds["wind_speed"].size))),
            ("wind_speed_grid_checksum", checksum_float64_le(ds["wind_speed"].values)),
            ("hash_algorithm", HASH_ALGORITHM),
            ("hash_schema_version", HASH_SCHEMA_VERSION),
        ]
    )

    return hash_inputs_id(items, unifier.fingerprint_method)


def _turbine_config_items(atlas) -> list[tuple[str, str]]:
    """Build turbine configuration items for inputs_id."""
    items: list[tuple[str, str]] = []
    turbines_cfg = atlas.turbines_configured

    items.append(("turbines_configured", _stable_json(list(turbines_cfg)) if turbines_cfg is not None else "default"))

    if turbines_cfg is not None:
        effective = sorted(turbines_cfg)
    else:
        effective = _default_turbines_from_resources(Path(atlas.path) / "resources")
    items.append(("turbines_effective", _stable_json(effective)))

    # Hash turbine YAML files
    resources_dir = Path(atlas.path) / "resources"
    sha_list = []
    for tid in effective:
        yaml_path = resources_dir / f"{tid}.yml"
        if yaml_path.exists():
            sha_list.append({"turbine_id": tid, "sha256": hashlib.sha256(yaml_path.read_bytes()).hexdigest()})
    items.append(("turbines_sha256", _stable_json(sha_list)))

    return items


def _compute_encoding(ds: xr.Dataset, chunk_policy: dict) -> dict:
    """Compute zarr encoding for dataset variables."""
    encoding = {}
    for var_name in ds.data_vars:
        var = ds[var_name]
        if "y" not in var.dims or "x" not in var.dims:
            continue
        var_chunks = []
        for dim in var.dims:
            if dim in chunk_policy:
                var_chunks.append(chunk_policy[dim])
            elif dim in ("y", "x"):
                var_chunks.append(chunk_policy.get(dim, var.sizes[dim]))
            else:
                var_chunks.append(var.sizes[dim])
        encoding[var_name] = {"chunks": tuple(var_chunks)}
    return encoding


def _write_wind_store(
    store_path: Path,
    ds: xr.Dataset,
    sources: list[dict],
    variables: list[dict],
    encoding: dict,
    grid_id: str,
    inputs_id: str,
    mask_policy: str,
    vp_snapshot: dict,
    chunk_policy: dict,
    unifier,
) -> None:
    """Atomic write of wind store."""
    git_info = get_git_info(Path.cwd())

    with atomic_dir(store_path) as tmp_path:
        ds.to_zarr(tmp_path, mode="w", encoding=encoding, consolidated=False)

        root = zarr.open_group(tmp_path, mode="a")
        root.attrs.update(
            {
                "store_state": "complete",
                "grid_id": grid_id,
                "inputs_id": inputs_id,
                "mask_policy": mask_policy,
                "requires_landscape_valid_mask": True,
                "unify_version": git_info["unify_version"],
                "code_dirty": git_info["code_dirty"],
                "chunk_policy": json.dumps(chunk_policy),
                "fingerprint_method": unifier.fingerprint_method,
                "cleo_vertical_policy_json": vp_snapshot["json"],
                "cleo_vertical_policy_checksum": vp_snapshot["sha256"],
                "wind_speed_grid_len": int(ds["wind_speed"].size),
                "wind_speed_grid_checksum": checksum_float64_le(ds["wind_speed"].values),
                "wind_speed_coord_source": "wind_store_coord",
                "hash_algorithm": HASH_ALGORITHM,
                "hash_schema_version": HASH_SCHEMA_VERSION,
            }
        )
        if "git_diff_hash" in git_info:
            root.attrs["git_diff_hash"] = git_info["git_diff_hash"]

        init_manifest(tmp_path)
        write_manifest_sources(tmp_path, sources)
        write_manifest_variables(tmp_path, variables)
