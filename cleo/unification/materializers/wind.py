"""Wind materialization policies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import xarray as xr
import zarr

from cleo.store import atomic_dir
from cleo.unification.fingerprint import (
    fingerprint_file,
    get_git_info,
    hash_grid_id,
    hash_inputs_id,
)
from cleo.unification.gwa_io import _assert_all_required_gwa_present, _open_gwa_raster
from cleo.unification.manifest import init_manifest, write_manifest_sources, write_manifest_variables
from cleo.unification.materializers.shared import _get_clip_geometry, _stable_json
from cleo.unification.turbines import _default_turbines_from_resources, _ingest_turbines_and_costs
from cleo.policies.vertical_policy import (
    HASH_ALGORITHM,
    HASH_SCHEMA_VERSION,
    checksum_float64_le,
    policy_snapshot,
)


def materialize_wind(unifier, atlas) -> None:
    """Materialize wind.zarr as a complete canonical store."""
    store_path = Path(atlas.path) / "wind.zarr"
    iso3 = atlas.country
    target_crs = atlas.crs
    chunk_policy = getattr(atlas, "chunk_policy", unifier.chunk_policy) or {"y": 1024, "x": 1024}
    mask_policy = "nan+valid_mask_in_landscape"
    vertical_policy_snapshot = policy_snapshot(getattr(atlas, "vertical_policy", None))

    req_files = _assert_all_required_gwa_present(atlas)
    clip_geom = _get_clip_geometry(atlas)

    ref_height = 100
    ref_path = None
    for sid, p in req_files:
        if sid == f"gwa:file:weibull_A:{ref_height}":
            ref_path = p
            break

    ref_da = _open_gwa_raster(
        atlas,
        ref_path,
        iso3=iso3,
        target_crs=target_crs,
        ref_da=None,
        clip_geom=clip_geom,
        resampling="bilinear",
    )

    sources = []
    weibull_A_arrays = []
    weibull_k_arrays = []
    rho_arrays = []

    weibull_A_source_ids = []
    weibull_k_source_ids = []
    rho_source_ids = []

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
            iso3=iso3,
            target_crs=target_crs,
            ref_da=ref_da,
            clip_geom=clip_geom,
            resampling="bilinear",
        )

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

    for bundle_name, source_ids in [
        ("gwa:bundle:weibull_A", weibull_A_source_ids),
        ("gwa:bundle:weibull_k", weibull_k_source_ids),
        ("gwa:bundle:rho", rho_source_ids),
    ]:
        bundle_fingerprint = hashlib.sha256(json.dumps(sorted(source_ids)).encode()).hexdigest()[:16]
        sources.append(
            {
                "source_id": bundle_name,
                "name": bundle_name.split(":")[-1],
                "kind": "bundle",
                "path": "",
                "params_json": json.dumps({"source_ids": sorted(source_ids)}),
                "fingerprint": bundle_fingerprint,
            }
        )

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

    ds_wind = xr.Dataset(
        {
            "weibull_A": weibull_A,
            "weibull_k": weibull_k,
            "rho": rho,
            "template": ref_da,
        }
    )

    ds_tech, tech_sources, tech_variables = _ingest_turbines_and_costs(atlas, unifier.fingerprint_method)
    sources.extend(tech_sources)

    ds = xr.merge([ds_wind, ds_tech])
    if "cleo_turbines_json" in ds_tech.attrs:
        ds.attrs["cleo_turbines_json"] = ds_tech.attrs["cleo_turbines_json"]

    variables = []
    for var_name, bundle_source in [
        ("weibull_A", "gwa:bundle:weibull_A"),
        ("weibull_k", "gwa:bundle:weibull_k"),
        ("rho", "gwa:bundle:rho"),
    ]:
        variables.append(
            {
                "variable_name": var_name,
                "source_id": bundle_source,
                "resampling_method": "bilinear",
                "nodata_policy": "nan",
                "dtype": str(ds[var_name].dtype),
            }
        )

    variables.append(
        {
            "variable_name": "template",
            "source_id": f"gwa:file:weibull_A:{ref_height}",
            "resampling_method": "bilinear",
            "nodata_policy": "nan",
            "dtype": str(ds["template"].dtype),
        }
    )

    variables.extend(tech_variables)

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
                    var_chunks.append(var.sizes[dim])
            encoding[var_name] = {"chunks": tuple(var_chunks)}

    transform = ref_da.rio.transform()
    transform_tuple = (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
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

    input_items = []
    for s in sources:
        input_items.append((s["source_id"], s["fingerprint"]))

    params_str = json.dumps(
        {
            "country": atlas.country,
            "target_crs": str(target_crs),
            "region": atlas.region,
            "chunk_policy": chunk_policy,
            "mask_policy": mask_policy,
            "fingerprint_method": unifier.fingerprint_method,
        },
        sort_keys=True,
    )
    params_fingerprint = hashlib.sha256(params_str.encode()).hexdigest()[:16]
    input_items.append(("unify:params", params_fingerprint))

    turbines_cfg = atlas.turbines_configured
    turbines_part = _stable_json(list(turbines_cfg)) if turbines_cfg is not None else "default"
    input_items.append(("turbines_configured", turbines_part))

    if turbines_cfg is not None:
        effective_turbines = sorted(turbines_cfg)
    else:
        resources_dir = Path(atlas.path) / "resources"
        effective_turbines = _default_turbines_from_resources(resources_dir)
    input_items.append(("turbines_effective", _stable_json(effective_turbines)))

    resources_dir = Path(atlas.path) / "resources"
    turbine_sha256_list = []
    for tid in effective_turbines:
        yaml_path = resources_dir / f"{tid}.yml"
        if yaml_path.exists():
            content_hash = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
            turbine_sha256_list.append({"turbine_id": tid, "sha256": content_hash})
    input_items.append(("turbines_sha256", _stable_json(turbine_sha256_list)))
    input_items.append(("vertical_policy_sha256", vertical_policy_snapshot["sha256"]))
    input_items.append(("wind_speed_grid_len", str(int(ds["wind_speed"].size))))
    input_items.append(("wind_speed_grid_checksum", checksum_float64_le(ds["wind_speed"].values)))
    input_items.append(("hash_algorithm", HASH_ALGORITHM))
    input_items.append(("hash_schema_version", HASH_SCHEMA_VERSION))

    inputs_id = hash_inputs_id(input_items, unifier.fingerprint_method)

    if store_path.exists():
        try:
            existing_root = zarr.open_group(store_path, mode="r")
            if (
                existing_root.attrs.get("store_state") == "complete"
                and existing_root.attrs.get("inputs_id") == inputs_id
            ):
                return
        except (OSError, ValueError, TypeError, KeyError):
            pass

    git_info = get_git_info(Path.cwd())

    with atomic_dir(store_path) as tmp_path:
        ds.to_zarr(tmp_path, mode="w", encoding=encoding, consolidated=False)

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
        root.attrs["fingerprint_method"] = unifier.fingerprint_method
        root.attrs["cleo_vertical_policy_json"] = vertical_policy_snapshot["json"]
        root.attrs["cleo_vertical_policy_checksum"] = vertical_policy_snapshot["sha256"]
        root.attrs["wind_speed_grid_len"] = int(ds["wind_speed"].size)
        root.attrs["wind_speed_grid_checksum"] = checksum_float64_le(ds["wind_speed"].values)
        root.attrs["wind_speed_coord_source"] = "wind_store_coord"
        root.attrs["hash_algorithm"] = HASH_ALGORITHM
        root.attrs["hash_schema_version"] = HASH_SCHEMA_VERSION

        init_manifest(tmp_path)
        write_manifest_sources(tmp_path, sources)
        write_manifest_variables(tmp_path, variables)
