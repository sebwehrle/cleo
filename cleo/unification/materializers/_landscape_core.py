"""Core landscape materialization function.

This private module contains the main materialize_landscape function that
creates the canonical landscape.zarr store from wind store alignment.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr

from cleo.policies.vertical_policy import HASH_ALGORITHM, HASH_SCHEMA_VERSION
from cleo.store import atomic_dir
from cleo.unification.store_io import open_zarr_dataset
from cleo.unification.fingerprint import (
    fingerprint_path_mtime_size,
    get_git_info,
    hash_inputs_id,
)
from cleo.unification.manifest import (
    init_manifest,
    write_manifest_sources,
    write_manifest_variables,
)
from cleo.unification.nuts_io import _read_nuts_region_catalog
from cleo.unification.raster_io import (
    _build_copdem_elevation,
    _open_local_elevation,
)
from cleo.unification.materializers.shared import _aoi_geom_or_none, _now_iso, _stable_json

logger = logging.getLogger(__name__)


def materialize_landscape(unifier, atlas) -> None:
    """Materialize landscape.zarr as a complete canonical store."""
    store_path = Path(atlas.path) / "landscape.zarr"
    wind_path = Path(atlas.path) / "wind.zarr"

    # Open wind canonical store (must be complete)
    wind = open_zarr_dataset(wind_path, chunk_policy=unifier.chunk_policy)

    if wind.attrs.get("store_state") != "complete":
        raise RuntimeError("wind.zarr is not complete; run Unifier.materialize_wind(atlas) first.")

    wind_grid_id = wind.attrs.get("grid_id") or ""
    wind_inputs_id = wind.attrs.get("inputs_id") or ""
    if not wind_grid_id or not wind_inputs_id:
        raise RuntimeError("wind.zarr missing grid_id/inputs_id; cannot materialize landscape.")

    # Choose wind_ref for alignment
    if "weibull_A" not in wind:
        raise RuntimeError("wind.zarr missing weibull_A; cannot define canonical grid.")

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
        wind_ref = wind_ref.rio.write_transform(wind_ref.rio.transform(recalc=True))

    # AOI geometry: same definition as wind
    aoi_gdf = _aoi_geom_or_none(atlas)

    # valid_mask derived from wind_ref (True where wind data is valid)
    valid_mask = wind_ref.notnull().rename("valid_mask")

    # Elevation: prefer local GeoTIFF (for offline use); else CopDEM path
    raw_country = Path(atlas.path) / "data" / "raw" / atlas.country
    local_elev = raw_country / f"{atlas.country}_elevation_w_bathymetry.tif"

    elev_meta: dict[str, Any] | None = None
    if local_elev.exists():
        elevation = _open_local_elevation(atlas, local_elev, wind_ref, aoi_gdf).rename("elevation")
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
                params_json=_stable_json(
                    {
                        "ref": "weibull_A",
                        "height": int(wind_ref["height"].values) if "height" in wind_ref.coords else None,
                    }
                ),
                fingerprint=hashlib.sha256(f"{wind_grid_id}:{wind_inputs_id}".encode("utf-8")).hexdigest(),
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
                    params_json=_stable_json({"clip": "aoi" if aoi_gdf is not None else "none"}),
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
                    fingerprint=hashlib.sha256(_stable_json(elev_meta).encode("utf-8")).hexdigest(),
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
                source_id=("elevation:local" if elev_kind == "local" else "elevation:copdem"),
                materialized_at=_now_iso(),
                resampling_method="bilinear",
                nodata_policy="nan",
                dtype=str(ds_land["elevation"].dtype),
            )
        )
        write_manifest_variables(tmp, vars_)
