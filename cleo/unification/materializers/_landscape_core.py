"""Core landscape materialization function.

This private module contains the main materialize_landscape function that
creates the canonical landscape.zarr store from wind store alignment.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr

from cleo.policies.vertical_policy import HASH_ALGORITHM, HASH_SCHEMA_VERSION
from cleo.store import atomic_dir
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
from cleo.unification.nuts_io import _read_nuts_area_catalog
from cleo.unification.raster_io import (
    _build_copdem_elevation,
    _open_local_elevation,
)
from cleo.unification.materializers.shared import _aoi_geom_or_none, _now_iso, _stable_json
from cleo.unification.materializers._helpers import (
    check_store_idempotent,
    get_wind_reference,
)

logger = logging.getLogger(__name__)
_AREA_CATALOG_ATTR = "cleo_area_catalog_json"
_AREA_CATALOG_COUNTRY_ATTR = "cleo_area_catalog_country_iso3"


@dataclass
class _ElevationResult:
    """Result of elevation data loading."""

    data: xr.DataArray
    kind: str  # "local" or "copdem"
    meta: dict[str, Any] | None
    local_path: Path | None


def materialize_landscape(unifier, atlas) -> None:
    """Materialize landscape.zarr as a complete canonical store."""
    store_path = Path(atlas.path) / "landscape.zarr"
    wind_path = Path(atlas.path) / "wind.zarr"

    # Get wind reference for grid alignment
    wind_ref = get_wind_reference(wind_path, atlas.crs, chunk_policy=unifier.chunk_policy)
    aoi_gdf = _aoi_geom_or_none(atlas)

    # Load elevation data
    elev_result = _load_elevation_data(atlas, wind_ref.ref_da, aoi_gdf)

    # Build dataset
    ds_land = _build_landscape_dataset(wind_ref, elev_result, unifier.chunk_policy)

    # Compute inputs_id for idempotency
    inputs_id, fingerprint_method = _compute_landscape_inputs_id(wind_ref, elev_result, aoi_gdf, unifier, atlas)

    # Idempotency check
    if check_store_idempotent(store_path, inputs_id=inputs_id, grid_id=wind_ref.grid_id):
        return

    # Atomic write
    _write_landscape_store(
        store_path=store_path,
        wind_path=wind_path,
        ds_land=ds_land,
        wind_ref=wind_ref,
        elev_result=elev_result,
        aoi_gdf=aoi_gdf,
        inputs_id=inputs_id,
        fingerprint_method=fingerprint_method,
        unifier=unifier,
        atlas=atlas,
    )


def _load_elevation_data(atlas, wind_ref_da: xr.DataArray, aoi_gdf) -> _ElevationResult:
    """Load elevation data from local file or CopDEM."""
    raw_country = Path(atlas.path) / "data" / "raw" / atlas.country
    local_elev = raw_country / f"{atlas.country}_elevation_w_bathymetry.tif"

    if local_elev.exists():
        elevation = _open_local_elevation(atlas, local_elev, wind_ref_da, aoi_gdf).rename("elevation")
        return _ElevationResult(data=elevation, kind="local", meta=None, local_path=local_elev)

    elevation, elev_meta = _build_copdem_elevation(atlas, wind_ref_da, aoi_gdf)
    elevation = elevation.rename("elevation")
    return _ElevationResult(data=elevation, kind="copdem", meta=elev_meta, local_path=None)


def _build_landscape_dataset(
    wind_ref,
    elev_result: _ElevationResult,
    chunk_policy: dict[str, int],
) -> xr.Dataset:
    """Build landscape dataset with valid_mask and elevation."""
    valid_mask = wind_ref.ref_da.notnull().rename("valid_mask")

    ds_land = xr.Dataset(
        coords={"y": wind_ref.dataset["y"], "x": wind_ref.dataset["x"]},
        data_vars={"valid_mask": valid_mask, "elevation": elev_result.data},
    )
    ds_land = ds_land.rio.write_crs(wind_ref.crs)
    ds_land = ds_land.rio.write_transform(wind_ref.transform)

    # Apply chunking
    chunk_y = chunk_policy.get("y", 1024)
    chunk_x = chunk_policy.get("x", 1024)
    ds_land = ds_land.chunk({"y": chunk_y, "x": chunk_x})

    # Ensure elevation NaN where valid_mask False
    ds_land["elevation"] = ds_land["elevation"].where(ds_land["valid_mask"], np.nan)

    return ds_land


def _compute_landscape_inputs_id(
    wind_ref,
    elev_result: _ElevationResult,
    aoi_gdf,
    unifier,
    atlas,
) -> tuple[str, str]:
    """Compute deterministic inputs_id for landscape store."""
    items: list[tuple[str, str]] = [
        ("wind:grid_id", wind_ref.grid_id),
        ("wind:inputs_id", wind_ref.inputs_id),
        ("mask_policy", "nan+valid_mask_in_landscape"),
        ("area", _stable_json(getattr(atlas, "area", None))),
        ("chunk_policy", _stable_json(unifier.chunk_policy)),
    ]

    wind_vp_checksum = str(wind_ref.dataset.attrs.get("cleo_vertical_policy_checksum", ""))
    if wind_vp_checksum:
        items.append(("wind:vertical_policy_checksum", wind_vp_checksum))

    if elev_result.kind == "local":
        items.extend(
            [
                ("elevation:kind", "legacy_tif"),
                ("elevation:path", str(elev_result.local_path)),
                ("elevation:fingerprint", fingerprint_path_mtime_size(elev_result.local_path)),
                ("elevation:clip", "aoi" if aoi_gdf is not None else "none"),
            ]
        )
        fingerprint_method = unifier.fingerprint_method
    else:
        meta = elev_result.meta
        if meta is None:
            raise RuntimeError("elev_result.meta muat be set when elev_result.kind != 'local'")

        items.extend(
            [
                ("elevation:kind", "copdem"),
                ("elevation:provider", meta["provider"]),
                ("elevation:version", meta["version"]),
                ("elevation:bbox_4326", _stable_json(meta["bbox_4326"])),
                ("elevation:tile_ids", _stable_json(meta["tile_ids"])),
                ("elevation:clip", meta["clip"]),
            ]
        )
        fingerprint_method = "copdem_tiles"

    inputs_id = hash_inputs_id(items, method=fingerprint_method)
    return inputs_id, fingerprint_method


def _write_landscape_store(
    *,
    store_path: Path,
    wind_path: Path,
    ds_land: xr.Dataset,
    wind_ref,
    elev_result: _ElevationResult,
    aoi_gdf,
    inputs_id: str,
    fingerprint_method: str,
    unifier,
    atlas,
) -> None:
    """Atomic write of landscape store with all attrs and manifest."""
    git = get_git_info(repo_root=Path(__file__).resolve().parents[1])

    with atomic_dir(store_path) as tmp:
        ds_land.to_zarr(tmp, mode="w", consolidated=False)
        g = zarr.open_group(tmp, mode="a")

        # Write core attrs
        _write_landscape_core_attrs(g, wind_ref, inputs_id, fingerprint_method, unifier, git)

        # Write optional wind attrs
        _write_wind_propagated_attrs(g, wind_ref.dataset)

        # Write area catalog
        _write_area_catalog_attr(g, atlas)

        # Write manifest
        init_manifest(tmp)
        _write_landscape_manifest(tmp, wind_path, wind_ref, elev_result, aoi_gdf, ds_land)


def _write_landscape_core_attrs(
    g: zarr.Group,
    wind_ref,
    inputs_id: str,
    fingerprint_method: str,
    unifier,
    git: dict[str, Any],
) -> None:
    """Write core landscape store attributes."""
    g.attrs.update(
        store_state="complete",
        grid_id=wind_ref.grid_id,
        inputs_id=inputs_id,
        unify_version=git["unify_version"],
        code_dirty=git["code_dirty"],
        chunk_policy=_stable_json(unifier.chunk_policy),
        fingerprint_method=fingerprint_method,
        hash_algorithm=HASH_ALGORITHM,
        hash_schema_version=HASH_SCHEMA_VERSION,
    )
    if git.get("git_diff_hash"):
        g.attrs["git_diff_hash"] = git["git_diff_hash"]


def _write_wind_propagated_attrs(g: zarr.Group, wind_ds: xr.Dataset) -> None:
    """Propagate relevant wind store attrs to landscape store."""
    propagate_if_present = [
        ("cleo_vertical_policy_checksum", str),
        ("cleo_vertical_policy_json", str),
        ("wind_speed_grid_len", None),
        ("wind_speed_grid_checksum", str),
        ("wind_speed_coord_source", str),
    ]

    for attr_name, check_type in propagate_if_present:
        val = wind_ds.attrs.get(attr_name)
        if val is None:
            continue
        if check_type is str and not isinstance(val, str):
            continue
        if check_type is str and not val:
            continue
        g.attrs[attr_name] = val


def _write_area_catalog_attr(g: zarr.Group, atlas) -> None:
    """Write area catalog attrs for the current atlas country.

    :param g: Target zarr root group.
    :type g: zarr.Group
    :param atlas: Atlas-like object exposing ``country``.
    :type atlas: Any
    """
    try:
        area_catalog = _read_nuts_area_catalog(atlas)
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError):
        logger.debug(
            "Failed to load NUTS area catalog for landscape attrs; continuing without.",
            extra={"atlas_path": str(atlas.path), "country": getattr(atlas, "country", None)},
            exc_info=True,
        )
        area_catalog = []

    if area_catalog:
        g.attrs[_AREA_CATALOG_ATTR] = _stable_json(area_catalog)
        g.attrs[_AREA_CATALOG_COUNTRY_ATTR] = str(getattr(atlas, "country", "")).strip().upper()


def _write_landscape_manifest(
    tmp: Path,
    wind_path: Path,
    wind_ref,
    elev_result: _ElevationResult,
    aoi_gdf,
    ds_land: xr.Dataset,
) -> None:
    """Write manifest sources and variables."""
    # Build sources
    sources = [
        dict(
            source_id="mask:derived_from_wind",
            name="valid_mask derived from wind weibull_A",
            kind="derived",
            path=str(wind_path),
            params_json=_stable_json(
                {
                    "ref": "weibull_A",
                    "height": int(wind_ref.ref_da["height"].values) if "height" in wind_ref.ref_da.coords else None,
                }
            ),
            fingerprint=hashlib.sha256(f"{wind_ref.grid_id}:{wind_ref.inputs_id}".encode("utf-8")).hexdigest(),
            created_at=_now_iso(),
        )
    ]

    if elev_result.kind == "local":
        sources.append(
            dict(
                source_id="elevation:local",
                name="local elevation GeoTIFF",
                kind="raster",
                path=str(elev_result.local_path),
                params_json=_stable_json({"clip": "aoi" if aoi_gdf is not None else "none"}),
                fingerprint=fingerprint_path_mtime_size(elev_result.local_path),
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
                params_json=_stable_json(elev_result.meta),
                fingerprint=hashlib.sha256(_stable_json(elev_result.meta).encode("utf-8")).hexdigest(),
                created_at=_now_iso(),
            )
        )

    write_manifest_sources(tmp, sources)

    # Build variables
    elev_source_id = "elevation:local" if elev_result.kind == "local" else "elevation:copdem"
    variables = [
        dict(
            variable_name="valid_mask",
            source_id="mask:derived_from_wind",
            materialized_at=_now_iso(),
            resampling_method="derived",
            nodata_policy="nan",
            dtype=str(ds_land["valid_mask"].dtype),
        ),
        dict(
            variable_name="elevation",
            source_id=elev_source_id,
            materialized_at=_now_iso(),
            resampling_method="bilinear",
            nodata_policy="nan",
            dtype=str(ds_land["elevation"].dtype),
        ),
    ]
    write_manifest_variables(tmp, variables)
