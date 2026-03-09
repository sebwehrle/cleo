"""Shared storage I/O helpers for orchestration layers."""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import warnings

import xarray as xr
import zarr

logger = logging.getLogger(__name__)

# Default chunk policy used when no explicit policy is provided
DEFAULT_CHUNK_POLICY: dict[str, int] = {"y": 1024, "x": 1024}


@dataclass(frozen=True)
class AreaMeta:
    """Area-store metadata used by cleanup policy."""

    created_at: datetime.datetime
    is_complete: bool
    wind_exists: bool
    landscape_exists: bool


def _read_stored_chunk_policy(store_path: Path) -> dict[str, int] | None:
    """Read chunk_policy from Zarr store attrs if available.

    :param store_path: Path to the Zarr store.
    :type store_path: pathlib.Path
    :returns: Stored chunk policy or None if not found/invalid.
    :rtype: dict[str, int] | None
    """
    try:
        g = zarr.open_group(store_path, mode="r")
        chunk_policy_json = g.attrs.get("chunk_policy")
        if chunk_policy_json is not None:
            return json.loads(chunk_policy_json)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        logger.debug(
            "Failed to read chunk_policy from store attrs.",
            extra={"store_path": str(store_path)},
            exc_info=True,
        )
    return None


def open_zarr_dataset(
    store_path: str | Path,
    *,
    chunk_policy: dict[str, int] | None = None,
) -> xr.Dataset:
    """Open a Zarr store with project-standard settings.

    If the store contains a ``chunk_policy`` attr, it is used for reading to
    ensure optimal chunk alignment. A warning is emitted if the requested
    chunk policy differs from the stored one.

    :param store_path: Path to the Zarr store directory.
    :type store_path: str | pathlib.Path
    :param chunk_policy: Requested chunk policy for reading. If None, uses
        stored chunk policy (if available) or DEFAULT_CHUNK_POLICY.
    :type chunk_policy: dict[str, int] | None
    :returns: Opened xarray Dataset with aligned chunking.
    :rtype: xarray.Dataset
    """
    path = Path(store_path)
    stored_chunks = _read_stored_chunk_policy(path)

    # Determine the configured policy (what the user expects)
    configured_policy = chunk_policy if chunk_policy is not None else DEFAULT_CHUNK_POLICY

    # Determine effective read policy
    if stored_chunks is not None:
        effective_chunks = stored_chunks
        # Warn if configured differs from stored
        if stored_chunks != configured_policy:
            warnings.warn(
                f"Stored chunk policy {stored_chunks} differs from configured "
                f"{configured_policy}. Using stored chunks for optimal read "
                f"performance. To silence this warning:\n"
                f"  - Set chunk_policy={stored_chunks} when creating Atlas, or\n"
                f"  - Rebuild stores: delete wind.zarr/landscape.zarr manually "
                f"and run atlas.build()",
                UserWarning,
                stacklevel=2,
            )
    else:
        # No stored policy - use configured (backward compat for old stores)
        effective_chunks = configured_policy

    return xr.open_zarr(path, consolidated=False, chunks=effective_chunks)


def resolve_active_landscape_store_path(atlas) -> Path:
    """Resolve active landscape store path (base or area) for an atlas-like object."""
    active_store_path = getattr(atlas, "_active_landscape_store_path", None)
    if callable(active_store_path):
        return Path(active_store_path())
    if hasattr(atlas, "landscape_store_path"):
        return Path(getattr(atlas, "landscape_store_path"))
    return Path(atlas.path) / "landscape.zarr"


def resolve_active_wind_store_path(atlas) -> Path:
    """Resolve active wind store path (base or area) for an atlas-like object.

    :param atlas: Atlas-like object exposing active or base wind store accessors.
    :returns: Resolved active wind store path.
    :rtype: pathlib.Path
    """
    active_store_path = getattr(atlas, "_active_wind_store_path", None)
    if callable(active_store_path):
        return Path(active_store_path())
    if hasattr(atlas, "wind_store_path"):
        return Path(getattr(atlas, "wind_store_path"))
    return Path(atlas.path) / "wind.zarr"


@lru_cache(maxsize=256)
def turbine_ids_from_json(payload: str) -> tuple[str, ...]:
    """Decode ``cleo_turbines_json`` payload into ordered turbine IDs.

    The payload string is used as cache key to avoid repeated JSON parsing in
    hot domain/result paths.
    """
    meta = json.loads(payload)
    return tuple(t["id"] for t in meta)


def write_netcdf_atomic(
    ds: xr.Dataset,
    out_path: str | Path,
    *,
    encoding: dict | None = None,
) -> Path:
    """Write a dataset atomically to NetCDF at ``out_path``."""
    out = Path(out_path)
    tmp = out.with_name(out.name + f".__tmp__{uuid4().hex}")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(tmp, encoding=encoding)
        os.replace(tmp, out)
    except (OSError, ValueError, RuntimeError, TypeError):
        if tmp.exists():
            tmp.unlink()
        raise
    return out


def read_zarr_group_attrs(store_path: str | Path) -> dict[str, object]:
    """Read root-group attrs from a Zarr store."""
    g = zarr.open_group(Path(store_path), mode="r")
    return dict(g.attrs)


def list_area_dirs(root: Path) -> list[Path]:
    """List candidate area directories under ``root``."""
    base = Path(root)
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def read_area_store_meta(area_dir: Path) -> AreaMeta:
    """Read completeness and timestamp metadata for an area directory."""
    wind_store = area_dir / "wind.zarr"
    land_store = area_dir / "landscape.zarr"

    wind_exists = wind_store.exists() and wind_store.is_dir()
    land_exists = land_store.exists() and land_store.is_dir()

    wind_complete = False
    land_complete = False

    if wind_exists:
        try:
            g_wind = zarr.open_group(wind_store, mode="r")
            wind_complete = g_wind.attrs.get("store_state") == "complete"
        except (OSError, ValueError, TypeError, KeyError):
            logger.debug(
                "Failed to read wind area store_state; treating as incomplete.",
                extra={"wind_store": str(wind_store)},
                exc_info=True,
            )

    if land_exists:
        try:
            g_land = zarr.open_group(land_store, mode="r")
            land_complete = g_land.attrs.get("store_state") == "complete"
        except (OSError, ValueError, TypeError, KeyError):
            logger.debug(
                "Failed to read landscape area store_state; treating as incomplete.",
                extra={"land_store": str(land_store)},
                exc_info=True,
            )

    created_at: datetime.datetime | None = None
    for store_path in (wind_store, land_store):
        try:
            g = zarr.open_group(store_path, mode="r")
            created_attr = g.attrs.get("created_at")
            if created_attr:
                created_at = datetime.datetime.fromisoformat(str(created_attr).replace("Z", "+00:00"))
                break
        except (OSError, ValueError, TypeError, KeyError):
            logger.debug(
                "Failed to read area store created_at; trying next fallback.",
                extra={"store_path": str(store_path)},
                exc_info=True,
            )

    if created_at is None:
        created_at = datetime.datetime.fromtimestamp(area_dir.stat().st_mtime)

    is_complete = wind_exists and land_exists and wind_complete and land_complete
    return AreaMeta(
        created_at=created_at,
        is_complete=is_complete,
        wind_exists=wind_exists,
        landscape_exists=land_exists,
    )


def delete_area_dir(area_dir: Path) -> None:
    """Delete an area directory recursively."""
    shutil.rmtree(area_dir)
