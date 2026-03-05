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

import xarray as xr
import zarr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AreaMeta:
    """Area-store metadata used by cleanup policy."""

    created_at: datetime.datetime
    is_complete: bool
    wind_exists: bool
    landscape_exists: bool


def open_zarr_dataset(
    store_path: str | Path,
    *,
    chunk_policy: dict[str, int] | None = None,
) -> xr.Dataset:
    """Open a Zarr store with project-standard settings."""
    return xr.open_zarr(Path(store_path), consolidated=False, chunks=chunk_policy)


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
