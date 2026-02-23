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
class RegionMeta:
    """Region-store metadata used by cleanup policy."""

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


def list_region_dirs(root: Path) -> list[Path]:
    """List candidate region directories under ``root``."""
    base = Path(root)
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def read_region_store_meta(region_dir: Path) -> RegionMeta:
    """Read completeness and timestamp metadata for a region directory."""
    wind_store = region_dir / "wind.zarr"
    land_store = region_dir / "landscape.zarr"

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
                "Failed to read wind region store_state; treating as incomplete.",
                extra={"wind_store": str(wind_store)},
                exc_info=True,
            )

    if land_exists:
        try:
            g_land = zarr.open_group(land_store, mode="r")
            land_complete = g_land.attrs.get("store_state") == "complete"
        except (OSError, ValueError, TypeError, KeyError):
            logger.debug(
                "Failed to read landscape region store_state; treating as incomplete.",
                extra={"land_store": str(land_store)},
                exc_info=True,
            )

    created_at: datetime.datetime | None = None
    for store_path in (wind_store, land_store):
        try:
            g = zarr.open_group(store_path, mode="r")
            created_attr = g.attrs.get("created_at")
            if created_attr:
                created_at = datetime.datetime.fromisoformat(
                    str(created_attr).replace("Z", "+00:00")
                )
                break
        except (OSError, ValueError, TypeError, KeyError):
            logger.debug(
                "Failed to read region store created_at; trying next fallback.",
                extra={"store_path": str(store_path)},
                exc_info=True,
            )

    if created_at is None:
        created_at = datetime.datetime.fromtimestamp(region_dir.stat().st_mtime)

    is_complete = wind_exists and land_exists and wind_complete and land_complete
    return RegionMeta(
        created_at=created_at,
        is_complete=is_complete,
        wind_exists=wind_exists,
        landscape_exists=land_exists,
    )


def delete_region_dir(region_dir: Path) -> None:
    """Delete a region directory recursively."""
    shutil.rmtree(region_dir)
