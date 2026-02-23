"""Shared storage I/O helpers for orchestration layers."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import xarray as xr


def open_zarr_dataset(
    store_path: str | Path,
    *,
    chunk_policy: dict[str, int] | None = None,
) -> xr.Dataset:
    """Open a Zarr store with project-standard settings."""
    return xr.open_zarr(Path(store_path), consolidated=False, chunks=chunk_policy)


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

