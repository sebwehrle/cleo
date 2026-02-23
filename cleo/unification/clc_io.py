"""CLC-specific I/O helpers for source metadata and wind-grid preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr
import yaml
from rasterio.enums import Resampling


def load_clc_codes(path: Path) -> dict[int, str]:
    """Load CLC code-to-label mapping from resources."""
    resource = Path(path) / "resources" / "clc_codes.yml"
    if not resource.exists():
        raise FileNotFoundError(
            f"Missing CLC mapping resource: {resource}. "
            "Run atlas.deploy_resources() to populate resources."
        )
    with open(resource, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("clc_codes", {})
    out: dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = str(v)
    return out


def wind_reference_template(atlas) -> xr.DataArray:
    """Get base wind template (2D y/x) for grid alignment."""
    wind_store = Path(atlas.path) / "wind.zarr"
    ds = xr.open_zarr(wind_store, consolidated=False, chunks=atlas.chunk_policy)
    if "weibull_A" not in ds.data_vars:
        raise RuntimeError("wind.zarr missing weibull_A; run atlas.build() first.")
    ref = ds["weibull_A"]
    if "height" in ref.dims:
        heights = np.asarray(ref.coords.get("height", xr.DataArray([])).data)
        if np.isin(100, heights).any():
            ref = ref.sel(height=100)
        else:
            ref = ref.isel(height=0)
    if ref.rio.crs is None:
        ref = ref.rio.write_crs(atlas.crs)
    return ref


def prepare_clc_to_wind_grid(
    *,
    source_path: Path,
    prepared_path: Path,
    ref: xr.DataArray,
    source_crs: str,
) -> Path:
    """Reproject/mask/crop CLC source raster to wind grid and write GeoTIFF."""
    valid = ref.notnull()

    clc = rxr.open_rasterio(source_path, parse_coordinates=True).squeeze(drop=True)
    if clc.rio.crs is None:
        clc = clc.rio.write_crs(source_crs)

    clc = clc.rio.reproject_match(ref, resampling=Resampling.nearest, nodata=np.nan)
    clc = clc.where(valid, np.nan)

    valid_np = np.asarray(valid.data, dtype=bool)
    y_any = np.where(valid_np.any(axis=1))[0]
    x_any = np.where(valid_np.any(axis=0))[0]
    if len(y_any) == 0 or len(x_any) == 0:
        raise RuntimeError("No valid wind cells found for CLC preparation.")

    clc = clc.isel(
        y=slice(int(y_any.min()), int(y_any.max()) + 1),
        x=slice(int(x_any.min()), int(x_any.max()) + 1),
    )
    clc = clc.astype(np.float32)
    clc = clc.rio.write_nodata(np.nan)
    clc.rio.to_raster(prepared_path)
    return prepared_path

