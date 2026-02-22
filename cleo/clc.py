"""Corine Land Cover (CLC) source handling and cache preparation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr
import yaml
from rasterio.enums import Resampling

from cleo.net import download_to_path


CLC_SOURCES: dict[str, dict[str, str]] = {
    "clc2018": {
        "filename": "U2018_CLC2018_V2020_20u1.tif",
        # Direct download URL is user/environment specific in many setups.
        # Pass url=... to Atlas.materialize_clc(...) when first populating cache.
        "url": "",
        "crs": "EPSG:3035",
    }
}


def source_cache_path(atlas_path: Path, source: str) -> Path:
    """Return cache path for the full CLC source raster."""
    if source not in CLC_SOURCES:
        raise ValueError(
            f"Unsupported CLC source {source!r}. Supported: {sorted(CLC_SOURCES)}"
        )
    return Path(atlas_path) / "data" / "raw" / "clc" / CLC_SOURCES[source]["filename"]


def prepared_country_path(atlas_path: Path, country: str, source: str) -> Path:
    """Return cache path for country-prepared CLC raster aligned to wind grid."""
    return (
        Path(atlas_path)
        / "data"
        / "raw"
        / str(country)
        / "clc"
        / f"{country}_{source}_windgrid.tif"
    )


def _load_clc_codes(path: Path) -> dict[int, str]:
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


def default_category_name(atlas_path: Path, code: int) -> str | None:
    """Return default variable name for a CLC code, or ``None`` if unknown."""
    labels = _load_clc_codes(Path(atlas_path))
    if int(code) not in labels:
        return None
    label = labels[int(code)].strip().casefold()
    name = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return name or None


def _wind_reference_template(atlas) -> xr.DataArray:
    """Get base wind template (2D y/x) for grid alignment."""
    wind_store = Path(atlas.path) / "wind.zarr"
    ds = xr.open_zarr(wind_store, consolidated=False, chunks=atlas.chunk_policy)
    if "weibull_A" not in ds.data_vars:
        raise RuntimeError("wind.zarr missing weibull_A; run atlas.materialize() first.")
    ref = ds["weibull_A"]
    if "height" in ref.dims:
        if 100 in ref.coords.get("height", xr.DataArray([])).values:
            ref = ref.sel(height=100)
        else:
            ref = ref.isel(height=0)
    if ref.rio.crs is None:
        ref = ref.rio.write_crs(atlas.crs)
    return ref


def materialize_clc(
    atlas,
    *,
    source: str = "clc2018",
    url: str | None = None,
    force_download: bool = False,
    force_prepare: bool = False,
) -> Path:
    """Prepare a country-cropped CLC raster aligned to wind/GWA grid.

    Returns path to prepared GeoTIFF in ``<atlas>/data/raw/<ISO3>/clc/``.
    """
    if source not in CLC_SOURCES:
        raise ValueError(
            f"Unsupported CLC source {source!r}. Supported: {sorted(CLC_SOURCES)}"
        )

    source_path = source_cache_path(Path(atlas.path), source)
    source_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = prepared_country_path(Path(atlas.path), atlas.country, source)
    prepared.parent.mkdir(parents=True, exist_ok=True)

    if prepared.exists() and not force_prepare:
        return prepared

    if (not source_path.exists()) or force_download:
        download_url = url or CLC_SOURCES[source].get("url")
        if not download_url:
            raise RuntimeError(
                f"No download URL configured for source {source!r}; "
                "provide url=... explicitly."
            )
        download_to_path(download_url, source_path)

    if not getattr(atlas, "_canonical_ready", False):
        atlas.materialize_canonical()

    ref = _wind_reference_template(atlas)
    valid = ref.notnull()

    clc = rxr.open_rasterio(source_path, parse_coordinates=True).squeeze(drop=True)
    if clc.rio.crs is None:
        clc = clc.rio.write_crs(CLC_SOURCES[source]["crs"])

    clc = clc.rio.reproject_match(ref, resampling=Resampling.nearest, nodata=np.nan)
    clc = clc.where(valid, np.nan)

    valid_np = np.asarray(valid.values, dtype=bool)
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
    clc.rio.to_raster(prepared)
    return prepared
