"""CLC-specific I/O helpers for source metadata and wind-grid preparation."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr
import yaml
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

from cleo.unification.store_io import open_zarr_dataset


def load_clc_codes(path: Path) -> dict[int, str]:
    """Load CLC code-to-label mapping from resources."""
    resource = Path(path) / "resources" / "clc_codes.yml"
    if not resource.exists():
        raise FileNotFoundError(
            f"Missing CLC mapping resource: {resource}. Run atlas.deploy_resources() to populate resources."
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
    ds = open_zarr_dataset(wind_store, chunk_policy=atlas.chunk_policy)
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


def _is_identity_pixel_grid(da: xr.DataArray) -> bool:
    """Return whether x/y coordinates match identity pixel-center indexing.

    :param da: Candidate raster array with x/y coordinates.
    :type da: xarray.DataArray
    :returns: ``True`` when coordinates are equivalent to identity transform
        pixel centers (0.5, 1.5, ...), allowing reversed y orientation.
    :rtype: bool
    """
    if "x" not in da.coords or "y" not in da.coords:
        return False
    x_vals = np.asarray(da.coords["x"].data, dtype=float)
    y_vals = np.asarray(da.coords["y"].data, dtype=float)
    if x_vals.ndim != 1 or y_vals.ndim != 1:
        return False
    x_expected = np.arange(x_vals.size, dtype=float) + 0.5
    y_expected = np.arange(y_vals.size, dtype=float) + 0.5
    x_ok = np.allclose(x_vals, x_expected, rtol=0.0, atol=1e-6)
    y_ok = np.allclose(y_vals, y_expected, rtol=0.0, atol=1e-6) or np.allclose(
        y_vals,
        y_expected[::-1],
        rtol=0.0,
        atol=1e-6,
    )
    return bool(x_ok and y_ok)


def prepare_clc_to_wind_grid(
    *,
    source_path: Path,
    prepared_path: Path,
    ref: xr.DataArray,
    source_crs: str,
    source_on_ref_grid: bool = False,
) -> Path:
    """Reproject/mask/crop CLC source raster to wind grid and write GeoTIFF.

    :param source_path: Path to source CLC raster.
    :type source_path: pathlib.Path
    :param prepared_path: Output path for prepared raster aligned to wind grid.
    :type prepared_path: pathlib.Path
    :param ref: Wind-grid reference template.
    :type ref: xarray.DataArray
    :param source_crs: Fallback source CRS used when source raster omits CRS metadata.
    :type source_crs: str
    :param source_on_ref_grid: If ``True``, interpret source as already sampled
        on the exact reference grid.
    :type source_on_ref_grid: bool
    :returns: Prepared raster path.
    :rtype: pathlib.Path
    :raises RuntimeError: If no valid wind cells exist for clipping.
    """
    valid = ref.notnull()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        clc_raw = rxr.open_rasterio(source_path, parse_coordinates=True)
        clc: xr.DataArray = clc_raw  # type: ignore[assignment]  # single-file returns DataArray
    if "band" in clc.dims:
        band_count = int(clc.sizes.get("band", 0))
        if band_count != 1:
            raise RuntimeError(
                "CLC source raster must be single-band categorical class codes, "
                f"but found {band_count} bands. "
                "This typically indicates rendered RGB(A) imagery instead of raw CLC classes."
            )
        clc = clc.squeeze("band", drop=True)
    clc = clc.squeeze(drop=True)
    clc = clc.astype(np.float32)

    inferred_ref_grid = (
        (not source_on_ref_grid)
        and clc.rio.crs is None
        and clc.sizes.get("y") == ref.sizes.get("y")
        and clc.sizes.get("x") == ref.sizes.get("x")
        and _is_identity_pixel_grid(clc)
    )
    use_ref_grid = source_on_ref_grid or inferred_ref_grid
    if use_ref_grid:
        if clc.sizes.get("y") != ref.sizes.get("y") or clc.sizes.get("x") != ref.sizes.get("x"):
            raise RuntimeError(
                "CLC source exported on reference grid has mismatched shape "
                f"({clc.sizes.get('y')}x{clc.sizes.get('x')} vs {ref.sizes.get('y')}x{ref.sizes.get('x')})."
            )
        clc = clc.assign_coords(y=ref.coords["y"].data, x=ref.coords["x"].data)
        clc = clc.rio.write_crs(source_crs)
    else:
        if clc.rio.crs is None:
            clc = clc.rio.write_crs(source_crs)
        # Integer source rasters need float destination to represent NaN nodata.
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
