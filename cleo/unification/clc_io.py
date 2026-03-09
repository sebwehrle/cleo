"""CLC-specific I/O helpers for source metadata and wind-grid preparation."""

from __future__ import annotations

import warnings
from importlib import resources
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


def raster_band_count(path: Path) -> int | None:
    """Read raster band count for lightweight source/prepared validation.

    :param path: Candidate raster path.
    :type path: pathlib.Path
    :returns: Band count for readable rasters, else ``None``.
    :rtype: int | None
    """
    try:
        import rasterio

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            with rasterio.open(path) as dataset:
                return int(dataset.count)
    except (ModuleNotFoundError, OSError, ValueError, RuntimeError, TypeError):
        return None


def _compact_clc_value_map(atlas_path: Path) -> dict[int, int]:
    """Return mapping from compact CLMS raster ids to canonical CLC codes.

    Some CLMS raster packages encode CLC categories as compact sequential ids
    ``1..44`` instead of canonical codes such as ``311``. The packaged
    ``clc_codes.yml`` resource defines the canonical ordering, so compact id
    ``1`` maps to the first listed code, compact id ``2`` to the second, and so on.

    :param atlas_path: Atlas root path used to load packaged CLC resources.
    :type atlas_path: pathlib.Path
    :returns: Compact-id to canonical-code mapping.
    :rtype: dict[int, int]
    """
    try:
        clc_codes = load_clc_codes(atlas_path)
    except FileNotFoundError:
        resource = resources.files("cleo.resources").joinpath("clc_codes.yml")
        with resource.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("clc_codes", {})
        clc_codes = {int(k): str(v).strip() for k, v in raw.items()}
    return {idx: code for idx, code in enumerate(clc_codes.keys(), start=1)}


def _expand_compact_clc_classes(clc: xr.DataArray, *, atlas_path: Path) -> xr.DataArray:
    """Convert compact CLC raster ids to canonical CLC codes when detected.

    :param clc: Prepared CLC raster candidate.
    :type clc: xarray.DataArray
    :param atlas_path: Atlas root path for loading canonical CLC code ordering.
    :type atlas_path: pathlib.Path
    :returns: CLC raster with canonical class codes.
    :rtype: xarray.DataArray
    """
    values = np.asarray(clc.values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return clc

    rounded = np.rint(finite)
    if not np.allclose(finite, rounded):
        return clc

    compact_ids = rounded.astype(np.int64, copy=False)
    if np.any(compact_ids < 1) or np.any(compact_ids > 44):
        return clc
    compact_map = _compact_clc_value_map(atlas_path)
    if np.any(compact_ids > len(compact_map)):
        return clc

    lookup = np.zeros(len(compact_map) + 1, dtype=np.float32)
    for compact_id, canonical_code in compact_map.items():
        lookup[compact_id] = float(canonical_code)

    out = values.astype(np.float32, copy=True)
    mask = np.isfinite(out)
    out[mask] = lookup[np.rint(out[mask]).astype(np.int64)]
    return xr.DataArray(out, coords=clc.coords, dims=clc.dims, attrs=clc.attrs, name=clc.name)


def prepare_clc_to_wind_grid(
    *,
    source_path: Path,
    prepared_path: Path,
    ref: xr.DataArray,
    valid_mask: xr.DataArray | None = None,
    atlas_path: Path,
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
    :param valid_mask: Optional canonical landscape validity mask on the same
        grid as ``ref``. If omitted, validity falls back to ``ref.notnull()``.
    :type valid_mask: xarray.DataArray | None
    :param atlas_path: Atlas root path for resource-backed CLC code normalization.
    :type atlas_path: pathlib.Path
    :param source_crs: Fallback source CRS used when source raster omits CRS metadata.
    :type source_crs: str
    :param source_on_ref_grid: If ``True``, interpret source as already sampled
        on the exact reference grid.
    :type source_on_ref_grid: bool
    :returns: Prepared raster path.
    :rtype: pathlib.Path
    :raises RuntimeError: If no valid wind cells exist for clipping.
    """
    valid = ref.notnull() if valid_mask is None else valid_mask.astype(bool)
    if "x" not in valid.dims or "y" not in valid.dims:
        raise ValueError("CLC valid_mask must include spatial dims 'y' and 'x'.")
    if not np.array_equal(np.asarray(valid.coords["y"].data), np.asarray(ref.coords["y"].data)):
        raise ValueError("CLC valid_mask y coordinates must match the wind reference grid.")
    if not np.array_equal(np.asarray(valid.coords["x"].data), np.asarray(ref.coords["x"].data)):
        raise ValueError("CLC valid_mask x coordinates must match the wind reference grid.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        clc_raw = rxr.open_rasterio(source_path, parse_coordinates=True)
        clc: xr.DataArray = clc_raw
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
    clc = _expand_compact_clc_classes(clc, atlas_path=atlas_path)
    clc = clc.astype(np.float32)
    clc = clc.rio.write_nodata(np.nan)
    clc.rio.to_raster(prepared_path)
    return prepared_path
