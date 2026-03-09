# spatial methods of the Atlas class
import logging
import numpy as np
import geopandas as gpd
import pyproj
import xarray as xr
from scipy.ndimage import distance_transform_edt

from typing import Literal
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Policy Helpers (dask-friendly automated validation)
# =============================================================================

Validation = Literal["auto", "full", "probe", "none"]


def _is_dask_backed(obj) -> bool:
    """
    Check if obj is backed by a dask array.

    :param obj: xarray DataArray, Dataset, or array-like
    :return: True if dask-backed, False otherwise
    """
    if hasattr(obj, "data"):
        arr = obj.data
    else:
        arr = obj

    # Check for dask array via module
    if hasattr(arr, "compute") and hasattr(arr, "__module__"):
        module = getattr(arr, "__module__", "") or ""
        if module.startswith("dask"):
            return True

    # Alternative: check __dask_graph__
    if hasattr(arr, "__dask_graph__"):
        return True

    return False


def _get_probe_points(ds: xr.Dataset, n: int = 8) -> list[tuple[float, float]]:
    """
    Get probe points for dask-friendly validation.

    Returns cached probe points from ds.attrs["cleo_probe_points"] if present.
    Otherwise, derives n spread-out valid indices from the "template" variable.

    :param ds: Dataset (may contain "template" variable)
    :param n: Number of probe points to derive (default 8)
    :return: List of (x, y) coordinate tuples. Empty list if "template" absent.
    """
    # Return cached probe points if present
    if "cleo_probe_points" in ds.attrs:
        return ds.attrs["cleo_probe_points"]

    # Template absent -> return empty (structural checks still run; no probe value check)
    if "template" not in ds.data_vars:
        return []

    template = ds["template"]

    # Get validity mask (True where valid data exists)
    if np.issubdtype(template.dtype, np.floating):
        mask = template.notnull()
    else:
        # For non-float types, assume all non-zero values are valid
        mask = template.astype(bool)

    # If mask is dask-backed, compute ONCE here (validation setup only)
    if _is_dask_backed(mask):
        mask = mask.compute()

    # Find valid indices
    valid_mask_np = mask.values
    valid_indices = np.argwhere(valid_mask_np)

    if len(valid_indices) == 0:
        return []

    # Choose n spread-out valid indices using linspace
    n_valid = len(valid_indices)
    if n_valid <= n:
        selected_indices = valid_indices
    else:
        idx_positions = np.linspace(0, n_valid - 1, n, dtype=int)
        selected_indices = valid_indices[idx_positions]

    # Map indices to (x, y) coordinates
    x_coords = template.coords["x"].values
    y_coords = template.coords["y"].values

    # Handle dimension order (y, x) or (x, y)
    dims = template.dims
    probe_points = []
    for idx in selected_indices:
        if dims == ("y", "x"):
            y_idx, x_idx = idx
        else:  # ("x", "y")
            x_idx, y_idx = idx
        probe_points.append((float(x_coords[x_idx]), float(y_coords[y_idx])))

    # Cache in attrs (does NOT mutate data_vars; attrs are metadata)
    ds.attrs["cleo_probe_points"] = probe_points

    return probe_points


def _probe_scalars(da: xr.DataArray, pts: list[tuple[float, float]]) -> np.ndarray:
    """
    Extract scalar values at probe points (dask-friendly).

    Only computes a handful of scalars via .sel().compute().

    :param da: DataArray to probe (may be dask-backed)
    :param pts: List of (x, y) coordinate tuples
    :return: 1D numpy array of probed values
    """
    if not pts:
        return np.array([])

    values = []
    for x, y in pts:
        val = da.sel(x=x, y=y, method="nearest").compute()
        values.append(float(val.values))

    return np.array(values)


def _validate_values(
    da: xr.DataArray,
    ds: xr.Dataset,
    *,
    validation: Validation = "auto",
    check_nan: bool = True,
    check_positive: bool = False,
    check_range: tuple[float, float] | None = None,
    context: str = "",
) -> None:
    """
    Validate DataArray values using policy-driven approach.

    Validation modes:
    - "auto" (default): probe if dask-backed, full if eager
    - "full": compute full-array reductions (may trigger dask compute)
    - "probe": use probe points from template (dask-friendly)
    - "none": skip value checks (structural checks still apply elsewhere)

    :param da: DataArray to validate
    :param ds: Dataset (for probe points from "template")
    :param validation: Validation policy
    :param check_nan: Check for NaN values
    :param check_positive: Check that all values are > 0
    :param check_range: Check that median is in (min, max) range
    :param context: Context string for error messages
    :raises ValueError: If validation fails
    """
    validation = _resolved_validation_mode(da, validation)
    if validation == "none":
        return
    if validation == "full":
        _validate_values_full(
            da,
            check_nan=check_nan,
            check_positive=check_positive,
            check_range=check_range,
            context=context,
        )
        return
    _validate_values_probe(
        da,
        ds,
        check_nan=check_nan,
        check_positive=check_positive,
        check_range=check_range,
        context=context,
    )


def _resolved_validation_mode(da: xr.DataArray, validation: Validation) -> Validation:
    """Resolve ``auto`` validation to a concrete mode.

    :param da: DataArray being validated.
    :param validation: Requested validation mode.
    :returns: Concrete validation mode.
    :rtype: Validation
    :raises ValueError: If ``validation`` is not one of the supported modes.
    """
    allowed: set[str] = {"auto", "full", "probe", "none"}
    if validation not in allowed:
        raise ValueError(f"validation must be one of {sorted(allowed)!r}; got {validation!r}")
    if validation != "auto":
        return validation
    return "probe" if _is_dask_backed(da) else "full"


def _validate_values_full(
    da: xr.DataArray,
    *,
    check_nan: bool,
    check_positive: bool,
    check_range: tuple[float, float] | None,
    context: str,
) -> None:
    """Run full-array value validation.

    :param da: DataArray to validate.
    :param check_nan: Check for NaN/Inf values.
    :param check_positive: Check that finite values are strictly positive.
    :param check_range: Check that finite-value median lies in range.
    :param context: Context string for error messages.
    :raises ValueError: If validation fails.
    """
    if check_nan:
        is_invalid = da.isnull() | np.isinf(da)
        if bool(is_invalid.any()):
            nan_count = int(is_invalid.sum())
            raise ValueError(f"Invalid values {context}: {nan_count} NaN/Inf values found.")

    finite_da = da.where(np.isfinite(da))

    if check_positive:
        min_val = finite_da.min(skipna=True)
        if not np.isnan(float(min_val)) and float(min_val) <= 0:
            raise ValueError(f"Invalid values {context}: values must be > 0. Found min={float(min_val):.6g}")

    if check_range is None:
        return
    median_val = finite_da.median(skipna=True)
    if np.isnan(float(median_val)):
        return
    lo, hi = check_range
    if not (lo <= float(median_val) <= hi):
        raise ValueError(
            f"Invalid values {context}: median={float(median_val):.6g} outside expected range [{lo}, {hi}]"
        )


def _validate_values_probe(
    da: xr.DataArray,
    ds: xr.Dataset,
    *,
    check_nan: bool,
    check_positive: bool,
    check_range: tuple[float, float] | None,
    context: str,
) -> None:
    """Run probe-point value validation.

    :param da: DataArray to validate.
    :param ds: Dataset providing probe-point context.
    :param check_nan: Check for NaN/Inf values.
    :param check_positive: Check that finite values are strictly positive.
    :param check_range: Check that finite-value median lies in range.
    :param context: Context string for error messages.
    :raises ValueError: If validation fails.
    """
    pts = _get_probe_points(ds)
    if not pts:
        return

    vals = _probe_scalars(da, pts)

    if check_nan and np.any(~np.isfinite(vals)):
        nan_count = int(np.sum(~np.isfinite(vals)))
        raise ValueError(f"Invalid values {context}: {nan_count}/{len(vals)} probe points are NaN/Inf.")

    finite_vals = vals[np.isfinite(vals)]

    if check_positive and finite_vals.size > 0 and np.min(finite_vals) <= 0:
        raise ValueError(f"Invalid values {context}: probe values must be > 0. Found min={np.min(finite_vals):.6g}")

    if check_range is None or finite_vals.size == 0:
        return
    median = float(np.nanmedian(finite_vals))
    lo, hi = check_range
    if not (lo <= median <= hi):
        raise ValueError(f"Invalid values {context}: probe median={median:.6g} outside expected range [{lo}, {hi}]")


# =============================================================================
# CRS Utilities (single chokepoint for all CRS operations)
# =============================================================================


def canonical_crs_str(crs_input) -> str:
    """
    Convert any CRS representation to a canonical string form.

    If the CRS has an EPSG code, returns "epsg:{code}" (lowercase).
    Otherwise, returns the CRS.to_string() fallback.

    :param crs_input: CRS in any form (str, int, pyproj.CRS, rasterio.crs.CRS, etc.)
    :return: Canonical CRS string
    :raises ValueError: If the input cannot be parsed as a CRS
    """
    try:
        crs = pyproj.CRS.from_user_input(crs_input)
    except (pyproj.exceptions.CRSError, TypeError, ValueError) as e:
        raise ValueError(f"Cannot parse CRS from input: {crs_input!r}") from e

    epsg = crs.to_epsg()
    if epsg is not None:
        return f"epsg:{epsg}"
    return crs.to_string()


def crs_equal(a, b) -> bool:
    """
    Check if two CRS representations are semantically equal.

    Handles different representations (EPSG codes, WKT, rasterio/pyproj objects, etc.)
    by parsing both through pyproj.CRS and comparing.

    :param a: First CRS (any form)
    :param b: Second CRS (any form)
    :return: True if semantically equal, False otherwise
    """
    try:
        crs_a = pyproj.CRS.from_user_input(a)
        crs_b = pyproj.CRS.from_user_input(b)
        return crs_a == crs_b
    except (pyproj.exceptions.CRSError, TypeError, ValueError):
        return False


def to_crs_if_needed(gdf: gpd.GeoDataFrame, dst_crs) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to dst_crs only if needed.

    :param gdf: Input GeoDataFrame
    :param dst_crs: Destination CRS (any form)
    :return: GeoDataFrame in dst_crs (may be same object if no reprojection needed)
    :raises ValueError: If gdf has no CRS set
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set; cannot reproject.")

    if crs_equal(gdf.crs, dst_crs):
        return gdf

    dst_canonical = canonical_crs_str(dst_crs)
    logger.debug(f"Reprojecting GeoDataFrame from {gdf.crs} to {dst_canonical}")
    return gdf.to_crs(dst_canonical)


def reproject_raster_if_needed(da: xr.DataArray, dst_crs, *, nodata=np.nan, resampling=None) -> xr.DataArray:
    """
    Reproject a raster DataArray to dst_crs only if needed.

    :param da: Input DataArray with rioxarray extension
    :param dst_crs: Destination CRS (any form)
    :param nodata: Nodata value for reprojection (default: np.nan)
    :param resampling: Resampling method (if None, uses rioxarray default)
    :return: DataArray in dst_crs (may be same object if no reprojection needed)
    :raises ValueError: If da has no CRS set
    """
    if da.rio.crs is None:
        raise ValueError("Raster DataArray has no CRS set; cannot reproject.")

    if crs_equal(da.rio.crs, dst_crs):
        return da

    dst_canonical = canonical_crs_str(dst_crs)
    logger.debug(f"Reprojecting raster from {da.rio.crs} to {dst_canonical}")

    if resampling is not None:
        return da.rio.reproject(dst_canonical, nodata=nodata, resampling=resampling)
    return da.rio.reproject(dst_canonical, nodata=nodata)


def coords_equal(da: xr.DataArray, template: xr.DataArray, *, dim: str) -> bool:
    """
    Check if a DataArray's coordinate matches template's coordinate exactly.

    :param da: DataArray to check
    :param template: Reference DataArray
    :param dim: Coordinate dimension to compare ('x' or 'y')
    :return: True if coords match exactly
    """
    if dim not in da.coords or dim not in template.coords:
        return False
    return np.array_equal(da.coords[dim].values, template.coords[dim].values)


def enforce_exact_grid(da: xr.DataArray, template: xr.DataArray, *, var_name: str) -> xr.DataArray:
    """
    Enforce that a raster DataArray has exact x/y coords matching template.

    Contract:
    - If da has both x and y dims, they MUST match template exactly.
    - No auto-reproject or reindex; raises ValueError on mismatch.
    - Non-raster DataArrays (missing x or y dim) pass through unchanged.

    :param da: DataArray to validate
    :param template: Reference template DataArray with canonical x/y coords
    :param var_name: Variable name for error messages
    :return: da unchanged if valid
    :raises ValueError: If x or y coords don't match template
    """
    # Only enforce for raster-like DataArrays (have both x and y dims)
    if "x" not in da.dims or "y" not in da.dims:
        return da

    # Check x coords
    if not coords_equal(da, template, dim="x"):
        da_x = da.coords["x"].values
        tpl_x = template.coords["x"].values
        raise ValueError(
            f"Grid mismatch for '{var_name}': x coords differ from template. "
            f"Expected shape {tpl_x.shape}, got {da_x.shape}. "
            f"Expected range [{tpl_x.min():.6f}, {tpl_x.max():.6f}], "
            f"got [{da_x.min():.6f}, {da_x.max():.6f}]."
        )

    # Check y coords
    if not coords_equal(da, template, dim="y"):
        da_y = da.coords["y"].values
        tpl_y = template.coords["y"].values
        raise ValueError(
            f"Grid mismatch for '{var_name}': y coords differ from template. "
            f"Expected shape {tpl_y.shape}, got {da_y.shape}. "
            f"Expected range [{tpl_y.min():.6f}, {tpl_y.max():.6f}], "
            f"got [{da_y.min():.6f}, {da_y.max():.6f}]."
        )

    return da


def _axis_spacing_meters(coords: xr.DataArray, *, dim: str) -> float:
    """Return absolute regular spacing for one spatial axis in meters."""
    vals = np.asarray(coords.values, dtype=np.float64)
    if vals.size <= 1:
        return 1.0

    diffs = np.diff(vals)
    if not np.all(np.isfinite(diffs)):
        raise ValueError(f"{dim} coordinates contain non-finite spacing values.")
    if np.any(diffs == 0):
        raise ValueError(f"{dim} coordinates contain repeated values; regular grid required.")
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(f"{dim} coordinates are not monotonic; regular grid required.")

    ref = float(diffs[0])
    tol = max(abs(ref) * 1e-6, 1e-9)
    if not np.allclose(diffs, ref, rtol=0.0, atol=tol):
        raise ValueError(f"{dim} coordinates are not regularly spaced; distance requires regular grid.")
    return abs(ref)


def _distance_2d_positive_mask(
    source_2d: np.ndarray,
    valid_mask_2d: np.ndarray,
    *,
    y_spacing_m: float,
    x_spacing_m: float,
) -> np.ndarray:
    """Compute 2D Euclidean distance to nearest finite positive cell."""
    finite = np.isfinite(source_2d)
    valid = valid_mask_2d.astype(bool, copy=False)
    inside = valid & finite
    targets = inside & (source_2d > 0)

    out = np.full(source_2d.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return out
    if not np.any(targets):
        return out

    dist = distance_transform_edt(~targets, sampling=(float(y_spacing_m), float(x_spacing_m)))
    out[valid] = dist[valid]
    return out


def _validate_distance_inputs(source: xr.DataArray, valid_mask: xr.DataArray) -> str | None:
    """Validate distance inputs and return the optional extra dimension.

    :param source: Source raster with spatial dimensions.
    :param valid_mask: Boolean mask on the same spatial grid.
    :returns: Extra non-spatial dimension name, if present.
    :rtype: str | None
    :raises ValueError: If input dimensions or coordinates are invalid.
    """
    if "x" not in source.dims or "y" not in source.dims:
        raise ValueError("distance source must include spatial dims 'y' and 'x'.")
    extra_dims = [d for d in source.dims if d not in ("y", "x")]
    if len(extra_dims) > 1:
        raise ValueError("distance source supports at most one non-spatial dimension.")

    if "x" not in valid_mask.dims or "y" not in valid_mask.dims:
        raise ValueError("valid_mask must include spatial dims 'y' and 'x'.")
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must be two-dimensional with dims ('y', 'x').")

    if not np.array_equal(source.coords["x"].values, valid_mask.coords["x"].values):
        raise ValueError("distance source x coordinates must match valid_mask exactly.")
    if not np.array_equal(source.coords["y"].values, valid_mask.coords["y"].values):
        raise ValueError("distance source y coordinates must match valid_mask exactly.")
    return extra_dims[0] if extra_dims else None


def _resolve_distance_context(
    source: xr.DataArray,
    valid_mask: xr.DataArray,
) -> tuple[object | None, np.ndarray, float, float]:
    """Resolve CRS, mask, and spacing context for distance computation.

    :param source: Source raster with spatial dimensions.
    :param valid_mask: Boolean mask on the same spatial grid.
    :returns: Tuple of source CRS, boolean valid mask, y spacing, and x spacing.
    :rtype: tuple[object | None, numpy.ndarray, float, float]
    :raises ValueError: If CRS or grid spacing is invalid.
    """
    src_crs = getattr(source.rio, "crs", None)
    mask_crs = getattr(valid_mask.rio, "crs", None)
    crs_input = src_crs if src_crs is not None else mask_crs
    if crs_input is None:
        raise ValueError("distance source/valid_mask CRS is missing.")

    try:
        crs_obj = pyproj.CRS.from_user_input(crs_input)
    except (pyproj.exceptions.CRSError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot parse CRS for distance computation: {crs_input!r}") from exc

    if not crs_obj.is_projected:
        raise ValueError("distance requires projected CRS with metric units.")

    axis_info = list(crs_obj.axis_info)
    if not axis_info:
        raise ValueError("distance requires projected CRS with metric units.")
    unit_factor = axis_info[0].unit_conversion_factor
    if unit_factor is None or not np.isfinite(unit_factor) or not np.isclose(unit_factor, 1.0):
        raise ValueError("distance requires projected CRS with metric units (meters).")

    y_spacing = _axis_spacing_meters(source.coords["y"], dim="y")
    x_spacing = _axis_spacing_meters(source.coords["x"], dim="x")
    valid_np = np.asarray(valid_mask.values, dtype=bool)
    return src_crs, valid_np, y_spacing, x_spacing


def _distance_result_2d(
    source: xr.DataArray,
    *,
    valid_np: np.ndarray,
    y_spacing: float,
    x_spacing: float,
) -> xr.DataArray:
    """Build a 2D distance result aligned to the source grid.

    :param source: 2D source raster.
    :param valid_np: Boolean valid-mask array.
    :param y_spacing: Grid spacing in meters along y.
    :param x_spacing: Grid spacing in meters along x.
    :returns: 2D distance result.
    :rtype: xarray.DataArray
    """
    dist_np = _distance_2d_positive_mask(
        np.asarray(source.values),
        valid_np,
        y_spacing_m=y_spacing,
        x_spacing_m=x_spacing,
    )
    return xr.DataArray(
        dist_np,
        dims=("y", "x"),
        coords={"y": source.coords["y"].values, "x": source.coords["x"].values},
        name=source.name,
    )


def _distance_result_stacked(
    source: xr.DataArray,
    *,
    extra_dim: str,
    valid_np: np.ndarray,
    y_spacing: float,
    x_spacing: float,
) -> xr.DataArray:
    """Build a distance result over one extra non-spatial dimension.

    :param source: Source raster with one extra dimension.
    :param extra_dim: Non-spatial dimension name.
    :param valid_np: Boolean valid-mask array.
    :param y_spacing: Grid spacing in meters along y.
    :param x_spacing: Grid spacing in meters along x.
    :returns: Distance result with the same dimension order as ``source``.
    :rtype: xarray.DataArray
    """
    values = [
        xr.DataArray(
            _distance_2d_positive_mask(
                np.asarray(source.isel({extra_dim: i}).values),
                valid_np,
                y_spacing_m=y_spacing,
                x_spacing_m=x_spacing,
            ),
            dims=("y", "x"),
            coords={"y": source.coords["y"].values, "x": source.coords["x"].values},
        )
        for i in range(source.sizes[extra_dim])
    ]
    dim_key = source.coords[extra_dim] if extra_dim in source.coords else extra_dim
    out = xr.concat(values, dim=dim_key)
    out = out.transpose(*source.dims)
    out.name = source.name
    return out


def distance_to_positive_mask(
    source: xr.DataArray,
    valid_mask: xr.DataArray,
) -> xr.DataArray:
    """Distance (meters) to nearest finite positive cell in ``source``.

    Contract:
    - ``source`` must have spatial dims ``("y", "x")`` with optional one extra non-spatial dim.
    - ``valid_mask`` must be on the exact same y/x grid and marks cells included in output.
    - CRS must be projected metric (meters).
    - Outside ``valid_mask`` output is NaN.
    """
    extra_dim = _validate_distance_inputs(source, valid_mask)
    src_crs, valid_np, y_spacing, x_spacing = _resolve_distance_context(source, valid_mask)

    if extra_dim is None:
        out = _distance_result_2d(
            source,
            valid_np=valid_np,
            y_spacing=y_spacing,
            x_spacing=x_spacing,
        )
    else:
        out = _distance_result_stacked(
            source,
            extra_dim=extra_dim,
            valid_np=valid_np,
            y_spacing=y_spacing,
            x_spacing=x_spacing,
        )

    if src_crs is not None:
        out = out.rio.write_crs(src_crs)

    out.attrs["units"] = "m"
    out.attrs["cleo:algo"] = "edt"
    out.attrs["cleo:algo_version"] = "1"
    out.attrs["cleo:distance_rule"] = "isfinite_and_gt_zero"
    return out


def _rio_clip_robust(da, geoms, *, drop: bool, all_touched_primary: bool = False):
    """
    Robust wrapper around rioxarray clip with NoDataInBounds fallback.

    Strategy:
    - Primary: all_touched=all_touched_primary, drop=drop
    - On NoDataInBounds: retry with all_touched=True, drop=drop
    - If still NoDataInBounds: raise ClipNoDataInBounds

    :param da: xarray DataArray with rioxarray accessor
    :param geoms: iterable of geometries for clipping
    :param drop: whether to drop pixels outside clip bounds
    :param all_touched_primary: all_touched setting for primary attempt
    :return: clipped DataArray
    :raises ClipNoDataInBounds: if geometry does not overlap raster bounds
    """
    from cleo.errors import ClipNoDataInBounds

    try:
        from rioxarray.exceptions import NoDataInBounds
    except ImportError:  # pragma: no cover
        # rioxarray not available; just attempt clip directly
        return da.rio.clip(geoms, all_touched=all_touched_primary, drop=drop)

    try:
        return da.rio.clip(geoms, all_touched=all_touched_primary, drop=drop)
    except NoDataInBounds:
        # Robust fallback for tiny / borderline polygons on coarse grids
        logger.warning(
            f"_rio_clip_robust: NoDataInBounds with all_touched={all_touched_primary}; retrying with all_touched=True."
        )
        try:
            return da.rio.clip(geoms, all_touched=True, drop=drop)
        except NoDataInBounds as e:
            raise ClipNoDataInBounds("Geometry does not overlap raster bounds") from e


def _validate_clip_inputs(self, clip_shape: gpd.GeoDataFrame) -> tuple[xr.Dataset, object]:
    """Validate clip inputs and return the source dataset plus raster CRS.

    :param self: Domain-like object exposing ``data`` and ``parent.crs``.
    :param clip_shape: Geometry used for clipping.
    :returns: Tuple ``(source_dataset, raster_crs)``.
    :raises ValueError: If the source dataset, CRS, or clip geometry is invalid.
    :raises TypeError: If ``clip_shape`` is not a GeoDataFrame.
    """
    if self.data is None:
        raise ValueError(f"There is no data in {self}")

    raster_crs = self.data.rio.crs
    if raster_crs is None:
        raise ValueError("Atlas data has no CRS set (self.data.rio.crs is None).")
    if self.parent is None or getattr(self.parent, "crs", None) is None:
        raise ValueError("Atlas parent CRS is missing (self.parent.crs is None).")
    if not crs_equal(raster_crs, self.parent.crs):
        raise ValueError(
            f"CRS inconsistency: data.rio.crs={raster_crs} but parent.crs={self.parent.crs}. This must never happen."
        )
    if not isinstance(clip_shape, gpd.GeoDataFrame):
        raise TypeError("clip_shape must be a geopandas.GeoDataFrame. Path arguments should be handled by the caller.")
    if clip_shape.empty:
        raise ValueError("Clipping geometry is empty.")
    if clip_shape.crs is None:
        raise ValueError("Clipping geometry CRS is missing (clip_shape.crs is None).")

    return self.data, raster_crs


def _repair_clip_geometry_feature(geom):
    """Repair one geometry feature if possible.

    :param geom: Shapely geometry or ``None``.
    :returns: Repaired geometry, the original geometry, or ``None``.
    """
    if geom is None:
        return None
    try:
        from shapely.make_valid import make_valid

        return make_valid(geom)
    except ImportError:
        try:
            return geom.buffer(0)
        except (AttributeError, TypeError, ValueError):
            return geom


def _repair_clip_shape_if_needed(clip_shape: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Repair invalid clip geometries when needed.

    :param clip_shape: Candidate clipping geometry.
    :returns: Original or repaired clipping geometry.
    :raises ValueError: If repair leaves the geometry invalid or empty.
    """
    invalid_mask = ~clip_shape.is_valid
    if not invalid_mask.any():
        return clip_shape

    n_invalid = int(invalid_mask.sum())
    logger.warning(f"Clipping geometry has {n_invalid} invalid feature(s); attempting auto-repair.")

    repaired = clip_shape.copy()
    repaired["geometry"] = repaired.geometry.apply(_repair_clip_geometry_feature)
    if repaired.empty or repaired.geometry.is_empty.all():
        raise ValueError("Clipping geometry became empty after auto-repair.")
    if not repaired.is_valid.all():
        raise ValueError("Clipping geometry remains invalid after auto-repair; aborting.")
    return repaired


def _build_clipped_dataset_template(source_ds: xr.Dataset, raster_crs) -> xr.Dataset:
    """Build an empty clipped dataset template with metadata preserved.

    :param source_ds: Source dataset being clipped.
    :param raster_crs: CRS to write onto the clipped dataset.
    :returns: Empty dataset with preserved attrs and non-spatial coords.
    """
    data_clipped = xr.Dataset().rio.write_crs(raster_crs.to_string())
    for name, coord in source_ds.coords.items():
        if name in ("x", "y", "spatial_ref"):
            continue
        if ("x" in coord.dims) or ("y" in coord.dims):
            continue
        data_clipped = data_clipped.assign_coords({name: coord})
    data_clipped.attrs = dict(source_ds.attrs)
    return data_clipped


def _clip_dataset_variables(source_ds: xr.Dataset, geoms: list, target_ds: xr.Dataset) -> xr.Dataset:
    """Clip all data variables from a source dataset into the target dataset.

    :param source_ds: Source dataset being clipped.
    :param geoms: Iterable of clip geometries.
    :param target_ds: Target dataset receiving clipped variables.
    :returns: Target dataset with clipped variables populated.
    :raises ValueError: If clipping fails or the geometry does not overlap the raster.
    """
    from cleo.errors import ClipNoDataInBounds

    for var_name, var in source_ds.data_vars.items():
        try:
            target_ds[var_name] = _rio_clip_robust(var, geoms, drop=True, all_touched_primary=False)
        except ClipNoDataInBounds as e:
            raise ValueError(f"Clipping geometry does not overlap raster bounds for variable '{var_name}'.") from e
        except (ValueError, TypeError, RuntimeError) as e:
            raise ValueError(f"Error clipping data variable '{var_name}'") from e
    return target_ds


# %% methods
def clip_to_geometry(self, clip_shape: gpd.GeoDataFrame) -> tuple[xr.Dataset, gpd.GeoDataFrame]:
    """
    Clip the atlas data to the provided GeoDataFrame geometry.

    This is a primitives-only function: it accepts only GeoDataFrame objects,
    not file paths. Path handling is done by callers (classes.py) which delegate
    raw I/O to unify.py.

    Contract:
    - Invalid geometries are auto-repaired (A2). If repair fails, raise.
    - CRS discipline:
        - self.data.rio.crs is the source of truth for raster clipping.
        - Atlas invariant is enforced: self.parent.crs must equal self.data.rio.crs.
        - clip geometry is reprojected to self.data.rio.crs before clipping.

    :param clip_shape: GeoDataFrame containing clipping geometry.
    :returns: Tuple ``(clipped_dataset, reprojected_clip_shape)``.
    :raises ValueError: If data/CRS/geometry is invalid or empty.
    :raises TypeError: If ``clip_shape`` is not a GeoDataFrame.
    """
    source_ds, raster_crs = _validate_clip_inputs(self, clip_shape)
    clip_shape = _repair_clip_shape_if_needed(clip_shape)

    try:
        clip_shape = to_crs_if_needed(clip_shape, self.parent.crs)
    except (ValueError, TypeError, pyproj.exceptions.CRSError) as e:
        raise ValueError(f"Error reprojecting clipping geometry to {self.parent.crs}") from e

    data_clipped = _build_clipped_dataset_template(source_ds, raster_crs)
    geoms = list(clip_shape.geometry)
    data_clipped = _clip_dataset_variables(source_ds, geoms, data_clipped)
    logger.info("Data clipped")
    return data_clipped, clip_shape


# %% functions
def bbox(self):
    """
    Get the bounding box of the object
    :param self: an object
    :return: (xmin, ymin, xmax, ymax) as Python floats
    """
    if hasattr(self, "coords") and hasattr(self.coords, "__getitem__"):
        return (
            float(self.coords["x"].min().item()),
            float(self.coords["y"].min().item()),
            float(self.coords["x"].max().item()),
            float(self.coords["y"].max().item()),
        )
    elif hasattr(self, "data") and hasattr(self.data, "coords") and hasattr(self.data.coords, "__getitem__"):
        return (
            float(self.data.coords["x"].min().item()),
            float(self.data.coords["y"].min().item()),
            float(self.data.coords["x"].max().item()),
            float(self.data.coords["y"].max().item()),
        )
    else:
        raise ValueError("Unsupported object type for bbox")


def reproject(self, new_crs: str) -> None:
    """
    Reproject the atlas' data to the provided coordinate reference system
    :param self: an instance of the Atlas class
    :param new_crs:  The new coordinate reference system to use
    :type new_crs:  str
    :raises ValueError: If the destination CRS is invalid.
    :raises TypeError: If data cannot be reprojected due to type issues.
    :raises RuntimeError: If reprojection fails at runtime.
    :raises OSError: If underlying raster IO operations fail.
    """
    # Canonicalize destination CRS
    dst_crs = canonical_crs_str(new_crs)

    # Check if reprojection is needed (semantic comparison)
    if self.data.rio.crs is not None and crs_equal(self.data.rio.crs, dst_crs):
        logger.info(f"Data already projected in CRS '{dst_crs}'")
        return

    def reproject_var(var):
        return var.rio.reproject(dst_crs=dst_crs, nodata=np.nan)

    try:
        # reproject each data variable in the dataset
        data_reproj = self.data.map(reproject_var, keep_attrs=True)
        logger.info(f"Reprojection of data variables to crs {dst_crs} completed")
        self.data = data_reproj
        # Update the parent's CRS attribute as canonical string (not rasterio object)
        if hasattr(self, "parent") and self.parent is not None:
            self.parent.crs = dst_crs
        # Also update self.crs if it exists (for backwards compatibility)
        if hasattr(self, "crs"):
            self.crs = dst_crs
    except (ValueError, TypeError, RuntimeError, OSError):
        logger.error(
            "Error during atlas reprojection.",
            extra={"dst_crs": dst_crs},
            exc_info=True,
        )
        raise


# %%
def save_to_geotiff(da: xr.DataArray, crs: str, save_path: Path, raster_name: str, nodata_value: int = -9999):
    """
    Save a 2D (x,y) DataArray as GeoTIFF.

    Contract:
    - da must have exactly two dims which are {'x','y'} (order may be ('y','x') or ('x','y')).
    - If NaNs exist: fill with nodata_value and write nodata metadata.
    - If no NaNs: still write the raster (B1); do not raise.
    """
    # Strict 2D spatial check (fixes precedence bug)
    dims = tuple(da.dims)
    if not (len(dims) == 2 and set(dims) == {"x", "y"}):
        raise TypeError(f"GeoTIFF export expects exactly dims ('x','y') in any order; got dims={dims}")

    # Ensure output dir exists
    save_path.mkdir(parents=True, exist_ok=True)

    # Handle NaNs -> nodata only when NaNs exist (B1)
    # bool() on 0-dim result triggers compute if dask-backed
    has_nan = da.isnull().any()
    if bool(has_nan):
        da = da.fillna(nodata_value)
        da.rio.write_nodata(nodata_value, inplace=True)

    # Always write CRS
    da.rio.write_crs(crs, inplace=True)

    if not raster_name.lower().endswith((".tif", ".tiff")):
        raster_name += ".tif"

    da.rio.to_raster(save_path / raster_name)
