# spatial methods of the Atlas class
import logging
import numpy as np
import geopandas as gpd
import pyproj
import rasterio.crs
import xarray as xr

from typing import Union
from pathlib import Path

logger = logging.getLogger(__name__)


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
    except Exception as e:
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
    except Exception:
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


# %% methods
def clip_to_geometry(self, clip_shape: Union[str, gpd.GeoDataFrame]) -> (xr.Dataset, gpd.GeoDataFrame):
    """
    Clip the atlas data to the provided geopandas shapefile / geometry.

    Contract:
    - If clip_shape is a path: it must be readable; otherwise raise immediately.
    - Invalid geometries are auto-repaired (A2). If repair fails, raise.
    - CRS discipline:
        - self.data.rio.crs is the source of truth for raster clipping.
        - Atlas invariant is enforced: self.parent.crs must equal self.data.rio.crs.
        - clip geometry is reprojected to self.data.rio.crs before clipping.
    """
    # Ensure data is loaded
    if self.data is None:
        raise ValueError(f"There is no data in {self}")

    # Enforce Atlas CRS consistency invariant (per Sebastian)
    if self.data.rio.crs is None:
        raise ValueError("Atlas data has no CRS set (self.data.rio.crs is None).")
    if self.parent is None or getattr(self.parent, "crs", None) is None:
        raise ValueError("Atlas parent CRS is missing (self.parent.crs is None).")
    # Semantic CRS comparison using centralized helper
    if not crs_equal(self.data.rio.crs, self.parent.crs):
        raise ValueError(
            f"CRS inconsistency: data.rio.crs={self.data.rio.crs} "
            f"but parent.crs={self.parent.crs}. This must never happen."
        )

    # Handle clip_shape argument type
    if isinstance(clip_shape, str):
        try:
            clip_shape = gpd.read_file(clip_shape)
        except Exception as e:
            raise ValueError(f"Cannot read clipping geometry from path: {clip_shape}") from e
    elif not isinstance(clip_shape, gpd.GeoDataFrame):
        raise TypeError("clip_shape must be a string path or a geopandas.GeoDataFrame")

    if clip_shape.empty:
        raise ValueError("Clipping geometry is empty.")

    if clip_shape.crs is None:
        raise ValueError("Clipping geometry CRS is missing (clip_shape.crs is None).")

    # A2: auto-repair invalid geometries (explicit and safe)
    invalid_mask = ~clip_shape.is_valid
    if invalid_mask.any():
        n_invalid = int(invalid_mask.sum())
        logger.warning(f"Clipping geometry has {n_invalid} invalid feature(s); attempting auto-repair.")

        # Prefer make_valid if available (shapely>=2); fallback to buffer(0)
        def _repair_geom(geom):
            if geom is None:
                return None
            try:
                from shapely.make_valid import make_valid  # shapely>=2
                return make_valid(geom)
            except Exception:
                # buffer(0) is a common repair for self-intersections
                try:
                    return geom.buffer(0)
                except Exception:
                    return geom

        clip_shape = clip_shape.copy()
        clip_shape["geometry"] = clip_shape.geometry.apply(_repair_geom)

        # Re-check validity and non-emptiness after repair
        if clip_shape.empty or clip_shape.geometry.is_empty.all():
            raise ValueError("Clipping geometry became empty after auto-repair.")
        if not clip_shape.is_valid.all():
            raise ValueError("Clipping geometry remains invalid after auto-repair; aborting.")

    # Reproject clip_shape to raster CRS if needed (using centralized helper)
    raster_crs = self.data.rio.crs
    try:
        clip_shape = to_crs_if_needed(clip_shape, self.parent.crs)
    except Exception as e:
        raise ValueError(f"Error reprojecting clipping geometry to {self.parent.crs}") from e

    # Clip each DataArray in the Dataset using the clip_shape geometry
    data_clipped = xr.Dataset().rio.write_crs(raster_crs.to_string())

    # rioxarray expects an iterable of geometries
    geoms = list(clip_shape.geometry)

    # Import lazily so cleo can still import in minimal envs/tests if needed
    try:
        from rioxarray.exceptions import NoDataInBounds
    except Exception:  # pragma: no cover
        NoDataInBounds = ()  # fallback type, will never match

    for var_name, var in self.data.data_vars.items():
        try:
            # Primary path: true crop to bounds
            data_clipped[var_name] = var.rio.clip(
                geoms,
                all_touched=False,
                drop=True,
            )
        except NoDataInBounds:
            # Robust fallback for tiny / borderline polygons on coarse grids:
            # keep full grid, but mask outside geometry. This prevents hard failures
            # due to window-crop edge cases.
            logger.warning(
                f"clip_to_geometry: NoDataInBounds for '{var_name}' with drop=True; "
                f"retrying with all_touched=True, drop=False."
            )
            try:
                data_clipped[var_name] = var.rio.clip(
                    geoms,
                    all_touched=True,
                    drop=False,
                )
            except NoDataInBounds as e:
                raise ValueError(
                    f"Clipping geometry does not overlap raster bounds for variable '{var_name}'."
                ) from e
        except Exception as e:
            raise ValueError(f"Error clipping data variable '{var_name}'") from e

    logger.info("Data clipped")
    return data_clipped, clip_shape


# %% functions
def bbox(self):
    """
    Get the bounding box of the object
    :param self: an object
    :return: (xmin, ymin, xmax, ymax) as Python floats
    """
    if hasattr(self, 'coords') and hasattr(self.coords, '__getitem__'):
        return (
            float(self.coords["x"].min().item()),
            float(self.coords["y"].min().item()),
            float(self.coords["x"].max().item()),
            float(self.coords["y"].max().item()),
        )
    elif hasattr(self, 'data') and hasattr(self.data, 'coords') and hasattr(self.data.coords, '__getitem__'):
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
        if hasattr(self, 'parent') and self.parent is not None:
            self.parent.crs = dst_crs
        # Also update self.crs if it exists (for backwards compatibility)
        if hasattr(self, 'crs'):
            self.crs = dst_crs
    except Exception as e:
        logger.error(f"Error during reprojecting atlas: {e}")


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
    if bool(da.isnull().any()):
        da = da.fillna(nodata_value)
        da.rio.write_nodata(nodata_value, inplace=True)

    # Always write CRS
    da.rio.write_crs(crs, inplace=True)

    if not raster_name.lower().endswith((".tif", ".tiff")):
        raster_name += ".tif"

    da.rio.to_raster(save_path / raster_name)
