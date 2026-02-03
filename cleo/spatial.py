# spatial methods of the Atlas class
import logging
import numpy as np
import geopandas as gpd
import rasterio.crs
import xarray as xr

from typing import Union
from pathlib import Path


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
    if self.data.rio.crs.to_string() != str(self.parent.crs):
        raise ValueError(
            f"CRS inconsistency: data.rio.crs={self.data.rio.crs.to_string()} "
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
        logging.warning(f"Clipping geometry has {n_invalid} invalid feature(s); attempting auto-repair.")

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

    # Reproject clip_shape to raster CRS if needed
    raster_crs = self.data.rio.crs
    if clip_shape.crs != raster_crs:
        logging.info(f"Reprojecting clip geometry from {clip_shape.crs} to {raster_crs.to_string()}")
        try:
            clip_shape = clip_shape.to_crs(raster_crs)
        except Exception as e:
            raise ValueError(f"Error reprojecting clipping geometry to {raster_crs.to_string()}") from e

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
            logging.warning(
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

    logging.info("Data clipped")
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

    def reproject_var(var):  # xr.DataArray):
        return var.rio.reproject(
            dst_crs=new_crs,
            nodata=np.nan
        )
    if self.crs == rasterio.crs.CRS.from_string(new_crs):
        logging.info(f"Data already in projected in CRS '{new_crs}'")
    else:
        try:
            # reproject each data variable in the dataset
            data_reproj = self.data.map(reproject_var, keep_attrs=True)
            logging.info(f"Reprojection of data variables to crs {new_crs} completed")
            self.data = data_reproj
            self.crs = rasterio.crs.CRS.from_string(new_crs)
        except Exception as e:
            logging.error(f"Error during reprojecting atlas: {e}")


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
