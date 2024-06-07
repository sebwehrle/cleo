# spatial methods of the Atlas class
import logging
import numpy as np
import geopandas as gpd
import rasterio.crs
import xarray as xr

from typing import Union


# %% methods
def clip_to_geometry(self, clip_shape: Union[str, gpd.GeoDataFrame]) -> (xr.Dataset, gpd.GeoDataFrame):
    """
    Clip the atlas data to the provided geopandas shapefile
    :param self: an Atlas-subclass instance that is to be clipped.
    :param clip_shape: a geopandas GeoDataFrame or a path-string pointing to a shapefile. Defines the area to clip to.
    :type clip_shape: Union[str, gpd.GeoDataFrame]
    :return: a xr.Dataset clipped to clip_shape and the gpd.GeoDataFrame used to clip the data.
    """
    # Handle clip_shape argument type
    if isinstance(clip_shape, str):
        try:
            clip_shape = gpd.read_file(clip_shape)
        except Exception as e:
            logging.error(f"Invalid geometry in the clipping shapefile: {e}")
    elif not isinstance(clip_shape, gpd.GeoDataFrame):
        raise TypeError(f"clip shape must be a string path or a gpd.GeoDataFrame")

    # Ensure the clip_shape geometry is valid
    if not all(clip_shape.is_valid):
        logging.error("Invalid geometry in clipping shape")

    # Ensure data is loaded
    if self.data is None:
        raise ValueError(f"There is no data in {self}")

    # Reproject clip_shape if CRS mismatch
    if self.data.rio.crs != clip_shape.crs:
        logging.info(f"Reprojecting clip_shape to {self.parent.crs}")
        try:
            clip_shape = clip_shape.to_crs(self.parent.crs)
        except Exception as e:
            logging.error(f"Error reprojecting clip_shape: {e}")

    # Clip each DataArray in the DataSet using the clip_shape geometry
    data_clipped = xr.Dataset()
    data_clipped = data_clipped.rio.write_crs(self.data.rio.crs.to_string())
    for var_name, var in self.data.data_vars.items():
        try:
            data_clipped[var_name] = var.rio.clip(clip_shape.geometry)
        except Exception as e:
            logging.error(f"Error clipping data variable {var_name}: {e}")
            continue
    # clipped_data = self.data.rio.clip(clip_shape, all_touched=True)
    logging.info(f"Data clipped")
    return data_clipped, clip_shape


# %% functions
def bbox(self):
    """
    Get the bounding box of the object
    :param self: an object
    :return: (xmin, ymin, xmax, ymax)
    """
    if hasattr(self, 'coords') and hasattr(self.coords, '__getitem__'):
        return (
            self.coords["x"].min(),
            self.coords["y"].min(),
            self.coords["x"].max(),
            self.coords["y"].max(),
        )
    elif hasattr(self, 'data') and hasattr(self.data, 'coords') and hasattr(self.data.coords, '__getitem__'):
        return (
            self.data.coords["x"].min(),
            self.data.coords["y"].min(),
            self.data.coords["x"].max(),
            self.data.coords["y"].max(),
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
