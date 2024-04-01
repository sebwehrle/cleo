# spatial methods of the Atlas class
import logging
import numpy as np
import geopandas as gpd
import rasterio.crs
import xarray as xr

from typing import Union


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


def clip_to_geometry(self, clip_shape: Union[str, gpd.GeoDataFrame]) -> (xr.Dataset, gpd.GeoDataFrame):
    """
    Clip the atlas data to the provided geopandas shapefile
    :param clip_shape_path: Path to the geopandas shapefile representing the clipping area
    :type clip_shape_path: str
    :param inplace: If True, the atlas data will be overwritten with the clipped data. Default is True.
    :type inplace: bool
    :return:
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

    # Ensure data is loaded and get atlas CRS
    if self.data is None:
        self._build_netcdf()

    # Reproject clip_shape if CRS mismatch
    if self.crs != clip_shape.crs:
        logging.info(f"Reprojecting clip_shape to {self.crs}")
        try:
            clip_shape = clip_shape.to_crs(self.crs)
        except Exception as e:
            logging.error(f"Error reprojecting clip_shape: {e}")

    # Clip each DataArray in the DataSet using the clip_shape geometry
    data_clipped = xr.Dataset()
    for var_name, var in self.data.data_vars.items():
        try:
            data_clipped[var_name] = var.rio.clip(clip_shape.geometry)
        except Exception as e:
            logging.error(f"Error clipping data variable {var_name}: {e}")
            continue
    # clipped_data = self.data.rio.clip(clip_shape, all_touched=True)
    logging.info(f"Data clipped")
    return data_clipped, clip_shape


def bbox(self):
    """
    Get the bounding box of the Atlas object
    :param self: an Atlas object
    :return: (xmin, ymin, xmax, ymax)
    """
    return (self.data.coords["x"].min(),
            self.data.coords["y"].min(),
            self.data.coords["x"].max(),
            self.data.coords["y"].max(),
            )
