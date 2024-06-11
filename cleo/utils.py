# %% imports
import sys
import requests
from requests.auth import HTTPProxyAuth
import logging
import logging.config
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path
from pint import UnitRegistry

from cleo.spatial import bbox


# %% methods
def add(self, other, name=None) -> None:
    """
    Merge other into self

    :param self: an instance of the WindAtlas- or Landscape-class (wrapping a xarray Dataset)
    :param other: an instance of the xarray.DataArray- or xarray.Dataset-class
    :param name: a name for the merged data variable
    :return:
    """
    # duck typing to check if other is a xarray.Dataset, xarray.DataArray
    if not hasattr(other, "dims"):
        raise TypeError(f"'{other}' must be an instance of the xr.Dataset- or xr.DataArray-class.")

    if self.data.rio.crs != other.rio.crs:
        other = other.rio.reproject(self.crs, nodata=np.nan)

    # clip other if necessary
    if bbox(self) != bbox(other):
        if self.parent.region is not None:
            other = other.rio.clip(self.parent.get_nuts_region(self.parent.region).geometry)
        else:
            other = other.rio.clip(self.parent.get_nuts_country().geometry)

    # check if spatial coordinates of self and other align
    if not (np.array_equal(self.data["x"].data, other["x"].data)
            and np.array_equal(self.data["y"].data, other["y"].data)):
        template = self.data["template"] if "template" in self.data.data_vars else self.data
        other = other.interp_like(template)

        logging.warning(f"Spatial coordinates do not align. 'other' interpolated like 'self'.")

    if name is not None:
        other.name = name

    # merge data
    self.data = xr.merge([self.data, other])
    logging.info(f"Merged '{name}' into '{self.data.attrs['country']}'-data.")


def convert(self, data_variable, to_unit, from_unit=None, inplace=False):
    """
    Convert a data variable from the current unit to the specified unit

    :param data_variable: name(s) of the data variable(s) to be converted
    :param to_unit: name of the unit to convert to
    :param from_unit: name of the unit from which to convert from
    :param inplace: if True (default) the data variable is updated in-place
    """
    ureg = UnitRegistry()

    if isinstance(data_variable, str):
        data_variable = [data_variable]
    elif not isinstance(data_variable, list):
        raise ValueError("data_variable must be a string or a list of strings.")

    converted_arrays = {}

    for var in data_variable:
        data_var = self.data[var]
        if from_unit is not None:
            unit = from_unit
        else:
            unit = data_var.attrs.get("unit")

        if unit is None:
            raise ValueError("No from-unit given. Cannot perform unit conversion.")

        converted_arrays[var] = xr.DataArray(
            (data_var.data * ureg(unit)).to(to_unit).magnitude,
            coords=data_var.coords,
            dims=data_var.dims,
            attrs={"unit": to_unit}
        )

    if inplace:
        self.data.update({var: converted_array for var, converted_array in converted_arrays.items()})
    else:
        return converted_arrays


def flatten(self, digits=5, exclude_template=True):
    """
    Converts data in a xarray.Dataset to a pandas.DataFrame in a slower but more memory efficient way than
    xarray.Dataset.to_dataframe. Rounding of coordinates facilitates merging across data variables. The default
    'digits' value of 5 results in a precision loss of at most about 50 cm when CRS is standard epsg:4326.

    :param self: an instance of the WindResourceAtlas- or SiteData-class
    :param digits: number of digits to round x and y coordinates to
    :param exclude_template: a boolean flag to exclude the template-data variable
    :return: a pandas.Dataframe with one column per data variable and non-spatial coordinate
    """

    collect_df = []

    data_variables = self.data.data_vars
    if exclude_template:
        data_variables = [data_var for data_var in data_variables if data_var != "template"]

    for var_name in data_variables:
        data_var = self.data[var_name]
        # drop non-dimensional coordinates
        non_dim_coords = set(data_var.coords) - set(data_var.dims)
        data_var = data_var.drop_vars(non_dim_coords)

        if {"x", "y"} == set(data_var.dims):
            df = data_var.to_dataframe().dropna()
            df.index = pd.MultiIndex.from_arrays([
                np.round(df.index.get_level_values("y"), digits),
                np.round(df.index.get_level_values("x"), digits),
            ])
            collect_df.append(df)

        elif len(data_var.dims) == 3 and {"x", "y"}.issubset(set(data_var.dims)):
            non_spatial_dim = next(iter(set(data_var.dims) - {"x", "y"}))

            for coord in data_var.coords[non_spatial_dim]:
                data_slice = data_var.sel({non_spatial_dim: coord})
                data_slice = data_slice.drop_vars(non_spatial_dim)
                data_slice.name = f"{var_name}_{non_spatial_dim}_{coord.data}"
                df = data_slice.to_dataframe().dropna()
                df.index = pd.MultiIndex.from_arrays([
                    np.round(df.index.get_level_values("y"), digits),
                    np.round(df.index.get_level_values("x"), digits),
                ])
                collect_df.append(df)

        else:
            raise ValueError(f"Error in {var_name}. Only 3-dimensional data with 'x' and 'y'-coordinates are supported")

    return pd.concat(collect_df, axis=1)


# %% functions
def stylish_tqdm(total, desc):
    """
    Create a stylish progress bar using tqdm.

    :param total: Total iterations for the progress bar.
    :type total: int
    :param desc: Description to be displayed alongside the progress bar.
    :type desc: str
    :return: A tqdm progress bar.
    :rtype: tqdm
    """
    return tqdm(
        total=total,
        desc=desc,
        bar_format='{l_bar}{bar}{r_bar}',
        ncols=80,
        file=sys.stdout,
        miniters=1,
        ascii=True,
        dynamic_ncols=True,
        leave=True,
        position=0
    )


def _process_chunk(self, processing_func, chunk_size, start_x, start_y, **kwargs):
    """
    Process a chunk of data.

    :param self: An instance of the Atlas-class.
    :param processing_func: A function that takes data chunks and properties as input and returns processed data.
    :type processing_func: Callable
    :param chunk_size: Size of the chunks along x and y coordinates for processing.
    :type chunk_size: int
    :param start_x: Starting index for the x-coordinate.
    :type start_x: int
    :param start_y: Starting index for the y-coordinate.
    :type start_y: int
    :param kwargs: Additional inputs to the processing_func.
    :type kwargs: dict
    :return: Processed data.
    :rtype: xarray.Dataset
    """
    end_x = min(start_x + chunk_size, len(self.data.coords["x"]))
    end_y = min(start_y + chunk_size, len(self.data.coords["y"]))

    data_chunks = {}
    for var_name, var in kwargs.items():
        if isinstance(var, (xr.DataArray, xr.Dataset)):
            data_chunks[var_name] = var.isel(x=slice(start_x, end_x), y=slice(start_y, end_y))

    all_args = {**data_chunks, **{k: v for k, v in kwargs.items() if k not in data_chunks}}
    processed_data = processing_func(**all_args)
    return processed_data


def compute_chunked(self, processing_func, chunk_size, **kwargs):
    """
    Process data in chunks and merge the results.

    :param self: an instance of the Atlas-class
    :param processing_func: processing_func (callable): A function that takes data chunks and properties as input and
    returns processed data.
    :type processing_func: Callable
    :param chunk_size: Size of the chunks along x and y coordinates for processing.
    :type chunk_size: int
    :param kwargs: inputs to the processing_func
    :type kwargs: dict
    :return Reassembled dataset containing the processed data.
    :rtype xarray.Dataset
    """
    reassembled_data = xr.Dataset()

    x_chunks = range(0, len(self.data.coords["x"]), chunk_size)
    y_chunks = range(0, len(self.data.coords["y"]), chunk_size)
    total_chunks = len(x_chunks) * len(y_chunks)

    with stylish_tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    _process_chunk,
                    self,
                    processing_func,
                    chunk_size,
                    start_x,
                    start_y,
                    **kwargs
                )
                for start_x in x_chunks
                for start_y in y_chunks
            ]

            for future in futures:
                processed_chunk = future.result()
                reassembled_data = xr.merge([reassembled_data, processed_chunk])
                pbar.update(1)

    # convert xr.Dataset to xr.DataArray by selecting first data variable in xr.Dataset
    reassembled_data_vars = list(reassembled_data.data_vars)
    return reassembled_data[reassembled_data_vars[0]]


def download_file(url, save_to=None, proxy=None, proxy_user=None, proxy_pass=None, overwrite=False):
    """
    Download a file from a given URL.

    Parameters:
    url (str): The URL of the file to download.
    filename (str, optional): The name to save the file as. If not provided, the file will be saved with its original
    name.
    proxy (str, optional): The URL and port of the proxy to use for the download.
    proxy_user (str, optional): The username for the proxy.
    proxy_pass (str, optional): The password for the proxy.
    overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.

    Returns:
    Bool -- True if the file was successfully downloaded
    """
    # If filename wasn't provided
    if not save_to:
        # Get the file name from the URL
        save_to = url.split("/")[-1]
    # Check if the file already exists and if we should overwrite it
    if Path(save_to).is_file() and not overwrite:
        logging.info(f"File {save_to} already exists and overwrite is set to False.")
        return
    # Set up the proxies for the request
    proxies = {"http": proxy, "https": proxy} if proxy else None
    auth = HTTPProxyAuth(proxy_user, proxy_pass) if proxy_user and proxy_pass else None
    # Set a custom User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
    }
    # Send an HTTP request to the URL of the file
    response = requests.get(url, stream=True, proxies=proxies, auth=auth, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Write the contents of the response to a file
        with open(save_to, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        return True
    else:
        logging.info(f"Failed to download file. HTTP response code: {response.status_code}")
        return False
