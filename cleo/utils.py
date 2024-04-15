# %% imports
import sys
import shutil
import urllib3
import certifi
import logging
import logging.config
import numpy as np
import pandas as pd
import xarray
import xarray as xr
from typing import Union
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path
from pint import UnitRegistry

from cleo.spatial import bbox


# %% functions
def stylish_tqdm(total, desc):
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


def download_file(url, save_to, proxy=None, proxy_user=None, proxy_pass=None, overwrite=False):
    """
    downloads a file from a specified url to disk
    :param overwrite:
    :param proxy_pass: proxy password
    :param proxy_user: proxy username
    :param proxy: proxy url:port
    :param url: url-string
    :param save_to: destination file name (string)
    :return:
    """
    dld = False
    if (not Path(save_to).is_file()) or (overwrite is True):
        if proxy is not None:
            default_headers = urllib3.make_headers(proxy_basic_auth=f'{proxy_user}:{proxy_pass}')
            http = urllib3.ProxyManager(proxy, proxy_headers=default_headers, ca_certs=certifi.where())
        else:
            http = urllib3.PoolManager(ca_certs=certifi.where())
        try:
            with http.request('GET', url.replace('"', '').replace(' ', ''),
                              preload_content=False) as r, open(save_to, 'wb') as out_file:
                shutil.copyfileobj(r, out_file)
        except:
            try:
                http = urllib3.PoolManager(cert_reqs='CERT_NONE')
                with http.request('GET', url.replace('"', '').replace(' ', ''),
                                  preload_content=False) as r, open(save_to, 'wb') as out_file:
                    shutil.copyfileobj(r, out_file)
            except:
                raise Exception
        dld = True
    return dld


def _process_chunk(self, processing_func, chunk_size, start_x, start_y, **kwargs):
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


def _setup_logging(self):
    """
    Setup logging in the logs directory
    :param self: an instance of the Atlas class
    """
    fname = self.path / "logs" / "logfile.log"
    colors = {
        'reset': '\33[m',
        'green': '\33[32m',
        'purple': '\33[35m',
        'orange': '\33[33m',
    }

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '[%(asctime)s] %(levelname)-4s - %(name)-4s - %(message)s'
            },
            'color': {
                'format': f"{colors['green']}[%(asctime)s]{colors['reset']} {colors['purple']}%(levelname)-5s{colors['reset']} - {colors['orange']}%(name)-5s{colors['reset']}: %(message)s"
            }
        },
        'handlers': {
            'stream': {
                'class': 'logging.StreamHandler',
                'formatter': 'color',
            }
        },
        'root': {
            'handlers': ['stream'],
            'level': logging.INFO,
        },
    }

    if fname:
        logging_config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'formatter': 'default',
            'level': logging.DEBUG,
            'filename': fname,
        }
        logging_config['root']['handlers'].append('file')

    logging.config.dictConfig(logging_config)


def flatten(self, digits=5, exclude_template=True):
    """
    Converts data in a xarray.Dataset to a pandas.DataFrame in a slower but more memory efficient way than
    xarray.Dataset.to_dataframe. Rounding of coordinates facilitates merging across data variables. The default
    'digits' value of 5 results in a precision loss of at most about 50 cm when CRS is standard epsg:4326.
    :param self: an instance of the WindResourceAtlas- or SiteData-class
    :param digits: number of digits to round x and y coordinates to
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


def add(self, other, name=None) -> None:
    """
    Merge other into self
    :param self: an instance of the WindScape- or GeoScape-class
    :param other: an instance of the xarray.DataArray- or xarray.Dataset-class
    :return:
    """
    # duck typing to check if other is an xarray.Dataset, xarray.DataArray, WindScape or GeoScape object
    if not hasattr(other, "dims"):
        raise TypeError(f"'{other}' must be an instance of the xr.Dataset- or xr.DataArray-class.")

    if self.data.rio.crs != other.rio.crs:
        xarray_data = other.rio.reproject(self.crs)

    # clip other if necessary
    if bbox(self) != bbox(other):
        other = other.rio.clip(self.clip_shape.geometry)

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
    logging.info(f"Merged '{name}' into '{self.country}'-data.")


def convert(self, data_variable, to_unit, inplace=False):
    """
    Convert a data variable from the current unit to the specified unit
    """
    ureg = UnitRegistry()

    if isinstance(data_variable, str):
        data_variable = [data_variable]
    elif not isinstance(data_variable, list):
        raise ValueError("data_variable must be a string or a list of strings.")

    converted_arrays = {}
    for var in data_variable:
        data_var = self.data[var]
        unit = data_var.attrs.get("unit")

        if unit is None:
            raise ValueError("DataArray has no unit. Cannot perform unit conversion.")

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
