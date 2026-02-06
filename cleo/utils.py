# %% imports
import requests
from requests.auth import HTTPProxyAuth
import logging
import logging.config
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from pint import UnitRegistry
_UREG = UnitRegistry()

from cleo.spatial import bbox

logger = logging.getLogger(__name__)


# %% private helpers
def _match_to_template(other: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    """
    Align a raster to a template grid via rio.reproject_match if needed.

    :param other: DataArray to align
    :param template: Reference DataArray with target x/y coordinates
    :return: Aligned DataArray with same x/y as template
    :raises ValueError: If other or template lacks x/y coords
    :raises ImportError: If rioxarray is not available and alignment is needed
    """
    # Validate inputs have x/y coords
    if "x" not in other.coords or "y" not in other.coords:
        raise ValueError(
            f"expected raster DataArray with x/y coords; got coords={list(other.coords)} dims={list(other.dims)}"
        )
    if "x" not in template.coords or "y" not in template.coords:
        raise ValueError(
            f"template must have x/y coords; got coords={list(template.coords)} dims={list(template.dims)}"
        )

    # Check if already aligned
    if (np.array_equal(other.coords["x"].values, template.coords["x"].values) and
            np.array_equal(other.coords["y"].values, template.coords["y"].values)):
        return other

    # Need rioxarray for reprojection
    try:
        import rioxarray  # noqa: F401
    except ImportError:
        raise ImportError(
            "Install rioxarray to align rasters to template grid: pip install rioxarray"
        )

    aligned = other.rio.reproject_match(template, nodata=np.nan)
    return aligned


# %% methods
def add(self, other, name=None) -> None:
    """
    Merge other into self (exact-grid contract, no silent misalignment).

    Rules:
    - If CRS differs: reproject.
    - If grid differs (even tiny float noise): align to self grid (no clipping).
    - Only clip if extents differ materially (bbox tolerance).
    - Final merge must be join="exact".
    """
    if not hasattr(other, "dims"):
        raise TypeError(f"'{other}' must be an instance of the xr.Dataset- or xr.DataArray-class.")

    # Normalize DataArray
    if isinstance(other, xr.Dataset):
        raise TypeError("add() currently expects a DataArray; got Dataset.")
    if not isinstance(other, xr.DataArray):
        raise TypeError(f"add() expects xarray.DataArray; got {type(other)}")

    if self.data is None:
        raise ValueError("self.data is None")

    # CRS discipline
    if self.data.rio.crs != other.rio.crs:
        other = other.rio.reproject(self.crs, nodata=np.nan)

    # Grid alignment (preferred over bbox checks; fixes tiny coord noise robustly)
    # Align to template if present, else align to first data var as grid template.
    if "template" in self.data.data_vars:
        template = self.data["template"]
    else:
        # Pick a stable reference grid from existing dataset variables
        first_var = next(iter(self.data.data_vars))
        template = self.data[first_var]

    other = _match_to_template(other, template)

    # Clip only if extents differ materially (not for float noise)
    b_self = bbox(self)
    b_other = bbox(other)
    tol = 1e-9  # in coordinate units; tiny enough for degree grids; prevents 1e-12 triggering
    if not all(np.isclose(bs, bo, rtol=0.0, atol=tol) for bs, bo in zip(b_self, b_other)):
        if self.parent.region is not None:
            other = other.rio.clip(self.parent.get_nuts_region(self.parent.region).geometry)
        else:
            other = other.rio.clip(self.parent.get_nuts_country().geometry)
        # After clipping, re-align again to template grid
        other = _match_to_template(other, template)

    if name is not None:
        other = other.copy()
        other.name = name

    self.data = xr.merge([self.data, other], join="exact", compat="no_conflicts")
    logger.info(f"Merged '{other.name}' into '{self.data.attrs.get('country', 'atlas')}'-data.")


def convert(self, data_variable, to_unit, from_unit=None, inplace=False):
    """
    Convert one or more data variables to a new unit.

    Contract:
    - Preserves existing attrs; updates only 'unit'.
    - Dask-friendly: computes a scalar factor once and multiplies the DataArray by that factor.
    """
    if isinstance(data_variable, str):
        data_variable = [data_variable]
    elif not isinstance(data_variable, list):
        raise ValueError("data_variable must be a string or a list of strings.")

    converted_arrays = {}

    for var in data_variable:
        data_var = self.data[var]

        unit = from_unit if from_unit is not None else data_var.attrs.get("unit")
        if unit is None:
            raise ValueError(f"No from-unit given and no 'unit' attr present for variable '{var}'.")

        # scalar conversion factor (1 * unit -> to_unit)
        factor = (1.0 * _UREG(unit)).to(to_unit).magnitude

        attrs = dict(data_var.attrs) if data_var.attrs is not None else {}
        attrs["unit"] = to_unit

        converted_arrays[var] = xr.DataArray(
            data_var * factor,
            coords=data_var.coords,
            dims=data_var.dims,
            attrs=attrs,
            name=data_var.name,
        )

    if inplace:
        self.data.update(converted_arrays)
    else:
        return converted_arrays


def flatten(self, digits=5, exclude_template=True):
    """
    Converts data in a xarray.Dataset to a pandas.DataFrame in a slower but more memory efficient way than
    xarray.Dataset.to_dataframe. Rounding of coordinates facilitates merging across data variables. The default
    'digits' value of 5 results in a precision loss of about 1.1 m in latitude per 1e-5 degrees; longitude scales
    by cos(latitude).

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


def download_file(url, save_to=None, proxy=None, proxy_user=None, proxy_pass=None, overwrite=False, timeout=(10, 60)):
    """
    Download a file from a given URL.

    Contract:
    - Atomic write: download to a temp file then rename on success.
    - If overwrite=False and target exists: return True (status quo).
    - On HTTP errors: return False and do not leave partial target file behind.
    """
    if not save_to:
        save_to = url.split("/")[-1]

    save_to = Path(save_to)

    if save_to.is_file() and not overwrite:
        logger.info(f"File {save_to} already exists and overwrite is set to False.")
        return True

    proxies = {"http": proxy, "https": proxy} if proxy else None
    auth = HTTPProxyAuth(proxy_user, proxy_pass) if proxy_user and proxy_pass else None
    headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"}

    tmp_path = save_to.with_suffix(save_to.suffix + ".tmp")

    try:
        response = requests.get(url, stream=True, proxies=proxies, auth=auth, headers=headers, timeout=timeout)
        response.raise_for_status()

        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

        tmp_path.replace(save_to)
        return True

    except Exception as e:
        logger.info(f"Failed to download file from {url}: {e}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False
