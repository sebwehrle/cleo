# %% imports
import numpy as np
import pandas as pd
import xarray as xr
from pint import UnitRegistry

_UREG = UnitRegistry()


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
                coord_value = coord.item() if hasattr(coord, "item") else coord.data
                data_slice.name = f"{var_name}_{non_spatial_dim}_{coord_value}"
                df = data_slice.to_dataframe().dropna()
                df.index = pd.MultiIndex.from_arrays([
                    np.round(df.index.get_level_values("y"), digits),
                    np.round(df.index.get_level_values("x"), digits),
                ])
                collect_df.append(df)

        else:
            raise ValueError(
                f"Error in {var_name}. Only 2D ('x','y') or 3D data with one non-spatial dimension plus 'x' and 'y' are supported"
            )
    if not collect_df:
        # No variables selected (e.g., template-only with exclude_template=True):
        # return an empty frame with rounded spatial index when available.
        if {"x", "y"}.issubset(set(self.data.coords)):
            y_vals = np.round(np.asarray(self.data.coords["y"].values), digits)
            x_vals = np.round(np.asarray(self.data.coords["x"].values), digits)
            index = pd.MultiIndex.from_product([y_vals, x_vals], names=["y", "x"])
            return pd.DataFrame(index=index)
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["y", "x"])
        return pd.DataFrame(index=empty_index)

    return pd.concat(collect_df, axis=1)
