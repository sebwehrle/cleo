# %% imports
import numpy as np
import pandas as pd
import xarray as xr


def _cast_binary_series_to_int(s: pd.Series) -> pd.Series:
    """Cast binary series (0/1 or bool) to nullable integer, else return unchanged."""
    if pd.api.types.is_bool_dtype(s.dtype):
        return s.astype("Int8")

    if not pd.api.types.is_numeric_dtype(s.dtype):
        return s

    non_na = s.dropna()
    if non_na.empty:
        return s

    vals = set(np.unique(non_na.to_numpy(dtype=np.float64)))
    if vals.issubset({0.0, 1.0}):
        return s.astype("Int8")
    return s


def flatten(
    data: xr.Dataset,
    digits=5,
    exclude_template=True,
    cast_binary_to_int=False,
    include_only=None,
):
    """
    Converts data in a xarray.Dataset to a pandas.DataFrame in a slower but more memory efficient way than
    xarray.Dataset.to_dataframe. Rounding of coordinates facilitates merging across data variables. The default
    'digits' value of 5 results in a precision loss of about 1.1 m in latitude per 1e-5 degrees; longitude scales
    by cos(latitude).

    :param data: dataset to flatten
    :param digits: number of digits to round x and y coordinates to
    :param exclude_template: a boolean flag to exclude the template-data variable
    :param cast_binary_to_int: if True, cast binary columns (bool or {0,1}) to nullable Int8.
        Continuous/non-binary columns are left unchanged.
    :param include_only: optional list of output column names to keep. Raises if any
        requested column is not present in the flattened output.
    :return: a pandas.Dataframe with one column per data variable and non-spatial coordinate
    """

    collect_df = []

    data_variables = data.data_vars
    if exclude_template:
        data_variables = [data_var for data_var in data_variables if data_var != "template"]

    for var_name in data_variables:
        data_var = data[var_name]
        # drop non-dimensional coordinates
        non_dim_coords = set(data_var.coords) - set(data_var.dims)
        data_var = data_var.drop_vars(non_dim_coords)

        if {"x", "y"} == set(data_var.dims):
            df = data_var.to_dataframe().dropna()
            df.index = pd.MultiIndex.from_arrays(
                [
                    np.round(df.index.get_level_values("y"), digits),
                    np.round(df.index.get_level_values("x"), digits),
                ]
            )
            collect_df.append(df)

        elif len(data_var.dims) == 3 and {"x", "y"}.issubset(set(data_var.dims)):
            non_spatial_dim = next(iter(set(data_var.dims) - {"x", "y"}))

            for coord in data_var.coords[non_spatial_dim]:
                data_slice = data_var.sel({non_spatial_dim: coord})
                data_slice = data_slice.drop_vars(non_spatial_dim)
                coord_value = coord.item() if hasattr(coord, "item") else coord.data
                data_slice.name = f"{var_name}_{non_spatial_dim}_{coord_value}"
                df = data_slice.to_dataframe().dropna()
                df.index = pd.MultiIndex.from_arrays(
                    [
                        np.round(df.index.get_level_values("y"), digits),
                        np.round(df.index.get_level_values("x"), digits),
                    ]
                )
                collect_df.append(df)

        else:
            raise ValueError(
                f"Error in {var_name}. Only 2D ('x','y') or 3D data with one non-spatial dimension plus 'x' and 'y' are supported"
            )
    if not collect_df:
        # No variables selected (e.g., template-only with exclude_template=True):
        # return an empty frame with rounded spatial index when available.
        if {"x", "y"}.issubset(set(data.coords)):
            y_vals = np.round(np.asarray(data.coords["y"].values), digits)
            x_vals = np.round(np.asarray(data.coords["x"].values), digits)
            index = pd.MultiIndex.from_product([y_vals, x_vals], names=["y", "x"])
            return pd.DataFrame(index=index)
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["y", "x"])
        return pd.DataFrame(index=empty_index)

    out = pd.concat(collect_df, axis=1)
    if cast_binary_to_int:
        for col in out.columns:
            out[col] = _cast_binary_series_to_int(out[col])
    if include_only is not None:
        include_only = list(include_only)
        missing = [c for c in include_only if c not in out.columns]
        if missing:
            raise ValueError(
                f"include_only contains unknown columns: {sorted(missing)!r}. "
                f"Available columns: {sorted(map(str, out.columns))!r}"
            )
        out = out.loc[:, include_only]
    return out
