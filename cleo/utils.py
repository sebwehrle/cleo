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
_UREG = UnitRegistry()

from cleo.spatial import bbox


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
    logging.info(f"Merged '{other.name}' into '{self.data.attrs.get('country', 'atlas')}'-data.")


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


def compute_chunked(
    self,
    processing_func,
    chunk_size: int,
    *,
    max_workers: int = 1,
    max_in_flight: int | None = None,
    show_progress: bool = True,
    **kwargs,
):
    """
    Process data in x/y chunks and reassemble results on the original x/y template grid.

    Contract:
    - processing_func must return either xr.DataArray or xr.Dataset.
    - Every returned DataArray (or every Dataset variable) MUST include both 'x' and 'y' dims
      (plus optional extra dims). Results that drop x/y (e.g. mean over x/y) are not assemble-able
      by this function and will raise.

    Exact-grid contract:
    - Returned chunk x/y coords must exactly match the corresponding slice of template coords
      from self.data.coords['x'/'y'].
    - Returned non-spatial dims (and their coords) must be identical across chunks.
    """

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0; got {chunk_size}")

    if "x" not in self.data.coords or "y" not in self.data.coords:
        raise ValueError("self.data must have 'x' and 'y' coordinates for compute_chunked().")

    template_x = self.data.coords["x"].values
    template_y = self.data.coords["y"].values
    nx_total = len(template_x)
    ny_total = len(template_y)

    x_starts = list(range(0, nx_total, chunk_size))
    y_starts = list(range(0, ny_total, chunk_size))
    total_chunks = len(x_starts) * len(y_starts)

    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1; got {max_workers}")

    if max_in_flight is None:
        max_in_flight = 2 * max_workers
    if max_in_flight < 1:
        raise ValueError(f"max_in_flight must be >= 1; got {max_in_flight}")

    # Will be initialized from the first completed chunk
    out_is_dataarray: bool | None = None
    out_da_name: str | None = None
    out_da_dims: tuple[str, ...] | None = None
    out_da_coords: dict[str, object] | None = None
    out_da_attrs: dict | None = None
    out_da_data = None  # numpy array

    out_ds_vars: dict[str, dict] | None = None  # var_name -> spec dict + numpy array

    def _where(start_x: int, start_y: int) -> str:
        return f"chunk(start_x={start_x}, start_y={start_y})"

    def _validate_and_get_xy_slices(chunk_obj, start_x: int, start_y: int):
        if "x" not in chunk_obj.dims or "y" not in chunk_obj.dims:
            raise ValueError(
                f"compute_chunked requires results to include dims 'x' and 'y'; got dims={chunk_obj.dims} at {_where(start_x, start_y)}"
            )

        nx = int(chunk_obj.sizes["x"])
        ny = int(chunk_obj.sizes["y"])

        # expected coords from template
        exp_x = template_x[start_x : start_x + nx]
        exp_y = template_y[start_y : start_y + ny]

        got_x = chunk_obj.coords["x"].values
        got_y = chunk_obj.coords["y"].values

        if not np.array_equal(got_x, exp_x):
            raise ValueError(
                f"Exact-grid violation: x-coords mismatch at {_where(start_x, start_y)} "
                f"(expected template slice)."
            )
        if not np.array_equal(got_y, exp_y):
            raise ValueError(
                f"Exact-grid violation: y-coords mismatch at {_where(start_x, start_y)} "
                f"(expected template slice)."
            )

        return nx, ny

    def _init_out_from_dataarray(chunk: xr.DataArray):
        nonlocal out_da_name, out_da_dims, out_da_coords, out_da_attrs, out_da_data

        out_da_name = chunk.name
        out_da_dims = tuple(chunk.dims)
        out_da_attrs = dict(chunk.attrs) if chunk.attrs is not None else {}

        coords = {}
        for d in out_da_dims:
            if d == "x":
                coords["x"] = template_x
            elif d == "y":
                coords["y"] = template_y
            else:
                if d not in chunk.coords:
                    raise ValueError(f"Missing coord for non-spatial dim '{d}' in chunk result.")
                coords[d] = chunk.coords[d].values
        out_da_coords = coords

        # allocate backing array
        shape = []
        for d in out_da_dims:
            if d == "x":
                shape.append(nx_total)
            elif d == "y":
                shape.append(ny_total)
            else:
                shape.append(len(coords[d]))

        dtype = chunk.data.dtype
        fill = np.nan if dtype.kind in ("f", "c") else 0
        out_da_data = np.full(shape, fill_value=fill, dtype=dtype)

    def _init_out_from_dataset(chunk: xr.Dataset):
        nonlocal out_ds_vars
        out_ds_vars = {}

        for var_name, var in chunk.data_vars.items():
            dims = tuple(var.dims)
            coords = {}
            for d in dims:
                if d == "x":
                    coords["x"] = template_x
                elif d == "y":
                    coords["y"] = template_y
                else:
                    if d not in var.coords:
                        raise ValueError(f"Missing coord for non-spatial dim '{d}' in var '{var_name}'.")
                    coords[d] = var.coords[d].values

            shape = []
            for d in dims:
                if d == "x":
                    shape.append(nx_total)
                elif d == "y":
                    shape.append(ny_total)
                else:
                    shape.append(len(coords[d]))

            dtype = var.data.dtype
            fill = np.nan if dtype.kind in ("f", "c") else 0
            data = np.full(shape, fill_value=fill, dtype=dtype)

            out_ds_vars[var_name] = {
                "dims": dims,
                "coords": coords,
                "attrs": dict(var.attrs) if var.attrs is not None else {},
                "data": data,
            }

    def _assert_non_xy_coords_match(expected_coords, got_obj, dims, var_label: str, start_x: int, start_y: int):
        for d in dims:
            if d in ("x", "y"):
                continue
            exp = expected_coords[d]
            got = got_obj.coords[d].values
            if not np.array_equal(got, exp):
                raise ValueError(
                    f"Non-spatial coord mismatch for dim '{d}' in {var_label} at {_where(start_x, start_y)}."
                )

    def _write_chunk_dataarray(chunk: xr.DataArray, start_x: int, start_y: int):
        nonlocal out_da_data
        assert out_da_dims is not None and out_da_coords is not None and out_da_data is not None

        # ensure x/y coords match template slice
        nx, ny = _validate_and_get_xy_slices(chunk, start_x, start_y)

        # ensure non-x/y coords match
        _assert_non_xy_coords_match(out_da_coords, chunk, out_da_dims, "DataArray", start_x, start_y)

        # write into backing array using dimension-ordered slices
        sl = []
        for d in out_da_dims:
            if d == "x":
                sl.append(slice(start_x, start_x + nx))
            elif d == "y":
                sl.append(slice(start_y, start_y + ny))
            else:
                sl.append(slice(None))

        chunk_t = chunk.transpose(*out_da_dims)
        out_da_data[tuple(sl)] = np.asarray(chunk_t.data)

    def _write_chunk_dataset(chunk: xr.Dataset, start_x: int, start_y: int):
        assert out_ds_vars is not None

        # ensure same set of variables
        for var_name in out_ds_vars.keys():
            if var_name not in chunk.data_vars:
                raise ValueError(f"Chunk result missing variable '{var_name}' at {_where(start_x, start_y)}.")
        for var_name in chunk.data_vars.keys():
            if var_name not in out_ds_vars:
                raise ValueError(f"Chunk result has unexpected variable '{var_name}' at {_where(start_x, start_y)}.")

        for var_name, spec in out_ds_vars.items():
            var = chunk[var_name]
            dims = spec["dims"]
            coords = spec["coords"]
            data = spec["data"]

            # must include x/y
            nx, ny = _validate_and_get_xy_slices(var, start_x, start_y)
            _assert_non_xy_coords_match(coords, var, dims, f"var '{var_name}'", start_x, start_y)

            sl = []
            for d in dims:
                if d == "x":
                    sl.append(slice(start_x, start_x + nx))
                elif d == "y":
                    sl.append(slice(start_y, start_y + ny))
                else:
                    sl.append(slice(None))

            var_t = var.transpose(*dims)
            data[tuple(sl)] = np.asarray(var_t.data)

    # Task generator (deterministic order)
    tasks = [(sx, sy) for sx in x_starts for sy in y_starts]

    def _run_one(start_x: int, start_y: int):
        return _process_chunk(self, processing_func, chunk_size, start_x, start_y, **kwargs), start_x, start_y

    pbar_ctx = stylish_tqdm(total=total_chunks, desc="Processing chunks") if show_progress else None

    try:
        if max_workers == 1:
            # serial, lowest overhead
            for sx, sy in tasks:
                chunk, sx0, sy0 = _run_one(sx, sy)

                if out_is_dataarray is None:
                    out_is_dataarray = isinstance(chunk, xr.DataArray)
                    if out_is_dataarray:
                        _init_out_from_dataarray(chunk)
                    else:
                        if not isinstance(chunk, xr.Dataset):
                            raise TypeError(f"processing_func must return xr.DataArray or xr.Dataset; got {type(chunk)}")
                        _init_out_from_dataset(chunk)

                if out_is_dataarray:
                    _write_chunk_dataarray(chunk, sx0, sy0)
                else:
                    _write_chunk_dataset(chunk, sx0, sy0)

                if pbar_ctx is not None:
                    pbar_ctx.update(1)

        else:
            # bounded parallelism
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                it = iter(tasks)
                in_flight = {}

                # prime queue
                for _ in range(min(max_in_flight, total_chunks)):
                    sx, sy = next(it, (None, None))
                    if sx is None:
                        break
                    in_flight[ex.submit(_run_one, sx, sy)] = (sx, sy)

                while in_flight:
                    # get one completed future
                    done = None
                    for fut in in_flight:
                        if fut.done():
                            done = fut
                            break
                    if done is None:
                        # wait on first future (blocks)
                        done = next(iter(in_flight))

                    sx0, sy0 = in_flight.pop(done)
                    chunk, sx1, sy1 = done.result()

                    if out_is_dataarray is None:
                        out_is_dataarray = isinstance(chunk, xr.DataArray)
                        if out_is_dataarray:
                            _init_out_from_dataarray(chunk)
                        else:
                            if not isinstance(chunk, xr.Dataset):
                                raise TypeError(f"processing_func must return xr.DataArray or xr.Dataset; got {type(chunk)}")
                            _init_out_from_dataset(chunk)

                    if out_is_dataarray:
                        _write_chunk_dataarray(chunk, sx1, sy1)
                    else:
                        _write_chunk_dataset(chunk, sx1, sy1)

                    if pbar_ctx is not None:
                        pbar_ctx.update(1)

                    # submit next
                    sx, sy = next(it, (None, None))
                    if sx is not None:
                        in_flight[ex.submit(_run_one, sx, sy)] = (sx, sy)

    finally:
        if pbar_ctx is not None:
            pbar_ctx.close()

    # materialize output object
    if out_is_dataarray:
        assert out_da_name is not None and out_da_dims is not None and out_da_coords is not None and out_da_data is not None
        out = xr.DataArray(
            out_da_data,
            dims=out_da_dims,
            coords=out_da_coords,
            name=out_da_name,
            attrs=out_da_attrs,
        )
        return out
    else:
        assert out_ds_vars is not None
        data_vars = {}
        # choose dataset-level coords as union of per-var coords (they should agree)
        ds_coords = {"x": template_x, "y": template_y}

        for var_name, spec in out_ds_vars.items():
            dims = spec["dims"]
            coords = spec["coords"]
            attrs = spec["attrs"]
            data = spec["data"]

            # attach any extra dim coords
            for d, v in coords.items():
                if d not in ("x", "y"):
                    if d in ds_coords and not np.array_equal(ds_coords[d], v):
                        raise ValueError(f"Internal error: inconsistent coord '{d}' across variables.")
                    ds_coords[d] = v

            data_vars[var_name] = xr.DataArray(data, dims=dims, coords={d: ds_coords[d] for d in dims}, attrs=attrs)

        return xr.Dataset(data_vars=data_vars, coords=ds_coords)


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
        logging.info(f"File {save_to} already exists and overwrite is set to False.")
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
        logging.info(f"Failed to download file from {url}: {e}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False
