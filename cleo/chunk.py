from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union
from tqdm import tqdm

import sys
import numpy as np
import xarray as xr


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

# ----------------------------
# Helpers: chunk planning
# ----------------------------

@dataclass(frozen=True)
class _ChunkGrid:
    template_x: np.ndarray
    template_y: np.ndarray
    chunk_size: int

    @property
    def nx_total(self) -> int:
        return int(len(self.template_x))

    @property
    def ny_total(self) -> int:
        return int(len(self.template_y))

    @property
    def x_starts(self) -> List[int]:
        return list(range(0, self.nx_total, self.chunk_size))

    @property
    def y_starts(self) -> List[int]:
        return list(range(0, self.ny_total, self.chunk_size))

    @property
    def tasks(self) -> List[Tuple[int, int]]:
        # deterministic order
        return [(sx, sy) for sx in self.x_starts for sy in self.y_starts]

    @property
    def total_chunks(self) -> int:
        return len(self.x_starts) * len(self.y_starts)

    def where(self, start_x: int, start_y: int) -> str:
        return f"chunk(start_x={start_x}, start_y={start_y})"

    def chunk_nx_ny(self, start_x: int, start_y: int) -> Tuple[int, int]:
        nx = min(self.chunk_size, self.nx_total - start_x)
        ny = min(self.chunk_size, self.ny_total - start_y)
        return int(nx), int(ny)

    def pixel_weight(self, start_x: int, start_y: int) -> float:
        nx, ny = self.chunk_nx_ny(start_x, start_y)
        return float(nx * ny)

    def expected_xy_coords(self, start_x: int, start_y: int, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
        exp_x = self.template_x[start_x : start_x + nx]
        exp_y = self.template_y[start_y : start_y + ny]
        return exp_x, exp_y


def _np_array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    # exact, not tolerant — per your “striping risk” contract
    return np.array_equal(a, b)


def _assert_non_xy_coords_match(
    expected_coords: Mapping[str, np.ndarray],
    got_obj: Union[xr.DataArray, xr.Dataset],
    dims: Tuple[str, ...],
    *,
    label: str,
    grid: _ChunkGrid,
    start_x: int,
    start_y: int,
) -> None:
    for d in dims:
        if d in ("x", "y"):
            continue
        if d not in got_obj.coords:
            raise ValueError(
                f"Missing coord for non-spatial dim '{d}' in {label} at {grid.where(start_x, start_y)}."
            )
        exp = expected_coords[d]
        got = got_obj.coords[d].values
        if not _np_array_equal(got, exp):
            raise ValueError(
                f"Non-spatial coord mismatch for dim '{d}' in {label} at {grid.where(start_x, start_y)}."
            )


def _validate_xy_exact_grid(
    obj: xr.DataArray,
    *,
    grid: _ChunkGrid,
    start_x: int,
    start_y: int,
) -> Tuple[int, int]:
    if "x" not in obj.dims or "y" not in obj.dims:
        raise ValueError(
            f"compute_chunked requires results to include dims 'x' and 'y'; got dims={obj.dims} at {grid.where(start_x, start_y)}"
        )

    nx = int(obj.sizes["x"])
    ny = int(obj.sizes["y"])

    exp_x, exp_y = grid.expected_xy_coords(start_x, start_y, nx, ny)
    got_x = obj.coords["x"].values
    got_y = obj.coords["y"].values

    if not _np_array_equal(got_x, exp_x):
        raise ValueError(
            f"Exact-grid violation: x-coords mismatch at {grid.where(start_x, start_y)} (expected template slice)."
        )
    if not _np_array_equal(got_y, exp_y):
        raise ValueError(
            f"Exact-grid violation: y-coords mismatch at {grid.where(start_x, start_y)} (expected template slice)."
        )

    return nx, ny


# ----------------------------
# Helpers: base interface
# ----------------------------

class _Combiner:
    def add(self, chunk: Union[xr.DataArray, xr.Dataset], start_x: int, start_y: int) -> None:
        raise NotImplementedError

    def finalize(self) -> Union[xr.DataArray, xr.Dataset]:
        raise NotImplementedError


# ----------------------------
# Map mode: DataArray
# ----------------------------

class _MapDAAssembler(_Combiner):
    def __init__(self, *, grid: _ChunkGrid):
        self._grid = grid
        self._name: Optional[str] = None
        self._dims: Optional[Tuple[str, ...]] = None
        self._coords: Optional[Dict[str, np.ndarray]] = None
        self._attrs: Dict[str, Any] = {}
        self._data: Optional[np.ndarray] = None

    def _init_from_first(self, chunk: xr.DataArray) -> None:
        self._name = chunk.name
        self._dims = tuple(chunk.dims)
        self._attrs = dict(chunk.attrs) if chunk.attrs is not None else {}

        coords: Dict[str, np.ndarray] = {}
        for d in self._dims:
            if d == "x":
                coords["x"] = self._grid.template_x
            elif d == "y":
                coords["y"] = self._grid.template_y
            else:
                if d not in chunk.coords:
                    raise ValueError(f"Missing coord for non-spatial dim '{d}' in chunk result.")
                coords[d] = chunk.coords[d].values
        self._coords = coords

        shape: List[int] = []
        for d in self._dims:
            if d == "x":
                shape.append(self._grid.nx_total)
            elif d == "y":
                shape.append(self._grid.ny_total)
            else:
                shape.append(len(coords[d]))

        dtype = chunk.data.dtype
        fill = np.nan if dtype.kind in ("f", "c") else 0
        self._data = np.full(shape, fill_value=fill, dtype=dtype)

    def add(self, chunk: xr.DataArray, start_x: int, start_y: int) -> None:
        if self._dims is None:
            self._init_from_first(chunk)

        assert self._dims is not None and self._coords is not None and self._data is not None

        nx, ny = _validate_xy_exact_grid(chunk, grid=self._grid, start_x=start_x, start_y=start_y)
        _assert_non_xy_coords_match(
            self._coords, chunk, self._dims, label="DataArray", grid=self._grid, start_x=start_x, start_y=start_y
        )

        sl: List[slice] = []
        for d in self._dims:
            if d == "x":
                sl.append(slice(start_x, start_x + nx))
            elif d == "y":
                sl.append(slice(start_y, start_y + ny))
            else:
                sl.append(slice(None))

        chunk_t = chunk.transpose(*self._dims)
        self._data[tuple(sl)] = np.asarray(chunk_t.data)

    def finalize(self) -> xr.DataArray:
        assert self._dims is not None and self._coords is not None and self._data is not None
        return xr.DataArray(
            self._data,
            dims=self._dims,
            coords=self._coords,
            name=self._name,
            attrs=self._attrs,
        )


# ----------------------------
# Map mode: Dataset
# ----------------------------

class _MapDSAssembler(_Combiner):
    def __init__(self, *, grid: _ChunkGrid):
        self._grid = grid
        self._vars: Optional[Dict[str, Dict[str, Any]]] = None  # var_name -> spec

    def _init_from_first(self, chunk: xr.Dataset) -> None:
        vars_spec: Dict[str, Dict[str, Any]] = {}

        for var_name, var in chunk.data_vars.items():
            dims = tuple(var.dims)

            coords: Dict[str, np.ndarray] = {}
            for d in dims:
                if d == "x":
                    coords["x"] = self._grid.template_x
                elif d == "y":
                    coords["y"] = self._grid.template_y
                else:
                    if d not in var.coords:
                        raise ValueError(f"Missing coord for non-spatial dim '{d}' in var '{var_name}'.")
                    coords[d] = var.coords[d].values

            shape: List[int] = []
            for d in dims:
                if d == "x":
                    shape.append(self._grid.nx_total)
                elif d == "y":
                    shape.append(self._grid.ny_total)
                else:
                    shape.append(len(coords[d]))

            dtype = var.data.dtype
            fill = np.nan if dtype.kind in ("f", "c") else 0
            data = np.full(shape, fill_value=fill, dtype=dtype)

            vars_spec[var_name] = {
                "dims": dims,
                "coords": coords,
                "attrs": dict(var.attrs) if var.attrs is not None else {},
                "data": data,
            }

        self._vars = vars_spec

    def add(self, chunk: xr.Dataset, start_x: int, start_y: int) -> None:
        if self._vars is None:
            self._init_from_first(chunk)

        assert self._vars is not None

        # Ensure same set of variables
        for var_name in self._vars.keys():
            if var_name not in chunk.data_vars:
                raise ValueError(f"Chunk result missing variable '{var_name}' at {self._grid.where(start_x, start_y)}.")
        for var_name in chunk.data_vars.keys():
            if var_name not in self._vars:
                raise ValueError(f"Chunk result has unexpected variable '{var_name}' at {self._grid.where(start_x, start_y)}.")

        for var_name, spec in self._vars.items():
            var = chunk[var_name]
            dims = spec["dims"]
            coords = spec["coords"]
            data = spec["data"]

            nx, ny = _validate_xy_exact_grid(var, grid=self._grid, start_x=start_x, start_y=start_y)
            _assert_non_xy_coords_match(
                coords, var, dims, label=f"var '{var_name}'", grid=self._grid, start_x=start_x, start_y=start_y
            )

            sl: List[slice] = []
            for d in dims:
                if d == "x":
                    sl.append(slice(start_x, start_x + nx))
                elif d == "y":
                    sl.append(slice(start_y, start_y + ny))
                else:
                    sl.append(slice(None))

            var_t = var.transpose(*dims)
            data[tuple(sl)] = np.asarray(var_t.data)

    def finalize(self) -> xr.Dataset:
        assert self._vars is not None

        ds_coords: Dict[str, np.ndarray] = {"x": self._grid.template_x, "y": self._grid.template_y}
        data_vars: Dict[str, xr.DataArray] = {}

        # attach extra dim coords (must agree across vars)
        for var_name, spec in self._vars.items():
            for d, v in spec["coords"].items():
                if d in ("x", "y"):
                    continue
                if d in ds_coords and not _np_array_equal(ds_coords[d], v):
                    raise ValueError(f"Internal error: inconsistent coord '{d}' across variables.")
                ds_coords[d] = v

        for var_name, spec in self._vars.items():
            dims = spec["dims"]
            attrs = spec["attrs"]
            data = spec["data"]
            data_vars[var_name] = xr.DataArray(data, dims=dims, coords={d: ds_coords[d] for d in dims}, attrs=attrs)

        return xr.Dataset(data_vars=data_vars, coords=ds_coords)


# ----------------------------
# Reduction mode: DataArray
# ----------------------------

class _ReduceDAAggregator(_Combiner):
    def __init__(self, *, grid: _ChunkGrid, reduction: str):
        self._grid = grid
        self._reduction = reduction  # "mean" or "sum"
        self._weight_sum: float = 0.0

        self._name: Optional[str] = None
        self._dims: Optional[Tuple[str, ...]] = None
        self._coords: Optional[Dict[str, np.ndarray]] = None
        self._attrs: Dict[str, Any] = {}
        self._accum: Optional[np.ndarray] = None  # float64

    def _init_from_first(self, chunk: xr.DataArray) -> None:
        if "x" in chunk.dims or "y" in chunk.dims:
            raise ValueError("Internal error: reduction output unexpectedly contains x/y.")

        self._name = chunk.name
        self._dims = tuple(chunk.dims)
        self._attrs = dict(chunk.attrs) if chunk.attrs is not None else {}

        coords: Dict[str, np.ndarray] = {}
        for d in self._dims:
            if d in ("x", "y"):
                raise ValueError("Internal error: reduction dims unexpectedly contain x/y.")
            if d not in chunk.coords:
                raise ValueError(f"Missing coord for non-spatial dim '{d}' in reduction chunk result.")
            coords[d] = chunk.coords[d].values
        self._coords = coords

        shape = tuple(len(coords[d]) for d in self._dims)
        self._accum = np.zeros(shape, dtype=np.float64)

    def add(self, chunk: xr.DataArray, start_x: int, start_y: int) -> None:
        if self._dims is None:
            self._init_from_first(chunk)

        assert self._dims is not None and self._coords is not None and self._accum is not None

        _assert_non_xy_coords_match(
            self._coords,
            chunk,
            self._dims,
            label="Reduction DataArray",
            grid=self._grid,
            start_x=start_x,
            start_y=start_y,
        )

        arr = np.asarray((chunk.transpose(*self._dims) if self._dims else chunk).data, dtype=np.float64)

        if self._reduction == "mean":
            w = self._grid.pixel_weight(start_x, start_y)
            self._accum += arr * w
            self._weight_sum += w
        elif self._reduction == "sum":
            self._accum += arr
        else:
            raise ValueError(f"Internal error: unsupported reduction {self._reduction!r}")

    def finalize(self) -> xr.DataArray:
        assert self._dims is not None and self._coords is not None and self._accum is not None

        data = self._accum
        if self._reduction == "mean":
            if self._weight_sum == 0:
                raise ValueError("Internal error: zero total weight in reduction.")
            data = data / self._weight_sum

        return xr.DataArray(
            data,
            dims=self._dims,
            coords=self._coords,
            name=self._name,
            attrs=self._attrs,
        )


# ----------------------------
# Reduction mode: Dataset
# ----------------------------

class _ReduceDSAggregator(_Combiner):
    def __init__(self, *, grid: _ChunkGrid, reduction: str):
        self._grid = grid
        self._reduction = reduction  # "mean" or "sum"
        self._weight_sum: float = 0.0
        self._vars: Optional[Dict[str, Dict[str, Any]]] = None  # var_name -> dims/coords/attrs/accum(float64)

    def _init_from_first(self, chunk: xr.Dataset) -> None:
        vars_spec: Dict[str, Dict[str, Any]] = {}

        for var_name, var in chunk.data_vars.items():
            dims = tuple(var.dims)
            if "x" in dims or "y" in dims:
                raise ValueError(
                    f"Reduction-mode Dataset must not include x/y dims, but var '{var_name}' has dims={dims}."
                )

            coords: Dict[str, np.ndarray] = {}
            for d in dims:
                if d not in var.coords:
                    raise ValueError(f"Missing coord for non-spatial dim '{d}' in var '{var_name}'.")
                coords[d] = var.coords[d].values

            shape = tuple(len(coords[d]) for d in dims)
            vars_spec[var_name] = {
                "dims": dims,
                "coords": coords,
                "attrs": dict(var.attrs) if var.attrs is not None else {},
                "accum": np.zeros(shape, dtype=np.float64),
            }

        self._vars = vars_spec

    def add(self, chunk: xr.Dataset, start_x: int, start_y: int) -> None:
        if self._vars is None:
            self._init_from_first(chunk)

        assert self._vars is not None

        for var_name in self._vars.keys():
            if var_name not in chunk.data_vars:
                raise ValueError(
                    f"Chunk reduction result missing variable '{var_name}' at {self._grid.where(start_x, start_y)}."
                )
        for var_name in chunk.data_vars.keys():
            if var_name not in self._vars:
                raise ValueError(
                    f"Chunk reduction result has unexpected variable '{var_name}' at {self._grid.where(start_x, start_y)}."
                )

        if self._reduction == "mean":
            w = self._grid.pixel_weight(start_x, start_y)
        else:
            w = 1.0

        for var_name, spec in self._vars.items():
            var = chunk[var_name]
            dims = spec["dims"]
            coords = spec["coords"]

            _assert_non_xy_coords_match(
                coords,
                var,
                dims,
                label=f"Reduction var '{var_name}'",
                grid=self._grid,
                start_x=start_x,
                start_y=start_y,
            )

            arr = np.asarray((var.transpose(*dims) if dims else var).data, dtype=np.float64)
            if self._reduction == "mean":
                spec["accum"] += arr * w
            else:
                spec["accum"] += arr

        if self._reduction == "mean":
            self._weight_sum += w  # here w == pixel_weight

    def finalize(self) -> xr.Dataset:
        assert self._vars is not None

        if self._reduction == "mean" and self._weight_sum == 0:
            raise ValueError("Internal error: zero total weight in reduction.")

        ds_coords: Dict[str, np.ndarray] = {}
        data_vars: Dict[str, xr.DataArray] = {}

        # unify coords across vars
        for var_name, spec in self._vars.items():
            for d, v in spec["coords"].items():
                if d in ds_coords and not _np_array_equal(ds_coords[d], v):
                    raise ValueError(f"Internal error: inconsistent coord '{d}' across reduction variables.")
                ds_coords[d] = v

        for var_name, spec in self._vars.items():
            dims = spec["dims"]
            attrs = spec["attrs"]
            accum = spec["accum"]
            data = (accum / self._weight_sum) if self._reduction == "mean" else accum
            data_vars[var_name] = xr.DataArray(data, dims=dims, coords={d: ds_coords[d] for d in dims}, attrs=attrs)

        return xr.Dataset(data_vars=data_vars, coords=ds_coords)


# ----------------------------
# Combiner selection
# ----------------------------

def _select_combiner(
    first_chunk: Union[xr.DataArray, xr.Dataset],
    *,
    allow_reduction: bool,
    reduction: str,
    grid: _ChunkGrid,
    start_x: int,
    start_y: int,
) -> _Combiner:
    where = grid.where(start_x, start_y)

    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}")

    if isinstance(first_chunk, xr.DataArray):
        has_xy = ("x" in first_chunk.dims) and ("y" in first_chunk.dims)
        if has_xy:
            return _MapDAAssembler(grid=grid)
        # reduction output
        if not allow_reduction:
            raise ValueError(
                f"compute_chunked requires results to include dims 'x' and 'y'; got dims={first_chunk.dims} at {where}"
            )
        return _ReduceDAAggregator(grid=grid, reduction=reduction)

    if isinstance(first_chunk, xr.Dataset):
        has_xy_flags = [("x" in v.dims) and ("y" in v.dims) for v in first_chunk.data_vars.values()]

        if not has_xy_flags:
            # empty Dataset is ambiguous/broken for this API
            raise ValueError(f"processing_func returned an empty Dataset at {where} (no data_vars).")

        if all(has_xy_flags):
            return _MapDSAssembler(grid=grid)

        if not any(has_xy_flags):
            # reduction Dataset
            if not allow_reduction:
                # allow_reduction=False: this is illegal; raise with x/y message
                # (This is the only branch that raises that message for Dataset reductions.)
                raise ValueError(
                    f"compute_chunked requires results to include dims 'x' and 'y'; got Dataset vars without x/y at {where}"
                )
            return _ReduceDSAggregator(grid=grid, reduction=reduction)

        # mixed vars: unsupported
        raise ValueError(
            "processing_func returned a Dataset with mixed variables (some with x/y, some without). "
            "This is unsupported."
        )

    raise TypeError(f"processing_func must return xr.DataArray or xr.Dataset; got {type(first_chunk)}")


# ----------------------------
# Main API
# ----------------------------

def compute_chunked(
    self,
    processing_func: Callable[..., Union[xr.DataArray, xr.Dataset]],
    chunk_size: int,
    *,
    allow_reduction: bool = False,
    reduction: str = "mean",
    max_workers: int = 1,
    max_in_flight: Optional[int] = None,
    show_progress: bool = True,
    **kwargs,
):
    """
    Process data in x/y chunks and reassemble results on the original x/y template grid,
    OR (if allowed) aggregate reductions that drop x/y.

    Contract:

    Map mode (assemble-able):
    - Returned DataArray (or every Dataset variable) MUST include both 'x' and 'y' dims (plus optional extra dims).
    - Exact-grid contract: returned chunk x/y coords must exactly match the corresponding slice of the template coords.
    - Non-spatial dims/coords must match exactly across chunks.

    Reduction mode (not assemble-able, only if allow_reduction=True):
    - Returned DataArray (or Dataset variables) do NOT include x/y dims.
    - Aggregation rule depends on `reduction`:
        * reduction="mean": pixel-weighted combine of chunk means:
              global = sum(chunk_mean * n_pixels) / sum(n_pixels)
        * reduction="sum": plain sum of chunk sums:
              global = sum(chunk_sum)
    - Non-spatial dims/coords must match exactly across chunks.

    Dataset rule:
    - Either all variables have x/y (map Dataset), OR none have x/y (reduction Dataset).
      Mixed datasets are rejected.
    """

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0; got {chunk_size}")

    if "x" not in self.data.coords or "y" not in self.data.coords:
        raise ValueError("self.data must have 'x' and 'y' coordinates for compute_chunked().")

    template_x = self.data.coords["x"].values
    template_y = self.data.coords["y"].values
    grid = _ChunkGrid(template_x=template_x, template_y=template_y, chunk_size=int(chunk_size))

    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1; got {max_workers}")

    if max_in_flight is None:
        max_in_flight = 2 * max_workers
    if max_in_flight < 1:
        raise ValueError(f"max_in_flight must be >= 1; got {max_in_flight}")

    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}")

    tasks = grid.tasks

    def _run_one(start_x: int, start_y: int):
        # _process_chunk is assumed to exist in this module (your current design)
        return _process_chunk(self, processing_func, chunk_size, start_x, start_y, **kwargs), start_x, start_y

    combiner: Optional[_Combiner] = None

    pbar_ctx = stylish_tqdm(total=grid.total_chunks, desc="Processing chunks") if show_progress else None
    try:
        if max_workers == 1:
            for sx, sy in tasks:
                chunk, sx0, sy0 = _run_one(sx, sy)

                if combiner is None:
                    combiner = _select_combiner(
                        chunk,
                        allow_reduction=allow_reduction,
                        reduction=reduction,
                        grid=grid,
                        start_x=sx0,
                        start_y=sy0,
                    )

                combiner.add(chunk, sx0, sy0)

                if pbar_ctx is not None:
                    pbar_ctx.update(1)

        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                it = iter(tasks)
                in_flight: Dict[Any, Tuple[int, int]] = {}

                # prime
                for _ in range(min(max_in_flight, grid.total_chunks)):
                    sx, sy = next(it, (None, None))
                    if sx is None:
                        break
                    fut = ex.submit(_run_one, sx, sy)
                    in_flight[fut] = (sx, sy)

                while in_flight:
                    done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                    for fut in done:
                        _ = in_flight.pop(fut)
                        chunk, sx0, sy0 = fut.result()

                        if combiner is None:
                            combiner = _select_combiner(
                                chunk,
                                allow_reduction=allow_reduction,
                                reduction=reduction,
                                grid=grid,
                                start_x=sx0,
                                start_y=sy0,
                            )

                        combiner.add(chunk, sx0, sy0)

                        if pbar_ctx is not None:
                            pbar_ctx.update(1)

                        # submit next
                        sx, sy = next(it, (None, None))
                        if sx is not None:
                            nfut = ex.submit(_run_one, sx, sy)
                            in_flight[nfut] = (sx, sy)

    finally:
        if pbar_ctx is not None:
            pbar_ctx.close()

    if combiner is None:
        # can happen if there are zero chunks (shouldn't, but keep defensive)
        raise ValueError("compute_chunked produced no chunks (unexpected).")

    return combiner.finalize()
