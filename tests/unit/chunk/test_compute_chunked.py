"""assess: test_compute_chunked.

Contracts:
- chunked computation must reassemble on an exact (x,y) grid without coord drift.
- extra non-spatial dims must be preserved and stitched correctly.
- serial and parallel execution must be bitwise-identical for deterministic funcs.
- Dataset-returning funcs must preserve all variables.
- reductions are only allowed when explicitly enabled.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from cleo.assess import compute_chunked
from tests.helpers.asserts import assert_same_coords


def _da_yx(values: np.ndarray, *, x: np.ndarray, y: np.ndarray, name: str) -> xr.DataArray:
    """Create a (y,x) DataArray with explicit coords (no implicit n=5 defaults)."""
    return xr.DataArray(values, dims=("y", "x"), coords={"x": x, "y": y}, name=name)


def _grid(*, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    return x, y


def test_compute_chunked_reassembles_exact_dataarray() -> None:
    x, y = _grid(nx=6, ny=5)
    v = _da_yx(
        np.arange(y.size * x.size, dtype=float).reshape(y.size, x.size),
        x=x,
        y=y,
        name="v",
    )
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    out = compute_chunked(self, processing_func=lambda v: v, chunk_size=2, v=v)

    assert isinstance(out, xr.DataArray)
    xr.testing.assert_identical(out, v)


def test_compute_chunked_raises_on_coord_drift() -> None:
    x, y = _grid(nx=6, ny=5)
    v = _da_yx(np.ones((y.size, x.size), dtype=float), x=x, y=y, name="v")
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    def drift_x(v: xr.DataArray) -> xr.DataArray:
        v2 = v.copy()
        return v2.assign_coords(x=v2.coords["x"] + 1e-6)

    with pytest.raises(ValueError, match="x-coords mismatch"):
        compute_chunked(self, processing_func=drift_x, chunk_size=2, v=v)


def test_compute_chunked_stitches_extra_dims() -> None:
    x, y = _grid(nx=6, ny=5)
    height = np.array([50.0, 100.0], dtype=float)

    v = xr.DataArray(
        np.arange(height.size * y.size * x.size, dtype=float).reshape(height.size, y.size, x.size),
        dims=("height", "y", "x"),
        coords={"height": height, "x": x, "y": y},
        name="v",
    )
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    out = compute_chunked(self, processing_func=lambda v: v * 2.0, chunk_size=2, v=v)

    assert isinstance(out, xr.DataArray)
    assert out.dims == ("height", "y", "x")
    assert_same_coords(out, v)
    np.testing.assert_allclose(out.values, (v * 2.0).values)


def test_compute_chunked_parallel_matches_serial_map_mode_with_extra_dim() -> None:
    # does not divide evenly by chunk_size -> exercises edge chunks
    x, y = _grid(nx=7, ny=5)
    rng = np.random.default_rng(0)
    v = _da_yx(rng.normal(size=(y.size, x.size)).astype(float), x=x, y=y, name="v")
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    def processing_func(v: xr.DataArray) -> xr.DataArray:
        out0 = (v + 1.0).astype(float)
        out1 = (v * 2.0).astype(float)
        out = xr.concat([out0, out1], dim="band").assign_coords(band=np.array([0, 1], dtype=int))
        return out.rename("out").transpose("band", "y", "x")

    chunk_size = 3

    serial = compute_chunked(
        self,
        processing_func=processing_func,
        chunk_size=chunk_size,
        max_workers=1,
        show_progress=False,
        v=v,
    )
    parallel = compute_chunked(
        self,
        processing_func=processing_func,
        chunk_size=chunk_size,
        max_workers=4,
        max_in_flight=8,
        show_progress=False,
        v=v,
    )

    xr.testing.assert_identical(serial, parallel)


def test_compute_chunked_preserves_dataset_vars() -> None:
    x, y = _grid(nx=4, ny=4)
    self = SimpleNamespace(data=xr.Dataset(coords={"x": x, "y": y}))

    dummy = _da_yx(np.ones((y.size, x.size), dtype=float), x=x, y=y, name="dummy")

    def processing_func(dummy: xr.DataArray) -> xr.Dataset:
        yy = dummy.coords["y"].values
        xx = dummy.coords["x"].values
        return xr.Dataset(
            {
                "a": xr.DataArray(np.ones((yy.size, xx.size), dtype=float), dims=("y", "x"), coords={"y": yy, "x": xx}),
                "b": xr.DataArray(
                    np.ones((yy.size, xx.size), dtype=float) * 2.0, dims=("y", "x"), coords={"y": yy, "x": xx}
                ),
            }
        )

    result = compute_chunked(self, processing_func, chunk_size=2, dummy=dummy)

    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"a", "b"}


def test_compute_chunked_returns_dataarray() -> None:
    x, y = _grid(nx=4, ny=4)
    self = SimpleNamespace(data=xr.Dataset(coords={"x": x, "y": y}))

    dummy = _da_yx(np.ones((y.size, x.size), dtype=float), x=x, y=y, name="dummy")

    def processing_func(dummy: xr.DataArray) -> xr.DataArray:
        yy = dummy.coords["y"].values
        xx = dummy.coords["x"].values
        return xr.DataArray(
            np.ones((yy.size, xx.size), dtype=float) * 3.0,
            dims=("y", "x"),
            coords={"y": yy, "x": xx},
            name="result",
        )

    result = compute_chunked(self, processing_func, chunk_size=2, dummy=dummy)

    assert isinstance(result, xr.DataArray)
    assert result.name == "result"


def test_compute_chunked_reduction_allowed_returns_scalar_sum() -> None:
    x, y = _grid(nx=6, ny=5)
    v = _da_yx(np.ones((y.size, x.size), dtype=float), x=x, y=y, name="v")
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    out = compute_chunked(
        self,
        processing_func=lambda v: v.sum(),
        chunk_size=2,
        allow_reduction=True,
        reduction="sum",
        v=v,
    )

    assert isinstance(out, xr.DataArray)
    assert out.dims == ()
    assert float(out.item()) == float(v.sum().item())


def test_compute_chunked_reduction_requires_xy_dims_when_not_allowed() -> None:
    x, y = _grid(nx=6, ny=5)
    v = _da_yx(np.arange(y.size * x.size, dtype=float).reshape(y.size, x.size), x=x, y=y, name="v")
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    with pytest.raises(ValueError, match=r"requires results to include dims 'x' and 'y'"):
        compute_chunked(self, processing_func=lambda v: v.mean(), chunk_size=2, allow_reduction=False, v=v)


def test_compute_chunked_weighted_mean_reduction_matches_global_mean() -> None:
    x, y = _grid(nx=6, ny=5)
    v = _da_yx(np.arange(y.size * x.size, dtype=float).reshape(y.size, x.size), x=x, y=y, name="v")
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    out = compute_chunked(self, processing_func=lambda v: v.mean(), chunk_size=2, allow_reduction=True, v=v)

    assert isinstance(out, xr.DataArray)
    assert out.dims == ()
    assert float(out.item()) == pytest.approx(float(v.mean().item()))


def test_compute_chunked_supports_extra_dims_time_y_x() -> None:
    rng = np.random.default_rng(0)
    time = np.arange(3, dtype=int)
    x, y = _grid(nx=6, ny=5)

    v = xr.DataArray(
        rng.normal(size=(time.size, y.size, x.size)),
        dims=("time", "y", "x"),
        coords={"time": time, "x": x, "y": y},
        name="v",
    )
    self = SimpleNamespace(data=xr.Dataset({"v": v}))

    out = compute_chunked(self, processing_func=lambda v: v, chunk_size=2, v=v)

    assert isinstance(out, xr.DataArray)
    assert out.dims == v.dims
    assert_same_coords(out, v)
    np.testing.assert_allclose(out.values, v.values)
