"""utils: flatten tests."""

from types import SimpleNamespace

import numpy as np
import xarray as xr
import pytest

from cleo.utils import flatten


def test_flatten_excludes_template_and_flattens_2d_and_3d():
    ds = xr.Dataset(
        {
            "template": (("y", "x"), np.ones((2, 2))),
            "v2": (("y", "x"), np.array([[1.0, 2.0], [3.0, np.nan]])),
            "v3": (("height", "y", "x"), np.array([
                [[10.0, 11.0], [12.0, np.nan]],
                [[20.0, 21.0], [22.0, np.nan]],
            ])),
        },
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0], "height": [50, 100]},
    )
    self = SimpleNamespace(data=ds)

    df = flatten(self)

    assert "template" not in df.columns
    assert "v2" in df.columns
    assert "v3_height_50" in df.columns
    assert "v3_height_100" in df.columns
    assert list(df.index.names) == ["y", "x"]


def test_flatten_can_include_template():
    ds = xr.Dataset(
        {"template": (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    self = SimpleNamespace(data=ds)

    df = flatten(self, exclude_template=False)

    assert "template" in df.columns
    assert len(df) == 4


def test_flatten_raises_for_non_spatial_variable_dims():
    ds = xr.Dataset(
        {"bad": (("t",), np.array([1.0, 2.0]))},
        coords={"t": [0, 1]},
    )
    self = SimpleNamespace(data=ds)

    with pytest.raises(
        ValueError,
        match="Only 2D \\('x','y'\\) or 3D data with one non-spatial dimension plus 'x' and 'y' are supported",
    ):
        flatten(self)


def test_flatten_template_only_returns_empty_frame_with_spatial_index():
    ds = xr.Dataset(
        {"template": (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    self = SimpleNamespace(data=ds)

    df = flatten(self, exclude_template=True)

    assert list(df.index.names) == ["y", "x"]
    assert df.shape == (4, 0)


def test_flatten_cast_binary_to_int_casts_only_binary_columns():
    ds = xr.Dataset(
        {
            "binary_bool": (("y", "x"), np.array([[True, False], [True, True]])),
            "binary_float": (("y", "x"), np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)),
            "continuous": (("y", "x"), np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)),
        },
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    self = SimpleNamespace(data=ds)

    df = flatten(self, cast_binary_to_int=True)

    assert str(df["binary_bool"].dtype) == "Int8"
    assert str(df["binary_float"].dtype) == "Int8"
    assert str(df["continuous"].dtype) != "Int8"


def test_flatten_include_only_selects_columns_in_order():
    ds = xr.Dataset(
        {
            "a": (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]])),
            "b": (("y", "x"), np.array([[5.0, 6.0], [7.0, 8.0]])),
        },
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    self = SimpleNamespace(data=ds)

    df = flatten(self, include_only=["b", "a"])
    assert list(df.columns) == ["b", "a"]


def test_flatten_include_only_unknown_column_raises():
    ds = xr.Dataset(
        {"a": (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    self = SimpleNamespace(data=ds)

    with pytest.raises(ValueError, match="include_only contains unknown columns"):
        flatten(self, include_only=["a", "missing"])
