"""Test that bbox() returns Python floats, not xarray DataArray scalars."""

import numpy as np
import xarray as xr

from cleo.spatial import bbox


def test_bbox_returns_floats():
    """bbox() should return a tuple of Python floats."""
    # Create a tiny DataArray with x/y coords
    da = xr.DataArray(
        data=np.ones((2, 3)),
        dims=["y", "x"],
        coords={"y": [0.0, 1.0], "x": [0.0, 0.5, 1.0]},
    )

    result = bbox(da)

    # Assert result is a tuple of 4 elements
    assert isinstance(result, tuple)
    assert len(result) == 4

    # Assert all elements are Python floats (or int), not xarray types
    for i, val in enumerate(result):
        assert isinstance(val, (float, int)), (
            f"Element {i} is {type(val).__name__}, expected float or int"
        )

    # Assert values are correct
    xmin, ymin, xmax, ymax = result
    assert xmin == 0.0
    assert ymin == 0.0
    assert xmax == 1.0
    assert ymax == 1.0


def test_bbox_equality_comparison():
    """bbox() results should be comparable with == and !=."""
    da = xr.DataArray(
        data=np.ones((2, 3)),
        dims=["y", "x"],
        coords={"y": [0.0, 1.0], "x": [0.0, 0.5, 1.0]},
    )

    b1 = bbox(da)
    b2 = bbox(da)

    # These comparisons should work without exceptions
    assert b1 == b2
    assert not (b1 != b2)


def test_bbox_with_data_attribute():
    """bbox() should work on objects with .data attribute."""

    class MockAtlas:
        def __init__(self):
            self.data = xr.Dataset(
                coords={"y": [10.0, 20.0], "x": [100.0, 150.0, 200.0]}
            )

    mock = MockAtlas()
    result = bbox(mock)

    assert isinstance(result, tuple)
    assert len(result) == 4
    assert all(isinstance(v, (float, int)) for v in result)

    xmin, ymin, xmax, ymax = result
    assert xmin == 100.0
    assert ymin == 10.0
    assert xmax == 200.0
    assert ymax == 20.0
