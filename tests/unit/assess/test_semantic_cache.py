"""assess: test_semantic_cache.

Tests for semantic signature-based caching:
- reuse: calling compute twice with same params -> sig unchanged (skipped recompute)
- invalidation: calling with different params -> sig differs
- unsigned recompute: removing cleo:* attrs forces recompute
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.helpers.factories import wind_speed_axis

import cleo.assess as assess


def _make_wind_atlas_stub(tmp_path):
    """Create minimal wind atlas stub for testing compute_mean_wind_speed."""
    # Build minimal dataset with template and wind_speed coords
    template = xr.DataArray(
        np.ones((3, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
    ).rio.write_crs("EPSG:4326")

    ds = xr.Dataset({"template": template})
    ds = ds.assign_coords(wind_speed=wind_speed_axis())
    ds.attrs["country"] = "AUT"

    parent = SimpleNamespace(
        country="AUT",
        crs="epsg:4326",
        path=tmp_path,
    )

    def _set_var(name, da):
        stub.data[name] = da

    stub = SimpleNamespace(
        data=ds,
        parent=parent,
        load_weibull_parameters=MagicMock(return_value=(
            xr.DataArray(np.full((3, 3), 7.0), dims=("y", "x"), coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]}),
            xr.DataArray(np.full((3, 3), 2.0), dims=("y", "x"), coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]}),
        )),
    )
    stub._set_var = _set_var

    return stub


def test_mean_wind_speed_reuse_same_height(tmp_path):
    """
    Calling compute_mean_wind_speed twice with same height should:
    - Return sig on first call
    - Skip recompute on second call (same sig)
    """
    stub = _make_wind_atlas_stub(tmp_path)

    # First call
    assess.compute_mean_wind_speed(stub, height=100)

    assert "mean_wind_speed" in stub.data.data_vars
    mws = stub.data["mean_wind_speed"]
    assert 100 in mws.coords["height"].values

    # Get sig from the height=100 slice
    slice_100 = mws.sel(height=100)
    sig1 = assess._get_sig(slice_100)
    assert sig1 is not None
    assert len(sig1) == 64

    # Second call (should skip)
    stub.load_weibull_parameters.reset_mock()
    assess.compute_mean_wind_speed(stub, height=100)

    # Sig should be unchanged
    mws2 = stub.data["mean_wind_speed"]
    slice_100_v2 = mws2.sel(height=100)
    sig2 = assess._get_sig(slice_100_v2)
    assert sig2 == sig1

    # load_weibull_parameters should NOT have been called again
    stub.load_weibull_parameters.assert_not_called()


def test_mean_wind_speed_invalidation_different_height(tmp_path):
    """
    Calling compute_mean_wind_speed with different heights should produce different sigs.
    """
    stub = _make_wind_atlas_stub(tmp_path)

    # First height
    assess.compute_mean_wind_speed(stub, height=50)
    sig_50 = assess._get_sig(stub.data["mean_wind_speed"].sel(height=50))
    assert sig_50 is not None

    # Second height
    assess.compute_mean_wind_speed(stub, height=100)
    sig_100 = assess._get_sig(stub.data["mean_wind_speed"].sel(height=100))
    assert sig_100 is not None

    # Sigs must differ (height is in params)
    assert sig_50 != sig_100


def test_mean_wind_speed_unsigned_recompute(tmp_path):
    """
    If cleo:* attrs are removed from existing var, next call should recompute.
    """
    stub = _make_wind_atlas_stub(tmp_path)

    # First call
    assess.compute_mean_wind_speed(stub, height=100)
    sig1 = assess._get_sig(stub.data["mean_wind_speed"].sel(height=100))
    assert sig1 is not None

    # Remove cleo:* attrs (modify in place for reliable clearing)
    mws = stub.data["mean_wind_speed"]
    for k in list(mws.attrs.keys()):
        if k.startswith("cleo:"):
            del mws.attrs[k]

    # Verify sig is now None
    assert assess._get_sig(stub.data["mean_wind_speed"].sel(height=100)) is None

    # Second call (should recompute)
    stub.load_weibull_parameters.reset_mock()
    assess.compute_mean_wind_speed(stub, height=100)

    # Sig should be restored
    sig2 = assess._get_sig(stub.data["mean_wind_speed"].sel(height=100))
    assert sig2 is not None
    assert sig2 == sig1

    # load_weibull_parameters should have been called (recompute happened)
    stub.load_weibull_parameters.assert_called()


def test_force_recompute(tmp_path):
    """
    force=True should always recompute even if sig matches.
    """
    stub = _make_wind_atlas_stub(tmp_path)

    # First call
    assess.compute_mean_wind_speed(stub, height=100)
    stub.load_weibull_parameters.reset_mock()

    # Second call with force=True
    assess.compute_mean_wind_speed(stub, height=100, force=True)

    # load_weibull_parameters should have been called (forced recompute)
    stub.load_weibull_parameters.assert_called()


def test_sig_helper_validation():
    """Test _get_sig validates signature format."""
    # Valid sig
    da = xr.DataArray([1, 2, 3])
    da.attrs["cleo:sig"] = "a" * 64
    assert assess._get_sig(da) == "a" * 64

    # Invalid: wrong length
    da.attrs["cleo:sig"] = "abc"
    assert assess._get_sig(da) is None

    # Invalid: not hex
    da.attrs["cleo:sig"] = "g" * 64
    assert assess._get_sig(da) is None

    # Missing
    del da.attrs["cleo:sig"]
    assert assess._get_sig(da) is None


def test_matches_helper():
    """Test _matches checks both presence and signature."""
    ds = xr.Dataset({"var1": xr.DataArray([1, 2, 3])})

    # No sig -> no match
    assert not assess._matches(ds, "var1", "any_sig")

    # Add valid sig
    ds["var1"].attrs["cleo:sig"] = "b" * 64
    assert assess._matches(ds, "var1", "b" * 64)
    assert not assess._matches(ds, "var1", "c" * 64)

    # Missing var
    assert not assess._matches(ds, "var2", "b" * 64)
