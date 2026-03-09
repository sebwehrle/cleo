"""spatial: test_validation_policy.

Tests for dask-safe validation policy helpers:
- _validate_values behavior under auto/full/probe/none modes
- Template-absent safety (no exceptions, structural checks still pass)
- Probe-based detection of invalid values at probe points
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cleo.spatial import (
    _is_dask_backed,
    _get_probe_points,
    _probe_scalars,
    _validate_values,
)


class TestIsDaskBacked:
    """Tests for _is_dask_backed helper."""

    def test_numpy_array_not_dask(self):
        """Numpy-backed DataArray is not dask-backed."""
        da = xr.DataArray(np.ones((3, 3)), dims=("y", "x"))
        assert not _is_dask_backed(da)

    def test_dask_array_is_dask(self):
        """Dask-backed DataArray is detected correctly."""
        dask = pytest.importorskip("dask.array")
        darr = dask.from_array(np.ones((3, 3)), chunks=2)
        da = xr.DataArray(darr, dims=("y", "x"))
        assert _is_dask_backed(da)


class TestGetProbePoints:
    """Tests for _get_probe_points helper."""

    def test_returns_empty_when_template_absent(self):
        """Returns empty list when template is not in dataset."""
        ds = xr.Dataset({"foo": xr.DataArray(np.ones((3, 3)), dims=("y", "x"))})
        pts = _get_probe_points(ds, n=4)
        assert pts == []

    def test_returns_points_from_valid_template(self):
        """Returns non-empty list when template has valid (finite) interior."""
        y = np.arange(10.0)
        x = np.arange(10.0)
        template = xr.DataArray(
            np.ones((10, 10), dtype=float),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name="template",
        )
        ds = xr.Dataset({"template": template})
        pts = _get_probe_points(ds, n=4)
        assert len(pts) == 4
        # All points should be tuples of (y, x)
        for pt in pts:
            assert len(pt) == 2

    def test_handles_template_with_nan_edges(self):
        """Template with NaN edges still returns valid interior points."""
        y = np.arange(10.0)
        x = np.arange(10.0)
        data = np.ones((10, 10), dtype=float)
        # Set edges to NaN
        data[0, :] = np.nan
        data[-1, :] = np.nan
        data[:, 0] = np.nan
        data[:, -1] = np.nan

        template = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x}, name="template")
        ds = xr.Dataset({"template": template})
        pts = _get_probe_points(ds, n=4)
        # Should still find valid interior points
        assert len(pts) == 4

    def test_caches_in_attrs(self):
        """Probe points are cached in ds.attrs['cleo_probe_points']."""
        y = np.arange(5.0)
        x = np.arange(5.0)
        template = xr.DataArray(
            np.ones((5, 5), dtype=float),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name="template",
        )
        ds = xr.Dataset({"template": template})
        assert "cleo_probe_points" not in ds.attrs

        pts = _get_probe_points(ds, n=3)
        assert "cleo_probe_points" in ds.attrs
        assert ds.attrs["cleo_probe_points"] == pts

        # Second call should return cached value
        pts2 = _get_probe_points(ds, n=3)
        assert pts2 == pts


class TestProbeScalars:
    """Tests for _probe_scalars helper."""

    def test_extracts_values_at_points(self):
        """Extracts scalar values at specified probe points."""
        y = np.arange(5.0)
        x = np.arange(5.0)
        data = np.arange(25.0).reshape((5, 5))
        da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})

        # pts are (x, y) tuples
        pts = [(1.0, 2.0), (3.0, 4.0)]
        vals = _probe_scalars(da, pts)

        assert len(vals) == 2
        assert vals[0] == data[2, 1]  # x=1, y=2 -> data[y, x] = data[2, 1]
        assert vals[1] == data[4, 3]  # x=3, y=4 -> data[y, x] = data[4, 3]

    def test_returns_empty_for_empty_points(self):
        """Returns empty array for empty probe points list."""
        da = xr.DataArray(np.ones((3, 3)), dims=("y", "x"))
        vals = _probe_scalars(da, [])
        assert len(vals) == 0

    def test_works_with_dask_backed(self):
        """Works correctly with dask-backed DataArrays."""
        dask = pytest.importorskip("dask.array")

        y = np.arange(5.0)
        x = np.arange(5.0)
        data = np.arange(25.0).reshape((5, 5))
        darr = dask.from_array(data, chunks=3)
        da = xr.DataArray(darr, dims=("y", "x"), coords={"y": y, "x": x})

        pts = [(2.0, 2.0)]
        vals = _probe_scalars(da, pts)

        assert len(vals) == 1
        assert vals[0] == data[2, 2]


class TestValidateValues:
    """Tests for _validate_values policy-driven validation."""

    def _make_ds_with_template(self, shape=(10, 10), nan_edges: bool = False) -> xr.Dataset:
        """Create dataset with template."""
        y = np.arange(shape[0], dtype=float)
        x = np.arange(shape[1], dtype=float)
        data = np.ones(shape, dtype=float)
        if nan_edges:
            data[0, :] = np.nan
            data[-1, :] = np.nan
            data[:, 0] = np.nan
            data[:, -1] = np.nan
        template = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x}, name="template")
        return xr.Dataset({"template": template})

    def test_validation_none_skips_checks(self):
        """validation='none' skips all value checks."""
        ds = self._make_ds_with_template()
        y = ds.coords["y"].values
        x = ds.coords["x"].values
        # Create DataArray with negative values (would fail check_positive)
        da = xr.DataArray(np.full((10, 10), -5.0), dims=("y", "x"), coords={"y": y, "x": x})

        # Should NOT raise even with check_positive=True
        _validate_values(da, ds, validation="none", check_positive=True, context="test")

    def test_invalid_validation_mode_raises(self):
        """Unsupported validation mode should raise explicitly."""
        ds = self._make_ds_with_template()
        y = ds.coords["y"].values
        x = ds.coords["x"].values
        da = xr.DataArray(np.ones((10, 10)), dims=("y", "x"), coords={"y": y, "x": x})

        with pytest.raises(ValueError, match="validation must be one of"):
            _validate_values(da, ds, validation="bogus", context="test")  # type: ignore[arg-type]

    def test_validation_full_checks_all_values(self):
        """validation='full' checks all values in array."""
        ds = self._make_ds_with_template()
        y = ds.coords["y"].values
        x = ds.coords["x"].values

        # Create valid DataArray
        da_valid = xr.DataArray(np.ones((10, 10)), dims=("y", "x"), coords={"y": y, "x": x})
        _validate_values(da_valid, ds, validation="full", check_positive=True, context="test")

        # Create DataArray with one negative value
        data = np.ones((10, 10))
        data[5, 5] = -1.0
        da_invalid = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})

        with pytest.raises(ValueError, match="must be > 0"):
            _validate_values(da_invalid, ds, validation="full", check_positive=True, context="test")

    def test_validation_probe_checks_probe_points_only(self):
        """validation='probe' only checks values at probe points."""
        ds = self._make_ds_with_template(shape=(20, 20), nan_edges=True)
        y = ds.coords["y"].values
        x = ds.coords["x"].values

        # Get probe points
        pts = _get_probe_points(ds, n=4)
        assert len(pts) == 4

        # Create DataArray with valid values at probe points but invalid elsewhere
        data = np.full((20, 20), 1.0)  # Valid everywhere initially
        # Set a non-probe point to invalid value
        data[0, 0] = -1.0  # Edge (NaN in template, not a probe point)
        da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})

        # Should pass because probe points are valid
        _validate_values(da, ds, validation="probe", check_positive=True, context="test")

    def test_validation_probe_detects_invalid_at_probe_point(self):
        """validation='probe' raises when probe point has invalid value."""
        ds = self._make_ds_with_template(shape=(20, 20))
        y = ds.coords["y"].values
        x = ds.coords["x"].values

        # Get probe points
        pts = _get_probe_points(ds, n=4)
        assert len(pts) > 0

        # Create DataArray with invalid value at first probe point
        data = np.ones((20, 20))
        py, px = pts[0]
        data[int(py), int(px)] = -1.0  # Invalid at probe point

        da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})

        with pytest.raises(ValueError, match="must be > 0"):
            _validate_values(da, ds, validation="probe", check_positive=True, context="test")

    def test_validation_auto_uses_probe_for_dask(self):
        """validation='auto' uses probe mode for dask-backed arrays."""
        dask = pytest.importorskip("dask.array")

        ds = self._make_ds_with_template(shape=(20, 20))
        y = ds.coords["y"].values
        x = ds.coords["x"].values

        # Create dask-backed DataArray with valid values
        darr = dask.from_array(np.ones((20, 20)), chunks=5)
        da = xr.DataArray(darr, dims=("y", "x"), coords={"y": y, "x": x})

        # Should NOT materialize the full array
        _validate_values(da, ds, validation="auto", check_positive=True, context="test")

        # Output should still be dask-backed (validation didn't compute it)
        assert _is_dask_backed(da)

    def test_validation_auto_uses_full_for_numpy(self):
        """validation='auto' uses full mode for numpy-backed arrays."""
        ds = self._make_ds_with_template(shape=(10, 10))
        y = ds.coords["y"].values
        x = ds.coords["x"].values

        # Create numpy-backed DataArray with one invalid value
        data = np.ones((10, 10))
        data[5, 5] = -1.0
        da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})

        # Should detect invalid value (full check)
        with pytest.raises(ValueError, match="must be > 0"):
            _validate_values(da, ds, validation="auto", check_positive=True, context="test")

    def test_template_absent_skips_probe(self):
        """When template is absent, probe mode skips value checks gracefully."""
        ds = xr.Dataset({"foo": xr.DataArray(np.ones((5, 5)), dims=("y", "x"))})

        da = xr.DataArray(
            np.full((5, 5), -1.0),  # Invalid values
            dims=("y", "x"),
        )

        # Should NOT raise - probe mode skips when template absent
        _validate_values(da, ds, validation="probe", check_positive=True, context="test")

    def test_check_range_validation(self):
        """check_range validates values are within specified bounds."""
        ds = self._make_ds_with_template(shape=(5, 5))
        y = ds.coords["y"].values
        x = ds.coords["x"].values

        # Valid range
        da_valid = xr.DataArray(np.full((5, 5), 1.0), dims=("y", "x"), coords={"y": y, "x": x})
        _validate_values(da_valid, ds, validation="full", check_range=(0.5, 2.0), context="test")

        # Out of range (too high)
        da_high = xr.DataArray(np.full((5, 5), 5.0), dims=("y", "x"), coords={"y": y, "x": x})
        with pytest.raises(ValueError, match="outside expected range"):
            _validate_values(da_high, ds, validation="full", check_range=(0.5, 2.0), context="test")

        # Out of range (too low)
        da_low = xr.DataArray(np.full((5, 5), 0.1), dims=("y", "x"), coords={"y": y, "x": x})
        with pytest.raises(ValueError, match="outside expected range"):
            _validate_values(da_low, ds, validation="full", check_range=(0.5, 2.0), context="test")

    def test_check_nan_validation(self):
        """check_nan validates that array contains no NaN values."""
        ds = self._make_ds_with_template(shape=(5, 5))
        y = ds.coords["y"].values
        x = ds.coords["x"].values

        # No NaN
        da_valid = xr.DataArray(np.ones((5, 5)), dims=("y", "x"), coords={"y": y, "x": x})
        _validate_values(da_valid, ds, validation="full", check_nan=True, context="test")

        # Has NaN
        data = np.ones((5, 5))
        data[2, 2] = np.nan
        da_nan = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})

        with pytest.raises(ValueError, match="NaN"):
            _validate_values(da_nan, ds, validation="full", check_nan=True, context="test")


class TestValidationWithDaskArrays:
    """Integration tests for validation with dask-backed arrays."""

    def test_dask_array_validation_does_not_compute_full(self):
        """Dask-backed arrays are NOT fully computed during validation."""
        dask = pytest.importorskip("dask.array")

        y = np.arange(100.0)
        x = np.arange(100.0)
        template = xr.DataArray(
            np.ones((100, 100), dtype=float),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name="template",
        )
        ds = xr.Dataset({"template": template})

        # Create large dask-backed array
        darr = dask.from_array(np.ones((100, 100)), chunks=10)
        da = xr.DataArray(darr, dims=("y", "x"), coords={"y": y, "x": x})

        # Validation should use probe mode (not compute full array)
        _validate_values(da, ds, validation="auto", check_positive=True, context="test")

        # Array should still be dask-backed
        assert _is_dask_backed(da)

    def test_probe_validation_catches_issues_at_sample_points(self):
        """Probe validation detects issues at sampled points."""
        dask = pytest.importorskip("dask.array")

        y = np.arange(50.0)
        x = np.arange(50.0)
        template = xr.DataArray(
            np.ones((50, 50), dtype=float),
            dims=("y", "x"),
            coords={"y": y, "x": x},
            name="template",
        )
        ds = xr.Dataset({"template": template})

        # Get probe points
        pts = _get_probe_points(ds, n=8)

        # Create data with invalid value at a probe point
        data = np.ones((50, 50)) * 1.0  # Valid everywhere
        if pts:
            py, px = pts[0]
            data[int(py), int(px)] = -1.0  # Invalid at first probe

        darr = dask.from_array(data, chunks=10)
        da = xr.DataArray(darr, dims=("y", "x"), coords={"y": y, "x": x})

        with pytest.raises(ValueError, match="must be > 0"):
            _validate_values(da, ds, validation="auto", check_positive=True, context="test")
