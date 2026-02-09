"""test_dask_level1.py

Minimal regression tests for Level-1 dask support:
- Verify compute_chunked is bypassed when inputs are dask-backed
- Verify outputs remain dask-lazy (not eagerly computed)
"""

from __future__ import annotations

from tests.helpers.optional import requires_dask

requires_dask()

import numpy as np
import pytest
import xarray as xr
import dask.array as da


class TestDaskLevel1:
    """Tests for dask-backed array handling in cleo.assess."""

    def test_is_dask_backed_detects_dask_arrays(self):
        """Verify _is_dask_backed correctly identifies dask-backed arrays."""
        from cleo.assess import _is_dask_backed

        # Create numpy-backed DataArray
        np_arr = xr.DataArray(np.ones((4, 4)), dims=("y", "x"))
        assert not _is_dask_backed(np_arr), "numpy array should not be detected as dask"

        # Create dask-backed DataArray
        dask_arr = np_arr.chunk({"y": 2, "x": 2})
        assert _is_dask_backed(dask_arr), "dask-chunked array should be detected as dask"

    def test_weibull_probability_density_stays_lazy(self):
        """Verify weibull_probability_density returns dask-backed output for dask inputs."""
        from cleo.assess import weibull_probability_density, _is_dask_backed

        # Create dask-backed Weibull parameters
        weibull_a = xr.DataArray(
            da.ones((8, 8), chunks=(4, 4)) * 8.0,
            dims=("y", "x"),
            name="weibull_a",
        )
        weibull_k = xr.DataArray(
            da.ones((8, 8), chunks=(4, 4)) * 2.0,
            dims=("y", "x"),
            name="weibull_k",
        )
        u = np.arange(0.0, 25.5, 0.5)

        # Call function
        pdf = weibull_probability_density(u, weibull_k, weibull_a)

        # Verify output is dask-backed (lazy)
        assert _is_dask_backed(pdf), "Output should remain dask-backed"
        assert "wind_speed" in pdf.dims, "Output should have wind_speed dim"
        assert pdf.dims == ("wind_speed", "y", "x"), f"Unexpected dims: {pdf.dims}"

    def test_capacity_factor_stays_lazy(self):
        """Verify capacity_factor returns dask-backed output for dask inputs."""
        from cleo.assess import capacity_factor, _is_dask_backed

        # Create dask-backed inputs
        u = np.arange(0.0, 25.5, 0.5)
        weibull_pdf = xr.DataArray(
            da.ones((len(u), 8, 8), chunks=(len(u), 4, 4)) * 0.01,
            dims=("wind_speed", "y", "x"),
            coords={"wind_speed": u},
            name="weibull_pdf",
        )
        wind_shear = xr.DataArray(
            da.zeros((8, 8), chunks=(4, 4)),
            dims=("y", "x"),
            name="wind_shear",
        )
        p_power_curve = np.where(u >= 4.0, np.minimum((u - 4) / 10, 1.0), 0.0)

        # Call function
        cf = capacity_factor(
            weibull_pdf=weibull_pdf,
            wind_shear=wind_shear,
            u_power_curve=u,
            p_power_curve=p_power_curve,
            h_turbine=100,
            h_reference=100,
            correction_factor=1.0,
        )

        # Verify output is dask-backed (lazy)
        assert _is_dask_backed(cf), "Output should remain dask-backed"
        assert cf.dims == ("y", "x"), f"Unexpected dims: {cf.dims}"

    def test_compute_chunked_bypassed_for_dask_inputs(self, monkeypatch):
        """
        Verify compute_chunked is NOT called when inputs are dask-backed.

        Monkeypatches compute_chunked to raise RuntimeError, then calls
        weibull_probability_density wrapper to ensure it takes the direct path.
        """
        from cleo.assess import weibull_probability_density, _is_dask_backed
        import cleo.assess

        # Monkeypatch compute_chunked to raise if called
        def raise_if_called(*args, **kwargs):
            raise RuntimeError("compute_chunked should NOT be called for dask inputs")

        monkeypatch.setattr(cleo.assess, "compute_chunked", raise_if_called)

        # Create dask-backed inputs
        weibull_a = xr.DataArray(
            da.ones((8, 8), chunks=(4, 4)) * 8.0,
            dims=("y", "x"),
        )
        weibull_k = xr.DataArray(
            da.ones((8, 8), chunks=(4, 4)) * 2.0,
            dims=("y", "x"),
        )
        u = np.arange(0.0, 25.5, 0.5)

        # This should NOT raise because dask path bypasses compute_chunked
        pdf = weibull_probability_density(u, weibull_k, weibull_a)

        # Verify it's still dask-backed
        assert _is_dask_backed(pdf), "Output should be dask-backed"

    def test_trapz_integration_stays_lazy(self):
        """Verify _trapz_over_wind_speed keeps output dask-backed."""
        from cleo.assess import _trapz_over_wind_speed, _is_dask_backed

        u = np.arange(0.0, 10.5, 0.5)
        u_da = xr.DataArray(u, dims=("wind_speed",), coords={"wind_speed": u})

        # Create dask-backed integrand
        integrand = xr.DataArray(
            da.ones((len(u), 4, 4), chunks=(len(u), 2, 2)),
            dims=("wind_speed", "y", "x"),
            coords={"wind_speed": u},
        )

        # Integrate
        result = _trapz_over_wind_speed(integrand, u_da)

        # Verify output is dask-backed and wind_speed dim is removed
        assert _is_dask_backed(result), "Output should remain dask-backed"
        assert "wind_speed" not in result.dims, "wind_speed should be integrated out"
        assert result.dims == ("y", "x"), f"Unexpected dims: {result.dims}"
