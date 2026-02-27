"""assess: test_weibull_pdf.

Oracles + contracts for Weibull PDF generation:
- analytic mass oracle via CDF
- finite-at-zero contract (u=0 -> 0 everywhere)
- dask-laziness contract (when inputs are dask-backed)
- vectorized formula spot-check on a small grid
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tests.helpers.oracles import assert_close, weibull_pdf as weibull_pdf_oracle, weibull_cdf as _weibull_cdf

from cleo.assess import weibull_probability_density


def test_weibull_pdf_integration_mass_matches_cdf_oracle() -> None:
    """
    Numerical integration of analytic Weibull PDF on [0, u_max] matches analytic CDF.
    """
    k = 2.0
    a = 8.0
    u_max = 5.0 * a
    u = np.linspace(0.0, u_max, 20001, dtype=np.float64)

    pdf = weibull_pdf_oracle(u, k=k, a=a)
    numerical_mass = float(np.trapezoid(pdf, x=u))
    oracle_mass = _weibull_cdf(u_max, k=k, a=a)

    assert abs(numerical_mass - oracle_mass) <= 5e-4


def test_weibull_probability_density_is_dask_lazy_when_inputs_are_dask() -> None:
    da = pytest.importorskip("dask.array")

    a = xr.DataArray(da.ones((10, 10), chunks=(5, 5)) * 7.0, dims=("y", "x"))
    k = xr.DataArray(da.ones((10, 10), chunks=(5, 5)) * 2.0, dims=("y", "x"))
    u = np.array([0.0, 5.0, 10.0, 15.0], dtype=float)

    pdf = weibull_probability_density(u, k, a)

    # Laziness contract: still dask-backed
    assert hasattr(pdf.data, "compute")
    # Shape/coord contract
    assert pdf.sizes["wind_speed"] == u.size
    assert np.array_equal(pdf.coords["wind_speed"].values, u)


@pytest.mark.parametrize(
    "k,a,u",
    [
        (0.5, 2.0, np.array([0.0, 1.0, 2.0, 3.0])),
        (1.0, 2.0, np.array([0.0, 1.0, 2.0])),
        (2.0, 8.0, np.array([0.0, 1.0, 5.0, 10.0])),
    ],
)
def test_weibull_probability_density_finite_and_zero_at_u0(k: float, a: float, u: np.ndarray) -> None:
    """
    Contract: PDF is finite everywhere and exactly 0 at wind_speed=0.0.

    Note: even for k=1 (exp distribution) where the analytic PDF at 0 is 1/a,
    the production contract here is "force 0 at u=0" to avoid singularities / infs.
    """
    weibull_k = xr.DataArray(np.full((2, 2), k, dtype=float), dims=("y", "x"))
    weibull_a = xr.DataArray(np.full((2, 2), a, dtype=float), dims=("y", "x"))

    pdf = weibull_probability_density(u, weibull_k, weibull_a)

    assert np.all(np.isfinite(pdf.values))
    assert float(pdf.sel(wind_speed=0.0).max()) == 0.0


def test_weibull_probability_density_integration_still_matches_cdf_oracle() -> None:
    k = 2.0
    a = 8.0
    u = np.linspace(0.0, 40.0, 201, dtype=float)

    weibull_k = xr.DataArray(np.full((2, 2), k, dtype=float), dims=("y", "x"))
    weibull_a = xr.DataArray(np.full((2, 2), a, dtype=float), dims=("y", "x"))

    pdf = weibull_probability_density(u, weibull_k, weibull_a)
    pdf_1d = pdf.isel(y=0, x=0).values

    numerical_mass = float(np.trapezoid(pdf_1d, x=u))
    oracle_mass = _weibull_cdf(float(u[-1]), k=k, a=a)

    assert abs(numerical_mass - oracle_mass) < 1e-3


def test_weibull_probability_density_vectorized_oracle_small_grid() -> None:
    a = xr.DataArray(
        np.array([[2.0, 4.0], [6.0, 8.0]], dtype=float), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]}
    )
    k = xr.DataArray(
        np.array([[2.0, 2.0], [2.0, 2.0]], dtype=float), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]}
    )
    u = np.array([0.0, 1.0, 2.0], dtype=float)

    pdf = weibull_probability_density(u, k, a)

    assert "wind_speed" in pdf.dims
    assert np.array_equal(pdf.coords["wind_speed"].values, u)

    # u=0 contract
    assert np.allclose(pdf.sel(wind_speed=0.0).values, 0.0)

    # Spot-check analytic formula at one cell and one wind speed
    u_val = 2.0
    a_val = 2.0
    k_val = 2.0
    expected = float(weibull_pdf_oracle(np.array([u_val]), k=k_val, a=a_val)[0])
    got = float(pdf.sel(wind_speed=u_val).isel(y=0, x=0).values)

    assert_close(got, expected, rtol=0.0, atol=1e-12)
