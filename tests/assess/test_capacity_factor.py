"""assess: test_capacity_factor.

Beautiful contract/oracle tests for:
- dimension order invariance of weibull_pdf
- grid contracts (>=2 points, matching u/p lengths, exact wind_speed labels)
- label-based alignment (NOT positional)
- shear scaling behavior
- analytic oracle for step curve
- scaling/rated-power invariant + compute_optimal_power_energy integration
- zero-shear reduces to direct integral
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pytest
import xarray as xr

from cleo.assess import capacity_factor, compute_optimal_power_energy
from tests.helpers.factories import da_xy
from tests.helpers.oracles import assert_close


# ----------------------------
# Small local helpers (module-private)
# ----------------------------

def _as_1d(a: Iterable[float]) -> np.ndarray:
    return np.asarray(list(a), dtype=float)


def _power_curve_linear(u: np.ndarray) -> np.ndarray:
    """Monotone linear power curve in [0, 1] on u ∈ [0, u.max()]."""
    u = np.asarray(u, dtype=float)
    umax = float(u.max()) if u.size else 1.0
    if umax == 0:
        return np.zeros_like(u)
    return np.clip(u / umax, 0.0, 1.0)


def _pdf_from_1d(pdf_1d: np.ndarray, *, y: int = 1, x: int = 1) -> xr.DataArray:
    """Broadcast a 1D wind_speed pdf to (wind_speed, y, x)."""
    pdf_1d = np.asarray(pdf_1d, dtype=float)
    data = np.broadcast_to(pdf_1d[:, None, None], (pdf_1d.size, y, x)).copy()
    return xr.DataArray(data, dims=("wind_speed", "y", "x"), coords={"y": np.arange(y), "x": np.arange(x)})


def _wind_shear_grid(*, ny: int, nx: int, alpha: float, name: str = "wind_shear") -> xr.DataArray:
    """
    Create a wind_shear DataArray with dims (y, x) and shape (ny, nx),
    using the tests' factory conventions (coords 0..n-1).
    """
    da = da_xy(n=max(ny, nx), name=name).astype(float)
    da[:] = alpha
    return da.isel(y=slice(0, ny), x=slice(0, nx))


# ----------------------------
# Dimension order invariance
# ----------------------------

@pytest.mark.parametrize(
    "dim_order",
    [
        ("wind_speed", "y", "x"),
        ("y", "x", "wind_speed"),
        ("y", "wind_speed", "x"),
    ],
)
def test_capacity_factor_accepts_weibull_pdf_any_dim_order(dim_order: tuple[str, str, str]) -> None:
    u = _as_1d([0.0, 5.0, 10.0, 15.0, 20.0])
    p = _as_1d([0.0, 0.2, 0.6, 0.9, 1.0])

    # Deterministic PDF (not random): any positive values are fine for this contract.
    pdf_standard = xr.DataArray(
        np.full((u.size, 2, 3), 0.1, dtype=float),
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": u, "y": [0.0, 1.0], "x": [0.0, 1.0, 2.0]},
    )
    pdf = pdf_standard.transpose(*dim_order)

    wind_shear = xr.DataArray(
        np.full((2, 3), 0.14, dtype=float),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0, 2.0]},
    )

    out = capacity_factor(
        weibull_pdf=pdf,
        wind_shear=wind_shear,
        u_power_curve=u,
        p_power_curve=p,
        h_turbine=100.0,
        h_reference=100.0,
    )

    assert out.shape == wind_shear.shape
    assert np.all(np.isfinite(out.values))


def test_capacity_factor_same_result_regardless_of_dim_order() -> None:
    u = _as_1d([0.0, 5.0, 10.0, 15.0, 20.0])
    p = _as_1d([0.0, 0.2, 0.6, 0.9, 1.0])

    # Use deterministic values (avoid RNG entirely).
    pdf_data = np.arange(u.size * 2 * 3, dtype=float).reshape((u.size, 2, 3))
    pdf_data = 0.001 + (pdf_data / pdf_data.max()) * 0.1  # strictly positive, bounded

    pdf_ws_first = xr.DataArray(
        pdf_data,
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": u, "y": [0.0, 1.0], "x": [0.0, 1.0, 2.0]},
    )
    pdf_ws_last = pdf_ws_first.transpose("y", "x", "wind_speed")

    wind_shear = xr.DataArray(
        np.full((2, 3), 0.14, dtype=float),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0, 2.0]},
    )

    cf1 = capacity_factor(pdf_ws_first, wind_shear, u, p, h_turbine=100.0, h_reference=100.0)
    cf2 = capacity_factor(pdf_ws_last, wind_shear, u, p, h_turbine=100.0, h_reference=100.0)

    np.testing.assert_allclose(cf1.values, cf2.values, rtol=1e-12, atol=0.0)


# ----------------------------
# Grid contracts: >=2 points, matching lengths
# ----------------------------

@pytest.mark.parametrize("n_wind_speeds", [0, 1])
def test_capacity_factor_requires_at_least_two_points(n_wind_speeds: int) -> None:
    u = np.linspace(0.0, 20.0, n_wind_speeds) if n_wind_speeds else np.array([], dtype=float)
    p = np.full_like(u, 0.5, dtype=float)

    if n_wind_speeds:
        pdf = xr.DataArray(
            np.full((n_wind_speeds, 2, 2), 0.1, dtype=float),
            dims=("wind_speed", "y", "x"),
            coords={"wind_speed": u, "y": [0.0, 1.0], "x": [0.0, 1.0]},
        )
    else:
        # Even if a caller passes a pdf without wind_speed dim, the function
        # should still reject on the curve contract.
        pdf = da_xy(n=2, name="pdf").astype(float)

    wind_shear = da_xy(n=2, name="wind_shear").astype(float)
    wind_shear[:] = 0.14

    with pytest.raises(ValueError, match="at least 2 points"):
        capacity_factor(pdf, wind_shear, u, p, h_turbine=100.0, h_reference=100.0)


def test_capacity_factor_two_points_works() -> None:
    u = _as_1d([0.0, 20.0])
    p = _as_1d([0.0, 1.0])

    pdf = xr.DataArray(
        np.full((2, 2, 2), 0.1, dtype=float),
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": u, "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    wind_shear = da_xy(n=2, name="wind_shear").astype(float)
    wind_shear[:] = 0.14

    out = capacity_factor(pdf, wind_shear, u, p, h_turbine=100.0, h_reference=100.0)
    assert out.shape == wind_shear.shape
    assert np.all(np.isfinite(out.values))


def test_capacity_factor_mismatched_curve_lengths_raises() -> None:
    u = _as_1d([0.0, 5.0, 10.0])
    p = _as_1d([0.0, 0.5])  # mismatch

    pdf = xr.DataArray(
        np.full((3, 2, 2), 0.1, dtype=float),
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": u, "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    wind_shear = da_xy(n=2, name="wind_shear").astype(float)

    with pytest.raises(ValueError, match="length"):
        capacity_factor(pdf, wind_shear, u, p, h_turbine=100.0, h_reference=100.0)


# ----------------------------
# Label alignment (NOT positional)
# ----------------------------

def test_capacity_factor_aligns_pdf_by_windspeed_label_not_position() -> None:
    u_asc = _as_1d([0.0, 5.0, 10.0, 15.0, 20.0])
    u_desc = u_asc[::-1].copy()

    p = u_asc / 20.0  # [0, .25, .5, .75, 1.0]

    def asymmetric_pdf(u: np.ndarray) -> np.ndarray:
        # integrates to 1 on [0, 12] if treated continuously; here it is an oracle-shape only
        return np.where(u < 12.0, 1.0 / 12.0, 0.0)

    pdf_desc = asymmetric_pdf(u_desc)  # values aligned with coords in descending order

    weibull_pdf = xr.DataArray(
        pdf_desc[:, None, None],
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": u_desc, "y": [0], "x": [0]},
    )

    wind_shear = da_xy(n=1, name="wind_shear").astype(float)  # s=1 regardless if alpha=0
    wind_shear[:] = 0.0

    # Oracle: explicitly align by label
    pdf_aligned = np.array([weibull_pdf.sel(wind_speed=uu).item() for uu in u_asc], dtype=float)
    integrand = pdf_aligned * p
    cf_oracle = float(np.trapezoid(integrand, x=u_asc))

    out = capacity_factor(
        weibull_pdf,
        wind_shear,
        u_asc,
        p,
        h_turbine=100.0,
        h_reference=100.0,
        correction_factor=1.0,
    )
    got = float(out.values.flat[0])

    assert np.isclose(got, cf_oracle, atol=1e-12, rtol=0.0), (
        f"Expected label-aligned CF {cf_oracle}, got {got}. "
        "If this fails, capacity_factor likely uses positional indexing (BUG)."
    )


def test_capacity_factor_allows_reordering_but_requires_exact_labels() -> None:
    pdf = xr.DataArray(
        np.ones((3, 1, 1), dtype=float),
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": np.array([2.0, 1.0, 0.0]), "y": [0], "x": [0]},  # descending labels
    )
    wind_shear = da_xy(n=1, name="wind_shear").astype(float)
    wind_shear[:] = 0.0

    u = _as_1d([0.0, 1.0, 2.0])
    p = _as_1d([0.0, 0.5, 1.0])

    out = capacity_factor(pdf, wind_shear, u, p, h_turbine=100.0, h_reference=100.0)
    assert out.shape == (1, 1)

    u_missing = _as_1d([0.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="wind_speed grid mismatch"):
        capacity_factor(pdf, wind_shear, u_missing, p, h_turbine=100.0, h_reference=100.0)


# ----------------------------
# Shear scaling contracts
# ----------------------------

def test_capacity_factor_shear_scaling_matches_manual_interp_oracle() -> None:
    u = _as_1d([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    pdf = xr.DataArray(
        np.full((u.size, 1, 1), 0.2, dtype=float),
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": u, "y": [0.0], "x": [0.0]},
    )

    wind_shear = da_xy(n=1, name="wind_shear").astype(float)
    wind_shear[:] = 1.0  # alpha=1

    p = np.minimum((u / 5.0) ** 2, 1.0)

    h_turbine = 200.0
    h_reference = 100.0
    s = (h_turbine / h_reference) ** 1.0  # 2

    out = capacity_factor(pdf, wind_shear, u, p, h_turbine=h_turbine, h_reference=h_reference)
    got = float(out.values.flat[0])

    u_scaled = u * s
    p_scaled = np.interp(u_scaled, u, p, left=0.0, right=0.0)
    integrand = 0.2 * p_scaled
    expected = float(np.trapezoid(integrand, x=u))

    assert np.isclose(got, expected, rtol=1e-12, atol=0.0)


def test_capacity_factor_zero_shear_reduces_to_direct_integral() -> None:
    k = 2.0
    a = 8.0

    u = np.linspace(0.0, 25.0, 501)
    p = _power_curve_linear(u)

    # analytic Weibull PDF (not normalized over truncated range; but oracle compares same calc)
    u_over_a = u / a
    pdf_1d = (k / a) * (u_over_a ** (k - 1)) * np.exp(-(u_over_a ** k))

    weibull_pdf = xr.DataArray(
        np.broadcast_to(pdf_1d[:, None, None], (u.size, 2, 3)).copy(),
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": u, "y": [0, 1], "x": [0, 1, 2]},
    )

    wind_shear = da_xy(n=2, name="wind_shear").astype(float)
    wind_shear[:] = 0.0
    wind_shear = wind_shear.broadcast_like(weibull_pdf.isel(wind_speed=0, drop=True))

    expected = np.trapezoid(weibull_pdf.values * p[:, None, None], x=u, axis=0)

    out = capacity_factor(
        weibull_pdf,
        wind_shear,
        u,
        p,
        h_turbine=100.0,
        h_reference=100.0,
        correction_factor=1.0,
    )

    assert out.dims == ("y", "x")
    assert out.name == "capacity_factor"
    np.testing.assert_allclose(out.values, expected, rtol=0.0, atol=1e-6)


# ----------------------------
# Analytic step-curve oracle (Weibull CDF)
# ----------------------------

def test_capacity_factor_step_curve_with_shear_matches_weibull_cdf_oracle() -> None:
    k = 2.0
    a = 8.0

    u_max = 5 * a  # 40.0
    u = np.linspace(0.0, u_max, 20001)

    u0 = 5.0  # hub-height threshold

    alpha = 0.14
    h_turbine = 150.0
    h_reference = 100.0
    s = (h_turbine / h_reference) ** alpha
    assert s > 1.0

    p = np.where(u >= u0, 1.0, 0.0)

    wind_shear = _wind_shear_grid(ny=2, nx=3, alpha=alpha)

    u_over_a = u / a
    pdf_raw = (k / a) * (u_over_a ** (k - 1)) * np.exp(-(u_over_a ** k))
    pdf_raw[0] = 0.0  # finite at u=0 for k>1 contract (avoid any accidental inf)

    # normalize on [0, u_max] to enforce mass=1 (this matches the test's stated oracle)
    mass = float(np.trapezoid(pdf_raw, x=u))
    pdf_1d = pdf_raw / mass
    assert np.isclose(np.trapezoid(pdf_1d, x=u), 1.0, atol=1e-10)

    weibull_pdf = xr.DataArray(
        np.broadcast_to(pdf_1d[:, None, None], (u.size, 2, 3)).copy(),
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": u, "y": [0, 1], "x": [0, 1, 2]},
    )

    def weibull_cdf(z: float) -> float:
        return float(1.0 - np.exp(-((z / a) ** k)))

    f_umax = weibull_cdf(u_max)
    f_u0_over_s = weibull_cdf(u0 / s)
    cf_expected = (f_umax - f_u0_over_s) / f_umax
    assert 0.0 < cf_expected < 1.0

    out = capacity_factor(
        weibull_pdf,
        wind_shear,
        u,
        p,
        h_turbine=h_turbine,
        h_reference=h_reference,
        correction_factor=1.0,
    )

    assert out.dims == ("y", "x")
    assert out.name == "capacity_factor"
    np.testing.assert_allclose(out.values, cf_expected, rtol=0.0, atol=5e-4)


# ----------------------------
# Scaling invariance + compute_optimal_power_energy integration
# ----------------------------

def test_capacity_factor_always_rated_power_curve_yields_cf_one_and_optimal_power_equals_rated() -> None:
    """
    Oracle:
      If p(u) == 1 for all u and pdf is normalized, then CF == 1.
      compute_optimal_power_energy should then pick rated power and compute energy = P * 8766 / 1e6.
    """
    k = 2.0
    a = 8.0

    u = np.linspace(0.0, 25.0, 501)
    p = np.ones_like(u, dtype=float)

    u_over_a = u / a
    pdf_1d = (k / a) * (u_over_a ** (k - 1)) * np.exp(-(u_over_a ** k))
    pdf_1d = pdf_1d / float(np.trapezoid(pdf_1d, x=u))
    assert np.isclose(np.trapezoid(pdf_1d, x=u), 1.0, atol=1e-9)

    weibull_pdf = xr.DataArray(
        pdf_1d[:, None, None].copy(),
        dims=("wind_speed", "y", "x"),
        coords={"wind_speed": u, "y": [0], "x": [0]},
    )

    wind_shear = da_xy(n=1, name="wind_shear").astype(float)
    wind_shear[:] = 0.0

    cf = capacity_factor(
        weibull_pdf,
        wind_shear,
        u,
        p,
        h_turbine=100.0,
        h_reference=100.0,
        correction_factor=1.0,
    )
    assert_close(cf.values, np.array([[1.0]]), rtol=0.0, atol=1e-6)

    rated_power_kw = 1234.0
    turbine = f"TestTurbine.{int(rated_power_kw)}"

    @dataclass
    class Dummy:
        data: xr.Dataset

        def get_turbine_attribute(self, _turbine: str, attr: str):
            if attr == "capacity":
                return rated_power_kw
            return None

        def _set_var(self, name, da):
            """Simple mock for _set_var - just assigns directly."""
            self.data[name] = da

    lcoe = xr.DataArray(
        [[[50.0]]],
        dims=("turbine", "y", "x"),
        coords={"turbine": [turbine], "y": [0], "x": [0]},
    )
    capacity_factors = cf.expand_dims(turbine=[turbine])

    self = Dummy(
        data=xr.Dataset(
            {
                "lcoe": lcoe,
                "capacity_factors": capacity_factors,
            }
        )
    )

    compute_optimal_power_energy(self)

    got_power = float(self.data["optimal_power"].values.item())
    assert np.isclose(got_power, rated_power_kw, rtol=0.0, atol=1e-6)

    got_energy = float(self.data["optimal_energy"].values.item())
    expected_energy = 1.0 * rated_power_kw * 8766.0 / 1e6
    assert np.isclose(got_energy, expected_energy, rtol=0.0, atol=1e-6)
