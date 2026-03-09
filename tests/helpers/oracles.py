from __future__ import annotations

from typing import Any

import numpy as np


__all__ = [
    # assertions
    "assert_close",
    # Weibull analytic oracles
    "weibull_pdf",
    "weibull_cdf",
    "weibull_tail",
    # wind / density helpers
    "density_speed_scale",
    # capacity-factor analytic oracles
    "cf_step",
    "cf_top_hat",
    "cf_step_density",
    "cf_top_hat_density",
]


# -----------------------------------------------------------------------------
# Assertions
# -----------------------------------------------------------------------------


def assert_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    msg: str | None = None,
) -> None:
    a = np.asarray(actual)
    e = np.asarray(expected)

    ok = np.allclose(a, e, rtol=rtol, atol=atol, equal_nan=True)
    if ok:
        return

    diff = np.abs(a - e)
    finite = np.isfinite(diff)
    max_abs = float(np.max(diff[finite])) if np.any(finite) else float("nan")

    denom = np.maximum(np.abs(e), np.abs(a))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = diff / np.where(denom == 0, 1.0, denom)
    finite_rel = np.isfinite(rel)
    max_rel = float(np.max(rel[finite_rel])) if np.any(finite_rel) else float("nan")

    detail = f"not close to oracle (max_abs={max_abs:g}, max_rel={max_rel:g}, rtol={rtol:g}, atol={atol:g})"
    raise AssertionError(detail if msg is None else f"{msg}: {detail}")


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


# -----------------------------------------------------------------------------
# Weibull distribution: analytic oracles
# -----------------------------------------------------------------------------


def weibull_cdf(u: Any, k: float, a: float) -> np.ndarray:
    r"""
    Weibull CDF:
      F(u)=0 for u<=0
      F(u)=1-exp(-(u/a)^k) for u>0
    """
    u = _as_float_array(u)
    out = np.zeros_like(u)

    mask = u > 0
    um = u[mask]
    out[mask] = 1.0 - np.exp(-((um / a) ** k))
    return out


def weibull_tail(u: Any, k: float, a: float) -> np.ndarray:
    r"""
    Weibull tail probability:
      P(U>=u)=1 for u<=0
      P(U>=u)=exp(-(u/a)^k) for u>0
    """
    u = _as_float_array(u)
    out = np.ones_like(u)

    mask = u > 0
    um = u[mask]
    out[mask] = np.exp(-((um / a) ** k))
    return out


def weibull_pdf(u: Any, k: float, a: float) -> np.ndarray:
    r"""
    Weibull PDF (cleo-friendly at u<=0):
      f(u)=0 for u<=0
      f(u)=(k/a)*(u/a)^(k-1)*exp(-(u/a)^k) for u>0
    """
    u = _as_float_array(u)
    out = np.zeros_like(u)

    mask = u > 0
    um = u[mask]
    out[mask] = (k / a) * (um / a) ** (k - 1.0) * np.exp(-((um / a) ** k))
    return out


# -----------------------------------------------------------------------------
# Air-density correction helpers and CF oracles
# -----------------------------------------------------------------------------


def density_speed_scale(rho: Any, *, rho0: float = 1.225) -> np.ndarray:
    r"""
    Density-to-speed scale:
      c = (rho/rho0)^(1/3)
    """
    rho = _as_float_array(rho)
    if rho0 <= 0:
        raise ValueError("rho0 must be positive.")
    if np.any(rho <= 0):
        raise ValueError("rho must be positive everywhere for density scaling.")
    return (rho / float(rho0)) ** (1.0 / 3.0)


def cf_step(u0: Any, k: float, a: float) -> np.ndarray:
    r"""
    Step curve CF oracle:
      CF = P(U>=u0) = exp(-(u0/a)^k) (for u0>0, else 1)
    """
    return weibull_tail(u0, k, a)


def cf_top_hat(u_in: Any, u_out: Any, k: float, a: float) -> np.ndarray:
    r"""
    Top-hat curve CF oracle:
      CF = P(u_in <= U < u_out) = F(u_out) - F(u_in)
    """
    u_in = _as_float_array(u_in)
    u_out = _as_float_array(u_out)
    return weibull_cdf(u_out, k, a) - weibull_cdf(u_in, k, a)


def cf_step_density(u0: Any, k: float, a: float, c: Any) -> np.ndarray:
    r"""
    Step CF with density correction inside lookup:
      u*c >= u0  <=>  u >= u0/c
      CF = P(U >= u0/c)
    """
    u0 = _as_float_array(u0)
    c = _as_float_array(c)
    if np.any(c <= 0):
        raise ValueError("c must be positive everywhere.")
    return weibull_tail(u0 / c, k, a)


def cf_top_hat_density(u_in: Any, u_out: Any, k: float, a: float, c: Any) -> np.ndarray:
    r"""
    Top-hat CF with density correction inside lookup:
      CF = F(u_out/c) - F(u_in/c)
    """
    u_in = _as_float_array(u_in)
    u_out = _as_float_array(u_out)
    c = _as_float_array(c)
    if np.any(c <= 0):
        raise ValueError("c must be positive everywhere.")
    return weibull_cdf(u_out / c, k, a) - weibull_cdf(u_in / c, k, a)
