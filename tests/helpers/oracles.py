from __future__ import annotations

import numpy as np


def weibull_pdf(u: np.ndarray, *, k: float, a: float) -> np.ndarray:
    # analytic oracle
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    mask = u >= 0
    um = u[mask]
    out[mask] = (k / a) * (um / a) ** (k - 1) * np.exp(-(um / a) ** k)
    return out


def assert_close(actual, expected, *, rtol=1e-12, atol=1e-12):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
        raise AssertionError("Values are not close to oracle.")
