"""Vertical-policy schema and deterministic fingerprint helpers.

This module defines the v0 vertical REWS/CF policy defaults and canonical
hashing utilities used for manifest attrs and inputs_id fingerprinting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

HASH_ALGORITHM = "sha256"
HASH_SCHEMA_VERSION = "v1"


DEFAULT_VERTICAL_POLICY: dict[str, Any] = {
    "vertical_policy_version": "mu_cv_continuous_momentmatch_v1",
    "compat_policy": "new_everywhere",
    "cf_mode": "direct_cf_quadrature",
    "max_query_height_m": 350.0,
    "max_query_height_policy": "error",
    "min_query_height_policy": "error",
    "mu_interp": "log_linear_lnmu_lnz",
    "mu_extrap": "top_weighted_regression_powerlaw",
    "mu_extrap_heights_m": [50.0, 100.0, 150.0, 200.0],
    "mu_extrap_weights": [1.0, 2.0, 3.0, 4.0],
    "alpha_min": -0.05,
    "alpha_max": 0.35,
    "alpha_fallback": "none",
    "alpha_fallback_tile_basis": "fixed_pixel",
    "alpha_fallback_tile_shape": [256, 256],
    "alpha_fallback_min_count": 2000,
    "alpha_fallback_stat": "median",
    "alpha_fallback_constant": 0.14,
    "k_interp": "log_linear_lncv_lnz_then_invert",
    "k_extrap": "constant_at_200",
    "k_lut_min": 0.5,
    "k_lut_max": 10.0,
    "k_lut_size": 4096,
    "rho0": 1.225,
    "rho_interp": "log_linear",
    "rho_extrap_above_200": "constant_at_200",
    "rho_scale_height_m": 8000.0,
    "rotor_integrator": "gauss_legendre_with_chord_weight",
    "rotor_nodes_n": 12,
    "rotor_match_moments": [1, 3],
    "power_curve_tail_policy": "auto_append_zero_at_cutout",
    "cutout_source": "constant_default",
    "cutout_default_mps": 25.0,
    "tail_append_method": "linear_to_zero",
    "power_curve_interp_semantics": "piecewise_linear_left0_right0",
    "min_valid_pixels_global": 10000,
    "min_valid_pixels_per_turbine": 2000,
    "min_valid_pixels_per_bin": 2000,
    "insufficient_sample_policy": "fail",
    "hash_algorithm": HASH_ALGORITHM,
    "hash_schema_version": HASH_SCHEMA_VERSION,
}

_POLICY_KEYS = frozenset(DEFAULT_VERTICAL_POLICY.keys())


def _normalize_for_json(value: Any) -> Any:
    """Convert values to stable JSON-compatible builtins."""
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, list):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_json_dumps(value: Any) -> str:
    """Serialize value as canonical JSON (deterministic)."""
    normalized = _normalize_for_json(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_hex_from_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_hex_from_json(value: Any) -> str:
    """Return SHA-256 hex digest for canonical JSON serialization."""
    payload = canonical_json_dumps(value).encode("utf-8")
    return sha256_hex_from_bytes(payload)


def checksum_float64_le(values: np.ndarray) -> str:
    """Return SHA-256 checksum of little-endian float64 bytes."""
    arr = np.asarray(values, dtype="<f8")
    return sha256_hex_from_bytes(arr.tobytes(order="C"))


def resolve_vertical_policy(policy_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized vertical policy with defaults + validated overrides."""
    resolved = dict(DEFAULT_VERTICAL_POLICY)
    if policy_overrides is None:
        return resolved

    unknown = sorted(set(policy_overrides.keys()) - _POLICY_KEYS)
    if unknown:
        raise ValueError(f"Unknown vertical policy keys: {unknown}")

    for key, value in policy_overrides.items():
        resolved[key] = _normalize_for_json(value)
    return resolved


def policy_snapshot(policy_overrides: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Build canonical policy snapshot payload and checksum."""
    resolved = resolve_vertical_policy(policy_overrides)
    payload = canonical_json_dumps(resolved)
    return {"json": payload, "sha256": sha256_hex_from_bytes(payload.encode("utf-8"))}
