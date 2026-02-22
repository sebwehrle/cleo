"""Unit tests for vertical policy schema + deterministic fingerprint helpers."""

from __future__ import annotations

import numpy as np
import pytest

from cleo.unification.vertical_policy import (
    HASH_ALGORITHM,
    HASH_SCHEMA_VERSION,
    canonical_json_dumps,
    checksum_float64_le,
    policy_snapshot,
    resolve_vertical_policy,
)


def test_canonical_json_dumps_is_deterministic_for_key_order() -> None:
    a = {"b": 2, "a": [3, {"d": 4, "c": 5}]}
    b = {"a": [3, {"c": 5, "d": 4}], "b": 2}
    assert canonical_json_dumps(a) == canonical_json_dumps(b)


def test_checksum_float64_le_is_deterministic_and_value_sensitive() -> None:
    x = np.array([0.0, 1.5, 2.5], dtype=np.float32)
    y = np.array([0.0, 1.5, 2.5], dtype=np.float64)
    z = np.array([0.0, 1.5, 3.5], dtype=np.float64)

    assert checksum_float64_le(x) == checksum_float64_le(y)
    assert checksum_float64_le(y) != checksum_float64_le(z)


def test_resolve_vertical_policy_applies_known_override() -> None:
    policy = resolve_vertical_policy({"cf_mode": "momentmatch_weibull", "rotor_nodes_n": 16})
    assert policy["cf_mode"] == "momentmatch_weibull"
    assert policy["rotor_nodes_n"] == 16
    assert policy["hash_algorithm"] == HASH_ALGORITHM
    assert policy["hash_schema_version"] == HASH_SCHEMA_VERSION


def test_resolve_vertical_policy_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown vertical policy keys"):
        resolve_vertical_policy({"unknown_key": 1})


def test_policy_snapshot_is_stable() -> None:
    s1 = policy_snapshot({"cf_mode": "direct_cf_quadrature"})
    s2 = policy_snapshot({"cf_mode": "direct_cf_quadrature"})
    assert s1 == s2
    assert "json" in s1 and "sha256" in s1
    assert len(s1["sha256"]) == 64
