"""Phase 6 tests for turbine power-curve tail policy and validation."""

from __future__ import annotations

import numpy as np
import pytest

from cleo.unification.turbines import (
    _normalize_power_curve_for_policy,
)


def _yaml(cutout_wind_speed=None) -> dict:
    data = {"manufacturer": "X", "model": "Y"}
    if cutout_wind_speed is not None:
        data["cutout_wind_speed"] = cutout_wind_speed
    return data


def test_strict_tail_requires_last_value_zero() -> None:
    u = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    p = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="must end at zero"):
        _normalize_power_curve_for_policy(
            turbine_id="T1",
            u=u,
            p=p,
            tail_policy="strict_zero_tail",
            cutout_source="constant_default",
            cutout_default_mps=25.0,
            yaml_data=_yaml(),
        )


def test_auto_append_adds_zero_tail_using_default_cutout() -> None:
    u = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    p = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    u2, p2 = _normalize_power_curve_for_policy(
        turbine_id="T1",
        u=u,
        p=p,
        tail_policy="auto_append_zero_at_cutout",
        cutout_source="constant_default",
        cutout_default_mps=30.0,
        yaml_data=_yaml(),
    )
    assert np.allclose(u2, np.array([0.0, 10.0, 20.0, 30.0]))
    assert np.allclose(p2, np.array([0.0, 1.0, 1.0, 0.0]))


def test_auto_append_uses_metadata_cutout_when_available() -> None:
    u = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    p = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    u2, p2 = _normalize_power_curve_for_policy(
        turbine_id="T1",
        u=u,
        p=p,
        tail_policy="auto_append_zero_at_cutout",
        cutout_source="from_turbine_metadata",
        cutout_default_mps=30.0,
        yaml_data=_yaml(cutout_wind_speed=26.0),
    )
    assert np.isclose(float(u2[-1]), 26.0)
    assert np.isclose(float(p2[-1]), 0.0)


def test_auto_append_requires_cutout_above_u_max() -> None:
    u = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    p = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="must be > last power-curve wind speed"):
        _normalize_power_curve_for_policy(
            turbine_id="T1",
            u=u,
            p=p,
            tail_policy="auto_append_zero_at_cutout",
            cutout_source="from_turbine_metadata",
            cutout_default_mps=25.0,
            yaml_data=_yaml(cutout_wind_speed=20.0),
        )


def test_validation_rejects_negative_power() -> None:
    u = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    p = np.array([0.0, -0.1, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="negative power"):
        _normalize_power_curve_for_policy(
            turbine_id="T1",
            u=u,
            p=p,
            tail_policy="auto_append_zero_at_cutout",
            cutout_source="constant_default",
            cutout_default_mps=25.0,
            yaml_data=_yaml(),
        )


def test_validation_rejects_above_rated_fraction() -> None:
    u = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    p = np.array([0.0, 1.1, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="above rated fraction"):
        _normalize_power_curve_for_policy(
            turbine_id="T1",
            u=u,
            p=p,
            tail_policy="auto_append_zero_at_cutout",
            cutout_source="constant_default",
            cutout_default_mps=25.0,
            yaml_data=_yaml(),
        )


def test_validation_rejects_non_monotone_wind_speed() -> None:
    u = np.array([0.0, 10.0, 10.0], dtype=np.float64)
    p = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="strictly increasing"):
        _normalize_power_curve_for_policy(
            turbine_id="T1",
            u=u,
            p=p,
            tail_policy="auto_append_zero_at_cutout",
            cutout_source="constant_default",
            cutout_default_mps=25.0,
            yaml_data=_yaml(),
        )
