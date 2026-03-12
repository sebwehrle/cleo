"""Tests for result-materialization metric normalization helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from cleo.results import normalize_metric_for_active_wind_store


def _active_wind_store() -> xr.Dataset:
    """Build a minimal active wind-store schema for normalization tests."""
    return xr.Dataset(
        {"template": (("y", "x"), np.ones((2, 2), dtype=np.float32))},
        coords={
            "turbine": np.array([10, 20], dtype=np.int64),
            "y": np.array([1.0, 0.0], dtype=np.float64),
            "x": np.array([0.0, 1.0], dtype=np.float64),
        },
        attrs={
            "cleo_turbines_json": json.dumps(
                [{"id": "T1"}, {"id": "T2"}],
                sort_keys=True,
                separators=(",", ":"),
            )
        },
    )


def test_normalize_metric_aligns_string_turbine_ids_to_store_labels() -> None:
    """String turbine IDs align onto the full active-store turbine axis."""
    existing_ds = _active_wind_store()
    computed = xr.DataArray(
        np.full((1, 2, 2), 7.0, dtype=np.float32),
        dims=("turbine", "y", "x"),
        coords={
            "turbine": np.array(["T2"], dtype=object),
            "y": np.array([1.0, 0.0], dtype=np.float64),
            "x": np.array([0.0, 1.0], dtype=np.float64),
        },
        name="capacity_factors",
    )

    normalized = normalize_metric_for_active_wind_store(
        metric="capacity_factors",
        da=computed,
        existing_ds=existing_ds,
    )

    assert normalized.sizes["turbine"] == 2
    assert normalized.coords["turbine"].values.tolist() == [10, 20]
    assert bool(np.isnan(normalized.sel(turbine=10).values).all()) is True
    np.testing.assert_allclose(normalized.sel(turbine=20).values, np.full((2, 2), 7.0, dtype=np.float32))


def test_normalize_metric_aligns_numeric_turbine_indices_to_store_labels() -> None:
    """Numeric turbine indices align onto the full active-store turbine axis."""
    existing_ds = _active_wind_store()
    computed = xr.DataArray(
        np.full((1, 2, 2), 5.0, dtype=np.float32),
        dims=("turbine", "y", "x"),
        coords={
            "turbine": np.array([1], dtype=np.int64),
            "y": np.array([1.0, 0.0], dtype=np.float64),
            "x": np.array([0.0, 1.0], dtype=np.float64),
        },
        name="capacity_factors",
    )

    normalized = normalize_metric_for_active_wind_store(
        metric="capacity_factors",
        da=computed,
        existing_ds=existing_ds,
    )

    assert normalized.coords["turbine"].values.tolist() == [10, 20]
    assert bool(np.isnan(normalized.sel(turbine=10).values).all()) is True
    np.testing.assert_allclose(normalized.sel(turbine=20).values, np.full((2, 2), 5.0, dtype=np.float32))


def test_normalize_metric_rejects_unknown_turbine_id() -> None:
    """Unknown turbine IDs still fail with an explicit alignment error."""
    existing_ds = _active_wind_store()
    computed = xr.DataArray(
        np.full((1, 2, 2), 3.0, dtype=np.float32),
        dims=("turbine", "y", "x"),
        coords={
            "turbine": np.array(["T9"], dtype=object),
            "y": np.array([1.0, 0.0], dtype=np.float64),
            "x": np.array([0.0, 1.0], dtype=np.float64),
        },
        name="capacity_factors",
    )

    with pytest.raises(ValueError, match="unknown turbine id"):
        normalize_metric_for_active_wind_store(
            metric="capacity_factors",
            da=computed,
            existing_ds=existing_ds,
        )


def test_normalize_metric_rewrites_turbine_axis_attrs_to_store_order() -> None:
    """Turbine-axis attrs reflect the active-store order after alignment."""
    existing_ds = _active_wind_store()
    computed = xr.DataArray(
        np.full((1, 2, 2), 7.0, dtype=np.float32),
        dims=("turbine", "y", "x"),
        coords={
            "turbine": np.array(["T2"], dtype=object),
            "y": np.array([1.0, 0.0], dtype=np.float64),
            "x": np.array([0.0, 1.0], dtype=np.float64),
        },
        attrs={
            "cleo:turbines_json": json.dumps(["T2"], ensure_ascii=True),
            "cleo:turbine_ids_json": json.dumps(["T2"], ensure_ascii=True),
        },
        name="lcoe",
    )

    normalized = normalize_metric_for_active_wind_store(
        metric="lcoe",
        da=computed,
        existing_ds=existing_ds,
    )

    assert json.loads(normalized.attrs["cleo:turbines_json"]) == ["T1", "T2"]
    assert json.loads(normalized.attrs["cleo:turbine_ids_json"]) == ["T1", "T2"]


def test_normalize_metric_remaps_min_lcoe_turbine_to_store_indices() -> None:
    """Subset-relative min-lcoe indices are remapped to the active-store axis."""
    existing_ds = _active_wind_store()
    computed = xr.DataArray(
        np.array([[0.0, np.nan], [0.0, np.nan]], dtype=np.float64),
        dims=("y", "x"),
        coords={
            "y": np.array([1.0, 0.0], dtype=np.float64),
            "x": np.array([0.0, 1.0], dtype=np.float64),
        },
        attrs={"cleo:turbine_ids_json": json.dumps(["T2"], ensure_ascii=True)},
        name="min_lcoe_turbine",
    )

    normalized = normalize_metric_for_active_wind_store(
        metric="min_lcoe_turbine",
        da=computed,
        existing_ds=existing_ds,
    )

    expected = np.array([[1.0, np.nan], [1.0, np.nan]], dtype=np.float64)
    np.testing.assert_allclose(normalized.values, expected, rtol=0.0, atol=0.0, equal_nan=True)
    assert json.loads(normalized.attrs["cleo:turbine_ids_json"]) == ["T1", "T2"]
