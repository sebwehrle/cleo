"""Unit tests for deterministic raster-source identity helpers."""

from __future__ import annotations

import numpy as np

from cleo.unification.fingerprint import semantic_raster_fingerprint


def test_semantic_raster_fingerprint_is_deterministic() -> None:
    values = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
    y = np.array([1.5, 0.5], dtype=np.float64)
    x = np.array([0.5, 1.5], dtype=np.float64)

    first = semantic_raster_fingerprint(
        values=values,
        y=y,
        x=x,
        dtype="float32",
        crs_wkt="EPSG:3035",
        categorical=False,
    )
    second = semantic_raster_fingerprint(
        values=values.copy(),
        y=y.copy(),
        x=x.copy(),
        dtype="float32",
        crs_wkt="EPSG:3035",
        categorical=False,
    )

    assert first == second


def test_semantic_raster_fingerprint_changes_with_values_or_categorical_flag() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    changed = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    y = np.array([1.5, 0.5], dtype=np.float64)
    x = np.array([0.5, 1.5], dtype=np.float64)

    baseline = semantic_raster_fingerprint(
        values=values,
        y=y,
        x=x,
        dtype="float32",
        crs_wkt="EPSG:3035",
        categorical=False,
    )
    changed_values = semantic_raster_fingerprint(
        values=changed,
        y=y,
        x=x,
        dtype="float32",
        crs_wkt="EPSG:3035",
        categorical=False,
    )
    categorical = semantic_raster_fingerprint(
        values=values,
        y=y,
        x=x,
        dtype="float32",
        crs_wkt="EPSG:3035",
        categorical=True,
    )

    assert baseline != changed_values
    assert baseline != categorical
