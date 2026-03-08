"""Unit tests for deterministic vector-source identity helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.helpers.optional import requires_geopandas

requires_geopandas()

import geopandas as gpd
from shapely.geometry import Polygon

from cleo.unification.materializers._landscape_vector import (
    _canonical_vector_source_artifact,
    _vector_semantic_hash,
)


def _sample_gdf(*, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    p1 = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
    p2 = Polygon([(1.0, 0.0), (1.0, 1.0), (2.0, 1.0), (2.0, 0.0)])
    return gpd.GeoDataFrame(
        {"overnight_stays": [10.0, 25.0]},
        geometry=[p1, p2],
        crs=crs,
    )


def test_vector_semantic_hash_is_deterministic() -> None:
    gdf = _sample_gdf()
    h1 = _vector_semantic_hash(gdf, column="overnight_stays", all_touched=False)
    h2 = _vector_semantic_hash(gdf, column="overnight_stays", all_touched=False)
    assert h1 == h2


def test_vector_semantic_hash_changes_with_feature_order() -> None:
    gdf = _sample_gdf()
    forward = _vector_semantic_hash(gdf, column="overnight_stays", all_touched=False)
    reverse = _vector_semantic_hash(
        gdf.iloc[::-1].reset_index(drop=True),
        column="overnight_stays",
        all_touched=False,
    )
    assert forward != reverse


def test_canonical_vector_source_artifact_path_and_hash(tmp_path: Path) -> None:
    atlas = SimpleNamespace(path=tmp_path, crs="epsg:3035")
    gdf = _sample_gdf()

    out_path, fingerprint = _canonical_vector_source_artifact(
        atlas,
        shape=gdf,
        column="overnight_stays",
        all_touched=True,
    )

    assert out_path.exists()
    assert out_path.parent == (tmp_path / "intermediates" / "vector_sources")
    assert out_path.suffix == ".geojson"
    assert out_path.stem == fingerprint


def test_canonical_vector_source_artifact_hash_matches_path_and_gdf(tmp_path: Path) -> None:
    atlas = SimpleNamespace(path=tmp_path, crs="epsg:3035")
    gdf = _sample_gdf()
    src = tmp_path / "overnight_stays.geojson"
    gdf.to_file(src, driver="GeoJSON")

    out_from_gdf, h_gdf = _canonical_vector_source_artifact(
        atlas,
        shape=gdf,
        column="overnight_stays",
        all_touched=False,
    )
    out_from_path, h_path = _canonical_vector_source_artifact(
        atlas,
        shape=src,
        column="overnight_stays",
        all_touched=False,
    )

    assert h_gdf == h_path
    assert out_from_gdf == out_from_path


def test_canonical_vector_source_artifact_requires_numeric_column(tmp_path: Path) -> None:
    atlas = SimpleNamespace(path=tmp_path, crs="epsg:3035")
    gdf = _sample_gdf()
    gdf["category"] = ["a", "b"]

    with pytest.raises(TypeError, match="must be numeric"):
        _canonical_vector_source_artifact(
            atlas,
            shape=gdf,
            column="category",
            all_touched=False,
        )


def test_canonical_vector_source_artifact_rejects_missing_column(tmp_path: Path) -> None:
    atlas = SimpleNamespace(path=tmp_path, crs="epsg:3035")
    gdf = _sample_gdf()

    with pytest.raises(ValueError, match="Column 'missing' not found"):
        _canonical_vector_source_artifact(
            atlas,
            shape=gdf,
            column="missing",
            all_touched=False,
        )


def test_vector_semantic_hash_default_column_uses_binary_value() -> None:
    gdf = _sample_gdf()
    h_none = _vector_semantic_hash(gdf, column=None, all_touched=False)
    h_col = _vector_semantic_hash(gdf, column="overnight_stays", all_touched=False)
    assert isinstance(h_none, str) and len(h_none) == 64
    assert h_none != h_col


def test_canonical_vector_source_artifact_rejects_nan_column_value(tmp_path: Path) -> None:
    atlas = SimpleNamespace(path=tmp_path, crs="epsg:3035")
    gdf = _sample_gdf()
    gdf.loc[1, "overnight_stays"] = np.nan

    with pytest.raises(ValueError, match="contains non-finite value"):
        _canonical_vector_source_artifact(
            atlas,
            shape=gdf,
            column="overnight_stays",
            all_touched=False,
        )
