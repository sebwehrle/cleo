"""Phase 4 coverage tests for CLC helper error and utility paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cleo.clc import default_category_name, materialize_clc, source_cache_path


def test_source_cache_path_raises_for_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported CLC source"):
        source_cache_path(tmp_path, "unknown-source")


def test_default_category_name_normalizes_label(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    with open(resources / "clc_codes.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"clc_codes": {"311": "Broad-leaved forest"}}, f)

    assert default_category_name(tmp_path, 311) == "broad_leaved_forest"


def test_default_category_name_returns_none_for_unknown_code(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    with open(resources / "clc_codes.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"clc_codes": {"311": "Broad-leaved forest"}}, f)

    assert default_category_name(tmp_path, 999) is None


def test_materialize_clc_requires_url_when_source_missing(tmp_path: Path) -> None:
    atlas = SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="epsg:3035",
        chunk_policy={"y": 2, "x": 2},
        _canonical_ready=True,
        build_canonical=lambda: None,
    )
    with pytest.raises(RuntimeError, match="No download URL configured"):
        materialize_clc(atlas, source="clc2018", url=None)
