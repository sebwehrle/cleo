"""Phase 4 unit tests for LandscapeDomain CLC helper API paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cleo.domains import LandscapeDomain


def _fake_atlas(tmp_path: Path):
    return SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="epsg:3035",
        chunk_policy={"y": 2, "x": 2},
        _canonical_ready=True,
        build_canonical=lambda: None,
        build_clc=lambda source="clc2018": tmp_path / "data" / "raw" / "AUT" / "clc" / f"{source}.tif",
        fingerprint_method="path_mtime_size",
    )


def test_add_clc_category_all_routes_to_add_with_categorical_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = _fake_atlas(tmp_path)
    domain = LandscapeDomain(atlas)
    calls: list[dict] = []

    def _capture_add(self, name, source_path, **kwargs):  # noqa: ANN001
        calls.append({"name": name, "source_path": source_path, **kwargs})

    monkeypatch.setattr(LandscapeDomain, "add", _capture_add)
    domain.add_clc_category("all", if_exists="noop", source="clc2018")

    assert len(calls) == 1
    assert calls[0]["name"] == "land_cover"
    assert calls[0]["params"] == {"categorical": True, "clc_source": "clc2018"}
    assert calls[0]["if_exists"] == "noop"


def test_add_clc_category_int_uses_default_name_and_clc_codes_param(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = _fake_atlas(tmp_path)
    domain = LandscapeDomain(atlas)
    calls: list[dict] = []

    monkeypatch.setattr("cleo.clc.default_category_name", lambda _path, _code: "forest")

    def _capture_add(self, name, source_path, **kwargs):  # noqa: ANN001
        calls.append({"name": name, "source_path": source_path, **kwargs})

    monkeypatch.setattr(LandscapeDomain, "add", _capture_add)
    domain.add_clc_category(311)

    assert len(calls) == 1
    assert calls[0]["name"] == "forest"
    assert calls[0]["params"]["clc_codes"] == [311]
    assert calls[0]["params"]["categorical"] is True


def test_add_clc_category_requires_name_for_multi_code(tmp_path: Path) -> None:
    atlas = _fake_atlas(tmp_path)
    domain = LandscapeDomain(atlas)
    with pytest.raises(ValueError, match="name is required"):
        domain.add_clc_category([311, 312])


def test_add_clc_category_multi_code_with_name_routes_to_add(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = _fake_atlas(tmp_path)
    domain = LandscapeDomain(atlas)
    calls: list[dict] = []

    def _capture_add(self, name, source_path, **kwargs):  # noqa: ANN001
        calls.append({"name": name, "source_path": source_path, **kwargs})

    monkeypatch.setattr(LandscapeDomain, "add", _capture_add)
    domain.add_clc_category([311, 312], name="forest_mask", if_exists="replace")

    assert len(calls) == 1
    assert calls[0]["name"] == "forest_mask"
    assert calls[0]["params"]["clc_codes"] == [311, 312]
    assert calls[0]["if_exists"] == "replace"


def test_add_clc_category_raises_for_invalid_categories_type(tmp_path: Path) -> None:
    atlas = _fake_atlas(tmp_path)
    domain = LandscapeDomain(atlas)
    with pytest.raises(ValueError, match="categories must be 'all', an int code"):
        domain.add_clc_category("forest")


def test_add_clc_category_raises_for_empty_list(tmp_path: Path) -> None:
    atlas = _fake_atlas(tmp_path)
    domain = LandscapeDomain(atlas)
    with pytest.raises(ValueError, match="categories must be 'all', an int code"):
        domain.add_clc_category([])


def test_add_clc_category_raises_for_non_integer_list_entry(tmp_path: Path) -> None:
    atlas = _fake_atlas(tmp_path)
    domain = LandscapeDomain(atlas)
    with pytest.raises(ValueError, match="invalid literal"):
        domain.add_clc_category(["not-an-int"])  # type: ignore[list-item]


def test_add_clc_category_raises_when_default_name_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = _fake_atlas(tmp_path)
    domain = LandscapeDomain(atlas)
    monkeypatch.setattr("cleo.clc.default_category_name", lambda _path, _code: None)
    with pytest.raises(ValueError, match="No default variable name known for CLC code"):
        domain.add_clc_category(999)


def test_add_rejects_legacy_materialize_kw(tmp_path: Path) -> None:
    atlas = _fake_atlas(tmp_path)
    domain = LandscapeDomain(atlas)
    with pytest.raises(TypeError, match="unexpected keyword argument 'materialize'"):
        domain.add("foo", tmp_path / "foo.tif", materialize=True)  # type: ignore[call-arg]


def test_add_clc_category_rejects_legacy_materialize_kw(tmp_path: Path) -> None:
    atlas = _fake_atlas(tmp_path)
    domain = LandscapeDomain(atlas)
    with pytest.raises(TypeError, match="unexpected keyword argument 'materialize'"):
        domain.add_clc_category("all", materialize=True)  # type: ignore[call-arg]
