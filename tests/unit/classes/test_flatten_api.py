"""Tests for Atlas.flatten() API."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from cleo.atlas import Atlas


def _atlas_with_domain_data(tmp_path, *, wind_ds: xr.Dataset, landscape_ds: xr.Dataset) -> Atlas:
    atlas = Atlas(tmp_path, country="AUT", crs="epsg:3035")
    atlas._wind_domain = SimpleNamespace(data=wind_ds)
    atlas._landscape_domain = SimpleNamespace(data=landscape_ds)
    return atlas


def test_atlas_flatten_wind_domain(tmp_path) -> None:
    wind = xr.Dataset(
        {
            "template": (("y", "x"), np.ones((2, 2))),
            "v2": (("y", "x"), np.array([[1.0, 2.0], [3.0, np.nan]])),
            "v3": (("height", "y", "x"), np.array([
                [[10.0, 11.0], [12.0, np.nan]],
                [[20.0, 21.0], [22.0, np.nan]],
            ])),
        },
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0], "height": [50, 100]},
    )
    landscape = xr.Dataset({"valid_mask": (("y", "x"), np.ones((2, 2), dtype=bool))}, coords={"x": [0.0, 1.0], "y": [1.0, 0.0]})
    atlas = _atlas_with_domain_data(tmp_path, wind_ds=wind, landscape_ds=landscape)

    df = atlas.flatten(domain="wind")

    assert "template" not in df.columns
    assert "v2" in df.columns
    assert "v3_height_50" in df.columns
    assert "v3_height_100" in df.columns
    assert list(df.index.names) == ["y", "x"]


def test_atlas_flatten_landscape_domain(tmp_path) -> None:
    wind = xr.Dataset({"dummy": (("y", "x"), np.ones((2, 2)))}, coords={"x": [0.0, 1.0], "y": [1.0, 0.0]})
    landscape = xr.Dataset(
        {"valid_mask": (("y", "x"), np.array([[True, False], [True, True]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    atlas = _atlas_with_domain_data(tmp_path, wind_ds=wind, landscape_ds=landscape)

    df = atlas.flatten(domain="landscape")

    assert "valid_mask" in df.columns
    assert list(df.index.names) == ["y", "x"]


def test_atlas_flatten_invalid_domain_raises(tmp_path) -> None:
    wind = xr.Dataset({"dummy": (("y", "x"), np.ones((2, 2)))}, coords={"x": [0.0, 1.0], "y": [1.0, 0.0]})
    landscape = xr.Dataset({"valid_mask": (("y", "x"), np.ones((2, 2), dtype=bool))}, coords={"x": [0.0, 1.0], "y": [1.0, 0.0]})
    atlas = _atlas_with_domain_data(tmp_path, wind_ds=wind, landscape_ds=landscape)

    with pytest.raises(ValueError, match="Unsupported domain"):
        atlas.flatten(domain="not-a-domain")


def test_atlas_flatten_both_domains_with_prefixes(tmp_path) -> None:
    wind = xr.Dataset(
        {"wvar": (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    landscape = xr.Dataset(
        {"lvar": (("y", "x"), np.array([[10.0, 20.0], [30.0, 40.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    atlas = _atlas_with_domain_data(tmp_path, wind_ds=wind, landscape_ds=landscape)

    df = atlas.flatten(domain="both")

    assert "wind__wvar" in df.columns
    assert "landscape__lvar" in df.columns
    assert list(df.index.names) == ["y", "x"]


def test_atlas_flatten_both_domains_collision_raises_without_prefix(tmp_path) -> None:
    wind = xr.Dataset(
        {"shared": (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    landscape = xr.Dataset(
        {"shared": (("y", "x"), np.array([[10.0, 20.0], [30.0, 40.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    atlas = _atlas_with_domain_data(tmp_path, wind_ds=wind, landscape_ds=landscape)

    with pytest.raises(ValueError, match="Column name collision"):
        atlas.flatten(domain="both", include_domain_prefix=False)


def test_atlas_flatten_cast_binary_to_int(tmp_path) -> None:
    wind = xr.Dataset(
        {"wbin": (("y", "x"), np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    landscape = xr.Dataset(
        {"lcont": (("y", "x"), np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    atlas = _atlas_with_domain_data(tmp_path, wind_ds=wind, landscape_ds=landscape)

    df = atlas.flatten(domain="both", cast_binary_to_int=True)

    assert str(df["wind__wbin"].dtype) == "Int8"
    assert str(df["landscape__lcont"].dtype) != "Int8"


def test_atlas_flatten_include_only_for_both_domain(tmp_path) -> None:
    wind = xr.Dataset(
        {"wvar": (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    landscape = xr.Dataset(
        {"lvar": (("y", "x"), np.array([[10.0, 20.0], [30.0, 40.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    atlas = _atlas_with_domain_data(tmp_path, wind_ds=wind, landscape_ds=landscape)

    df = atlas.flatten(domain="both", include_only=["landscape__lvar"])
    assert list(df.columns) == ["landscape__lvar"]


def test_atlas_flatten_include_only_unknown_raises(tmp_path) -> None:
    wind = xr.Dataset(
        {"wvar": (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    landscape = xr.Dataset(
        {"lvar": (("y", "x"), np.array([[10.0, 20.0], [30.0, 40.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    atlas = _atlas_with_domain_data(tmp_path, wind_ds=wind, landscape_ds=landscape)

    with pytest.raises(ValueError, match="include_only contains unknown columns"):
        atlas.flatten(domain="both", include_only=["missing"])


def test_validate_flatten_schema_success_and_failure(tmp_path) -> None:
    wind = xr.Dataset(
        {"wvar": (("y", "x"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    landscape = xr.Dataset(
        {"lvar": (("y", "x"), np.array([[10.0, 20.0], [30.0, 40.0]]))},
        coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    atlas = _atlas_with_domain_data(tmp_path, wind_ds=wind, landscape_ds=landscape)
    df = atlas.flatten(domain="both")

    Atlas.validate_flatten_schema(df, ["wind__wvar", "landscape__lvar"])
    with pytest.raises(ValueError, match="Missing required flatten columns"):
        Atlas.validate_flatten_schema(df, ["wind__wvar", "missing"])
