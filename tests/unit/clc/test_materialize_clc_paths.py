"""Phase 4 unit tests for CLC materialize fast/error paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tests.helpers.optional import requires_rioxarray
from cleo.clc import CLC_SOURCES, materialize_clc, prepared_country_path

requires_rioxarray()


def test_materialize_clc_returns_cached_prepared_without_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepared_country_path(tmp_path, "AUT", "clc2018")
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text("cached", encoding="utf-8")

    called = {"canonical": 0}
    atlas = SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="epsg:3035",
        chunk_policy={"y": 2, "x": 2},
        _canonical_ready=False,
        materialize_canonical=lambda: called.__setitem__("canonical", called["canonical"] + 1),
    )

    out = materialize_clc(atlas, source="clc2018", force_prepare=False)
    assert out == prepared
    assert called["canonical"] == 0


def test_materialize_clc_raises_when_no_valid_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create source raster placeholder path and skip download by ensuring it exists.
    source_path = tmp_path / "data" / "raw" / "clc" / CLC_SOURCES["clc2018"]["filename"]
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("placeholder", encoding="utf-8")

    atlas = SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="epsg:3035",
        chunk_policy={"y": 2, "x": 2},
        _canonical_ready=True,
        materialize_canonical=lambda: None,
    )

    # Wind template with all-NaN => no valid cells.
    y = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([10.0, 11.0], dtype=np.float64)
    ref = xr.DataArray(
        np.full((2, 2), np.nan, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")

    clc = xr.DataArray(
        np.full((2, 2), 231.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="clc",
    ).rio.write_crs("EPSG:3035")

    monkeypatch.setattr("cleo.clc._wind_reference_template", lambda _atlas: ref)

    class _FakeRaster:
        def squeeze(self, drop=True):  # noqa: ANN001, ARG002
            return clc

    monkeypatch.setattr(
        "cleo.unification.clc_io.rxr.open_rasterio",
        lambda *a, **k: _FakeRaster(),
    )

    with pytest.raises(RuntimeError, match="No valid wind cells found"):
        materialize_clc(atlas, source="clc2018")


def test_materialize_clc_raises_for_unknown_source(tmp_path: Path) -> None:
    atlas = SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="epsg:3035",
        chunk_policy={"y": 2, "x": 2},
        _canonical_ready=True,
        materialize_canonical=lambda: None,
    )
    with pytest.raises(ValueError, match="Unsupported CLC source"):
        materialize_clc(atlas, source="clc2099")


def test_materialize_clc_force_prepare_rebuilds_not_cached(tmp_path: Path) -> None:
    prepared = prepared_country_path(tmp_path, "AUT", "clc2018")
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text("cached", encoding="utf-8")

    atlas = SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="epsg:3035",
        chunk_policy={"y": 2, "x": 2},
        _canonical_ready=True,
        materialize_canonical=lambda: None,
    )

    # force_prepare=True must not return cached file directly; it should proceed
    # and here fail because no source URL/file is available.
    with pytest.raises(RuntimeError, match="No download URL configured"):
        materialize_clc(atlas, source="clc2018", force_prepare=True)
