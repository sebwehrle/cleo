"""Phase 4 unit tests for CLC materialize fast/error paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

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
        build_canonical=lambda: called.__setitem__("canonical", called["canonical"] + 1),
    )

    out = materialize_clc(atlas, source="clc2018", force_prepare=False)
    assert out == prepared
    assert called["canonical"] == 0


def test_materialize_clc_raises_when_no_valid_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        build_canonical=lambda: None,
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

    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)

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
        build_canonical=lambda: None,
    )
    with pytest.raises(ValueError, match="Unsupported CLC source"):
        materialize_clc(atlas, source="clc2099")


def test_materialize_clc_uses_deterministic_default_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atlas = SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="epsg:3035",
        chunk_policy={"y": 2, "x": 2},
        _canonical_ready=True,
        build_canonical=lambda: None,
    )

    ref = xr.DataArray(
        np.ones((2, 3), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")

    captured: dict[str, str | Path] = {}
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)

    def _fake_download(url: str, out_path: Path) -> None:
        captured["url"] = url
        captured["source_path"] = out_path
        out_path.write_text("source", encoding="utf-8")

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)
    monkeypatch.setattr(
        "cleo.clc.prepare_clc_to_wind_grid",
        lambda **kwargs: kwargs["prepared_path"],
    )

    out = materialize_clc(atlas, source="clc2018")
    assert out == prepared_country_path(tmp_path, "AUT", "clc2018")
    assert "url" in captured

    parsed = urlparse(str(captured["url"]))
    assert parsed.path.endswith("/Corine/CLC2018_LAEA/MapServer/export")
    query = parse_qs(parsed.query)
    assert query["f"] == ["image"]
    assert query["format"] == ["tiff"]
    assert query["layers"] == ["show:1"]
    assert query["size"] == ["3,2"]
    assert query["bboxSR"] == ["3035"]
    assert query["imageSR"] == ["3035"]
    bbox = [float(v) for v in query["bbox"][0].split(",")]
    assert bbox == pytest.approx([5.0, 50.0, 35.0, 250.0])


def test_materialize_clc_force_prepare_rebuilds_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepared_country_path(tmp_path, "AUT", "clc2018")
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text("cached", encoding="utf-8")

    atlas = SimpleNamespace(
        path=tmp_path,
        country="AUT",
        crs="epsg:3035",
        chunk_policy={"y": 2, "x": 2},
        _canonical_ready=True,
        build_canonical=lambda: None,
    )

    ref = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": np.array([200.0, 100.0], dtype=np.float64),
            "x": np.array([10.0, 20.0], dtype=np.float64),
        },
        name="weibull_A",
    ).rio.write_crs("EPSG:3035")
    calls = {"downloaded": 0, "prepared": 0}
    monkeypatch.setattr("cleo.clc.wind_reference_template", lambda _atlas: ref)

    def _fake_download(_url: str, out_path: Path) -> None:
        calls["downloaded"] += 1
        out_path.write_text("source", encoding="utf-8")

    def _fake_prepare(**kwargs):
        calls["prepared"] += 1
        return kwargs["prepared_path"]

    monkeypatch.setattr("cleo.clc.download_to_path", _fake_download)
    monkeypatch.setattr("cleo.clc.prepare_clc_to_wind_grid", _fake_prepare)

    out = materialize_clc(atlas, source="clc2018", force_prepare=True)
    assert out == prepared
    assert calls["downloaded"] == 1
    assert calls["prepared"] == 1
