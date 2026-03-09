"""Phase 3 coverage tests for unification manifest and I/O helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from tests.helpers.optional import requires_rasterio, requires_rioxarray

requires_rasterio()
requires_rioxarray()

import rasterio
import cleo.unification.manifest as M
from cleo.unification.gwa_io import (
    _assert_all_required_gwa_present,
    _load_or_fetch_gwa_crs,
    _open_gwa_raster,
    ensure_required_gwa_files,
)


def _write_tif(
    path: Path,
    data: np.ndarray,
    *,
    crs: str = "EPSG:3035",
    nodata: float | None = None,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=str(data.dtype),
        crs=crs,
        nodata=nodata,
        transform=rasterio.transform.from_bounds(0, 0, 2, 2, data.shape[1], data.shape[0]),
    ) as dst:
        dst.write(data, 1)


def test_read_manifest_invalid_json_returns_defaults(tmp_path: Path) -> None:
    store = tmp_path / "store.zarr"
    root = zarr.open_group(store, mode="w")
    root.attrs["cleo_manifest_sources_json"] = "{invalid"
    root.attrs["cleo_manifest_variables_json"] = "[]"

    out = M._read_manifest(store)
    assert out == {"version": 1, "sources": [], "variables": []}


def test_init_manifest_swallows_open_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fail_open(*args, **kwargs):  # noqa: ANN002, ANN003
        raise OSError("cannot open store")

    monkeypatch.setattr("cleo.unification.manifest.zarr.open_group", _fail_open)
    M.init_manifest(tmp_path / "missing-store.zarr")


def test_write_manifest_sources_adds_created_at(tmp_path: Path) -> None:
    store = tmp_path / "store.zarr"
    zarr.open_group(store, mode="w")
    M.init_manifest(store)

    M.write_manifest_sources(
        store,
        [
            {
                "source_id": "s1",
                "name": "a",
                "kind": "raster",
                "path": "/tmp/a",
                "params_json": "{}",
                "fingerprint": "abc",
            }
        ],
    )
    manifest = M._read_manifest(store)
    assert len(manifest["sources"]) == 1
    assert "created_at" in manifest["sources"][0]


def test_load_or_fetch_gwa_crs_raises_if_cache_write_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    atlas = SimpleNamespace(path=tmp_path)
    monkeypatch.setattr("cleo.unification.gwa_io.fetch_gwa_crs", lambda _iso3: "EPSG:3035")

    def _fail_write(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _fail_write)

    with pytest.raises(RuntimeError, match="failed to persist cache"):
        _load_or_fetch_gwa_crs(atlas, "AUT")


def test_assert_all_required_gwa_present_lists_missing_files(tmp_path: Path) -> None:
    atlas = SimpleNamespace(path=tmp_path, country="AUT")
    with pytest.raises(FileNotFoundError, match="Missing required GWA files"):
        _assert_all_required_gwa_present(atlas)


def test_ensure_required_gwa_files_downloads_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Auto-download path creates all missing required GWA files."""
    atlas = SimpleNamespace(path=tmp_path, country="AUT")
    calls: list[tuple[str, Path]] = []

    def _fake_download(url: str, dest: Path, **kwargs) -> Path:  # noqa: ANN003
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_tif(path, np.ones((2, 2), dtype=np.float32))
        calls.append((url, path))
        return path

    monkeypatch.setattr("cleo.unification.gwa_io.cleo.net.download_to_path", _fake_download)

    req = ensure_required_gwa_files(atlas, auto_download=True)

    assert len(req) == 15
    assert len(calls) == 15
    assert all(path.exists() for _sid, path in req)
    assert all(url.startswith("https://globalwindatlas.info/api/gis/country/AUT/") for url, _path in calls)


def test_ensure_required_gwa_files_raises_when_download_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Auto-download path still fails fast with complete missing-file list."""
    atlas = SimpleNamespace(path=tmp_path, country="AUT")
    calls: list[tuple[str, Path]] = []

    def _fail_download(url: str, dest: Path, **kwargs) -> None:  # noqa: ANN003
        calls.append((url, Path(dest)))
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("cleo.unification.gwa_io.cleo.net.download_to_path", _fail_download)

    with pytest.raises(FileNotFoundError, match="Missing required GWA files"):
        ensure_required_gwa_files(atlas, auto_download=True)

    assert len(calls) == 15


def test_open_gwa_raster_converts_nodata_to_nan(tmp_path: Path) -> None:
    atlas = SimpleNamespace(path=tmp_path)
    raster_path = tmp_path / "nodata.tif"
    data = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    _write_tif(raster_path, data, nodata=0.0)

    da = _open_gwa_raster(
        atlas,
        raster_path,
        iso3="AUT",
        target_crs="EPSG:3035",
    )
    assert np.isnan(da.values[0, 1])
    assert float(da.values[0, 0]) == 1.0
