"""classes: test_loading_contracts.

Covers:
- load_and_extract_from_dict: must not change cwd; must reject zip-slip paths
- Atlas.load: must fall back to legacy filenames when no timestamped files exist
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import cleo.classes as C


def _mini_ds() -> xr.Dataset:
    return xr.Dataset(
        {"a": (("y", "x"), np.ones((2, 2), dtype=np.float32))},
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )


def _mk_landscape_atlas_stub(tmp_path: Path) -> C._LandscapeAtlas:
    """
    Create a minimal _LandscapeAtlas instance without heavy __init__.
    Ensure any optional proxy attributes exist for download/extract flows.
    """
    atlas = C._LandscapeAtlas.__new__(C._LandscapeAtlas)
    atlas.path = Path(tmp_path)
    atlas.parent = SimpleNamespace(proxy=None, proxy_user=None, proxy_pass=None)
    return atlas


def test_load_and_extract_from_dict_does_not_change_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Prepare a zip file we can "download"
    src_zip = tmp_path / "src.zip"
    with zipfile.ZipFile(src_zip, "w") as z:
        z.writestr("ok.txt", "hello")

    # Monkeypatch download_file to copy local zip
    def fake_download_file(url, dest, proxy=None, proxy_user=None, proxy_pass=None):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(Path(url).read_bytes())
        return True

    monkeypatch.setattr(C, "download_file", fake_download_file, raising=True)

    atlas = _mk_landscape_atlas_stub(tmp_path)

    cwd_before = os.getcwd()
    dest_dir = tmp_path / "dest"

    atlas.load_and_extract_from_dict({"downloaded.zip": (str(dest_dir), str(src_zip))})

    assert os.getcwd() == cwd_before
    assert (dest_dir / "ok.txt").is_file()


def test_load_and_extract_from_dict_rejects_zip_slip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        z.writestr("../evil.txt", "x")

    def fake_download_file(url, dest, proxy=None, proxy_user=None, proxy_pass=None):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(Path(url).read_bytes())
        return True

    monkeypatch.setattr(C, "download_file", fake_download_file, raising=True)

    atlas = _mk_landscape_atlas_stub(tmp_path)

    with pytest.raises(ValueError, match="Unsafe zip member path"):
        atlas.load_and_extract_from_dict({"downloaded.zip": (str(tmp_path / "dest2"), str(bad_zip))})


class _DummySub:
    def __init__(self) -> None:
        self.data = None


def test_load_falls_back_to_legacy_files(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # legacy names expected by code when no timestamped matches exist
    ds = _mini_ds()
    ds.to_netcdf(processed / "WindAtlas_AUT.nc")
    ds.close()

    ds = _mini_ds()
    ds.to_netcdf(processed / "LandscapeAtlas_AUT.nc")
    ds.close()

    a = C.Atlas.__new__(C.Atlas)
    a._path = Path(tmp_path)
    a.country = "AUT"
    a.region = None

    # Satisfy materialization guards without invoking heavy materialize()
    a._materialized = True
    a._wind = _DummySub()
    a._landscape = _DummySub()

    a.load(region="None", scenario="default", timestamp="latest")

    assert a.wind.data is not None
    assert a.landscape.data is not None
