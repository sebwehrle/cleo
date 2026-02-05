import os
from pathlib import Path
import pytest
import xarray as xr

from cleo.classes import Atlas


class _DummySubAtlas:
    def __init__(self):
        self.data = xr.Dataset(attrs={"country": "AUT", "region": "None"})


def test_save_creates_processed_dir(tmp_path, monkeypatch):
    # Create Atlas instance without running __init__
    atlas = Atlas.__new__(Atlas)
    atlas._path = Path(tmp_path)
    atlas.country = "AUT"
    atlas.region = None
    atlas.index_file = atlas._path / "data" / "index.txt"
    atlas.wind = _DummySubAtlas()
    atlas.landscape = _DummySubAtlas()

    # Avoid real NetCDF writing; just touch the file path
    def _fake_to_netcdf(self, path, *args, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"dummy")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _fake_to_netcdf, raising=True)

    # Remove processed dir if exists
    processed = atlas._path / "data" / "processed"
    if processed.exists():
        for p in processed.rglob("*"):
            if p.is_file():
                p.unlink()
        processed.rmdir()

    atlas.save()

    assert processed.is_dir()
