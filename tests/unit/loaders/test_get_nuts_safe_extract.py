"""
loaders: load_nuts safe-extract tests.
Split from merged test_resources_and_safe_extract.py
"""

import zipfile
from types import SimpleNamespace

import pytest
import cleo.loaders as L


def test_load_nuts_rejects_zip_slip(tmp_path, monkeypatch):
    resolution = "03M"
    year = 2021
    crs = 4326
    file_collection = f"ref-nuts-{year}-{resolution}.shp.zip"
    file_name = f"NUTS_RG_{resolution}_{year}_{crs}.shp.zip"

    nuts_path = tmp_path / "data" / "nuts"
    nuts_path.mkdir(parents=True, exist_ok=True)

    outer_zip_path = nuts_path / file_collection

    inner_bytes_path = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner_bytes_path, "w") as inner:
        inner.writestr("../pwned.txt", "x")

    with zipfile.ZipFile(outer_zip_path, "w") as outer:
        outer.write(inner_bytes_path, arcname=file_name)

    def mock_download_file(url, fpath):
        return True

    monkeypatch.setattr(L, "download_file", mock_download_file)

    parent = SimpleNamespace(path=tmp_path)
    self = SimpleNamespace(parent=parent)

    with pytest.raises(ValueError, match="Unsafe zip member"):
        L.load_nuts(self, resolution=resolution, year=year, crs=crs)

    assert not (nuts_path.parent / "pwned.txt").exists()
