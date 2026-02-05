import os
import zipfile
from pathlib import Path

import pytest

import cleo.classes as C


def test_load_and_extract_from_dict_does_not_change_cwd(tmp_path, monkeypatch):
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

    atlas = C._LandscapeAtlas.__new__(C._LandscapeAtlas)

    cwd_before = os.getcwd()
    dest_dir = tmp_path / "dest"

    atlas.load_and_extract_from_dict({"downloaded.zip": (str(dest_dir), str(src_zip))})

    assert os.getcwd() == cwd_before
    assert (dest_dir / "ok.txt").is_file()


def test_load_and_extract_from_dict_rejects_zip_slip(tmp_path, monkeypatch):
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        z.writestr("../evil.txt", "x")

    def fake_download_file(url, dest, proxy=None, proxy_user=None, proxy_pass=None):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(Path(url).read_bytes())
        return True

    monkeypatch.setattr(C, "download_file", fake_download_file, raising=True)

    atlas = C._LandscapeAtlas.__new__(C._LandscapeAtlas)

    with pytest.raises(ValueError, match="Unsafe zip member path"):
        atlas.load_and_extract_from_dict({"downloaded.zip": (str(tmp_path / "dest2"), str(bad_zip))})
