"""
loaders: load_gwa creates raw dir tests.
Split from merged test_gwa.py
"""

from pathlib import Path

import cleo.loaders


class MockParent:
    def __init__(self, path, country):
        self.path = path
        self.country = country


class MockSelf:
    def __init__(self, parent):
        self.parent = parent


def test_load_gwa_creates_raw_dir(tmp_path, monkeypatch):
    recorded = []

    def mock_download_to_path(url, fpath, **kwargs):
        recorded.append((url, Path(fpath)))
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        Path(fpath).touch()
        return Path(fpath)

    monkeypatch.setattr(cleo.loaders, "download_to_path", mock_download_to_path)

    parent = MockParent(path=tmp_path, country="AUT")
    dummy = MockSelf(parent)

    cleo.loaders.load_gwa(dummy)

    raw_dir = tmp_path / "data" / "raw" / "AUT"
    assert raw_dir.is_dir()
    assert len(recorded) > 0
