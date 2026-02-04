"""Oracle test: load_gwa must create data/raw/<ISO3> if missing."""

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

    def mock_download_file(url, fpath):
        recorded.append((url, Path(fpath)))
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        Path(fpath).touch()
        return True

    monkeypatch.setattr(cleo.loaders, "download_file", mock_download_file)

    parent = MockParent(path=tmp_path, country="AUT")
    dummy = MockSelf(parent)

    # raw dir intentionally NOT created here
    cleo.loaders.load_gwa(dummy)

    raw_dir = tmp_path / "data" / "raw" / "AUT"
    assert raw_dir.is_dir()
    assert len(recorded) > 0
