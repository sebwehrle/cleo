"""Tests for download_file: no file written on HTTP error, correct return values."""
import pytest
from cleo.utils import download_file


class MockResponse:
    def __init__(self, status_code=200, content_chunks=None):
        self.status_code = status_code
        self._chunks = content_chunks or []

    def iter_content(self, chunk_size=1024):
        yield from self._chunks

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise Exception(f"HTTP {self.status_code}")

    # If production uses context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_file_http_403_no_file_written(tmp_path, monkeypatch):
    """HTTP 403 should return False and not create any file."""
    save_path = tmp_path / "bad.tif"

    def mock_get(url, **kwargs):
        return MockResponse(status_code=403, content_chunks=[b"error body"])

    monkeypatch.setattr("cleo.utils.requests.get", mock_get)

    result = download_file("http://example.com/file.tif", save_to=save_path, overwrite=True)

    assert result is False
    assert not save_path.exists()


def test_download_file_http_200_writes_file(tmp_path, monkeypatch):
    """HTTP 200 should return True and write file contents."""
    save_path = tmp_path / "good.tif"

    def mock_get(url, **kwargs):
        return MockResponse(status_code=200, content_chunks=[b"OK"])

    monkeypatch.setattr("cleo.utils.requests.get", mock_get)

    result = download_file("http://example.com/file.tif", save_to=save_path, overwrite=True)

    assert result is True
    assert save_path.exists()
    assert save_path.read_bytes() == b"OK"
