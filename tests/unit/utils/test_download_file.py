"""
utils: download_file tests.
Split from merged test_downloads_and_templates.py
"""

import pytest
from unittest.mock import patch, MagicMock

from cleo.utils import download_file


class _Resp:
    def __init__(self):
        self.status_code = 200

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1024):
        yield b"partial"
        raise RuntimeError("boom")


def test_download_file_atomic_no_partial_left(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        return _Resp()

    monkeypatch.setattr("cleo.net.http_get", fake_get)

    target = tmp_path / "x.bin"
    ok = download_file("https://example.com/x.bin", save_to=target, overwrite=True)
    assert ok is False
    assert not target.exists()
    assert not (tmp_path / "x.bin.tmp").exists()


class MockResponse:
    def __init__(self, status_code=200, content_chunks=None):
        self.status_code = status_code
        self._chunks = content_chunks or []

    def iter_content(self, chunk_size=1024):
        yield from self._chunks

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise Exception(f"HTTP {self.status_code}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_file_http_403_no_file_written(tmp_path, monkeypatch):
    """HTTP 403 should return False and not create any file."""
    save_path = tmp_path / "bad.tif"

    def mock_get(url, **kwargs):
        return MockResponse(status_code=403, content_chunks=[b"error body"])

    monkeypatch.setattr("cleo.net.http_get", mock_get)

    result = download_file("http://example.com/file.tif", save_to=save_path, overwrite=True)

    assert result is False
    assert not save_path.exists()


def test_download_file_http_200_writes_file(tmp_path, monkeypatch):
    """HTTP 200 should return True and write file contents."""
    save_path = tmp_path / "good.tif"

    def mock_get(url, **kwargs):
        return MockResponse(status_code=200, content_chunks=[b"OK"])

    monkeypatch.setattr("cleo.net.http_get", mock_get)

    result = download_file("http://example.com/file.tif", save_to=save_path, overwrite=True)

    assert result is True
    assert save_path.exists()
    assert save_path.read_bytes() == b"OK"


def test_download_file_passes_default_timeout(tmp_path):
    """download_file should pass default timeout (10, 60) to http_get."""
    captured_kwargs = {}

    def mock_get(url, **kwargs):
        captured_kwargs.update(kwargs)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=[b"test"])
        return mock_response

    with patch("cleo.net.http_get", side_effect=mock_get):
        download_file("http://example.com/file.txt", save_to=tmp_path / "x", overwrite=True)

    assert "timeout" in captured_kwargs, "timeout should be passed to http_get"
    assert captured_kwargs["timeout"] == (10, 60), f"Expected (10, 60), got {captured_kwargs['timeout']}"


def test_download_file_passes_custom_timeout(tmp_path):
    """download_file should pass custom timeout to http_get."""
    captured_kwargs = {}

    def mock_get(url, **kwargs):
        captured_kwargs.update(kwargs)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=[b"test"])
        return mock_response

    with patch("cleo.net.http_get", side_effect=mock_get):
        download_file(
            "http://example.com/file.txt",
            save_to=tmp_path / "x",
            overwrite=True,
            timeout=(5, 30),
        )

    assert "timeout" in captured_kwargs
    assert captured_kwargs["timeout"] == (5, 30), f"Expected (5, 30), got {captured_kwargs['timeout']}"
