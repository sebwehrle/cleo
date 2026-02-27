"""net: download_to_path tests."""

import pytest

from cleo.net import download_to_path


class _Response:
    def __init__(self, status_code=200, chunks=None, raise_error=None):
        self.status_code = status_code
        self._chunks = chunks or []
        self._raise_error = raise_error
        self.closed = False

    def raise_for_status(self):
        if self._raise_error is not None:
            raise self._raise_error

    def iter_content(self, chunk_size=8192):
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


def test_download_to_path_atomic_cleanup_on_stream_error(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        return _Response(status_code=200, chunks=[b"partial", RuntimeError("boom")])

    monkeypatch.setattr("cleo.net.http_get", fake_get)

    target = tmp_path / "x.bin"
    with pytest.raises(RuntimeError, match="boom"):
        download_to_path("https://example.com/x.bin", target, overwrite=True)

    assert not target.exists()
    assert not (tmp_path / "x.bin.tmp").exists()


def test_download_to_path_http_403_raises_no_file_written(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        return _Response(status_code=403, raise_error=Exception("HTTP 403"))

    monkeypatch.setattr("cleo.net.http_get", fake_get)

    target = tmp_path / "bad.tif"
    with pytest.raises(Exception, match="HTTP 403"):
        download_to_path("https://example.com/bad.tif", target, overwrite=True)

    assert not target.exists()


def test_download_to_path_http_404_raises_filenotfound(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        return _Response(status_code=404)

    monkeypatch.setattr("cleo.net.http_get", fake_get)

    target = tmp_path / "missing.tif"
    with pytest.raises(FileNotFoundError, match="HTTP 404"):
        download_to_path("https://example.com/missing.tif", target, overwrite=True)

    assert not target.exists()


def test_download_to_path_http_200_writes_file(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        return _Response(status_code=200, chunks=[b"OK"])

    monkeypatch.setattr("cleo.net.http_get", fake_get)

    target = tmp_path / "good.tif"
    out = download_to_path("https://example.com/good.tif", target, overwrite=True)

    assert out == target
    assert target.exists()
    assert target.read_bytes() == b"OK"


def test_download_to_path_passes_default_timeout(tmp_path, monkeypatch):
    captured = {}

    def fake_get(*args, **kwargs):
        captured.update(kwargs)
        return _Response(status_code=200, chunks=[b"test"])

    monkeypatch.setattr("cleo.net.http_get", fake_get)

    download_to_path("https://example.com/file.txt", tmp_path / "x", overwrite=True)

    assert "timeout" in captured
    assert captured["timeout"] == (10, 60)


def test_download_to_path_passes_custom_timeout(tmp_path, monkeypatch):
    captured = {}

    def fake_get(*args, **kwargs):
        captured.update(kwargs)
        return _Response(status_code=200, chunks=[b"test"])

    monkeypatch.setattr("cleo.net.http_get", fake_get)

    download_to_path(
        "https://example.com/file.txt",
        tmp_path / "x",
        overwrite=True,
        timeout=(5, 30),
    )

    assert "timeout" in captured
    assert captured["timeout"] == (5, 30)


def test_download_to_path_skip_existing_without_overwrite(tmp_path, monkeypatch):
    target = tmp_path / "exists.bin"
    target.write_bytes(b"existing")

    def fake_get(*args, **kwargs):
        raise AssertionError("http_get should not be called")

    monkeypatch.setattr("cleo.net.http_get", fake_get)

    out = download_to_path("https://example.com/exists.bin", target, overwrite=False)

    assert out == target
    assert target.read_bytes() == b"existing"
