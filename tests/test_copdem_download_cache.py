"""Test Copernicus DEM tile download and caching."""

import pytest
from pathlib import Path

from cleo.copdem import (
    copdem_tile_url,
    copdem_tile_cache_path,
    download_copdem_tile,
)


class MockResponse:
    """Mock response object for requests.get."""

    def __init__(self, status_code=200, content=b"FAKE_TIF"):
        self.status_code = status_code
        self._content = content

    def iter_content(self, chunk_size=8192):
        yield self._content


TILE_ID = "Copernicus_DSM_COG_10_N46_00_E009_00_DEM"
ISO3 = "AUT"


def test_copdem_tile_url():
    """Test that copdem_tile_url returns the correct S3 URL."""
    url = copdem_tile_url(TILE_ID)
    expected = f"https://copernicus-dem-30m.s3.amazonaws.com/{TILE_ID}/{TILE_ID}.tif"
    assert url == expected


def test_copdem_tile_cache_path(tmp_path):
    """Test that copdem_tile_cache_path returns the correct path structure."""
    path = copdem_tile_cache_path(tmp_path, ISO3, TILE_ID)
    expected = tmp_path / ISO3 / "copdem" / TILE_ID / f"{TILE_ID}.tif"
    assert path == expected


def test_download_copdem_tile_success(monkeypatch, tmp_path):
    """Test successful download of a Copernicus DEM tile."""
    call_count = {"count": 0}
    fake_content = b"FAKE_TIF_CONTENT"

    def mock_get(url, stream=False, timeout=None, **kwargs):
        call_count["count"] += 1
        return MockResponse(status_code=200, content=fake_content)

    monkeypatch.setattr("cleo.copdem.requests.get", mock_get)

    # Download tile
    result_path = download_copdem_tile(tmp_path, ISO3, TILE_ID)

    # Assert path is correct
    expected_path = tmp_path / ISO3 / "copdem" / TILE_ID / f"{TILE_ID}.tif"
    assert result_path == expected_path

    # Assert file exists with expected content
    assert result_path.exists()
    assert result_path.read_bytes() == fake_content

    # Assert requests.get was called once
    assert call_count["count"] == 1


def test_download_copdem_tile_cache_hit(monkeypatch, tmp_path):
    """Test that cached files are not re-downloaded."""
    call_count = {"count": 0}
    fake_content = b"FAKE_TIF_CONTENT"

    def mock_get(url, stream=False, timeout=None, **kwargs):
        call_count["count"] += 1
        return MockResponse(status_code=200, content=fake_content)

    monkeypatch.setattr("cleo.copdem.requests.get", mock_get)

    # First download
    path1 = download_copdem_tile(tmp_path, ISO3, TILE_ID)
    assert call_count["count"] == 1

    # Second download with overwrite=False (default)
    path2 = download_copdem_tile(tmp_path, ISO3, TILE_ID, overwrite=False)
    assert call_count["count"] == 1  # Should NOT have called requests.get again
    assert path1 == path2


def test_download_copdem_tile_overwrite(monkeypatch, tmp_path):
    """Test that overwrite=True re-downloads the file."""
    call_count = {"count": 0}

    def mock_get(url, stream=False, timeout=None, **kwargs):
        call_count["count"] += 1
        return MockResponse(status_code=200, content=f"CONTENT_{call_count['count']}".encode())

    monkeypatch.setattr("cleo.copdem.requests.get", mock_get)

    # First download
    download_copdem_tile(tmp_path, ISO3, TILE_ID)
    assert call_count["count"] == 1

    # Second download with overwrite=True
    path = download_copdem_tile(tmp_path, ISO3, TILE_ID, overwrite=True)
    assert call_count["count"] == 2
    assert path.read_bytes() == b"CONTENT_2"


def test_download_copdem_tile_not_found(monkeypatch, tmp_path):
    """Test that HTTP 404 raises FileNotFoundError."""

    def mock_get(url, stream=False, timeout=None, **kwargs):
        return MockResponse(status_code=404)

    monkeypatch.setattr("cleo.copdem.requests.get", mock_get)

    with pytest.raises(FileNotFoundError) as exc_info:
        download_copdem_tile(tmp_path, ISO3, TILE_ID)

    assert TILE_ID in str(exc_info.value)
    assert "404" in str(exc_info.value)
