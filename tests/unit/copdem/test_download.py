"""copdem: test_download.
Merged test file (imports preserved per chunk).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cleo.copdem import (
    copdem_tile_cache_path,
    copdem_tile_url,
    download_copdem_tile,
    download_copdem_tiles_for_bbox,
    tiles_for_bbox,
)

# --- merged from tests/_staging/test_copdem_download_cache.py ---


class MockResponse:
    """Mock response object for requests.get."""

    def __init__(self, status_code: int = 200, content: bytes = b"FAKE_TIF"):
        self.status_code = status_code
        self._content = content
        self.closed = False

    def iter_content(self, chunk_size: int = 8192):
        yield self._content

    def raise_for_status(self) -> None:
        if self.status_code >= 400 and self.status_code != 404:
            raise Exception(f"HTTP Error: {self.status_code}")

    def close(self) -> None:
        self.closed = True


TILE_ID = "Copernicus_DSM_COG_10_N46_00_E009_00_DEM"
ISO3 = "AUT"


def test_copdem_tile_url() -> None:
    """Test that copdem_tile_url returns the correct S3 URL."""
    url = copdem_tile_url(TILE_ID)
    expected = f"https://copernicus-dem-30m.s3.amazonaws.com/{TILE_ID}/{TILE_ID}.tif"
    assert url == expected


def test_copdem_tile_cache_path(tmp_path: Path) -> None:
    """Test that copdem_tile_cache_path returns the correct path structure."""
    path = copdem_tile_cache_path(tmp_path, ISO3, TILE_ID)
    expected = tmp_path / "data" / "raw" / ISO3 / "copdem" / TILE_ID / f"{TILE_ID}.tif"
    assert path == expected


def test_download_copdem_tile_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test successful download of a Copernicus DEM tile."""
    call_count = {"count": 0}
    fake_content = b"FAKE_TIF_CONTENT"
    last_response = {"obj": None}

    def mock_get(url, stream=False, timeout=None, **kwargs):
        call_count["count"] += 1
        resp = MockResponse(status_code=200, content=fake_content)
        last_response["obj"] = resp
        return resp

    monkeypatch.setattr("cleo.copdem.requests.get", mock_get)

    # Download tile
    result_path = download_copdem_tile(tmp_path, ISO3, TILE_ID)

    # Assert path is correct
    expected_path = tmp_path / "data" / "raw" / ISO3 / "copdem" / TILE_ID / f"{TILE_ID}.tif"
    assert result_path == expected_path

    # Assert file exists with expected content
    assert result_path.exists()
    assert result_path.read_bytes() == fake_content

    # Assert requests.get was called once
    assert call_count["count"] == 1

    # Assert response was closed
    assert last_response["obj"].closed is True


def test_download_copdem_tile_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_download_copdem_tile_overwrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_download_copdem_tile_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that HTTP 404 raises FileNotFoundError."""

    def mock_get(url, stream=False, timeout=None, **kwargs):
        return MockResponse(status_code=404)

    monkeypatch.setattr("cleo.copdem.requests.get", mock_get)

    with pytest.raises(FileNotFoundError) as exc_info:
        download_copdem_tile(tmp_path, ISO3, TILE_ID)

    assert TILE_ID in str(exc_info.value)
    assert "404" in str(exc_info.value)


# --- merged from tests/_staging/test_copdem_download_cleans_part_on_failure.py ---


class FlakyResponse:
    def __init__(self):
        self.status_code = 200
        self.closed = False
        self._yielded = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        if not self._yielded:
            self._yielded = True
            yield b"PARTIAL"
        raise RuntimeError("network dropped")

    def close(self):
        self.closed = True


def test_download_copdem_tile_cleans_part(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    iso3 = "AUT"
    tile_id = "Copernicus_DSM_COG_10_N46_00_E009_00_DEM"
    resp = FlakyResponse()

    def mock_get(url, stream=False, timeout=None, **kwargs):
        return resp

    monkeypatch.setattr("cleo.copdem.requests.get", mock_get)

    with pytest.raises(RuntimeError, match="network dropped"):
        download_copdem_tile(tmp_path, iso3, tile_id, overwrite=True, timeout_s=0.001)

    dest = copdem_tile_cache_path(tmp_path, iso3, tile_id)
    part = dest.with_suffix(".tif.part")
    assert not part.exists()
    assert resp.closed is True


# --- merged from tests/_staging/test_copdem_download_tiles_for_bbox.py ---


def test_download_copdem_tiles_for_bbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Test that download_copdem_tiles_for_bbox downloads all tiles for a bbox
    in the correct (lexicographic) order.

    Bbox: min_lon=9.2, max_lon=11.1, min_lat=46.0, max_lat=47.0
    Expected:
        lon_deg: floor(9.2)=9, ceil(11.1)-1=11 => [9, 10, 11] (3 values)
        lat_deg: floor(46.0)=46, ceil(47.0)-1=46 => [46] (1 value)
        Total: 3 tiles
    """
    min_lon = 9.2
    max_lon = 11.1
    min_lat = 46.0
    max_lat = 47.0
    iso3 = "AUT"

    # Oracle: compute expected tile_ids independently
    def oracle_tile_id(lat_deg, lon_deg):
        ns = "N" if lat_deg >= 0 else "S"
        ew = "E" if lon_deg >= 0 else "W"
        return f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_{ew}{abs(lon_deg):03d}_00_DEM"

    expected_tile_ids = []
    for lat_deg in [46]:
        for lon_deg in [9, 10, 11]:
            expected_tile_ids.append(oracle_tile_id(lat_deg, lon_deg))
    expected_tile_ids.sort()

    calls: list[tuple[str, str, bool]] = []

    def mock_download_copdem_tile(base_dir, iso3_arg, tile_id, *, overwrite=False):
        calls.append((iso3_arg, tile_id, overwrite))
        return Path(base_dir) / "data" / "raw" / iso3_arg / "copdem" / tile_id / f"{tile_id}.tif"

    monkeypatch.setattr("cleo.copdem.download_copdem_tile", mock_download_copdem_tile)

    result_paths = download_copdem_tiles_for_bbox(
        tmp_path,
        iso3,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        overwrite=False,
    )

    assert len(calls) == 3, f"Expected 3 calls, got {len(calls)}"

    actual_tile_ids = [call[1] for call in calls]
    assert actual_tile_ids == expected_tile_ids, (
        f"Tile IDs mismatch:\nExpected: {expected_tile_ids}\nGot: {actual_tile_ids}"
    )

    for iso3_arg, _, overwrite_arg in calls:
        assert iso3_arg == iso3
        assert overwrite_arg is False

    expected_paths = [
        tmp_path / "data" / "raw" / iso3 / "copdem" / tile_id / f"{tile_id}.tif"
        for tile_id in expected_tile_ids
    ]
    assert result_paths == expected_paths


def test_download_copdem_tiles_for_bbox_overwrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that overwrite parameter is passed through correctly."""
    calls: list[tuple[str, str, bool]] = []

    def mock_download_copdem_tile(base_dir, iso3_arg, tile_id, *, overwrite=False):
        calls.append((iso3_arg, tile_id, overwrite))
        return Path(base_dir) / "data" / "raw" / iso3_arg / "copdem" / tile_id / f"{tile_id}.tif"

    monkeypatch.setattr("cleo.copdem.download_copdem_tile", mock_download_copdem_tile)

    download_copdem_tiles_for_bbox(
        tmp_path,
        "AUT",
        min_lon=9.2,
        min_lat=46.0,
        max_lon=11.1,
        max_lat=47.0,
        overwrite=True,
    )

    for _, _, overwrite_arg in calls:
        assert overwrite_arg is True


def test_download_copdem_tiles_for_bbox_determinism(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that the function returns tiles in deterministic order."""
    calls1: list[str] = []
    calls2: list[str] = []

    def make_mock(calls_list):
        def mock_download_copdem_tile(base_dir, iso3_arg, tile_id, *, overwrite=False):
            calls_list.append(tile_id)
            return Path(base_dir) / "data" / "raw" / iso3_arg / "copdem" / tile_id / f"{tile_id}.tif"

        return mock_download_copdem_tile

    monkeypatch.setattr("cleo.copdem.download_copdem_tile", make_mock(calls1))
    result1 = download_copdem_tiles_for_bbox(
        tmp_path, "AUT", min_lon=9.2, min_lat=46.0, max_lon=11.1, max_lat=47.0
    )

    monkeypatch.setattr("cleo.copdem.download_copdem_tile", make_mock(calls2))
    result2 = download_copdem_tiles_for_bbox(
        tmp_path, "AUT", min_lon=9.2, min_lat=46.0, max_lon=11.1, max_lat=47.0
    )

    assert calls1 == calls2, "Tile download order should be deterministic"
    assert result1 == result2, "Returned paths should be deterministic"


def test_tiles_for_bbox_degenerate_raises() -> None:
    with pytest.raises(ValueError, match="Degenerate bbox"):
        tiles_for_bbox(10.0, 45.0, 10.0, 46.0)
