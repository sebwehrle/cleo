"""Guardrail test: download_copdem_tile cleans up *.part on failure."""

from pathlib import Path

import pytest

from cleo.copdem import download_copdem_tile, copdem_tile_cache_path


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


def test_download_copdem_tile_cleans_part(monkeypatch, tmp_path):
    ISO3 = "AUT"
    TILE_ID = "Copernicus_DSM_COG_10_N46_00_E009_00_DEM"
    resp = FlakyResponse()

    def mock_get(url, stream=False, timeout=None, **kwargs):
        return resp

    monkeypatch.setattr("cleo.copdem.requests.get", mock_get)

    with pytest.raises(RuntimeError, match="network dropped"):
        download_copdem_tile(tmp_path, ISO3, TILE_ID, overwrite=True, timeout_s=0.001)

    dest = copdem_tile_cache_path(tmp_path, ISO3, TILE_ID)
    part = dest.with_suffix(".tif.part")
    assert not part.exists()
    assert resp.closed is True
