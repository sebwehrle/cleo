"""Test GWA v4 CRS inference from country GeoJSON metadata."""

import numpy as np
import rasterio
import rioxarray as rxr
import tempfile
import os
from unittest.mock import MagicMock

from cleo.loaders import fetch_gwa_crs, ensure_crs_from_gwa


class MockResponse:
    """Mock response object for requests.get."""

    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")


def test_fetch_gwa_crs_parses_response(monkeypatch):
    """Test that fetch_gwa_crs correctly parses CRS from GeoJSON response."""
    mock_json = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}}

    def mock_get(url, headers=None, **kwargs):
        return MockResponse(mock_json)

    monkeypatch.setattr("cleo.loaders.requests.get", mock_get)

    result = fetch_gwa_crs("AUT")

    assert result == "urn:ogc:def:crs:OGC:1.3:CRS84"


def test_ensure_crs_sets_crs_when_missing(monkeypatch, tmp_path):
    """Test that ensure_crs_from_gwa sets CRS when raster has no CRS."""
    mock_json = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}}

    def mock_get(url, headers=None, **kwargs):
        return MockResponse(mock_json)

    monkeypatch.setattr("cleo.loaders.requests.get", mock_get)

    # Create a tiny GeoTIFF without CRS
    tiff_path = tmp_path / "no_crs.tif"
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)

    with rasterio.open(
        tiff_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=data.dtype,
        crs=None,  # No CRS
        transform=rasterio.transform.from_bounds(0, 0, 1, 1, 2, 2),
    ) as dst:
        dst.write(data, 1)

    # Open via rioxarray and confirm rio.crs is None
    ds = rxr.open_rasterio(tiff_path)
    assert ds.rio.crs is None, "Expected raster to have no CRS initially"

    # Call ensure_crs_from_gwa
    result = ensure_crs_from_gwa(ds, "AUT")

    # Assert CRS is now set
    assert result.rio.crs is not None, "CRS should be set after ensure_crs_from_gwa"


def test_ensure_crs_does_not_call_api_when_crs_present(monkeypatch, tmp_path):
    """Test that ensure_crs_from_gwa does not call API when raster already has CRS."""
    call_count = {"count": 0}

    def mock_get(url, headers=None, **kwargs):
        call_count["count"] += 1
        return MockResponse({"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}})

    monkeypatch.setattr("cleo.loaders.requests.get", mock_get)

    # Create a tiny GeoTIFF with CRS (EPSG:4326)
    tiff_path = tmp_path / "with_crs.tif"
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)

    with rasterio.open(
        tiff_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",  # Has CRS
        transform=rasterio.transform.from_bounds(0, 0, 1, 1, 2, 2),
    ) as dst:
        dst.write(data, 1)

    # Open via rioxarray and confirm rio.crs is set
    ds = rxr.open_rasterio(tiff_path)
    assert ds.rio.crs is not None, "Expected raster to have CRS"

    # Call ensure_crs_from_gwa
    result = ensure_crs_from_gwa(ds, "AUT")

    # Assert requests.get was NOT called
    assert call_count["count"] == 0, "API should not be called when CRS is already present"

    # Assert CRS is still set
    assert result.rio.crs is not None


def test_fetch_gwa_crs_raises_on_missing_crs(monkeypatch):
    """Test that fetch_gwa_crs raises ValueError when CRS is missing from response."""
    mock_json = {"some": "data", "but": "no_crs"}

    def mock_get(url, headers=None, **kwargs):
        return MockResponse(mock_json)

    monkeypatch.setattr("cleo.loaders.requests.get", mock_get)

    try:
        fetch_gwa_crs("AUT")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "CRS missing" in str(e)
