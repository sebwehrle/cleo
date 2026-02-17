"""
loaders: fetch_gwa_crs GeoJSON parsing tests.
Split from merged test_gwa.py
"""

import json
import pytest

from cleo.loaders import fetch_gwa_crs


class MockResponse:
    """Mock response object for http_get."""

    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")


def test_fetch_gwa_crs_handles_string_encoded_json(monkeypatch):
    inner_json = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    }
    outer_string = json.dumps(inner_json)

    def mock_get(url, headers=None, **kwargs):
        return MockResponse(outer_string)

    monkeypatch.setattr("cleo.net.http_get", mock_get)

    result = fetch_gwa_crs("AUT")
    assert result == "urn:ogc:def:crs:OGC:1.3:CRS84"


def test_fetch_gwa_crs_handles_dict_json(monkeypatch):
    json_dict = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    }

    def mock_get(url, headers=None, **kwargs):
        return MockResponse(json_dict)

    monkeypatch.setattr("cleo.net.http_get", mock_get)

    result = fetch_gwa_crs("AUT")
    assert result == "urn:ogc:def:crs:OGC:1.3:CRS84"


def test_fetch_gwa_crs_raises_on_missing_crs_in_string(monkeypatch):
    inner_json = {"type": "FeatureCollection", "features": []}
    outer_string = json.dumps(inner_json)

    def mock_get(url, headers=None, **kwargs):
        return MockResponse(outer_string)

    monkeypatch.setattr("cleo.net.http_get", mock_get)

    with pytest.raises(ValueError) as exc:
        fetch_gwa_crs("AUT")

    msg = str(exc.value)
    assert "CRS missing" in msg
    assert "AUT" in msg
