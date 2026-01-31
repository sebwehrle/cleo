"""Test GWA GeoJSON parsing handles both string and dict responses."""

import json

from cleo.loaders import fetch_gwa_crs


class MockResponse:
    """Mock response object for requests.get."""

    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")


def test_fetch_gwa_crs_handles_string_encoded_json(monkeypatch):
    """
    Test that fetch_gwa_crs correctly parses when response.json() returns
    a string that itself is JSON text (double-encoded JSON).
    """
    # Inner JSON object as a dict
    inner_json = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        }
    }
    # Outer response: response.json() returns the JSON as a STRING
    outer_string = json.dumps(inner_json)

    def mock_get(url, headers=None):
        return MockResponse(outer_string)

    monkeypatch.setattr("cleo.loaders.requests.get", mock_get)

    result = fetch_gwa_crs("AUT")

    assert result == "urn:ogc:def:crs:OGC:1.3:CRS84"


def test_fetch_gwa_crs_handles_dict_json(monkeypatch):
    """
    Test that fetch_gwa_crs correctly parses when response.json() returns
    a dict directly (normal JSON parsing).
    """
    # response.json() returns a dict directly
    json_dict = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        }
    }

    def mock_get(url, headers=None):
        return MockResponse(json_dict)

    monkeypatch.setattr("cleo.loaders.requests.get", mock_get)

    result = fetch_gwa_crs("AUT")

    assert result == "urn:ogc:def:crs:OGC:1.3:CRS84"


def test_fetch_gwa_crs_raises_on_missing_crs_in_string(monkeypatch):
    """Test that fetch_gwa_crs raises ValueError when CRS is missing (string case)."""
    inner_json = {"type": "FeatureCollection", "features": []}
    outer_string = json.dumps(inner_json)

    def mock_get(url, headers=None):
        return MockResponse(outer_string)

    monkeypatch.setattr("cleo.loaders.requests.get", mock_get)

    try:
        fetch_gwa_crs("AUT")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "CRS missing" in str(e)
        assert "AUT" in str(e)
