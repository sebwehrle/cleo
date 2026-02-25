"""Phase 3 loaders error-path tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import cleo.loaders as L
from cleo.unification.gwa_io import fetch_gwa_crs
from cleo.net import RequestException


class _MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def test_fetch_gwa_crs_wraps_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RequestException("timeout")

    monkeypatch.setattr("cleo.net.http_get", _raise)
    with pytest.raises(RuntimeError, match="Failed to fetch GWA CRS for AUT"):
        fetch_gwa_crs("AUT")


def test_fetch_gwa_crs_raises_on_invalid_double_encoded_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def _mock_get(*args, **kwargs):  # noqa: ANN002, ANN003
        return _MockResponse("{not-valid-json")

    monkeypatch.setattr("cleo.net.http_get", _mock_get)
    with pytest.raises(ValueError, match="Failed to parse GWA GeoJSON response"):
        fetch_gwa_crs("AUT")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"resolution": "BAD", "year": 2021, "crs": 4326}, "Invalid resolution"),
        ({"resolution": "03M", "year": 1999, "crs": 4326}, "Invalid year"),
        ({"resolution": "03M", "year": 2021, "crs": 9999}, "Invalid crs"),
    ],
)
def test_load_nuts_validates_inputs(kwargs: dict, expected: str, tmp_path) -> None:
    parent = SimpleNamespace(path=tmp_path)
    self = SimpleNamespace(parent=parent)
    with pytest.raises(ValueError, match=expected):
        L.load_nuts(self, **kwargs)


def test_get_clc_codes_reverse_reads_reverse_mapping(tmp_path) -> None:
    resources = tmp_path / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    payload = {"clc_codes": {"111": "urban"}, "clc_reverse": {"urban": 111}}
    (resources / "clc_codes.yml").write_text(json.dumps(payload), encoding="utf-8")

    parent = SimpleNamespace(path=tmp_path)
    self = SimpleNamespace(parent=parent)
    reverse = L.get_clc_codes(self, reverse=True)
    assert reverse["urban"] == 111

