"""net: http_post tests."""

from __future__ import annotations

from cleo.net import http_post


def test_http_post_forwards_json_payload_and_timeout(monkeypatch) -> None:
    captured: dict = {}

    class _Response:
        status_code = 200

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr("cleo.net.requests.post", fake_post)

    out = http_post(
        "https://example.com/login",
        headers={"Authorization": "Bearer token"},
        json_body={"username": "alice"},
        timeout=(5, 15),
        stream=False,
    )

    assert out.status_code == 200
    assert captured["url"] == "https://example.com/login"
    assert captured["headers"] == {"Authorization": "Bearer token"}
    assert captured["json"] == {"username": "alice"}
    assert captured["timeout"] == (5, 15)
    assert captured["stream"] is False


def test_http_post_forwards_form_payload(monkeypatch) -> None:
    captured: dict = {}

    class _Response:
        status_code = 200

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr("cleo.net.requests.post", fake_post)

    out = http_post(
        "https://example.com/oauth2-token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data_body={"grant_type": "client_credentials"},
        timeout=(3, 9),
        stream=False,
    )

    assert out.status_code == 200
    assert captured["url"] == "https://example.com/oauth2-token"
    assert captured["headers"] == {"Content-Type": "application/x-www-form-urlencoded"}
    assert captured["data"] == {"grant_type": "client_credentials"}
    assert captured["timeout"] == (3, 9)
    assert captured["stream"] is False
