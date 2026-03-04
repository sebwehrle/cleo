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
