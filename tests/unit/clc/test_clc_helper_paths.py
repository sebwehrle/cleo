"""Focused unit tests for CLC helper branches and error handling."""

from __future__ import annotations

import builtins
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import xarray as xr

import cleo.clc as clc
from tests.helpers.clc_fixtures import (
    JsonResponseStub,
    TEST_OAUTH_SERVICE_KEY,
    legacy_service_key,
    oauth_service_key,
    write_nested_zip_with_member,
    write_zip_with_member,
)


def _selected_dataset_payload(
    *, file_item: dict[str, object], dataset_id: str = "dataset-1"
) -> dict[str, list[dict[str, object]]]:
    """Build a minimal CLMS dataset-search payload with one dataset item."""
    return {
        "items": [
            {
                "UID": dataset_id,
                "downloadable_files": {"items": [file_item]},
            }
        ]
    }


def _sign_assertion_with_test_key(**kwargs) -> bytes:
    """Call the openssl assertion helper with the shared test private key."""
    return clc._sign_clms_assertion_with_openssl(
        private_key=TEST_OAUTH_SERVICE_KEY["private_key"],
        **kwargs,
    )


def test_canonical_landscape_valid_mask_raises_when_store_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = SimpleNamespace(path=tmp_path)

    def _missing_store(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError("missing")

    monkeypatch.setattr(clc, "open_zarr_dataset", _missing_store)

    with pytest.raises(RuntimeError, match="canonical landscape store"):
        clc._canonical_landscape_valid_mask(atlas)


def test_canonical_landscape_valid_mask_raises_when_valid_mask_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = SimpleNamespace(path=tmp_path)
    monkeypatch.setattr(clc, "open_zarr_dataset", lambda *args, **kwargs: xr.Dataset())

    with pytest.raises(RuntimeError, match="missing valid_mask"):
        clc._canonical_landscape_valid_mask(atlas)


def test_canonical_landscape_valid_mask_returns_dataset_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas = SimpleNamespace(path=tmp_path)
    ds = xr.Dataset({"valid_mask": (("y", "x"), [[True]])})
    monkeypatch.setattr(clc, "open_zarr_dataset", lambda *args, **kwargs: ds)

    out = clc._canonical_landscape_valid_mask(atlas)
    assert out.equals(ds["valid_mask"])


def test_json_object_from_response_rejects_non_object_payload() -> None:
    response = JsonResponseStub(payload=["not", "an", "object"])

    with pytest.raises(RuntimeError, match="returned non-object JSON payload"):
        clc._json_object_from_response(response, context="test")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5, 5),
        ("7", 7),
        (0, None),
        (-4, None),
        ("bad", None),
        (None, None),
    ],
)
def test_coerce_positive_int_handles_valid_and_invalid_inputs(value, expected) -> None:
    assert clc._coerce_positive_int(value) == expected


def test_normalize_service_key_fields_rejects_non_dict_and_missing_values() -> None:
    assert clc._normalize_service_key_fields([], fields=("service_name",)) is None
    assert clc._normalize_service_key_fields({"service_name": " "}, fields=("service_name",)) is None


def test_normalize_service_key_payload_supports_oauth_schema() -> None:
    payload = oauth_service_key()

    assert clc._normalize_service_key_payload(payload) == {
        "auth_kind": "oauth_service_key",
        **payload,
    }


def test_load_clms_service_key_payload_rejects_bad_inline_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLEO_CLMS_SERVICE_KEY_JSON", "{not-json")

    with pytest.raises(RuntimeError, match="not valid JSON"):
        clc._load_clms_service_key_payload()


def test_load_clms_service_key_payload_rejects_invalid_inline_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLEO_CLMS_SERVICE_KEY_JSON", json.dumps({"service_name": "svc"}))

    with pytest.raises(RuntimeError, match="service_name/secret/username"):
        clc._load_clms_service_key_payload()


def test_load_clms_service_key_payload_rejects_missing_key_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLEO_CLMS_SERVICE_KEY_JSON", raising=False)
    monkeypatch.setenv("CLEO_CLMS_SERVICE_KEY_PATH", "/tmp/does-not-exist.json")

    with pytest.raises(RuntimeError, match="points to a missing file"):
        clc._load_clms_service_key_payload()


def test_load_clms_service_key_payload_rejects_invalid_key_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path = tmp_path / "service-key.json"
    key_path.write_text("{bad", encoding="utf-8")
    monkeypatch.delenv("CLEO_CLMS_SERVICE_KEY_JSON", raising=False)
    monkeypatch.setenv("CLEO_CLMS_SERVICE_KEY_PATH", str(key_path))

    with pytest.raises(RuntimeError, match="file is not valid JSON"):
        clc._load_clms_service_key_payload()


def test_load_clms_service_key_payload_loads_legacy_path_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path = tmp_path / "service-key.json"
    key_path.write_text(
        json.dumps(legacy_service_key()),
        encoding="utf-8",
    )
    monkeypatch.delenv("CLEO_CLMS_SERVICE_KEY_JSON", raising=False)
    monkeypatch.delenv("CLEO_CLMS_SERVICE_KEY_PATH", raising=False)
    monkeypatch.setenv("CLMS_API_SERVICE_KEY", str(key_path))

    assert clc._load_clms_service_key_payload() == {"auth_kind": "legacy", **legacy_service_key()}


def test_load_clms_service_key_payload_rejects_invalid_file_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path = tmp_path / "service-key.json"
    key_path.write_text(json.dumps({"service_name": "svc"}), encoding="utf-8")
    monkeypatch.delenv("CLEO_CLMS_SERVICE_KEY_JSON", raising=False)
    monkeypatch.setenv("CLEO_CLMS_SERVICE_KEY_PATH", str(key_path))

    with pytest.raises(RuntimeError, match="Path:"):
        clc._load_clms_service_key_payload()


def test_load_clms_service_key_payload_returns_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLEO_CLMS_SERVICE_KEY_JSON", raising=False)
    monkeypatch.delenv("CLEO_CLMS_SERVICE_KEY_PATH", raising=False)
    monkeypatch.delenv("CLMS_API_SERVICE_KEY", raising=False)

    assert clc._load_clms_service_key_payload() is None


def test_sign_clms_assertion_with_openssl_raises_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing_binary(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError("openssl")

    monkeypatch.setattr(clc.subprocess, "run", _missing_binary)

    with pytest.raises(RuntimeError, match="requires PyJWT or an available openssl binary"):
        _sign_assertion_with_test_key(signing_input=b"abc")


def test_sign_clms_assertion_with_openssl_surfaces_subprocess_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clc.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b"", stderr=b"bad key"),
    )

    with pytest.raises(RuntimeError, match="openssl output: bad key"):
        _sign_assertion_with_test_key(signing_input=b"abc")


def test_sign_clms_assertion_with_openssl_uses_stdout_when_stderr_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        clc.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b"bad key", stderr=b""),
    )

    with pytest.raises(RuntimeError, match="openssl output: bad key"):
        _sign_assertion_with_test_key(signing_input=b"abc")


def test_sign_clms_assertion_with_openssl_returns_signature_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        clc.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"signature", stderr=b""),
    )

    assert _sign_assertion_with_test_key(signing_input=b"abc") == b"signature"


def test_sign_clms_oauth_assertion_uses_jwt_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jwt = SimpleNamespace(encode=lambda *args, **kwargs: "signed-token")
    monkeypatch.setitem(sys.modules, "jwt", fake_jwt)

    out = clc._sign_clms_oauth_assertion(
        header={"kid": "kid"},
        claims={"sub": "user"},
        private_key=TEST_OAUTH_SERVICE_KEY["private_key"],
    )

    assert out == "signed-token"


def test_sign_clms_oauth_assertion_falls_back_to_openssl(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "jwt":
            raise ModuleNotFoundError("jwt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(
        clc,
        "_sign_clms_assertion_with_openssl",
        lambda **kwargs: b"\x01\x02",
    )

    token = clc._sign_clms_oauth_assertion(
        header={"kid": "kid"},
        claims={"sub": "user"},
        private_key=TEST_OAUTH_SERVICE_KEY["private_key"],
    )

    assert token.count(".") == 2


def test_request_oauth_access_token_returns_none_for_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    response = JsonResponseStub(payload=["unexpected"], status_code=202)
    monkeypatch.setattr(clc, "http_post", lambda *args, **kwargs: response)

    status_code, payload = clc._request_oauth_access_token(
        token_uri="https://token.example.test",
        payload={"grant_type": "client_credentials"},
    )

    assert status_code == 202
    assert payload is None
    assert response.closed is True


def test_request_oauth_access_token_handles_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    response = JsonResponseStub(payload=..., json_error=ValueError("bad json"))
    monkeypatch.setattr(clc, "http_post", lambda *args, **kwargs: response)

    status_code, payload = clc._request_oauth_access_token(
        token_uri="https://token.example.test",
        payload={"grant_type": "client_credentials"},
    )

    assert status_code == 200
    assert payload is None


def test_oauth_access_token_falls_back_between_request_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    service_payload = oauth_service_key()
    monkeypatch.setattr(clc, "_oauth_assertion_claim_candidates", lambda payload: [{"sub": "user"}])
    monkeypatch.setattr(clc, "_sign_clms_oauth_assertion", lambda **kwargs: "signed-token")

    requests_seen: list[dict[str, str]] = []

    def _fake_request(*, token_uri: str, payload: dict[str, str]):
        del token_uri
        requests_seen.append(payload)
        if payload["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer":
            return 400, {"error": "invalid_grant", "error_description": "jwt rejected"}
        return 200, {"access_token": "access-token"}

    monkeypatch.setattr(clc, "_request_oauth_access_token", _fake_request)

    assert clc._oauth_access_token(service_payload) == "access-token"
    assert [payload["grant_type"] for payload in requests_seen] == [
        "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "client_credentials",
    ]


def test_oauth_access_token_raises_with_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    service_payload = oauth_service_key()
    monkeypatch.setattr(clc, "_oauth_assertion_claim_candidates", lambda payload: [{"sub": "user"}])
    monkeypatch.setattr(clc, "_sign_clms_oauth_assertion", lambda **kwargs: "signed-token")
    monkeypatch.setattr(
        clc,
        "_request_oauth_access_token",
        lambda **kwargs: (401, {"error": "invalid_client", "error_description": "bad assertion"}),
    )

    with pytest.raises(RuntimeError, match="bad assertion"):
        clc._oauth_access_token(service_payload)


def test_oauth_access_token_uses_http_status_when_payload_not_object(monkeypatch: pytest.MonkeyPatch) -> None:
    service_payload = oauth_service_key()
    monkeypatch.setattr(clc, "_oauth_assertion_claim_candidates", lambda payload: [{"sub": "user"}])
    monkeypatch.setattr(clc, "_sign_clms_oauth_assertion", lambda **kwargs: "signed-token")
    monkeypatch.setattr(clc, "_request_oauth_access_token", lambda **kwargs: (503, None))

    with pytest.raises(RuntimeError, match="HTTP 503"):
        clc._oauth_access_token(service_payload)


def test_legacy_access_token_raises_without_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clc, "_http_post_json", lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError, match="did not include access_token"):
        clc._legacy_access_token(legacy_service_key(service_name="svc", secret="sec"))  # pragma: allowlist secret


def test_clms_access_token_prefers_existing_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLEO_CLMS_ACCESS_TOKEN", "env-token")

    assert clc._clms_access_token() == "env-token"


def test_clms_access_token_rejects_unknown_service_key_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLEO_CLMS_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(clc, "_load_clms_service_key_payload", lambda: {"auth_kind": "unknown"})

    with pytest.raises(RuntimeError, match="Unsupported CLMS service-key auth schema"):
        clc._clms_access_token()


def test_candidate_search_items_handles_list_and_invalid_payloads() -> None:
    assert clc._candidate_search_items([{"UID": "dataset-1"}, "ignored"]) == [{"UID": "dataset-1"}]
    assert clc._candidate_search_items("invalid") == []


def test_dataset_identifier_returns_none_when_missing() -> None:
    assert clc._dataset_identifier({"name": "dataset"}) is None


def test_downloadable_file_items_handles_empty_dict_and_list_payloads() -> None:
    assert clc._downloadable_file_items({"items": "invalid"}) == []
    assert clc._downloadable_file_items([{"UID": "file-1"}, "ignored"]) == [{"UID": "file-1"}]


def test_file_identifier_returns_none_when_missing() -> None:
    assert clc._file_identifier({"name": "file"}) is None


def test_select_clms_dataset_file_matches_exact_filename() -> None:
    payload = _selected_dataset_payload(
        file_item={
            "UID": "file-1",
            "downloadable_file_name": "U2018_CLC2018_V2020_20u1.tif",
        }
    )

    assert clc._select_clms_dataset_file(
        search_payload=payload,
        expected_filename="U2018_CLC2018_V2020_20u1.tif",
    ) == ("dataset-1", "file-1")


def test_select_clms_dataset_file_skips_invalid_dataset_and_files() -> None:
    payload = {
        "items": [
            {"downloadable_files": {"items": [{"UID": "file-1", "file": "U2018_CLC2018_V2020_20u1.tif"}]}},
            {"UID": "dataset-2", "downloadable_files": {"items": [{"UID": "", "file": ""}]}},
        ]
    }

    assert (
        clc._select_clms_dataset_file(
            search_payload=payload,
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )
        is None
    )


def test_select_clms_dataset_file_matches_expected_stem() -> None:
    payload = _selected_dataset_payload(file_item={"UID": "file-1", "file": "u2018_clc2018_v2020_20u1"})

    assert clc._select_clms_dataset_file(
        search_payload=payload,
        expected_filename="U2018_CLC2018_V2020_20u1.tif",
    ) == ("dataset-1", "file-1")


def test_select_clms_dataset_file_uses_raster_geotiff_fallback() -> None:
    payload = [
        {
            "UID": "dataset-1",
            "downloadable_files": [
                {
                    "@id": "file-1",
                    "downloadable_file_name": "clc2018_raster_100m_bundle",
                    "format": "GeoTIFF",
                    "type": "Raster",
                    "path": "H:\\CLC\\2018\\bundle.zip",
                }
            ],
        }
    ]

    assert clc._select_clms_dataset_file(
        search_payload=payload,
        expected_filename="U2018_CLC2018_V2020_20u1.tif",
    ) == ("dataset-1", "file-1")


def test_extract_download_urls_handles_nested_payload_shapes() -> None:
    payload = {
        "output": {
            "downloads": [
                {"downloadURL": " https://download.example.test/one.zip "},
                {"download_url": "https://download.example.test/two.zip"},
            ]
        }
    }

    assert clc._extract_download_urls(payload) == [
        "https://download.example.test/one.zip",
        "https://download.example.test/two.zip",
    ]


def test_extract_download_urls_recurses_lists_and_nested_values() -> None:
    payload = [
        "ignored",
        {
            "other": {
                "DownloadURLs": [
                    "https://download.example.test/one.zip",
                    " https://download.example.test/two.zip ",
                ]
            }
        },
    ]

    assert clc._extract_download_urls(payload) == [
        "https://download.example.test/one.zip",
        "https://download.example.test/two.zip",
    ]


def test_artifact_url_candidates_deduplicates_raw_and_normalized_urls() -> None:
    assert clc._artifact_url_candidates(
        [
            "",
            "https://copernicus-fme.eea.europa.eu/fmedatadownload/results/123.zip",
            "https://copernicus-fme.eea.europa.eu/clmsdatadownload/results/123.zip",
        ]
    ) == [
        "https://copernicus-fme.eea.europa.eu/fmedatadownload/results/123.zip",
        "https://copernicus-fme.eea.europa.eu/clmsdatadownload/results/123.zip",
    ]


def test_normalize_clms_artifact_url_keeps_empty_string_empty() -> None:
    assert clc._normalize_clms_artifact_url("  ") == ""


def test_finished_request_urls_filters_items_by_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        clc,
        "_http_get_json",
        lambda *args, **kwargs: {
            "items": [
                "ignored",
                {"TaskID": "other-task", "DownloadURL": ["https://download.example.test/other.zip"]},
                {"task_id": "target-task", "DownloadURL": ["https://download.example.test/target.zip"]},
            ]
        },
    )

    assert clc._finished_request_urls(access_token="token", task_id="target-task") == [
        "https://download.example.test/target.zip"
    ]


def test_finished_request_urls_returns_empty_when_items_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clc, "_http_get_json", lambda *args, **kwargs: {"items": "invalid"})
    assert clc._finished_request_urls(access_token="token", task_id="target-task") == []


def test_finished_request_urls_returns_empty_when_no_task_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        clc,
        "_http_get_json",
        lambda *args, **kwargs: {"items": [{"TaskID": "other-task", "DownloadURL": ["https://download.example.test"]}]},
    )
    assert clc._finished_request_urls(access_token="token", task_id="target-task") == []


def test_submit_clms_download_task_uses_request_token_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        clc,
        "_http_get_json",
        lambda *args, **kwargs: {
            "items": [{"UID": "dataset-1", "downloadable_files": {"items": [{"UID": "file-1", "file": "CLC"}]}}]
        },
    )
    monkeypatch.setattr(clc, "_select_clms_dataset_file", lambda **kwargs: ("dataset-1", "file-1"))
    monkeypatch.setattr(clc, "_http_post_json", lambda *args, **kwargs: {"requestToken": "request-token"})

    assert (
        clc._submit_clms_download_task(
            source="clc2018",
            access_token="token",
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )
        == "request-token"
    )


def test_submit_clms_download_task_raises_when_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clc, "_http_get_json", lambda *args, **kwargs: {"items": []})

    with pytest.raises(RuntimeError, match="did not return downloadable file"):
        clc._submit_clms_download_task(
            source="clc2018",
            access_token="token",
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )


def test_submit_clms_download_task_raises_without_task_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        clc,
        "_http_get_json",
        lambda *args, **kwargs: {"items": [{"UID": "dataset-1", "downloadable_files": {"items": [{"UID": "file-1"}]}}]},
    )
    monkeypatch.setattr(clc, "_select_clms_dataset_file", lambda **kwargs: ("dataset-1", "file-1"))
    monkeypatch.setattr(clc, "_http_post_json", lambda *args, **kwargs: {"TaskIds": ["invalid", {}]})

    with pytest.raises(RuntimeError, match="did not include a task identifier"):
        clc._submit_clms_download_task(
            source="clc2018",
            access_token="token",
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )


def test_request_clms_download_url_raises_when_finished_without_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clc, "_submit_clms_download_task", lambda **kwargs: "task-1")
    monkeypatch.setattr(clc, "_clms_status_payload", lambda **kwargs: {"Status": "finished_ok"})
    monkeypatch.setattr(clc, "_finished_request_urls", lambda **kwargs: [])

    with pytest.raises(RuntimeError, match="finished but no download URL"):
        clc._request_clms_download_url(
            source="clc2018",
            access_token="token",
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )


def test_request_clms_download_url_raises_when_request_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clc, "_submit_clms_download_task", lambda **kwargs: "task-1")
    monkeypatch.setattr(clc, "_clms_status_payload", lambda **kwargs: {"Status": "failed"})

    with pytest.raises(RuntimeError, match="request failed"):
        clc._request_clms_download_url(
            source="clc2018",
            access_token="token",
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )


def test_request_clms_download_url_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clc, "_submit_clms_download_task", lambda **kwargs: "task-1")
    monkeypatch.setattr(clc, "_clms_status_payload", lambda **kwargs: {"Status": "running"})
    monkeypatch.setattr(clc.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(clc, "_CLMS_POLL_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(clc, "_CLMS_POLL_SLEEP_SECONDS", 1.0)

    with pytest.raises(RuntimeError, match="Timed out waiting for CLMS request completion"):
        clc._request_clms_download_url(
            source="clc2018",
            access_token="token",
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )


def test_clms_artifact_request_headers_returns_original_headers_for_non_fme_host() -> None:
    headers = {"Authorization": "Bearer token"}
    assert (
        clc._clms_artifact_request_headers(headers, download_url="https://download.example.test/artifact.zip")
        == headers
    )


def test_zip_member_candidates_prefers_yeared_raster_when_exact_name_missing(tmp_path: Path) -> None:
    zip_path = tmp_path / "artifact.zip"
    write_zip_with_member(zip_path, "bundle/CLC2018_RASTER_100m.tif", b"data")

    with zipfile.ZipFile(zip_path, "r") as archive:
        assert clc._zip_member_candidates(
            archive,
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        ) == ["bundle/CLC2018_RASTER_100m.tif"]


def test_zip_member_candidates_prefers_exact_filename(tmp_path: Path) -> None:
    zip_path = tmp_path / "artifact.zip"
    write_zip_with_member(zip_path, "bundle/U2018_CLC2018_V2020_20u1.tif", b"data")

    with zipfile.ZipFile(zip_path, "r") as archive:
        assert clc._zip_member_candidates(
            archive,
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        ) == ["bundle/U2018_CLC2018_V2020_20u1.tif"]


def test_zip_member_candidates_returns_all_tiffs_when_no_preferred_match(tmp_path: Path) -> None:
    zip_path = tmp_path / "artifact.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("one.tif", b"one")
        archive.writestr("two.tif", b"two")

    with zipfile.ZipFile(zip_path, "r") as archive:
        assert clc._zip_member_candidates(
            archive,
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        ) == ["one.tif", "two.tif"]


def test_extract_tif_from_clms_zip_extracts_direct_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "artifact.zip"
    dest_path = tmp_path / "out.tif"
    write_zip_with_member(zip_path, "bundle/U2018_CLC2018_V2020_20u1.tif", b"tif-bytes")

    out = clc._extract_tif_from_clms_zip(
        zip_path=zip_path,
        dest_path=dest_path,
        expected_filename="U2018_CLC2018_V2020_20u1.tif",
    )

    assert out == dest_path
    assert dest_path.read_bytes() == b"tif-bytes"


def test_extract_tif_from_clms_zip_raises_without_tiff(tmp_path: Path) -> None:
    zip_path = tmp_path / "artifact.zip"
    write_zip_with_member(zip_path, "README.txt", b"not-a-raster")

    with pytest.raises(RuntimeError, match="did not contain a TIFF file"):
        clc._extract_tif_from_clms_zip(
            zip_path=zip_path,
            dest_path=tmp_path / "out.tif",
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )


def test_extract_tif_from_clms_zip_raises_when_nested_zip_has_no_tiff(tmp_path: Path) -> None:
    zip_path = tmp_path / "artifact.zip"
    write_nested_zip_with_member(zip_path, "nested.zip", "README.txt", b"not-a-raster")

    with pytest.raises(RuntimeError, match="did not contain a TIFF file"):
        clc._extract_tif_from_clms_zip(
            zip_path=zip_path,
            dest_path=tmp_path / "out.tif",
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )


def test_download_clms_artifact_downloads_direct_tiff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        out_path.write_bytes(b"tif-bytes")

    monkeypatch.setattr(clc, "download_to_path", _fake_download)

    out = clc._download_clms_artifact(
        download_url="https://download.example.test/source.tif",
        source_path=tmp_path / "source.tif",
        headers={"Authorization": "Bearer token"},
        expected_filename="U2018_CLC2018_V2020_20u1.tif",
    )

    assert out.read_bytes() == b"tif-bytes"
    assert captured["url"] == "https://download.example.test/source.tif"
    assert captured["headers"] == {"Authorization": "Bearer token"}


def test_download_clms_artifact_extracts_zip_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clc.time, "sleep", lambda *args, **kwargs: None)

    def _fake_download(url: str, out_path: Path, **kwargs) -> None:
        del url, kwargs
        write_zip_with_member(out_path, "bundle/U2018_CLC2018_V2020_20u1.tif", b"tif-bytes")

    monkeypatch.setattr(clc, "download_to_path", _fake_download)
    out = clc._download_clms_artifact(
        download_url="https://download.example.test/source.zip",
        source_path=tmp_path / "source.tif",
        headers=None,
        expected_filename="U2018_CLC2018_V2020_20u1.tif",
    )

    assert out.read_bytes() == b"tif-bytes"


def test_download_clms_artifact_reraises_non_retryable_request_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _failing_download(*args, **kwargs):
        del args, kwargs
        err = clc.RequestException("boom")
        err.response = SimpleNamespace(status_code=500)
        raise err

    monkeypatch.setattr(clc, "download_to_path", _failing_download)

    with pytest.raises(clc.RequestException, match="boom"):
        clc._download_clms_artifact(
            download_url="https://download.example.test/artifact.zip",
            source_path=tmp_path / "source.tif",
            headers=None,
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )


def test_download_clms_artifact_raises_after_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clc.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(clc, "_CLMS_ARTIFACT_READY_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(clc, "_CLMS_ARTIFACT_READY_SLEEP_SECONDS", 1.0)
    monkeypatch.setattr(
        clc,
        "download_to_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("not ready")),
    )

    with pytest.raises(RuntimeError, match="did not become downloadable"):
        clc._download_clms_artifact(
            download_url="https://download.example.test/artifact.zip",
            source_path=tmp_path / "source.tif",
            headers=None,
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )


def test_download_clms_artifact_retries_bad_zip_then_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clc.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(clc, "_CLMS_ARTIFACT_READY_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(clc, "_CLMS_ARTIFACT_READY_SLEEP_SECONDS", 1.0)

    def _bad_zip_download(url: str, out_path: Path, **kwargs) -> None:
        del url, kwargs
        out_path.write_bytes(b"not-a-zip")

    monkeypatch.setattr(clc, "download_to_path", _bad_zip_download)

    with pytest.raises(RuntimeError, match="did not become downloadable"):
        clc._download_clms_artifact(
            download_url="https://download.example.test/artifact.zip",
            source_path=tmp_path / "source.tif",
            headers=None,
            expected_filename="U2018_CLC2018_V2020_20u1.tif",
        )
