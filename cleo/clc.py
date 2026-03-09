"""Corine Land Cover (CLC) source handling and cache preparation."""

from __future__ import annotations

import base64
import json
import io
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

from cleo.net import RequestException, download_to_path, http_get, http_post
from cleo.unification.clc_io import (
    load_clc_codes,
    prepare_clc_to_wind_grid,
    raster_band_count,
    wind_reference_template,
)
from cleo.unification.store_io import open_zarr_dataset


_CLMS_API_BASE_URL = "https://land.copernicus.eu/api"
_CLMS_POLL_SLEEP_SECONDS = 2.0
_CLMS_POLL_MAX_ATTEMPTS = 180
_CLMS_SOURCE_QUERY_KEY: dict[str, str] = {"clc2018": "CLC2018"}
_CLMS_LEGACY_SERVICE_KEY_FIELDS = ("service_name", "secret", "username")
_CLMS_OAUTH_SERVICE_KEY_FIELDS = ("client_id", "private_key", "user_id", "key_id", "token_uri")
_CLMS_JWT_EXP_SECONDS = 300
_CLMS_ARTIFACT_READY_SLEEP_SECONDS = 10.0
_CLMS_ARTIFACT_READY_MAX_ATTEMPTS = 18


CLC_SOURCES: dict[str, dict[str, str]] = {
    "clc2018": {
        "filename": "U2018_CLC2018_V2020_20u1.tif",
        # If left empty, materialize_clc() resolves a CLMS API-backed default
        # file download URL for the matching product/version.
        "url": "",
        "crs": "EPSG:3035",
    }
}


def source_cache_path(atlas_path: Path, source: str) -> Path:
    """Return cache path for the full CLC source raster."""
    if source not in CLC_SOURCES:
        raise ValueError(f"Unsupported CLC source {source!r}. Supported: {sorted(CLC_SOURCES)}")
    return Path(atlas_path) / "data" / "raw" / "clc" / CLC_SOURCES[source]["filename"]


def prepared_country_path(atlas_path: Path, country: str, source: str) -> Path:
    """Return cache path for country-prepared CLC raster aligned to wind grid."""
    return Path(atlas_path) / "data" / "raw" / str(country) / "clc" / f"{country}_{source}_windgrid.tif"


def _canonical_landscape_valid_mask(atlas, *, chunk_policy: dict[str, int] | None = None):
    """Load the canonical landscape validity mask for CLC preparation.

    :param atlas: Atlas instance whose canonical landscape store should be read.
    :param chunk_policy: Optional chunk policy for dataset opening.
    :returns: Canonical ``valid_mask`` DataArray.
    :rtype: xarray.DataArray
    :raises RuntimeError: If the canonical landscape store is missing
        ``valid_mask`` after ``build_canonical()``.
    """
    landscape_store = Path(atlas.path) / "landscape.zarr"
    try:
        land = open_zarr_dataset(landscape_store, chunk_policy=chunk_policy)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "CLC preparation requires the canonical landscape store. "
            "Run atlas.build() or atlas.build_canonical() before build_clc()."
        ) from exc
    if "valid_mask" not in land:
        raise RuntimeError("landscape.zarr missing valid_mask after canonical build.")
    return land["valid_mask"]


def default_category_name(atlas_path: Path, code: int) -> str | None:
    """Return default variable name for a CLC code, or ``None`` if unknown."""
    labels = load_clc_codes(Path(atlas_path))
    if int(code) not in labels:
        return None
    label = labels[int(code)].strip().casefold()
    name = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return name or None


def _json_object_from_response(response, *, context: str) -> dict[str, Any]:
    """Decode a JSON object from an HTTP response.

    :param response: HTTP response object.
    :param context: Human-readable operation context for error messages.
    :returns: Decoded JSON dictionary.
    :rtype: dict[str, typing.Any]
    :raises RuntimeError: If response body is not a JSON object.
    """
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} returned non-object JSON payload.")
    return payload


def _http_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: tuple[int, int] | int = (10, 60),
) -> dict[str, Any]:
    """Run GET request and parse JSON object payload.

    :param url: Endpoint URL.
    :param headers: Optional request headers.
    :param timeout: HTTP timeout configuration.
    :returns: Parsed JSON response body.
    :rtype: dict[str, typing.Any]
    :raises requests.RequestException: On network or HTTP errors.
    :raises RuntimeError: If response payload is invalid.
    """
    response = http_get(url, headers=headers, timeout=timeout, stream=False)
    try:
        response.raise_for_status()
        return _json_object_from_response(response, context=f"GET {url}")
    finally:
        response.close()


def _http_post_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: tuple[int, int] | int = (10, 60),
) -> dict[str, Any]:
    """Run JSON POST request and parse JSON object payload.

    :param url: Endpoint URL.
    :param payload: JSON body payload.
    :param headers: Optional request headers.
    :param timeout: HTTP timeout configuration.
    :returns: Parsed JSON response body.
    :rtype: dict[str, typing.Any]
    :raises requests.RequestException: On network or HTTP errors.
    :raises RuntimeError: If response payload is invalid.
    """
    response = http_post(url, headers=headers, json_body=payload, timeout=timeout, stream=False)
    try:
        response.raise_for_status()
        return _json_object_from_response(response, context=f"POST {url}")
    finally:
        response.close()


def _coerce_positive_int(value: Any) -> int | None:
    """Convert a value into positive integer if possible.

    :param value: Candidate value.
    :returns: Positive integer value, or ``None``.
    :rtype: int | None
    """
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    if out <= 0:
        return None
    return out


def _normalize_service_key_fields(payload: Any, *, fields: tuple[str, ...]) -> dict[str, str] | None:
    """Normalize a service-key payload against one required field set.

    :param payload: Decoded JSON payload.
    :param fields: Required non-empty fields.
    :returns: Normalized payload subset or ``None`` if invalid.
    :rtype: dict[str, str] | None
    """
    if not isinstance(payload, dict):
        return None
    out: dict[str, str] = {}
    for field in fields:
        value = str(payload.get(field, "")).strip()
        if not value:
            return None
        out[field] = value
    return out


def _normalize_service_key_payload(payload: Any) -> dict[str, str] | None:
    """Normalize CLMS service-key payload for supported auth schemas.

    Supports the older CLMS ``service_name/secret/username`` schema and the
    current Copernicus service-key schema based on
    ``client_id/private_key/user_id/key_id/token_uri``.

    :param payload: Decoded JSON payload.
    :returns: Normalized payload with ``auth_kind`` marker or ``None`` if invalid.
    :rtype: dict[str, str] | None
    """
    legacy_payload = _normalize_service_key_fields(payload, fields=_CLMS_LEGACY_SERVICE_KEY_FIELDS)
    if legacy_payload is not None:
        return {"auth_kind": "legacy", **legacy_payload}

    oauth_payload = _normalize_service_key_fields(payload, fields=_CLMS_OAUTH_SERVICE_KEY_FIELDS)
    if oauth_payload is not None:
        return {"auth_kind": "oauth_service_key", **oauth_payload}
    return None


def _service_key_schema_error_message(env_name: str) -> str:
    """Return an error message describing the supported CLMS key schemas.

    :param env_name: Environment variable being validated.
    :returns: Human-readable schema error message.
    :rtype: str
    """
    return (
        f"{env_name} must contain either non-empty "
        "service_name/secret/username fields or "
        "client_id/private_key/user_id/key_id/token_uri fields."
    )


def _load_clms_service_key_payload() -> dict[str, str] | None:
    """Load CLMS service key from supported environment variables.

    Supports:
    - ``CLEO_CLMS_SERVICE_KEY_JSON`` (inline JSON)
    - ``CLEO_CLMS_SERVICE_KEY_PATH`` (path to JSON file)
    - ``CLMS_API_SERVICE_KEY`` (path to JSON file; CLMS docs convention)

    :returns: Normalized service-key payload or ``None`` if not configured.
    :rtype: dict[str, str] | None
    :raises RuntimeError: If configured payload/file is malformed.
    """
    inline = os.getenv("CLEO_CLMS_SERVICE_KEY_JSON", "").strip()
    if inline:
        try:
            payload = json.loads(inline)
        except json.JSONDecodeError as exc:
            raise RuntimeError("CLEO_CLMS_SERVICE_KEY_JSON is not valid JSON.") from exc
        normalized = _normalize_service_key_payload(payload)
        if normalized is None:
            raise RuntimeError(_service_key_schema_error_message("CLEO_CLMS_SERVICE_KEY_JSON"))
        return normalized

    for env_name in ("CLEO_CLMS_SERVICE_KEY_PATH", "CLMS_API_SERVICE_KEY"):
        path_raw = os.getenv(env_name, "").strip()
        if not path_raw:
            continue
        path = Path(path_raw).expanduser()
        if not path.exists():
            raise RuntimeError(f"{env_name} points to a missing file: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{env_name} file is not valid JSON: {path}") from exc
        normalized = _normalize_service_key_payload(payload)
        if normalized is None:
            raise RuntimeError(f"{_service_key_schema_error_message(env_name)} Path: {path}")
        return normalized
    return None


def _base64url_encode(data: bytes) -> str:
    """Encode bytes using unpadded base64url encoding.

    :param data: Input bytes.
    :returns: Base64url-encoded text without padding.
    :rtype: str
    """
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sign_clms_assertion_with_openssl(*, signing_input: bytes, private_key: str) -> bytes:
    """Sign a JWT assertion using the local ``openssl`` binary.

    :param signing_input: Header and claims bytes joined by ``b"."``.
    :param private_key: PEM-encoded RSA private key text.
    :returns: Raw RSA-SHA256 signature bytes.
    :rtype: bytes
    :raises RuntimeError: If ``openssl`` is unavailable or signing fails.
    """
    with tempfile.TemporaryDirectory(prefix="cleo-clms-") as tmpdir:
        tmp_path = Path(tmpdir)
        key_path = tmp_path / "clms_service_key.pem"
        payload_path = tmp_path / "clms_assertion.bin"
        key_path.write_text(private_key, encoding="utf-8")
        os.chmod(key_path, 0o600)
        payload_path.write_bytes(signing_input)
        try:
            proc = subprocess.run(
                [
                    "openssl",
                    "dgst",
                    "-binary",
                    "-sha256",
                    "-sign",
                    str(key_path),
                    str(payload_path),
                ],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Current Copernicus service-key auth requires PyJWT or an available openssl binary."
            ) from exc
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="ignore").strip()
            if not detail:
                detail = proc.stdout.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(
                "Failed to sign CLMS OAuth assertion from service key."
                + (f" openssl output: {detail}" if detail else "")
            )
        return proc.stdout


def _sign_clms_oauth_assertion(
    *,
    header: dict[str, str],
    claims: dict[str, Any],
    private_key: str,
) -> str:
    """Sign a CLMS OAuth JWT assertion using PyJWT or ``openssl`` fallback.

    :param header: JWT header fields.
    :param claims: JWT claims payload.
    :param private_key: PEM-encoded RSA private key text.
    :returns: Compact JWT assertion.
    :rtype: str
    :raises RuntimeError: If signing support is unavailable.
    """
    try:
        import jwt
    except ModuleNotFoundError:
        header_b64 = _base64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        claims_b64 = _base64url_encode(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        signing_input = f"{header_b64}.{claims_b64}".encode("ascii")
        signature = _sign_clms_assertion_with_openssl(signing_input=signing_input, private_key=private_key)
        return f"{header_b64}.{claims_b64}.{_base64url_encode(signature)}"

    token = jwt.encode(claims, private_key, algorithm="RS256", headers=header)
    return str(token)


def _request_oauth_access_token(
    *,
    token_uri: str,
    payload: dict[str, str],
) -> tuple[int, dict[str, Any] | None]:
    """Submit a form-encoded OAuth token request.

    :param token_uri: OAuth token endpoint.
    :param payload: Form body.
    :returns: Tuple of HTTP status code and optional JSON object payload.
    :rtype: tuple[int, dict[str, typing.Any] | None]
    """
    response = http_post(
        token_uri,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data_body=payload,
        timeout=(10, 60),
        stream=False,
    )
    try:
        try:
            payload_out = response.json()
        except ValueError:
            payload_out = None
        if not isinstance(payload_out, dict):
            payload_out = None
        return int(response.status_code), payload_out
    finally:
        response.close()


def _oauth_assertion_claim_candidates(service_payload: dict[str, str]) -> list[dict[str, Any]]:
    """Build claim variants for the documented Copernicus service-key schema.

    :param service_payload: Normalized OAuth-style service-key payload.
    :returns: Candidate claim dictionaries.
    :rtype: list[dict[str, typing.Any]]
    """
    now = int(time.time())
    common = {
        "aud": service_payload["token_uri"],
        "iat": now,
        "exp": now + _CLMS_JWT_EXP_SECONDS,
        "jti": str(uuid.uuid4()),
    }
    return [
        {**common, "iss": service_payload["client_id"], "sub": service_payload["user_id"]},
        {**common, "iss": service_payload["user_id"], "sub": service_payload["user_id"]},
    ]


def _oauth_access_token(service_payload: dict[str, str]) -> str:
    """Exchange a current Copernicus service key for a bearer token.

    :param service_payload: Normalized OAuth-style service-key payload.
    :returns: Access token string.
    :rtype: str
    :raises RuntimeError: If token exchange fails for all supported request variants.
    """
    token_uri = service_payload["token_uri"]
    header = {"alg": "RS256", "typ": "JWT", "kid": service_payload["key_id"]}
    last_error = "no response payload"

    for claims in _oauth_assertion_claim_candidates(service_payload):
        assertion = _sign_clms_oauth_assertion(
            header=header,
            claims=claims,
            private_key=service_payload["private_key"],
        )
        request_payloads = [
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": assertion,
            },
            {
                "grant_type": "client_credentials",
                "client_id": service_payload["client_id"],
                "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                "client_assertion": assertion,
            },
        ]
        for request_payload in request_payloads:
            status_code, response_payload = _request_oauth_access_token(token_uri=token_uri, payload=request_payload)
            access_token = ""
            if isinstance(response_payload, dict):
                access_token = str(response_payload.get("access_token", "")).strip()
                if access_token:
                    return access_token
                error_parts = [
                    str(response_payload.get("error", "")).strip(),
                    str(response_payload.get("error_description", "")).strip(),
                ]
                detail = " ".join(part for part in error_parts if part)
                last_error = detail or f"HTTP {status_code}"
            else:
                last_error = f"HTTP {status_code}"

    raise RuntimeError(
        f"CLMS OAuth token exchange failed for the current Copernicus service-key schema. Last response: {last_error}"
    )


def _legacy_access_token(service_payload: dict[str, str]) -> str:
    """Exchange a legacy CLMS service key for a bearer token.

    :param service_payload: Normalized legacy service-key payload.
    :returns: Access token string.
    :rtype: str
    :raises RuntimeError: If the login response lacks ``access_token``.
    """
    login_payload = _http_post_json(f"{_CLMS_API_BASE_URL}/@login", payload=service_payload)
    access_token = str(login_payload.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("CLMS login response did not include access_token.")
    return access_token


def _clms_access_token() -> str:
    """Resolve CLMS access token from environment or service-key login.

    Security notes:
        - Prefer file-based keys (``CLEO_CLMS_SERVICE_KEY_PATH``) over inline JSON
          to avoid credentials in shell history or process listings.
        - Set restrictive permissions on key files: ``chmod 600 <keyfile>``.
        - Never commit service keys to version control; add patterns to ``.gitignore``.
        - For CI/CD, use secret injection (e.g., GitHub Secrets to env vars).
        - Credentials are only transmitted over HTTPS to the CLMS API.

    :returns: Access token string.
    :rtype: str
    :raises RuntimeError: If CLMS authentication cannot be resolved.
    """
    token = os.getenv("CLEO_CLMS_ACCESS_TOKEN", "").strip()
    if token:
        return token

    service_payload = _load_clms_service_key_payload()
    if service_payload is None:
        raise RuntimeError(
            "CLC default download requires CLMS authentication. "
            "Set CLEO_CLMS_ACCESS_TOKEN or provide service key via "
            "CLEO_CLMS_SERVICE_KEY_JSON / CLEO_CLMS_SERVICE_KEY_PATH / CLMS_API_SERVICE_KEY."
        )
    auth_kind = service_payload.get("auth_kind", "").strip()
    if auth_kind == "legacy":
        return _legacy_access_token(service_payload)
    if auth_kind == "oauth_service_key":
        return _oauth_access_token(service_payload)
    raise RuntimeError("Unsupported CLMS service-key auth schema.")


def _candidate_search_items(payload: Any) -> list[dict[str, Any]]:
    """Extract dataset search items from current or older CLMS response shapes.

    :param payload: JSON payload from CLMS dataset search.
    :returns: Dataset item dictionaries.
    :rtype: list[dict[str, typing.Any]]
    """
    if isinstance(payload, dict):
        for key in ("items", "datasets"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _dataset_identifier(dataset: dict[str, Any]) -> str | None:
    """Extract the canonical dataset identifier from a CLMS search item.

    :param dataset: Dataset search item.
    :returns: Dataset identifier string or ``None``.
    :rtype: str | None
    """
    for key in ("UID", "uid", "id", "@id"):
        value = str(dataset.get(key, "")).strip()
        if value:
            return value
    return None


def _downloadable_file_items(payload: Any) -> list[dict[str, Any]]:
    """Extract downloadable-file items from current or older CLMS payload shapes.

    :param payload: Downloadable-files payload from a dataset record.
    :returns: Downloadable-file item dictionaries.
    :rtype: list[dict[str, typing.Any]]
    """
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _file_identifier(downloadable_file: dict[str, Any]) -> str | None:
    """Extract the canonical prepackaged-file identifier from CLMS metadata.

    :param downloadable_file: Downloadable-file metadata object.
    :returns: File identifier string or ``None``.
    :rtype: str | None
    """
    for key in ("UID", "uid", "id", "@id"):
        value = str(downloadable_file.get(key, "")).strip()
        if value:
            return value
    return None


def _select_clms_dataset_file(
    *,
    search_payload: Any,
    expected_filename: str,
) -> tuple[str, str] | None:
    """Select dataset/file identifiers for an expected CLC filename.

    :param search_payload: Dataset search payload from CLMS API.
    :param expected_filename: Expected source filename.
    :returns: ``(dataset_id, file_id)`` tuple if found, else ``None``.
    :rtype: tuple[str, str] | None
    """
    datasets = _candidate_search_items(search_payload)
    if not datasets:
        return None

    expected_cf = expected_filename.casefold()
    expected_year_match = re.search(r"(19|20)\d{2}", expected_filename)
    expected_year = expected_year_match.group(0) if expected_year_match else ""

    fallback: tuple[str, str] | None = None
    for dataset in datasets:
        dataset_id = _dataset_identifier(dataset)
        downloadable_files = _downloadable_file_items(dataset.get("downloadable_files"))
        if dataset_id is None or not downloadable_files:
            continue

        for downloadable_file in downloadable_files:
            file_id = _file_identifier(downloadable_file)
            downloadable_name = str(downloadable_file.get("downloadable_file_name", "")).strip()
            if not downloadable_name:
                downloadable_name = str(downloadable_file.get("file", "")).strip()
            file_path = str(downloadable_file.get("path", "")).strip().replace("\\", "/")
            downloadable_type = str(downloadable_file.get("type", "")).strip().casefold()
            downloadable_format = str(downloadable_file.get("format", "")).strip().casefold()
            if file_id is None or not downloadable_name:
                continue
            if downloadable_name.casefold() == expected_cf:
                return dataset_id, file_id
            expected_stem = Path(expected_filename).stem.casefold()
            if downloadable_name.casefold() == expected_stem:
                return dataset_id, file_id
            if (
                fallback is None
                and downloadable_type == "raster"
                and downloadable_format in {"geotiff", "tif", "tiff"}
                and (
                    not expected_year
                    or expected_year in downloadable_name.casefold()
                    or expected_year in file_path.casefold()
                )
            ):
                fallback = (dataset_id, file_id)
    return fallback


def _extract_download_urls(payload: Any) -> list[str]:
    """Extract download URL list from CLMS task-status payloads.

    :param payload: CLMS task-status payload or nested output structure.
    :returns: Ordered list of non-empty URLs.
    :rtype: list[str]
    """
    if isinstance(payload, list):
        urls: list[str] = []
        for item in payload:
            urls.extend(_extract_download_urls(item))
        return urls
    if not isinstance(payload, dict):
        return []
    candidates = (
        payload.get("downloadURL"),
        payload.get("downloadUrl"),
        payload.get("download_url"),
        payload.get("DownloadURL"),
        payload.get("DownloadUrl"),
        payload.get("downloadURLs"),
        payload.get("DownloadURLs"),
        payload.get("output"),
    )
    for candidate in candidates:
        if isinstance(candidate, str):
            url = candidate.strip()
            if url:
                return [url]
        if isinstance(candidate, list):
            urls = [str(item).strip() for item in candidate if str(item).strip()]
            if urls:
                return urls
        if isinstance(candidate, dict):
            urls = _extract_download_urls(candidate)
            if urls:
                return urls
    for value in payload.values():
        if isinstance(value, (dict, list)):
            urls = _extract_download_urls(value)
            if urls:
                return urls
    return []


def _normalize_clms_artifact_url(url: str) -> str:
    """Normalize CLMS artifact URLs to the documented machine-download path.

    The official docs show ``.../clmsdatadownload/results/...`` while live task
    payloads may currently expose ``.../fmedatadownload/results/...``.

    :param url: Raw CLMS artifact URL.
    :returns: Normalized artifact URL.
    :rtype: str
    """
    normalized = str(url).strip()
    if not normalized:
        return normalized
    return normalized.replace("/fmedatadownload/", "/clmsdatadownload/")


def _artifact_url_candidates(urls: list[str]) -> list[str]:
    """Return raw and normalized artifact URL candidates in preference order.

    :param urls: Raw artifact URLs from CLMS.
    :returns: Ordered unique candidate URLs.
    :rtype: list[str]
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw_url in urls:
        raw = str(raw_url).strip()
        if not raw:
            continue
        for candidate in (raw, _normalize_clms_artifact_url(raw)):
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


def _status_payload_urls(status_payload: dict[str, Any]) -> list[str]:
    """Extract raw and normalized artifact URL candidates from a task-status payload.

    :param status_payload: CLMS status payload.
    :returns: Ordered artifact URL candidates.
    :rtype: list[str]
    """
    return _artifact_url_candidates(_extract_download_urls(status_payload))


def _finished_request_urls(*, access_token: str, task_id: str) -> list[str]:
    """Look up finished CLMS requests using the documented search endpoint.

    :param access_token: CLMS bearer token.
    :param task_id: Task identifier returned by ``@datarequest_post``.
    :returns: Normalized artifact URLs found in the finished-request search results.
    :rtype: list[str]
    """
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    search_url = (
        f"{_CLMS_API_BASE_URL}/@datarequest_search?status=Finished_ok&sort_on=modified&sort_order=descending&b_size=50"
    )
    payload = _http_get_json(search_url, headers=headers)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_task_id = str(item.get("TaskID", item.get("taskid", item.get("task_id", "")))).strip()
        if item_task_id != task_id:
            continue
        urls = _status_payload_urls(item)
        if urls:
            return urls
    return []


def _submit_clms_download_task(*, source: str, access_token: str, expected_filename: str) -> str:
    """Submit a CLMS pre-packaged download request and return its task id.

    :param source: CLC source key.
    :param access_token: CLMS bearer token.
    :param expected_filename: Expected downloadable file name.
    :returns: CLMS task identifier.
    :rtype: str
    :raises RuntimeError: If source/file cannot be resolved or request fails.
    """
    query_key = _CLMS_SOURCE_QUERY_KEY.get(source, source)
    encoded_key = quote_plus(query_key)
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}

    search_url = (
        f"{_CLMS_API_BASE_URL}/@search?"
        f"portal_type=DataSet&SearchableText={encoded_key}"
        "&metadata_fields=UID&metadata_fields=downloadable_files&b_size=300"
    )
    datasets_payload = _http_get_json(search_url, headers=headers)
    selected = _select_clms_dataset_file(search_payload=datasets_payload, expected_filename=expected_filename)
    if selected is None:
        raise RuntimeError(
            f"CLMS API did not return downloadable file {expected_filename!r}. Provide url=... explicitly."
        )
    dataset_id, file_id = selected

    request_payload = _http_post_json(
        f"{_CLMS_API_BASE_URL}/@datarequest_post",
        payload={"Datasets": [{"DatasetID": dataset_id, "FileID": file_id}]},
        headers={**headers, "Content-Type": "application/json"},
    )
    task_ids = request_payload.get("TaskIds")
    request_token = ""
    if isinstance(task_ids, list):
        for item in task_ids:
            if not isinstance(item, dict):
                continue
            request_token = str(item.get("TaskID", "")).strip()
            if request_token:
                break
    if not request_token:
        request_token = str(request_payload.get("requestToken", "")).strip()
    if not request_token:
        raise RuntimeError("CLMS data-request response did not include a task identifier.")
    return request_token


def _clms_status_payload(*, access_token: str, task_id: str) -> dict[str, Any]:
    """Fetch the current CLMS status payload for a submitted task.

    :param access_token: CLMS bearer token.
    :param task_id: CLMS task identifier.
    :returns: Raw status payload.
    :rtype: dict[str, typing.Any]
    """
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    request_url = f"{_CLMS_API_BASE_URL}/@datarequest_status_get?TaskID={quote_plus(task_id)}"
    return _http_get_json(request_url, headers=headers)


def _request_clms_download_url(*, source: str, access_token: str, expected_filename: str) -> str:
    """Resolve final CLMS file URL via authenticated data-request workflow.

    :param source: CLC source key.
    :param access_token: CLMS bearer token.
    :param expected_filename: Expected downloadable file name.
    :returns: Final downloadable URL for the requested source file.
    :rtype: str
    :raises RuntimeError: If source/file cannot be resolved or request fails.
    """
    request_token = _submit_clms_download_task(
        source=source,
        access_token=access_token,
        expected_filename=expected_filename,
    )

    for attempt in range(_CLMS_POLL_MAX_ATTEMPTS):
        status_payload = _clms_status_payload(access_token=access_token, task_id=request_token)
        status = str(status_payload.get("Status", status_payload.get("status", ""))).strip().casefold()
        if status in {"succeeded", "success", "completed", "done", "finished_ok"}:
            urls = _status_payload_urls(status_payload)
            if not urls:
                urls = _finished_request_urls(access_token=access_token, task_id=request_token)
            if urls:
                return urls[0]
            raise RuntimeError("CLMS request finished but no download URL was returned.")
        if status in {"failed", "error", "finished_ko"}:
            raise RuntimeError(f"CLMS request failed for source {source!r}.")
        if attempt < (_CLMS_POLL_MAX_ATTEMPTS - 1):
            time.sleep(_CLMS_POLL_SLEEP_SECONDS)

    raise RuntimeError(
        f"Timed out waiting for CLMS request completion after "
        f"{_CLMS_POLL_MAX_ATTEMPTS * _CLMS_POLL_SLEEP_SECONDS:.0f} seconds."
    )


def _resolve_default_clc_download(source: str) -> tuple[str, dict[str, str] | None]:
    """Resolve default CLC source download URL and request headers.

    :param source: CLC source key.
    :returns: Tuple of ``(url, headers)`` for file download.
    :rtype: tuple[str, dict[str, str] | None]
    :raises RuntimeError: If default source resolution fails.
    """
    access_token = _clms_access_token()
    expected_filename = CLC_SOURCES[source]["filename"]
    url = _request_clms_download_url(source=source, access_token=access_token, expected_filename=expected_filename)
    return url, {"Authorization": f"Bearer {access_token}"}


def _is_zip_artifact_url(url: str) -> bool:
    """Return whether a download URL points to a ZIP artifact.

    :param url: Download URL.
    :returns: ``True`` if the URL path ends with ``.zip``.
    :rtype: bool
    """
    return urlparse(url).path.lower().endswith(".zip")


def _clms_artifact_request_headers(headers: dict[str, str] | None, *, download_url: str) -> dict[str, str] | None:
    """Build artifact-download headers, using browser-like navigation headers for FME.

    :param headers: Existing request headers.
    :param download_url: Artifact URL being fetched.
    :returns: Headers for the artifact request.
    :rtype: dict[str, str] | None
    """
    parsed = urlparse(download_url)
    host = parsed.netloc.casefold()
    if host != "copernicus-fme.eea.europa.eu":
        return headers
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }


def _zip_member_candidates(zip_file: zipfile.ZipFile, *, expected_filename: str) -> list[str]:
    """Return candidate TIFF members from a CLMS ZIP artifact.

    :param zip_file: Open ZIP archive.
    :param expected_filename: Preferred output filename.
    :returns: Ordered candidate member names.
    :rtype: list[str]
    """
    members = [name for name in zip_file.namelist() if not name.endswith("/")]
    tif_members = [name for name in members if Path(name).suffix.casefold() in {".tif", ".tiff"}]
    if not tif_members:
        return []

    expected_name_cf = expected_filename.casefold()
    expected_stem_cf = Path(expected_filename).stem.casefold()
    ordered: list[str] = []
    for name in tif_members:
        base = Path(name).name.casefold()
        stem = Path(name).stem.casefold()
        if base == expected_name_cf or stem == expected_stem_cf:
            ordered.append(name)
    if ordered:
        return ordered

    expected_year_match = re.search(r"(19|20)\d{2}", expected_filename)
    expected_year = expected_year_match.group(0) if expected_year_match else ""
    for name in tif_members:
        name_cf = name.casefold()
        if "clc" in name_cf and "raster" in name_cf and (not expected_year or expected_year in name_cf):
            ordered.append(name)
    if ordered:
        return ordered

    if len(tif_members) == 1:
        return tif_members
    return tif_members


def _nested_zip_member_candidates(zip_file: zipfile.ZipFile) -> list[str]:
    """Return nested ZIP members that may contain the actual raster package.

    :param zip_file: Open ZIP archive.
    :returns: Ordered nested ZIP member names.
    :rtype: list[str]
    """
    members = [name for name in zip_file.namelist() if not name.endswith("/")]
    zip_members = [name for name in members if Path(name).suffix.casefold() == ".zip"]
    preferred = [name for name in zip_members if "raster" in name.casefold() or "clc" in name.casefold()]
    return preferred or zip_members


def _extract_tif_from_clms_zip(*, zip_path: Path, dest_path: Path, expected_filename: str) -> Path:
    """Extract a CLC TIFF from a downloaded CLMS ZIP package.

    :param zip_path: Downloaded ZIP artifact path.
    :param dest_path: Destination GeoTIFF path.
    :param expected_filename: Preferred TIFF filename inside the ZIP.
    :returns: ``dest_path`` after extraction.
    :rtype: pathlib.Path
    :raises RuntimeError: If the ZIP does not contain a suitable TIFF member.
    """
    with zipfile.ZipFile(zip_path, "r") as archive:
        candidates = _zip_member_candidates(archive, expected_filename=expected_filename)
        member_name = ""
        if candidates:
            member_name = candidates[0]
            source_archive = archive
        else:
            nested_zip_members = _nested_zip_member_candidates(archive)
            if not nested_zip_members:
                raise RuntimeError(f"CLMS ZIP artifact did not contain a TIFF file: {zip_path}")
            nested_name = nested_zip_members[0]
            nested_bytes = archive.read(nested_name)
            source_archive = zipfile.ZipFile(io.BytesIO(nested_bytes), "r")
            nested_candidates = _zip_member_candidates(source_archive, expected_filename=expected_filename)
            if not nested_candidates:
                raise RuntimeError(f"CLMS ZIP artifact did not contain a TIFF file: {zip_path}")
            member_name = nested_candidates[0]
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        try:
            with source_archive.open(member_name, "r") as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp_path.replace(dest_path)
        finally:
            if source_archive is not archive:
                source_archive.close()
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    return dest_path


def _download_clms_artifact(
    *,
    download_url: str,
    source_path: Path,
    headers: dict[str, str] | None,
    expected_filename: str,
) -> Path:
    """Download a CLMS source artifact and normalize it to the cached TIFF path.

    Handles both direct TIFF downloads and the current CLMS pre-packaged ZIP
    artifacts delivered through the FME handoff service.

    :param download_url: Resolved artifact URL.
    :param source_path: Destination TIFF cache path.
    :param headers: Optional request headers for the artifact request.
    :param expected_filename: Expected TIFF filename for ZIP extraction.
    :returns: Cached TIFF path.
    :rtype: pathlib.Path
    :raises RuntimeError: If the artifact never becomes ready or cannot be normalized.
    """
    if not _is_zip_artifact_url(download_url):
        request_headers = _clms_artifact_request_headers(headers, download_url=download_url)
        download_to_path(download_url, source_path, headers=request_headers, overwrite=True)
        return source_path

    artifact_path = source_path.with_suffix(".zip")
    candidate_urls = _artifact_url_candidates([download_url])
    last_exc: Exception | None = None
    for attempt in range(_CLMS_ARTIFACT_READY_MAX_ATTEMPTS):
        for candidate_url in candidate_urls:
            request_headers = _clms_artifact_request_headers(headers, download_url=candidate_url)
            try:
                download_to_path(candidate_url, artifact_path, headers=request_headers, overwrite=True)
                extracted = _extract_tif_from_clms_zip(
                    zip_path=artifact_path,
                    dest_path=source_path,
                    expected_filename=expected_filename,
                )
                try:
                    artifact_path.unlink()
                except OSError:
                    pass
                return extracted
            except FileNotFoundError as exc:
                last_exc = exc
            except RequestException as exc:
                last_exc = exc
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code not in {403, 404}:
                    raise
            except zipfile.BadZipFile as exc:
                last_exc = exc
        if attempt < (_CLMS_ARTIFACT_READY_MAX_ATTEMPTS - 1):
            time.sleep(_CLMS_ARTIFACT_READY_SLEEP_SECONDS)
    detail = str(last_exc) if last_exc is not None else "artifact did not become ready"
    raise RuntimeError(
        "CLMS pre-packaged ZIP artifact was created but did not become downloadable "
        f"after {_CLMS_ARTIFACT_READY_MAX_ATTEMPTS * _CLMS_ARTIFACT_READY_SLEEP_SECONDS:.0f} seconds. "
        f"Last error: {detail}"
    ) from last_exc


def materialize_clc(
    atlas,
    *,
    source: str = "clc2018",
    url: str | None = None,
    force_download: bool = False,
    force_prepare: bool = False,
) -> Path:
    """Prepare a country-cropped CLC raster aligned to wind/GWA grid.

    :param atlas: Atlas-like object with path/country/build state.
    :param source: CLC source identifier.
    :param url: Optional explicit source URL override.
    :param force_download: If ``True``, re-download source raster.
    :param force_prepare: If ``True``, rebuild prepared country cache.
    :returns: Path to prepared GeoTIFF in ``<atlas>/data/raw/<ISO3>/clc/``.
    :rtype: pathlib.Path
    :raises ValueError: If ``source`` is unknown.
    :raises RuntimeError: If required source download/auth metadata is unavailable.
    """
    if source not in CLC_SOURCES:
        raise ValueError(f"Unsupported CLC source {source!r}. Supported: {sorted(CLC_SOURCES)}")

    source_path = source_cache_path(Path(atlas.path), source)
    source_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = prepared_country_path(Path(atlas.path), atlas.country, source)
    prepared.parent.mkdir(parents=True, exist_ok=True)

    if prepared.exists() and not force_prepare:
        prepared_band_count = raster_band_count(prepared)
        if prepared_band_count in (None, 1):
            return prepared
        force_prepare = True

    source_refresh_reason: str | None = None
    source_exists = source_path.exists()
    source_band_count = raster_band_count(source_path) if source_exists else None
    if source_exists and source_band_count != 1:
        force_download = True
        if source_band_count is None:
            source_refresh_reason = f"Cached CLC source is unreadable or invalid: {source_path}"
        else:
            source_refresh_reason = (
                f"Cached CLC source has {source_band_count} bands but CLC sources must be single-band: {source_path}"
            )

    canonical_ready = bool(getattr(atlas, "_canonical_ready", False))
    ref = None
    download_headers: dict[str, str] | None = None
    if (not source_exists) or force_download:
        download_url = url or CLC_SOURCES[source].get("url")
        if not download_url:
            try:
                download_url, download_headers = _resolve_default_clc_download(source)
            except RuntimeError as exc:
                if source_refresh_reason is not None:
                    raise RuntimeError(
                        f"{source_refresh_reason}. CLEO attempted to refresh the cache but "
                        f"could not resolve the authenticated CLMS download. {exc}"
                    ) from exc
                raise
        if not download_url:
            raise RuntimeError(f"No download URL configured for source {source!r}; provide url=... explicitly.")
        _download_clms_artifact(
            download_url=download_url,
            source_path=source_path,
            headers=download_headers,
            expected_filename=CLC_SOURCES[source]["filename"],
        )

    if not canonical_ready:
        atlas.build_canonical()
        canonical_ready = True

    if ref is None:
        ref = wind_reference_template(atlas)
    valid_mask = _canonical_landscape_valid_mask(atlas, chunk_policy=atlas.chunk_policy)
    return prepare_clc_to_wind_grid(
        source_path=source_path,
        prepared_path=prepared,
        ref=ref,
        valid_mask=valid_mask,
        atlas_path=Path(atlas.path),
        source_crs=CLC_SOURCES[source]["crs"],
    )
