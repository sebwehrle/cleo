"""Corine Land Cover (CLC) source handling and cache preparation."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from cleo.net import download_to_path, http_get, http_post
from cleo.unification.clc_io import (
    load_clc_codes,
    prepare_clc_to_wind_grid,
    wind_reference_template,
)


_CLMS_API_BASE_URL = "https://api.clms-copernicus.eu/api"
_CLMS_POLL_SLEEP_SECONDS = 2.0
_CLMS_POLL_MAX_ATTEMPTS = 180
_CLMS_SOURCE_QUERY_KEY: dict[str, str] = {"clc2018": "CLC2018"}
_CLMS_SERVICE_KEY_FIELDS = ("service_name", "secret", "username")


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


def _normalize_service_key_payload(payload: Any) -> dict[str, str] | None:
    """Normalize CLMS service-key payload.

    :param payload: Decoded JSON payload.
    :returns: Normalized payload or ``None`` if invalid.
    :rtype: dict[str, str] | None
    """
    if not isinstance(payload, dict):
        return None
    out: dict[str, str] = {}
    for field in _CLMS_SERVICE_KEY_FIELDS:
        value = str(payload.get(field, "")).strip()
        if not value:
            return None
        out[field] = value
    return out


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
            raise RuntimeError(
                "CLEO_CLMS_SERVICE_KEY_JSON must contain non-empty service_name/secret/username fields."
            )
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
            raise RuntimeError(
                f"{env_name} file must contain non-empty service_name/secret/username fields: {path}"
            )
        return normalized
    return None


def _clms_access_token() -> str:
    """Resolve CLMS access token from environment or service-key login.

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

    login_payload = _http_post_json(f"{_CLMS_API_BASE_URL}/@login", payload=service_payload)
    access_token = str(login_payload.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("CLMS login response did not include access_token.")
    return access_token


def _select_clms_dataset_file(
    *,
    datasets: Any,
    expected_filename: str,
) -> tuple[int, int] | None:
    """Select dataset/file identifiers for an expected CLC filename.

    :param datasets: ``datasets`` payload list from CLMS API.
    :param expected_filename: Expected source filename.
    :returns: ``(dataset_id, file_id)`` tuple if found, else ``None``.
    :rtype: tuple[int, int] | None
    """
    if not isinstance(datasets, list):
        return None

    expected_cf = expected_filename.casefold()
    expected_year_match = re.search(r"(19|20)\d{2}", expected_filename)
    expected_year = expected_year_match.group(0) if expected_year_match else ""

    fallback: tuple[int, int] | None = None
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        dataset_id = _coerce_positive_int(dataset.get("id"))
        downloadable_files = dataset.get("downloadable_files")
        if dataset_id is None or not isinstance(downloadable_files, list):
            continue

        for downloadable_file in downloadable_files:
            if not isinstance(downloadable_file, dict):
                continue
            file_id = _coerce_positive_int(downloadable_file.get("id"))
            downloadable_name = str(downloadable_file.get("downloadable_file_name", "")).strip()
            if file_id is None or not downloadable_name:
                continue
            if downloadable_name.casefold() == expected_cf:
                return dataset_id, file_id
            if (
                fallback is None
                and downloadable_name.casefold().endswith(".tif")
                and (not expected_year or expected_year in downloadable_name)
            ):
                fallback = (dataset_id, file_id)
    return fallback


def _extract_download_urls(output_payload: Any) -> list[str]:
    """Extract download URL list from CLMS request-result output payload.

    :param output_payload: ``output`` field from CLMS data-request result.
    :returns: Ordered list of non-empty URLs.
    :rtype: list[str]
    """
    if not isinstance(output_payload, dict):
        return []
    candidates = (
        output_payload.get("downloadURL"),
        output_payload.get("downloadUrl"),
        output_payload.get("download_url"),
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
    return []


def _request_clms_download_url(*, source: str, access_token: str, expected_filename: str) -> str:
    """Resolve final CLMS file URL via authenticated data-request workflow.

    :param source: CLC source key.
    :param access_token: CLMS bearer token.
    :param expected_filename: Expected downloadable file name.
    :returns: Final downloadable URL for the requested source file.
    :rtype: str
    :raises RuntimeError: If source/file cannot be resolved or request fails.
    """
    query_key = _CLMS_SOURCE_QUERY_KEY.get(source, source)
    encoded_key = quote_plus(query_key)
    headers = {"Authorization": f"Bearer {access_token}"}

    datasets_payload = _http_get_json(f"{_CLMS_API_BASE_URL}/datasets?key={encoded_key}", headers=headers)
    selected = _select_clms_dataset_file(datasets=datasets_payload.get("datasets"), expected_filename=expected_filename)
    if selected is None:
        raise RuntimeError(
            f"CLMS API did not return downloadable file {expected_filename!r}. "
            "Provide url=... explicitly."
        )
    dataset_id, file_id = selected

    request_payload = _http_post_json(
        f"{_CLMS_API_BASE_URL}/@datarequest_post",
        payload={"DatasetID": dataset_id, "FileID": [file_id]},
        headers=headers,
    )
    request_token = str(request_payload.get("requestToken", "")).strip()
    if not request_token:
        raise RuntimeError("CLMS data-request response did not include requestToken.")

    request_url = f"{_CLMS_API_BASE_URL}/@datarequest_results/{request_token}"
    for attempt in range(_CLMS_POLL_MAX_ATTEMPTS):
        status_payload = _http_get_json(request_url, headers=headers)
        status = str(status_payload.get("status", "")).strip().casefold()
        if status in {"succeeded", "success", "completed", "done"}:
            urls = _extract_download_urls(status_payload.get("output"))
            if urls:
                return urls[0]
            raise RuntimeError("CLMS request finished but no download URL was returned.")
        if status in {"failed", "error"}:
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


def _raster_band_count(path: Path) -> int | None:
    """Read raster band count for validation.

    :param path: Raster path.
    :returns: Band count when readable, else ``None``.
    :rtype: int | None
    """
    try:
        import rasterio

        with rasterio.open(path) as ds:
            return int(ds.count)
    except (OSError, ValueError, RuntimeError, TypeError):
        return None


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
        prepared_band_count = _raster_band_count(prepared)
        if prepared_band_count in (None, 1):
            return prepared
        force_prepare = True

    source_band_count = _raster_band_count(source_path) if source_path.exists() else None
    if source_band_count is not None and source_band_count != 1:
        force_download = True

    canonical_ready = bool(getattr(atlas, "_canonical_ready", False))
    ref = None
    download_headers: dict[str, str] | None = None
    if (not source_path.exists()) or force_download:
        download_url = url or CLC_SOURCES[source].get("url")
        if not download_url:
            download_url, download_headers = _resolve_default_clc_download(source)
        if not download_url:
            raise RuntimeError(f"No download URL configured for source {source!r}; provide url=... explicitly.")
        download_to_path(download_url, source_path, headers=download_headers, overwrite=True)

    if not canonical_ready:
        atlas.build_canonical()
        canonical_ready = True

    if ref is None:
        ref = wind_reference_template(atlas)
    return prepare_clc_to_wind_grid(
        source_path=source_path,
        prepared_path=prepared,
        ref=ref,
        source_crs=CLC_SOURCES[source]["crs"],
    )
