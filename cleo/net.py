"""Network boundary module for all HTTP/network I/O.

All network operations in Cleo must go through this module.
This provides a single point of control for:
- HTTP requests (GET, POST, etc.)
- File downloads with atomic writes
- Proxy/auth configuration

No other Cleo module should import requests/urllib/httpx/aiohttp directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from requests.auth import HTTPProxyAuth as _HTTPProxyAuth
from requests.exceptions import RequestException as _RequestException

# Re-export for use by other modules
HTTPProxyAuth = _HTTPProxyAuth
RequestException = _RequestException

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from requests import Response


def http_get(
    url: str,
    *,
    headers: dict | None = None,
    timeout: tuple[int, int] | int = (10, 60),
    stream: bool = True,
    proxies: dict | None = None,
    auth: HTTPProxyAuth | None = None,
) -> "Response":
    """Make an HTTP GET request.

    This is the central entry point for all HTTP GET requests in Cleo.

    :param url: URL to fetch.
    :param headers: Optional HTTP headers dictionary.
    :param timeout: Connection/read timeout as ``(connect, read)`` tuple or single int.
    :param stream: If ``True``, stream response body.
    :param proxies: Optional proxy mapping (for example ``{"http": ..., "https": ...}``).
    :param auth: Optional proxy authentication object.
    :returns: ``requests.Response`` object.
    :raises requests.RequestException: On network errors.
    """
    return requests.get(
        url,
        headers=headers,
        timeout=timeout,
        stream=stream,
        proxies=proxies,
        auth=auth,
    )


def http_post(
    url: str,
    *,
    headers: dict | None = None,
    json_body: dict | list | None = None,
    data_body: dict | list | None = None,
    timeout: tuple[int, int] | int = (10, 60),
    stream: bool = True,
    proxies: dict | None = None,
    auth: HTTPProxyAuth | None = None,
) -> "Response":
    """Make an HTTP POST request.

    This is the central entry point for all HTTP POST requests in Cleo.

    :param url: URL to post to.
    :param headers: Optional HTTP headers dictionary.
    :param json_body: Optional JSON request payload.
    :param data_body: Optional form/body payload.
    :param timeout: Connection/read timeout as ``(connect, read)`` tuple or single int.
    :param stream: If ``True``, stream response body.
    :param proxies: Optional proxy mapping (for example ``{"http": ..., "https": ...}``).
    :param auth: Optional proxy authentication object.
    :returns: ``requests.Response`` object.
    :raises requests.RequestException: On network errors.
    """
    return requests.post(
        url,
        headers=headers,
        json=json_body,
        data=data_body,
        timeout=timeout,
        stream=stream,
        proxies=proxies,
        auth=auth,
    )


def download_to_path(
    url: str,
    dest: Path,
    *,
    headers: dict | None = None,
    timeout: tuple[int, int] | int = (10, 60),
    proxies: dict | None = None,
    auth: HTTPProxyAuth | None = None,
    overwrite: bool = False,
    chunk_size: int = 8192,
) -> Path:
    """Download a file from URL to destination path with atomic write.

    Downloads to a temporary file (.tmp suffix) then atomically renames
    to the final destination on success. On failure, cleans up partial files.

    :param url: URL to download from.
    :param dest: Destination file path.
    :param headers: Optional HTTP headers dictionary.
    :param timeout: Connection/read timeout as ``(connect, read)`` tuple or single int.
    :param proxies: Optional proxy configuration dictionary.
    :param auth: Optional proxy authentication object.
    :param overwrite: If ``False`` and destination exists, skip download.
    :param chunk_size: Streaming chunk size in bytes.
    :returns: Path to the downloaded file (same as ``dest``).
    :raises requests.RequestException: On network errors.
    :raises requests.HTTPError: On non-404 HTTP error responses.
    :raises FileNotFoundError: If server returns HTTP 404.
    """
    dest = Path(dest)

    if dest.exists() and not overwrite:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    response = None

    try:
        response = http_get(
            url,
            headers=headers,
            timeout=timeout,
            stream=True,
            proxies=proxies,
            auth=auth,
        )

        if response.status_code == 404:
            raise FileNotFoundError(f"Resource not found: {url} (HTTP 404)")

        response.raise_for_status()

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        tmp_path.replace(dest)
        return dest

    except (RequestException, FileNotFoundError, OSError, ValueError, RuntimeError, TypeError):
        # Best-effort cleanup of partial file
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logger.debug(
                "Failed to remove temporary download file during cleanup.",
                extra={"tmp_path": str(tmp_path), "url": url},
                exc_info=True,
            )
        raise

    finally:
        if response is not None:
            try:
                response.close()
            except (RequestException, OSError, ValueError, RuntimeError):
                logger.debug(
                    "Failed to close HTTP response cleanly after download.",
                    extra={"url": url, "dest": str(dest)},
                    exc_info=True,
                )
