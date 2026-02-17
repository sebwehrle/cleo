"""Network boundary module for all HTTP/network I/O.

All network operations in Cleo must go through this module.
This provides a single point of control for:
- HTTP requests (GET, POST, etc.)
- File downloads with atomic writes
- Proxy/auth configuration

No other Cleo module should import requests/urllib/httpx/aiohttp directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import requests
from requests.auth import HTTPProxyAuth as _HTTPProxyAuth
from requests.exceptions import RequestException as _RequestException

# Re-export for use by other modules
HTTPProxyAuth = _HTTPProxyAuth
RequestException = _RequestException

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

    Args:
        url: The URL to fetch.
        headers: Optional HTTP headers dict.
        timeout: Connection/read timeout as (connect, read) tuple or single int.
        stream: If True, stream the response body (default True for large files).
        proxies: Optional proxy configuration dict ({"http": ..., "https": ...}).
        auth: Optional authentication object.

    Returns:
        requests.Response object. Caller is responsible for:
        - Calling response.raise_for_status() if needed
        - Closing the response when done (especially with stream=True)

    Raises:
        requests.RequestException: On network errors.
    """
    return requests.get(
        url,
        headers=headers,
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

    Args:
        url: The URL to download from.
        dest: Destination file path.
        headers: Optional HTTP headers dict.
        timeout: Connection/read timeout as (connect, read) tuple or single int.
        proxies: Optional proxy configuration dict.
        auth: Optional authentication object.
        overwrite: If False and dest exists, return dest without downloading.
        chunk_size: Size of chunks to read at a time (default 8192).

    Returns:
        Path to the downloaded file (same as dest).

    Raises:
        requests.RequestException: On network errors.
        requests.HTTPError: On HTTP error responses (4xx, 5xx).
        FileNotFoundError: If server returns 404.
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

    except Exception:
        # Best-effort cleanup of partial file
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise

    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
