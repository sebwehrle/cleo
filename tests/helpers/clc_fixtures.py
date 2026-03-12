"""Shared test helpers for CLC-related unit and integration tests."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any


TEST_LEGACY_SERVICE_KEY = {
    "service_name": "service",
    "secret": "secret",  # pragma: allowlist secret
    "username": "user",
}

TEST_OAUTH_SERVICE_KEY = {
    "client_id": "client",
    "private_key": "private",  # pragma: allowlist secret
    "user_id": "user",
    "key_id": "kid",
    "token_uri": "https://token.example.test",
}


class JsonResponseStub:
    """Minimal HTTP response stub for CLMS helper-path tests."""

    def __init__(
        self,
        *,
        payload: Any = ...,
        status_code: int = 200,
        json_error: Exception | None = None,
    ) -> None:
        self._payload = payload
        self._json_error = json_error
        self.status_code = status_code
        self.closed = False

    def json(self) -> Any:
        """Return configured payload or raise configured decode error."""
        if self._json_error is not None:
            raise self._json_error
        return self._payload

    def raise_for_status(self) -> None:
        """Raise on HTTP error status codes."""
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self) -> None:
        """Record close calls."""
        self.closed = True


def legacy_service_key(**overrides: str) -> dict[str, str]:
    """Return a copy of the legacy CLMS service-key fixture."""
    return {**TEST_LEGACY_SERVICE_KEY, **overrides}


def oauth_service_key(**overrides: str) -> dict[str, str]:
    """Return a copy of the OAuth-style CLMS service-key fixture."""
    return {**TEST_OAUTH_SERVICE_KEY, **overrides}


def write_zip_with_member(zip_path: Path, member_name: str, content: bytes) -> None:
    """Write a zip archive with a single member."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(member_name, content)


def write_nested_zip_with_member(
    zip_path: Path,
    nested_zip_name: str,
    member_name: str,
    content: bytes,
) -> None:
    """Write a zip archive containing a nested zip member."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    nested_buffer = io.BytesIO()
    with zipfile.ZipFile(nested_buffer, "w") as nested:
        nested.writestr(member_name, content)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(nested_zip_name, nested_buffer.getvalue())
