"""Global pytest fixtures and test-wide contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Ensure the local package is importable even when not installed.
# Supports both "src/" layout and flat-package layout.
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))
else:
    sys.path.insert(0, str(_REPO))


@pytest.fixture(autouse=True)
def _forbid_network_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent accidental network access."""
    import requests

    def _blocked(*args, **kwargs):  # noqa: ANN001
        raise AssertionError(
            "Network access is forbidden in tests. Patch cleo.*.requests.get or the relevant helper."
        )

    monkeypatch.setattr(requests, "request", _blocked, raising=True)
    monkeypatch.setattr(requests.sessions.Session, "request", _blocked, raising=True)
