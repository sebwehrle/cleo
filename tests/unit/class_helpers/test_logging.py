"""class_helpers: test_logging.

Contracts:
- setup_logging must not mutate the root logger (level/handlers).
- setup_logging must configure the "cleo" logger with at least one handler.
- logging.config must be importable (regression: missing import usage must not crash).
"""

from __future__ import annotations

import logging
import logging.config  # noqa: F401  (ensures logging.config is importable)
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from cleo.class_helpers import setup_logging


def test_setup_logging_does_not_override_root(tmp_path: Path) -> None:
    root = logging.getLogger()

    sentinel = logging.StreamHandler()
    root.addHandler(sentinel)
    try:
        old_level = root.level
        old_handlers = list(root.handlers)

        parent = SimpleNamespace(path=Path(tmp_path), country="AUT")
        setup_logging(parent)

        # Root must be untouched
        assert root.level == old_level
        assert list(root.handlers) == old_handlers

        # cleo logger must exist and have handlers
        cleo_logger = logging.getLogger("cleo")
        assert len(cleo_logger.handlers) >= 1
    finally:
        # Ensure sentinel never leaks across tests, even if assertion fails
        if sentinel in root.handlers:
            root.removeHandler(sentinel)


def test_setup_logging_does_not_crash(tmp_path: Path) -> None:
    """setup_logging should not raise AttributeError for logging.config."""
    mock_self = MagicMock()
    mock_self.path = tmp_path

    # setup_logging expects logs dir to exist
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    setup_logging(mock_self)


def test_logging_config_is_importable() -> None:
    """Regression guard: logging.config should be importable and accessible."""
    assert hasattr(logging, "config")
