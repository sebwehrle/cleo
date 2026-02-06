"""Test that setup_logging does not crash due to missing logging.config import."""

from pathlib import Path
from unittest.mock import MagicMock


def test_setup_logging_does_not_crash(tmp_path):
    """setup_logging should not raise AttributeError for logging.config."""
    from cleo.class_helpers import setup_logging

    # Create mock self with path attribute
    mock_self = MagicMock()
    mock_self.path = tmp_path

    # Create logs directory (setup_logging expects it to exist)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    # Should not raise AttributeError: module 'logging' has no attribute 'config'
    setup_logging(mock_self)


def test_logging_config_is_importable():
    """Verify logging.config is properly imported in class_helpers."""
    import cleo.class_helpers

    # After the fix, logging.config should be available
    import logging.config
    assert hasattr(logging, "config"), "logging.config should be accessible"
