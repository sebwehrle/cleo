import logging

from types import SimpleNamespace
from pathlib import Path

from cleo.class_helpers import setup_logging


def test_setup_logging_does_not_override_root(tmp_path):
    root = logging.getLogger()
    sentinel = logging.StreamHandler()
    root.addHandler(sentinel)
    old_level = root.level
    old_handlers = list(root.handlers)

    parent = SimpleNamespace(path=Path(tmp_path), country="AUT")
    # setup_logging expects self.path and self.country
    setup_logging(parent)

    # root must be untouched
    assert root.level == old_level
    assert root.handlers == old_handlers

    # cleo logger must exist and have handlers
    cleo_logger = logging.getLogger("cleo")
    assert len(cleo_logger.handlers) >= 1

    # cleanup sentinel (avoid leaking across tests)
    root.removeHandler(sentinel)
