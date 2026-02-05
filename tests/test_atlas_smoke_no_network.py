"""
Smoke test: Atlas instantiation must NOT trigger network calls.
Construction must only create directories and deploy resources.
Actual data fetching requires explicit materialize() call.
"""
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from cleo.classes import Atlas


def test_atlas_init_creates_expected_dirs_offline(tmp_path, monkeypatch):
    """
    Atlas.__init__ must NOT call any network functions.
    Verify expected directories are created without network access.
    """
    # Block all network-related functions - they should NOT be called during __init__
    def _block_network(*args, **kwargs):
        raise AssertionError("Network call attempted during Atlas.__init__")

    monkeypatch.setattr("cleo.utils.download_file", _block_network)
    monkeypatch.setattr("cleo.loaders.requests.get", _block_network)

    # Create Atlas - this should NOT trigger network calls
    atlas = Atlas(tmp_path, "AUT", "EPSG:3035")

    # Verify expected directories exist
    assert (tmp_path / "data" / "raw" / "AUT").is_dir(), "raw/AUT directory missing"
    assert (tmp_path / "data" / "processed").is_dir(), "processed directory missing"
    assert (tmp_path / "logs").is_dir(), "logs directory missing"

    # Verify index file is created (may be empty)
    index_file = tmp_path / "data" / "index.jsonl"
    assert index_file.exists(), "index.jsonl file missing"

    # Verify resources were deployed
    resources_dir = tmp_path / "resources"
    assert resources_dir.is_dir(), "resources directory missing"
    yml_files = list(resources_dir.glob("*.yml"))
    assert len(yml_files) >= 1, "No resource YAML files deployed"


def test_atlas_init_does_not_modify_root_logger(tmp_path, monkeypatch):
    """
    Atlas.__init__ must not add handlers to the root logger.
    Only the 'cleo' logger namespace should be configured.
    """
    # Block network
    def _block_network(*args, **kwargs):
        raise AssertionError("Network call attempted")

    monkeypatch.setattr("cleo.utils.download_file", _block_network)
    monkeypatch.setattr("cleo.loaders.requests.get", _block_network)

    # Capture root logger state before
    root_logger = logging.getLogger()
    handlers_before = list(root_logger.handlers)
    handler_ids_before = {id(h) for h in handlers_before}

    # Create Atlas
    atlas = Atlas(tmp_path, "AUT", "EPSG:3035")

    # Verify root logger handlers unchanged
    handlers_after = list(root_logger.handlers)
    handler_ids_after = {id(h) for h in handlers_after}

    assert handler_ids_before == handler_ids_after, (
        f"Root logger handlers changed. Before: {len(handlers_before)}, After: {len(handlers_after)}"
    )


def test_atlas_materialize_deferred(tmp_path, monkeypatch):
    """
    Atlas.materialize() is where data loading happens.
    Without calling materialize(), wind/landscape should not be available.
    """
    # Block network during init
    def _block_network(*args, **kwargs):
        raise AssertionError("Network call attempted during __init__")

    monkeypatch.setattr("cleo.utils.download_file", _block_network)
    monkeypatch.setattr("cleo.loaders.requests.get", _block_network)

    atlas = Atlas(tmp_path, "AUT", "EPSG:3035")

    # Without materialize, accessing wind/landscape should raise
    with pytest.raises(RuntimeError, match="materialize"):
        _ = atlas.wind

    with pytest.raises(RuntimeError, match="materialize"):
        _ = atlas.landscape
