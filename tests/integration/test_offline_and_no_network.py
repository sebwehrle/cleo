"""integration: test_offline_and_no_network.
Merged test file (imports preserved per chunk).
"""

import logging
import pytest
from pathlib import Path
from cleo.atlas import Atlas


def test_atlas_init_creates_expected_dirs_offline(tmp_path, monkeypatch):
    """
    Atlas.__init__ must NOT call any network functions.
    Verify expected directories are created without network access.
    """

    # Block all network-related functions - they should NOT be called during __init__
    def _block_network(*args, **kwargs):
        raise AssertionError("Network call attempted during Atlas.__init__")

    monkeypatch.setattr("cleo.net.download_to_path", _block_network)
    monkeypatch.setattr("cleo.net.http_get", _block_network)

    # Create Atlas - this should NOT trigger network calls
    atlas = Atlas(tmp_path, "AUT", "EPSG:3035")

    # Verify expected directories exist
    assert (tmp_path / "data" / "raw" / "AUT").is_dir(), "raw/AUT directory missing"
    assert (tmp_path / "logs").is_dir(), "logs directory missing"

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

    monkeypatch.setattr("cleo.net.download_to_path", _block_network)
    monkeypatch.setattr("cleo.net.http_get", _block_network)

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


def test_atlas_build_deferred(tmp_path, monkeypatch):
    """
    Atlas.build() is where data loading happens.
    Without calling build(), wind/landscape should not be available.
    """

    # Block network during init
    def _block_network(*args, **kwargs):
        raise AssertionError("Network call attempted during __init__")

    monkeypatch.setattr("cleo.net.download_to_path", _block_network)
    monkeypatch.setattr("cleo.net.http_get", _block_network)

    atlas = Atlas(tmp_path, "AUT", "EPSG:3035")

    # Public API: wind/landscape return domain objects without error
    # But accessing .data raises FileNotFoundError since store doesn't exist
    assert atlas.wind is not None  # WindDomain object
    assert atlas.landscape is not None  # LandscapeDomain object

    # Accessing data without canonical stores raises
    with pytest.raises(FileNotFoundError, match="Wind store missing"):
        _ = atlas.wind.data

    with pytest.raises(FileNotFoundError, match="Landscape store missing"):
        _ = atlas.landscape.data


class MockParent:
    """Minimal mock of the Atlas parent object."""

    def __init__(self, path, country):
        self.path = path
        self.country = country


class MockSelf:
    """Minimal mock of the WindAtlas self object."""

    def __init__(self, parent):
        self.parent = parent


def test_load_gwa_never_downloads_elevation(tmp_path, monkeypatch):
    """
    Verify load_gwa never calls download_file with elevation_w_bathymetry URL.

    Oracle: No recorded URL should contain "elevation_w_bathymetry".
    """
    # Set up directory structure
    path_raw = tmp_path / "data" / "raw" / "AUT"
    path_raw.mkdir(parents=True, exist_ok=True)

    # Track all download URLs
    recorded_urls = []

    def mock_download_to_path(url, fpath, **kwargs):
        recorded_urls.append(url)
        # Create empty file so subsequent checks see it as downloaded
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        Path(fpath).touch()
        return Path(fpath)

    import cleo.loaders

    monkeypatch.setattr(cleo.loaders, "download_to_path", mock_download_to_path)

    parent = MockParent(path=tmp_path, country="AUT")
    dummy = MockSelf(parent)

    # Run the GWA bootstrap
    cleo.loaders.load_gwa(dummy)

    # Oracle: no URL should contain "elevation_w_bathymetry"
    elevation_urls = [u for u in recorded_urls if "elevation_w_bathymetry" in u]
    assert len(elevation_urls) == 0, f"download_file was called with elevation_w_bathymetry URL(s): {elevation_urls}"


def test_load_gwa_skips_elevation_even_with_legacy_file(tmp_path, monkeypatch):
    """
    Verify load_gwa never downloads elevation even when legacy file exists.

    Oracle: No recorded URL should contain "elevation_w_bathymetry".
    """
    # Set up directory structure
    path_raw = tmp_path / "data" / "raw" / "AUT"
    path_raw.mkdir(parents=True, exist_ok=True)

    # Create legacy elevation file
    legacy_file = path_raw / "AUT_elevation_w_bathymetry.tif"
    legacy_file.touch()

    # Track all download URLs
    recorded_urls = []

    def mock_download_to_path(url, fpath, **kwargs):
        recorded_urls.append(url)
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        Path(fpath).touch()
        return Path(fpath)

    import cleo.loaders

    monkeypatch.setattr(cleo.loaders, "download_to_path", mock_download_to_path)

    parent = MockParent(path=tmp_path, country="AUT")
    dummy = MockSelf(parent)

    # Run the GWA bootstrap
    cleo.loaders.load_gwa(dummy)

    # Oracle: no URL should contain "elevation_w_bathymetry"
    elevation_urls = [u for u in recorded_urls if "elevation_w_bathymetry" in u]
    assert len(elevation_urls) == 0, f"download_file was called with elevation_w_bathymetry URL(s): {elevation_urls}"
