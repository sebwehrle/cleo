"""Oracle test ensuring GWA bootstrap never downloads elevation_w_bathymetry.

Elevation must come from local legacy file or Copernicus DEM, not GWA download.
"""

from pathlib import Path


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

    def mock_download_file(url, fpath):
        recorded_urls.append(url)
        # Create empty file so subsequent checks see it as downloaded
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        Path(fpath).touch()
        return True

    import cleo.loaders
    monkeypatch.setattr(cleo.loaders, "download_file", mock_download_file)

    parent = MockParent(path=tmp_path, country="AUT")
    dummy = MockSelf(parent)

    # Run the GWA bootstrap
    cleo.loaders.load_gwa(dummy)

    # Oracle: no URL should contain "elevation_w_bathymetry"
    elevation_urls = [u for u in recorded_urls if "elevation_w_bathymetry" in u]
    assert len(elevation_urls) == 0, (
        f"download_file was called with elevation_w_bathymetry URL(s): {elevation_urls}"
    )


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

    def mock_download_file(url, fpath):
        recorded_urls.append(url)
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        Path(fpath).touch()
        return True

    import cleo.loaders
    monkeypatch.setattr(cleo.loaders, "download_file", mock_download_file)

    parent = MockParent(path=tmp_path, country="AUT")
    dummy = MockSelf(parent)

    # Run the GWA bootstrap
    cleo.loaders.load_gwa(dummy)

    # Oracle: no URL should contain "elevation_w_bathymetry"
    elevation_urls = [u for u in recorded_urls if "elevation_w_bathymetry" in u]
    assert len(elevation_urls) == 0, (
        f"download_file was called with elevation_w_bathymetry URL(s): {elevation_urls}"
    )
