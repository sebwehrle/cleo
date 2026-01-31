"""Test download_copdem_tiles_for_bbox helper."""

from pathlib import Path

from cleo.copdem import download_copdem_tiles_for_bbox, tiles_for_bbox


def test_download_copdem_tiles_for_bbox(monkeypatch, tmp_path):
    """
    Test that download_copdem_tiles_for_bbox downloads all tiles for a bbox
    in the correct (lexicographic) order.

    Bbox: min_lon=9.2, max_lon=11.1, min_lat=46.0, max_lat=47.0
    Expected:
        lon_deg: floor(9.2)=9, ceil(11.1)-1=11 => [9, 10, 11] (3 values)
        lat_deg: floor(46.0)=46, ceil(47.0)-1=46 => [46] (1 value)
        Total: 3 tiles
    """
    min_lon = 9.2
    max_lon = 11.1
    min_lat = 46.0
    max_lat = 47.0
    iso3 = "AUT"

    # Oracle: compute expected tile_ids independently
    def oracle_tile_id(lat_deg, lon_deg):
        ns = "N" if lat_deg >= 0 else "S"
        ew = "E" if lon_deg >= 0 else "W"
        return f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_{ew}{abs(lon_deg):03d}_00_DEM"

    expected_tile_ids = []
    for lat_deg in [46]:  # lat0=46, lat1=46
        for lon_deg in [9, 10, 11]:  # lon0=9, lon1=11
            expected_tile_ids.append(oracle_tile_id(lat_deg, lon_deg))
    expected_tile_ids.sort()

    # Track calls to download_copdem_tile
    calls = []

    def mock_download_copdem_tile(base_dir, iso3, tile_id, *, overwrite=False):
        calls.append((iso3, tile_id, overwrite))
        # Return a deterministic fake path
        fake_path = Path(base_dir) / iso3 / "copdem" / tile_id / f"{tile_id}.tif"
        return fake_path

    monkeypatch.setattr("cleo.copdem.download_copdem_tile", mock_download_copdem_tile)

    # Call the function under test
    result_paths = download_copdem_tiles_for_bbox(
        tmp_path,
        iso3,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        overwrite=False,
    )

    # Assert download_copdem_tile called exactly 3 times
    assert len(calls) == 3, f"Expected 3 calls, got {len(calls)}"

    # Assert tile_ids match expected in lexicographic order
    actual_tile_ids = [call[1] for call in calls]
    assert actual_tile_ids == expected_tile_ids, (
        f"Tile IDs mismatch:\nExpected: {expected_tile_ids}\nGot: {actual_tile_ids}"
    )

    # Assert iso3 and overwrite are correct in all calls
    for iso3_arg, tile_id, overwrite_arg in calls:
        assert iso3_arg == iso3
        assert overwrite_arg is False

    # Assert returned paths match expected
    expected_paths = [
        tmp_path / iso3 / "copdem" / tile_id / f"{tile_id}.tif"
        for tile_id in expected_tile_ids
    ]
    assert result_paths == expected_paths


def test_download_copdem_tiles_for_bbox_overwrite(monkeypatch, tmp_path):
    """Test that overwrite parameter is passed through correctly."""
    calls = []

    def mock_download_copdem_tile(base_dir, iso3, tile_id, *, overwrite=False):
        calls.append((iso3, tile_id, overwrite))
        return Path(base_dir) / iso3 / "copdem" / tile_id / f"{tile_id}.tif"

    monkeypatch.setattr("cleo.copdem.download_copdem_tile", mock_download_copdem_tile)

    download_copdem_tiles_for_bbox(
        tmp_path,
        "AUT",
        min_lon=9.2,
        min_lat=46.0,
        max_lon=11.1,
        max_lat=47.0,
        overwrite=True,
    )

    # Assert overwrite=True was passed to all calls
    for _, _, overwrite_arg in calls:
        assert overwrite_arg is True


def test_download_copdem_tiles_for_bbox_determinism(monkeypatch, tmp_path):
    """Test that the function returns tiles in deterministic order."""
    calls1 = []
    calls2 = []

    def make_mock(calls_list):
        def mock_download_copdem_tile(base_dir, iso3, tile_id, *, overwrite=False):
            calls_list.append(tile_id)
            return Path(base_dir) / iso3 / "copdem" / tile_id / f"{tile_id}.tif"
        return mock_download_copdem_tile

    monkeypatch.setattr("cleo.copdem.download_copdem_tile", make_mock(calls1))
    result1 = download_copdem_tiles_for_bbox(
        tmp_path, "AUT", min_lon=9.2, min_lat=46.0, max_lon=11.1, max_lat=47.0
    )

    monkeypatch.setattr("cleo.copdem.download_copdem_tile", make_mock(calls2))
    result2 = download_copdem_tiles_for_bbox(
        tmp_path, "AUT", min_lon=9.2, min_lat=46.0, max_lon=11.1, max_lat=47.0
    )

    assert calls1 == calls2, "Tile download order should be deterministic"
    assert result1 == result2, "Returned paths should be deterministic"
