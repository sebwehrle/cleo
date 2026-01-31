"""Test Copernicus DEM tile selection helper."""

from cleo.copdem import tiles_for_bbox


def test_tiles_for_bbox_austria():
    """
    Test tile selection for Austria bounding box.

    Oracle bbox (from PO's Austria raster bounds):
        min_lon=9.50875, max_lon=17.18125, min_lat=46.35125, max_lat=49.04125

    Expected integer degree ranges:
        lon_deg = 9..17 inclusive (9 values)
        lat_deg = 46..49 inclusive (4 values)
        expected count = 36
    """
    min_lon = 9.50875
    max_lon = 17.18125
    min_lat = 46.35125
    max_lat = 49.04125

    # Oracle: inline formatting function (do not import from cleo.copdem)
    def oracle_tile_id(lat_deg, lon_deg):
        ns = "N" if lat_deg >= 0 else "S"
        ew = "E" if lon_deg >= 0 else "W"
        abs_lat = abs(lat_deg)
        abs_lon = abs(lon_deg)
        return f"Copernicus_DSM_COG_10_{ns}{abs_lat:02d}_00_{ew}{abs_lon:03d}_00_DEM"

    # Build expected list using oracle
    expected_tiles = []
    for lat_deg in range(46, 50):  # 46..49 inclusive
        for lon_deg in range(9, 18):  # 9..17 inclusive
            expected_tiles.append(oracle_tile_id(lat_deg, lon_deg))
    expected_tiles.sort()

    # Call the function under test
    result = tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)

    # Assert correct count
    assert len(result) == 36, f"Expected 36 tiles, got {len(result)}"

    # Assert exact equality with oracle
    assert result == expected_tiles, f"Tile list mismatch:\nExpected: {expected_tiles}\nGot: {result}"


def test_tiles_for_bbox_determinism():
    """Test that tiles_for_bbox returns identical results on repeated calls."""
    min_lon = 9.50875
    max_lon = 17.18125
    min_lat = 46.35125
    max_lat = 49.04125

    result1 = tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)
    result2 = tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)

    assert result1 == result2, "tiles_for_bbox should be deterministic"
