from pathlib import Path
from cleo.classes import Atlas


def test_index_parsing_allows_windows_drive_colon(tmp_path):
    atlas = Atlas.__new__(Atlas)
    atlas.index_file = Path(tmp_path) / "index.txt"

    line = "WindAtlas:AUT:None:default:C:\\data\\WindAtlas_AUT_None_default_20260101T000000.nc:20260101T000000\n"
    atlas.index_file.write_text(line, encoding="utf-8")

    entries = atlas._read_index()
    assert len(entries) == 1
    assert entries[0][0] == "WindAtlas"
    assert entries[0][4].startswith("C:\\data\\WindAtlas_")
    assert entries[0][5] == "20260101T000000"
