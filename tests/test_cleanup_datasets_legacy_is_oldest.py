from pathlib import Path
from cleo.classes import Atlas


def test_cleanup_datasets_legacy_is_oldest(tmp_path):
    atlas = Atlas.__new__(Atlas)
    atlas._path = Path(tmp_path)
    atlas.country = "AUT"
    atlas.region = None
    atlas.index_file = atlas._path / "data" / "index.txt"
    atlas.index_file.parent.mkdir(parents=True, exist_ok=True)

    # Create two fake files, one legacy and one timestamped
    legacy_file = atlas._path / "data" / "processed" / "legacy.nc"
    new_file = atlas._path / "data" / "processed" / "new.nc"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_text("x")
    new_file.write_text("y")

    atlas.index_file.write_text(
        f"WindAtlas:AUT:None:default:{legacy_file}:legacy\n"
        f"WindAtlas:AUT:None:default:{new_file}:20260101T000000\n",
        encoding="utf-8",
    )

    atlas.cleanup_datasets()

    assert not legacy_file.exists()
    assert new_file.exists()

    remaining = atlas._read_index()
    assert len(remaining) == 1
    assert remaining[0][4] == str(new_file)
    assert remaining[0][5] == "20260101T000000"
