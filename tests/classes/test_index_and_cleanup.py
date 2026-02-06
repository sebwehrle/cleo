"""classes: test_index_and_cleanup.

Covers:
- index parsing: legacy colon format + JSONL, including Windows drive-colon paths
- timestamp ordering: "legacy" is oldest
- cleanup: keep latest per (subclass,country,region,scenario); delete older files and rewrite index
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

import cleo.classes as C


@contextmanager
def _chdir(path: Path):
    """Temporary cwd change (used by other tests too; must be a real contextmanager)."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _mk_unmaterialized_atlas(tmp_path: Path) -> C.Atlas:
    """Create a minimal Atlas instance without running heavy __init__/materialize."""
    a = C.Atlas.__new__(C.Atlas)
    a._path = Path(tmp_path)
    a.country = "AUT"
    a.region = None
    a._crs = "epsg:3035"
    return a


def test_cleanup_datasets_legacy_is_oldest(tmp_path: Path) -> None:
    atlas = _mk_unmaterialized_atlas(tmp_path)
    atlas.index_file = atlas._path / "data" / "index.txt"
    atlas.index_file.parent.mkdir(parents=True, exist_ok=True)

    # Create two fake files, one legacy and one timestamped
    legacy_file = atlas._path / "data" / "processed" / "legacy.nc"
    new_file = atlas._path / "data" / "processed" / "new.nc"
    _touch(legacy_file, b"x")
    _touch(new_file, b"y")

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


def test_index_roundtrip_and_cleanup_keeps_latest(tmp_path: Path) -> None:
    atlas = _mk_unmaterialized_atlas(tmp_path)

    with _chdir(tmp_path):
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        # create dummy dataset files referenced by index
        f_old = Path("old.nc")
        f_new = Path("new.nc")
        f_leg = Path("legacy.nc")
        for p in (f_old, f_new, f_leg):
            _touch(p, b"x")

        atlas.index_file = data_dir / "index.jsonl"

        # write JSONL lines directly (includes legacy + 2 timestamps)
        lines = [
            {
                "subclass": "WindAtlas",
                "country": "AUT",
                "region": "None",
                "scenario": "default",
                "path": str(f_old),
                "timestamp": "20200101T000000",
            },
            {
                "subclass": "WindAtlas",
                "country": "AUT",
                "region": "None",
                "scenario": "default",
                "path": str(f_new),
                "timestamp": "20210101T000000",
            },
            {
                "subclass": "LandscapeAtlas",
                "country": "AUT",
                "region": "None",
                "scenario": "default",
                "path": str(f_leg),
                "timestamp": "legacy",
            },
        ]
        atlas.index_file.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

        entries = atlas._read_index()
        assert len(entries) == 3

        # cleanup should keep latest per subclass:
        # - WindAtlas: keep 20210101..., delete 20200101...
        # - LandscapeAtlas: only legacy exists -> keep it
        atlas.cleanup_datasets(scenario="default")

        assert not f_old.exists()
        assert f_new.exists()
        assert f_leg.exists()

        # ensure index rewritten to only 2 entries
        rewritten = atlas._read_index()
        assert len(rewritten) == 2


def test_parse_index_jsonl_roundtrip() -> None:
    line = (
        '{"subclass":"WindAtlas","country":"AUT","region":"None","scenario":"default",'
        '"path":"C:\\\\data\\\\x.nc","timestamp":"20260101T000000"}'
    )
    e = C._parse_index_line(line)
    assert e[4].startswith("C:\\")
    assert e[5] == "20260101T000000"


def test_parse_legacy_colon_with_windows_path() -> None:
    legacy = "WindAtlas:AUT:None:default:C:\\data\\WindAtlas_AUT.nc:20260101T000000"
    e = C._parse_index_line(legacy)
    assert e[0] == "WindAtlas"
    assert e[4] == "C:\\data\\WindAtlas_AUT.nc"
    assert e[5] == "20260101T000000"


def test_timestamp_key_legacy_is_oldest() -> None:
    assert C._timestamp_key("legacy") < C._timestamp_key("20200101T000000")
    with pytest.raises(ValueError):
        C._timestamp_key("not-a-timestamp")


def test_index_parsing_allows_windows_drive_colon(tmp_path: Path) -> None:
    atlas = _mk_unmaterialized_atlas(tmp_path)
    atlas.index_file = Path(tmp_path) / "index.txt"

    line = (
        "WindAtlas:AUT:None:default:"
        "C:\\data\\WindAtlas_AUT_None_default_20260101T000000.nc:20260101T000000\n"
    )
    atlas.index_file.write_text(line, encoding="utf-8")

    entries = atlas._read_index()
    assert len(entries) == 1
    assert entries[0][0] == "WindAtlas"
    assert entries[0][4].startswith("C:\\data\\WindAtlas_")
    assert entries[0][5] == "20260101T000000"
