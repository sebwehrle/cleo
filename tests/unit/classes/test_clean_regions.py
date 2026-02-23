"""Tests for Atlas.clean_regions()."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import zarr

from cleo.atlas import Atlas


def _write_region_catalog(path: Path, mapping: dict[str, str]) -> None:
    """Write region catalog attrs used by Atlas region resolution."""
    g = zarr.open_group(str(path / "landscape.zarr"), mode="w")
    rows = [
        {"name": name.title(), "name_norm": name, "nuts_id": nuts_id, "level": 2}
        for name, nuts_id in mapping.items()
    ]
    g.attrs["cleo_region_catalog_json"] = json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    )


def _create_region_store_set(
    root: Path,
    region_id: str,
    *,
    complete: bool,
    created_at: str | None = None,
    wind_only: bool = False,
) -> Path:
    """Create a minimal region store directory with wind/landscape zarr roots."""
    region_dir = root / "regions" / region_id
    region_dir.mkdir(parents=True, exist_ok=True)

    wind = zarr.open_group(str(region_dir / "wind.zarr"), mode="w")
    wind.attrs["store_state"] = "complete" if complete else "incomplete"
    if created_at is not None:
        wind.attrs["created_at"] = created_at

    if not wind_only:
        land = zarr.open_group(str(region_dir / "landscape.zarr"), mode="w")
        land.attrs["store_state"] = "complete" if complete else "incomplete"
        if created_at is not None:
            land.attrs["created_at"] = created_at

    return region_dir


def test_clean_regions_all_runs(tmp_path: Path) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")

    d1 = _create_region_store_set(tmp_path, "AT13", complete=True)
    d2 = _create_region_store_set(tmp_path, "AT12", complete=True)

    deleted = atlas.clean_regions()

    assert deleted == 2
    assert not d1.exists()
    assert not d2.exists()


def test_clean_regions_region_filter_by_name(tmp_path: Path) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")

    _write_region_catalog(tmp_path, {"wien": "AT13"})
    d_wien = _create_region_store_set(tmp_path, "AT13", complete=True)
    d_other = _create_region_store_set(tmp_path, "AT12", complete=True)

    deleted = atlas.clean_regions(region="Wien")

    assert deleted == 1
    assert not d_wien.exists()
    assert d_other.exists()


def test_clean_regions_include_incomplete_flag(tmp_path: Path) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")

    d_incomplete = _create_region_store_set(
        tmp_path,
        "AT13",
        complete=False,
        wind_only=True,
    )

    deleted_skip = atlas.clean_regions(include_incomplete=False)
    assert deleted_skip == 0
    assert d_incomplete.exists()

    deleted_drop = atlas.clean_regions(include_incomplete=True)
    assert deleted_drop == 1
    assert not d_incomplete.exists()


def test_clean_regions_older_than(tmp_path: Path) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")

    old_ts = "2020-01-01T00:00:00+00:00"
    new_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    d_old = _create_region_store_set(tmp_path, "AT11", complete=True, created_at=old_ts)
    d_new = _create_region_store_set(tmp_path, "AT12", complete=True, created_at=new_ts)

    deleted = atlas.clean_regions(older_than="2021-01-01")

    assert deleted == 1
    assert not d_old.exists()
    assert d_new.exists()
