"""Unit tests for cleanup policy helpers."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import pytest

from cleo.atlas_policies.cleanup import (
    parse_older_than,
    resolve_area_cleanup_id,
    select_area_dirs_for_cleanup,
    select_result_stores_for_cleanup,
)


@dataclass(frozen=True)
class _Meta:
    created_at: datetime.datetime
    is_complete: bool


def test_parse_older_than_accepts_iso_and_date() -> None:
    iso = parse_older_than("2021-01-01T00:00:00+00:00")
    day = parse_older_than("2021-01-01")
    assert iso == datetime.datetime(2021, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    assert day == datetime.datetime(2021, 1, 1, 0, 0)
    assert parse_older_than(None) is None


def test_select_result_stores_filters_metric_and_older_than() -> None:
    stores = [
        Path("/tmp/run/metric_a.zarr"),
        Path("/tmp/run/metric_b.zarr"),
    ]
    dt_map = {
        stores[0]: datetime.datetime(2020, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
        stores[1]: datetime.datetime(2022, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
    }
    threshold = datetime.datetime(2021, 1, 1, 0, 0)

    selected, scanned = select_result_stores_for_cleanup(
        stores=stores,
        metric_name="metric_a",
        threshold_dt=threshold,
        read_store_datetime=lambda p: dt_map[p],
    )

    assert scanned == 2
    assert selected == [stores[0]]


def test_select_region_dirs_filters_incomplete_and_older_than() -> None:
    dirs = [Path("/tmp/areas/AT11"), Path("/tmp/areas/AT12"), Path("/tmp/areas/AT13")]
    meta = {
        dirs[0]: _Meta(created_at=datetime.datetime(2020, 1, 1), is_complete=False),
        dirs[1]: _Meta(created_at=datetime.datetime(2020, 1, 1), is_complete=True),
        dirs[2]: _Meta(created_at=datetime.datetime(2022, 1, 1), is_complete=True),
    }
    threshold = datetime.datetime(2021, 1, 1)

    selected, scanned = select_area_dirs_for_cleanup(
        area_dirs=dirs,
        include_incomplete=False,
        threshold_dt=threshold,
        read_area_meta=lambda p: meta[p],
    )

    assert scanned == 3
    assert selected == [dirs[1]]


def test_resolve_area_cleanup_id_validates_and_resolves() -> None:
    rid = resolve_area_cleanup_id(
        area=" Wien ",
        resolve_area_name=lambda name: ("wien", "AT13", 2),
    )
    assert rid == "AT13"
    assert resolve_area_cleanup_id(area=None, resolve_area_name=lambda name: ("n", "id", 2)) is None

    with pytest.raises(ValueError, match="area must be a string or None"):
        resolve_area_cleanup_id(area=1, resolve_area_name=lambda name: ("n", "id", 2))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="area cannot be empty"):
        resolve_area_cleanup_id(area="   ", resolve_area_name=lambda name: ("n", "id", 2))
