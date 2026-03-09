"""Cleanup policy helpers for Atlas delegation."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Callable, Protocol


class AreaMetaLike(Protocol):
    """Typed contract required by area cleanup policy."""

    created_at: datetime.datetime
    is_complete: bool


def parse_older_than(older_than: str | None) -> datetime.datetime | None:
    """Parse cleanup threshold from ISO datetime or YYYY-MM-DD date."""
    if not older_than:
        return None
    try:
        return datetime.datetime.fromisoformat(older_than)
    except ValueError:
        return datetime.datetime.strptime(older_than, "%Y-%m-%d")


def _is_older_than_threshold(
    store_dt: datetime.datetime,
    threshold_dt: datetime.datetime,
) -> bool:
    """Return True when ``store_dt`` is strictly older than ``threshold_dt``."""
    cmp_threshold = threshold_dt
    cmp_store = store_dt
    if cmp_store.tzinfo is not None and cmp_threshold.tzinfo is None:
        cmp_threshold = cmp_threshold.replace(tzinfo=datetime.timezone.utc)
    elif cmp_store.tzinfo is None and cmp_threshold.tzinfo is not None:
        cmp_store = cmp_store.replace(tzinfo=datetime.timezone.utc)
    return cmp_store < cmp_threshold


def resolve_area_cleanup_id(
    *,
    area: object,
    resolve_area_name: Callable[[str], tuple[str, str, int]],
) -> str | None:
    """Normalize optional area filter to a resolved area ID."""
    if area is None:
        return None
    if not isinstance(area, str):
        raise ValueError(f"area must be a string or None, got {type(area).__name__}")
    area_stripped = area.strip()
    if not area_stripped:
        raise ValueError("area cannot be empty or whitespace-only")
    _name_norm, area_id, _level = resolve_area_name(area_stripped)
    return area_id


def select_result_stores_for_cleanup(
    *,
    stores: list[Path],
    metric_name: str | None,
    threshold_dt: datetime.datetime | None,
    read_store_datetime: Callable[[Path], datetime.datetime],
) -> tuple[list[Path], int]:
    """Choose result stores to delete using metric and age policy."""
    selected: list[Path] = []
    scanned = len(stores)

    for store in stores:
        if metric_name is not None and store.name != f"{metric_name}.zarr":
            continue

        if threshold_dt is not None:
            store_dt = read_store_datetime(store)
            if not _is_older_than_threshold(store_dt, threshold_dt):
                continue

        selected.append(store)

    return selected, scanned


def select_area_dirs_for_cleanup(
    *,
    area_dirs: list[Path],
    include_incomplete: bool,
    threshold_dt: datetime.datetime | None,
    read_area_meta: Callable[[Path], AreaMetaLike],
) -> tuple[list[Path], int]:
    """Choose area directories to delete using completeness and age policy."""
    selected: list[Path] = []
    scanned = len(area_dirs)

    for area_dir in area_dirs:
        meta = read_area_meta(area_dir)
        if not include_incomplete and not meta.is_complete:
            continue
        if threshold_dt is not None and not _is_older_than_threshold(meta.created_at, threshold_dt):
            continue
        selected.append(area_dir)

    return selected, scanned
