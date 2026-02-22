"""Tests for Atlas.clean_results()."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
import zarr

from cleo.atlas import Atlas


def _create_result_store(
    root: Path,
    *,
    run_id: str,
    metric_name: str,
    created_at: str | None = None,
) -> Path:
    """Create a minimal persisted result store with optional created_at attr."""
    store_dir = root / "results" / run_id / f"{metric_name}.zarr"
    store_dir.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(store_dir), mode="w")
    group.attrs["store_state"] = "complete"
    if created_at is not None:
        group.attrs["created_at"] = created_at
    return store_dir


def test_clean_results_older_than_handles_mixed_timezone_awareness(tmp_path: Path) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")

    old_ts = "2020-01-01T00:00:00+00:00"
    new_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    old_store = _create_result_store(
        tmp_path,
        run_id="run-old",
        metric_name="metric",
        created_at=old_ts,
    )
    new_store = _create_result_store(
        tmp_path,
        run_id="run-new",
        metric_name="metric",
        created_at=new_ts,
    )

    deleted = atlas.clean_results(older_than="2021-01-01")

    assert deleted == 1
    assert not old_store.exists()
    assert new_store.exists()


def test_clean_results_rejects_run_id_path_traversal(tmp_path: Path) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")

    with pytest.raises(ValueError, match="run_id"):
        atlas.clean_results(run_id="../outside")


def test_clean_results_rejects_metric_name_path_traversal(tmp_path: Path) -> None:
    atlas = Atlas(tmp_path, "AUT", "epsg:3035")

    with pytest.raises(ValueError, match="metric_name"):
        atlas.clean_results(metric_name="../outside")
