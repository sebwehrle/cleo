"""
Atomic directory operations for safe store management.

Provides Windows-safe atomic directory replacement with single-writer assumption.
"""

from __future__ import annotations

import os
import shutil
import uuid
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


def _unique_tmp_sibling(path: Path) -> Path:
    """Create a sibling temp dir path using uuid4.

    Args:
        path: The path to create a sibling for.

    Returns:
        A unique sibling path with .__tmp__<uuid> suffix.
    """
    return path.parent / f"{path.name}.__tmp__{uuid.uuid4().hex}"


def replace_dir_atomic(tmp_dir: Path, dst_dir: Path) -> None:
    """Replace dst_dir with tmp_dir atomically (Windows-safe, single-writer).

    This function assumes:
    - Single writer: no concurrent writes to dst_dir.
    - All file handles to zarr/xarray objects in dst_dir are closed.

    Steps:
    1. If dst_dir exists, rename it to a backup sibling.
    2. Rename tmp_dir -> dst_dir via os.replace.
    3. Delete the backup directory.

    If any step fails, best-effort cleanup is attempted.

    :param tmp_dir: Temporary directory containing new content.
    :param dst_dir: Destination directory to replace.
    :returns: ``None``
    :raises OSError: If replacement fails.
    """
    backup_dir: Path | None = None

    try:
        if dst_dir.exists():
            backup_dir = dst_dir.parent / f"{dst_dir.name}.__backup__{uuid.uuid4().hex}"
            os.replace(dst_dir, backup_dir)

        os.replace(tmp_dir, dst_dir)

        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)

    except OSError as e:
        # Best-effort cleanup
        if tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir)
            except OSError:
                logger.debug(
                    "Failed to clean temporary directory after atomic replace failure.",
                    extra={"tmp_dir": str(tmp_dir), "dst_dir": str(dst_dir)},
                    exc_info=True,
                )

        if backup_dir is not None and backup_dir.exists():
            try:
                # Try to restore backup if dst_dir doesn't exist
                if not dst_dir.exists():
                    os.replace(backup_dir, dst_dir)
                else:
                    shutil.rmtree(backup_dir)
            except OSError:
                logger.debug(
                    "Failed to restore/remove backup directory during atomic replace rollback.",
                    extra={
                        "backup_dir": str(backup_dir),
                        "dst_dir": str(dst_dir),
                        "tmp_dir": str(tmp_dir),
                    },
                    exc_info=True,
                )

        raise OSError(
            f"Failed to atomically replace directory '{dst_dir}' with '{tmp_dir}': {e}"
        ) from e


@contextmanager
def atomic_dir(dst_dir: Path) -> Generator[Path, None, None]:
    """Context manager for atomic directory creation/replacement.

    Creates a temporary sibling directory, yields it for population,
    then atomically replaces dst_dir with the temp directory on exit.

    Usage:
        with atomic_dir(Path("/path/to/store")) as tmp:
            # Write to tmp directory
            (tmp / "data.txt").write_text("content")
        # On exit, tmp is atomically moved to /path/to/store

    :param dst_dir: Destination directory path.
    :yields: Temporary directory path to populate.
    :raises OSError: If directory creation or replacement fails.
    """
    tmp_dir = _unique_tmp_sibling(dst_dir)
    committed = False
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield tmp_dir
        replace_dir_atomic(tmp_dir, dst_dir)
        committed = True
    finally:
        if not committed and tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir)
            except OSError:
                logger.debug(
                    "Failed to clean temporary directory in atomic_dir context cleanup.",
                    extra={"tmp_dir": str(tmp_dir), "dst_dir": str(dst_dir)},
                    exc_info=True,
                )
