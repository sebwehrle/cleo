"""
Atomic directory operations for safe store management.

Provides Windows-safe atomic directory replacement with single-writer lock enforcement.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import socket
import uuid
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# Lock file name used by single_writer_lock
_LOCK_FILE_NAME = ".cleo_write_lock"


class StoreLockError(OSError):
    """Raised when a store write lock cannot be acquired."""

    def __init__(self, message: str, lock_path: Path, lock_metadata: dict | None = None):
        super().__init__(message)
        self.lock_path = lock_path
        self.lock_metadata = lock_metadata


def _read_lock_metadata(lock_path: Path) -> dict | None:
    """Read lock metadata from lock file, returning None if unreadable."""
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _write_lock_file(lock_path: Path, target_dir: Path) -> None:
    """Write lock file with metadata using O_CREAT|O_EXCL for atomicity."""
    metadata = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "target_dir": str(target_dir),
    }
    payload = json.dumps(metadata, indent=2, ensure_ascii=True)

    # Use O_CREAT|O_EXCL for atomic creation (fails if file exists)
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)


@contextmanager
def single_writer_lock(
    target_dir: str | Path,
    *,
    force: bool = False,
) -> Generator[Path, None, None]:
    """Context manager enforcing single-writer access to a store directory.

    Creates a lock file in target_dir to prevent concurrent writes. The lock
    file contains JSON metadata: created_at, pid, hostname, target_dir.

    Usage:
        with single_writer_lock(Path("/path/to/store.zarr")) as store_path:
            # Write to store_path safely
            ...
        # Lock is released on exit

    Parameters
    ----------
    target_dir : str or Path
        The store directory to lock for writing.
    force : bool, default False
        If True, forcibly remove an existing lock and acquire a new one.
        Use with caution: this may cause data corruption if another writer
        is actually active.

    Yields
    ------
    Path
        The target directory path (same as input, for convenience).

    Raises
    ------
    StoreLockError
        If the lock cannot be acquired (lock file exists and force=False).
        The error message includes lock metadata and remediation steps.

    Notes
    -----
    - Lock files are named `.cleo_write_lock` inside target_dir.
    - The lock is advisory; it only works if all writers use this mechanism.
    - No automatic stale-lock cleanup: use force=True if you're certain no
      other writer is active.
    """
    target = Path(target_dir)
    lock_path = target / _LOCK_FILE_NAME

    # Ensure target directory exists
    target.mkdir(parents=True, exist_ok=True)

    # Handle existing lock
    if lock_path.exists():
        if not force:
            lock_meta = _read_lock_metadata(lock_path)
            meta_str = json.dumps(lock_meta, indent=2) if lock_meta else "(unreadable)"
            raise StoreLockError(
                f"Store write lock exists at {lock_path}\n"
                f"Lock metadata:\n{meta_str}\n\n"
                f"Another process may be writing to this store.\n"
                f"If you're certain no other writer is active, either:\n"
                f"  1. Delete the lock file: rm {lock_path}\n"
                f"  2. Use force=True to override the lock",
                lock_path=lock_path,
                lock_metadata=lock_meta,
            )
        else:
            # Force mode: remove existing lock
            logger.warning(
                "Forcibly removing existing write lock.",
                extra={"lock_path": str(lock_path), "target_dir": str(target)},
            )
            try:
                lock_path.unlink()
            except OSError as e:
                raise StoreLockError(
                    f"Failed to remove existing lock file: {e}",
                    lock_path=lock_path,
                ) from e

    # Acquire lock
    try:
        _write_lock_file(lock_path, target)
    except FileExistsError:
        # Race condition: another process created the lock between our check and write
        lock_meta = _read_lock_metadata(lock_path)
        raise StoreLockError(
            f"Lock acquisition race: another process acquired the lock at {lock_path}",
            lock_path=lock_path,
            lock_metadata=lock_meta,
        )
    except OSError as e:
        raise StoreLockError(
            f"Failed to create lock file at {lock_path}: {e}",
            lock_path=lock_path,
        ) from e

    try:
        yield target
    finally:
        # Release lock
        try:
            if lock_path.exists():
                lock_path.unlink()
        except OSError:
            logger.warning(
                "Failed to remove write lock on exit.",
                extra={"lock_path": str(lock_path), "target_dir": str(target)},
                exc_info=True,
            )


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

        raise OSError(f"Failed to atomically replace directory '{dst_dir}' with '{tmp_dir}': {e}") from e


@contextmanager
def atomic_dir(
    dst_dir: Path,
    *,
    use_lock: bool = True,
    force_lock: bool = False,
) -> Generator[Path, None, None]:
    """Context manager for atomic directory creation/replacement.

    Creates a temporary sibling directory, yields it for population,
    then atomically replaces dst_dir with the temp directory on exit.

    Usage:
        with atomic_dir(Path("/path/to/store")) as tmp:
            # Write to tmp directory
            (tmp / "data.txt").write_text("content")
        # On exit, tmp is atomically moved to /path/to/store

    Parameters
    ----------
    dst_dir : Path
        Destination directory path.
    use_lock : bool, default True
        If True, acquire a single-writer lock on dst_dir before writing.
    force_lock : bool, default False
        If True and use_lock=True, forcibly override any existing lock.

    Yields
    ------
    Path
        Temporary directory path to populate.

    Raises
    ------
    OSError
        If directory creation or replacement fails.
    StoreLockError
        If use_lock=True and lock cannot be acquired.
    """
    tmp_dir = _unique_tmp_sibling(dst_dir)
    committed = False

    # Ensure parent directory exists for lock file placement
    dst_dir.parent.mkdir(parents=True, exist_ok=True)

    def _do_atomic_write():
        nonlocal committed
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

    if use_lock:
        # Lock is placed in a sibling location to avoid being inside the replaced dir
        lock_dir = dst_dir.parent / f".{dst_dir.name}.lock"
        with single_writer_lock(lock_dir, force=force_lock):
            yield from _do_atomic_write()
    else:
        yield from _do_atomic_write()
