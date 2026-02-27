"""Tests for single-writer lock functionality."""

import json
import os
import pytest
from pathlib import Path

from cleo.store import single_writer_lock, StoreLockError, _LOCK_FILE_NAME


class TestSingleWriterLock:
    """Tests for single_writer_lock context manager."""

    def test_creates_lock_file(self, tmp_path: Path) -> None:
        """Lock file is created when entering the context."""
        target = tmp_path / "store"

        with single_writer_lock(target):
            lock_path = target / _LOCK_FILE_NAME
            assert lock_path.exists()
            # Verify lock file contains expected JSON fields
            content = json.loads(lock_path.read_text())
            assert "created_at" in content
            assert "pid" in content
            assert "hostname" in content
            assert "target_dir" in content
            assert content["pid"] == os.getpid()

    def test_removes_lock_file_on_exit(self, tmp_path: Path) -> None:
        """Lock file is removed when exiting the context normally."""
        target = tmp_path / "store"

        with single_writer_lock(target):
            lock_path = target / _LOCK_FILE_NAME
            assert lock_path.exists()

        assert not lock_path.exists()

    def test_removes_lock_file_on_exception(self, tmp_path: Path) -> None:
        """Lock file is removed when exiting the context via exception."""
        target = tmp_path / "store"
        lock_path = target / _LOCK_FILE_NAME

        with pytest.raises(ValueError, match="test error"):
            with single_writer_lock(target):
                assert lock_path.exists()
                raise ValueError("test error")

        assert not lock_path.exists()

    def test_blocks_second_writer(self, tmp_path: Path) -> None:
        """Second writer is blocked when lock already exists."""
        target = tmp_path / "store"

        with single_writer_lock(target):
            # Try to acquire a second lock - should fail
            with pytest.raises(StoreLockError) as exc_info:
                with single_writer_lock(target):
                    pass

            # Verify error message is informative
            assert "lock exists" in str(exc_info.value).lower()
            assert str(target / _LOCK_FILE_NAME) in str(exc_info.value)

    def test_force_overrides_existing_lock(self, tmp_path: Path) -> None:
        """force=True allows overriding an existing lock."""
        target = tmp_path / "store"
        lock_path = target / _LOCK_FILE_NAME

        # Create a stale lock
        target.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({
            "created_at": "2020-01-01T00:00:00Z",
            "pid": 99999,
            "hostname": "stale-host",
            "target_dir": str(target),
        }))

        # force=True should succeed
        with single_writer_lock(target, force=True):
            # Verify lock was replaced with new metadata
            content = json.loads(lock_path.read_text())
            assert content["pid"] == os.getpid()
            assert content["hostname"] != "stale-host"

    def test_creates_target_directory(self, tmp_path: Path) -> None:
        """Target directory is created if it doesn't exist."""
        target = tmp_path / "new" / "nested" / "store"
        assert not target.exists()

        with single_writer_lock(target):
            assert target.exists()

    def test_yields_target_path(self, tmp_path: Path) -> None:
        """Context manager yields the target path for convenience."""
        target = tmp_path / "store"

        with single_writer_lock(target) as locked_path:
            assert locked_path == target

    def test_lock_metadata_includes_required_fields(self, tmp_path: Path) -> None:
        """Lock metadata includes all required fields per contract."""
        target = tmp_path / "store"

        with single_writer_lock(target):
            lock_path = target / _LOCK_FILE_NAME
            content = json.loads(lock_path.read_text())

            # Required fields per PR6 spec
            assert "created_at" in content
            assert "pid" in content
            assert "hostname" in content
            assert "target_dir" in content

            # Verify field types
            assert isinstance(content["created_at"], str)
            assert isinstance(content["pid"], int)
            assert isinstance(content["hostname"], str)
            assert isinstance(content["target_dir"], str)


class TestStoreLockError:
    """Tests for StoreLockError exception."""

    def test_has_lock_path_attribute(self, tmp_path: Path) -> None:
        """StoreLockError includes the lock_path attribute."""
        target = tmp_path / "store"

        with single_writer_lock(target):
            with pytest.raises(StoreLockError) as exc_info:
                with single_writer_lock(target):
                    pass

            assert exc_info.value.lock_path == target / _LOCK_FILE_NAME

    def test_has_lock_metadata_attribute(self, tmp_path: Path) -> None:
        """StoreLockError includes the lock_metadata attribute."""
        target = tmp_path / "store"

        with single_writer_lock(target):
            with pytest.raises(StoreLockError) as exc_info:
                with single_writer_lock(target):
                    pass

            # Metadata should be the original lock's metadata
            meta = exc_info.value.lock_metadata
            assert meta is not None
            assert meta["pid"] == os.getpid()


class TestAtomicDirWithLock:
    """Tests for atomic_dir integration with single-writer lock."""

    def test_atomic_dir_uses_lock_by_default(self, tmp_path: Path) -> None:
        """atomic_dir acquires a lock by default."""
        from cleo.store import atomic_dir

        target = tmp_path / "store"

        # Start atomic_dir, which should hold the lock
        with atomic_dir(target) as tmp:
            # Lock should exist in sibling location
            lock_dir = target.parent / f".{target.name}.lock"
            lock_path = lock_dir / _LOCK_FILE_NAME
            assert lock_path.exists()

            # Write some content
            (tmp / "data.txt").write_text("content")

        # After exit, lock should be released
        assert not lock_path.exists()
        # And content should be in target
        assert (target / "data.txt").read_text() == "content"

    def test_atomic_dir_lock_blocks_concurrent_writes(self, tmp_path: Path) -> None:
        """atomic_dir's lock blocks concurrent atomic_dir on same target."""
        from cleo.store import atomic_dir

        target = tmp_path / "store"

        with atomic_dir(target) as tmp:
            (tmp / "data.txt").write_text("first")

            # Try to start another atomic_dir - should fail due to lock
            with pytest.raises(StoreLockError):
                with atomic_dir(target):
                    pass

    def test_atomic_dir_can_disable_lock(self, tmp_path: Path) -> None:
        """atomic_dir can be used without lock via use_lock=False."""
        from cleo.store import atomic_dir

        target = tmp_path / "store"

        with atomic_dir(target, use_lock=False) as tmp:
            lock_dir = target.parent / f".{target.name}.lock"
            lock_path = lock_dir / _LOCK_FILE_NAME
            # No lock should exist
            assert not lock_path.exists()

            (tmp / "data.txt").write_text("content")

        assert (target / "data.txt").read_text() == "content"
