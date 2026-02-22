"""Tests for atomic directory replacement operations."""

import pytest
from pathlib import Path

from cleo.store import replace_dir_atomic, atomic_dir, _unique_tmp_sibling


class TestUniqueeTmpSibling:
    """Tests for _unique_tmp_sibling."""

    def test_creates_sibling_path(self, tmp_path: Path) -> None:
        """Sibling path is in same parent directory."""
        target = tmp_path / "mydir"
        sibling = _unique_tmp_sibling(target)

        assert sibling.parent == target.parent
        assert sibling.name.startswith("mydir.__tmp__")

    def test_unique_each_call(self, tmp_path: Path) -> None:
        """Each call produces a unique path."""
        target = tmp_path / "mydir"
        siblings = {_unique_tmp_sibling(target) for _ in range(10)}

        assert len(siblings) == 10


class TestReplaceDirAtomic:
    """Tests for replace_dir_atomic."""

    def test_replace_existing_directory(self, tmp_path: Path) -> None:
        """Replacing existing dst_dir with tmp_dir works correctly."""
        # Setup: dst_dir with original file
        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        (dst_dir / "old_file.txt").write_text("old content")

        # Setup: tmp_dir with new file
        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir()
        (tmp_dir / "new_file.txt").write_text("new content")

        # Act
        replace_dir_atomic(tmp_dir, dst_dir)

        # Assert: dst_dir now has new content
        assert dst_dir.exists()
        assert (dst_dir / "new_file.txt").exists()
        assert (dst_dir / "new_file.txt").read_text() == "new content"

        # Assert: old file is gone
        assert not (dst_dir / "old_file.txt").exists()

        # Assert: tmp_dir no longer exists (was renamed)
        assert not tmp_dir.exists()

    def test_replace_nonexistent_directory(self, tmp_path: Path) -> None:
        """Creating dst_dir when it doesn't exist."""
        dst_dir = tmp_path / "new_dst"
        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir()
        (tmp_dir / "file.txt").write_text("content")

        replace_dir_atomic(tmp_dir, dst_dir)

        assert dst_dir.exists()
        assert (dst_dir / "file.txt").read_text() == "content"
        assert not tmp_dir.exists()

    def test_preserves_nested_structure(self, tmp_path: Path) -> None:
        """Nested directory structure is preserved after replacement."""
        dst_dir = tmp_path / "dst"
        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir()

        # Create nested structure in tmp
        (tmp_dir / "subdir").mkdir()
        (tmp_dir / "subdir" / "nested.txt").write_text("nested")
        (tmp_dir / "root.txt").write_text("root")

        replace_dir_atomic(tmp_dir, dst_dir)

        assert (dst_dir / "root.txt").read_text() == "root"
        assert (dst_dir / "subdir" / "nested.txt").read_text() == "nested"


class TestAtomicDirContextManager:
    """Tests for atomic_dir context manager."""

    def test_creates_and_replaces_directory(self, tmp_path: Path) -> None:
        """Context manager creates tmp, yields it, then replaces dst."""
        dst_dir = tmp_path / "store"

        with atomic_dir(dst_dir) as tmp:
            # tmp should exist and be different from dst
            assert tmp.exists()
            assert tmp != dst_dir
            assert tmp.parent == dst_dir.parent

            # Write content to tmp
            (tmp / "data.txt").write_text("test data")

            # dst should not exist yet
            assert not dst_dir.exists()

        # After context: dst should exist with content
        assert dst_dir.exists()
        assert (dst_dir / "data.txt").read_text() == "test data"

    def test_replaces_existing_directory(self, tmp_path: Path) -> None:
        """Context manager replaces existing dst_dir."""
        dst_dir = tmp_path / "store"
        dst_dir.mkdir()
        (dst_dir / "old.txt").write_text("old")

        with atomic_dir(dst_dir) as tmp:
            (tmp / "new.txt").write_text("new")

        assert (dst_dir / "new.txt").read_text() == "new"
        assert not (dst_dir / "old.txt").exists()

    def test_cleanup_on_exception(self, tmp_path: Path) -> None:
        """Temp directory is cleaned up if an exception occurs."""
        dst_dir = tmp_path / "store"

        with pytest.raises(ValueError, match="test error"):
            with atomic_dir(dst_dir) as tmp:
                tmp_path_saved = tmp
                (tmp / "file.txt").write_text("content")
                raise ValueError("test error")

        # tmp should be cleaned up
        assert not tmp_path_saved.exists()
        # dst should not have been created
        assert not dst_dir.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created if needed."""
        dst_dir = tmp_path / "nested" / "path" / "store"

        with atomic_dir(dst_dir) as tmp:
            (tmp / "file.txt").write_text("content")

        assert dst_dir.exists()
        assert (dst_dir / "file.txt").read_text() == "content"
