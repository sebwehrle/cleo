"""Smoke tests for basic import sanity."""

from __future__ import annotations


def test_import_cleo() -> None:
    """Verify cleo package imports without error."""
    import cleo

    assert hasattr(cleo, "Atlas")


def test_import_version() -> None:
    """Verify __version__ is accessible."""
    from cleo import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_atlas_class_exists() -> None:
    """Verify Atlas class is importable and callable."""
    from cleo import Atlas

    assert callable(Atlas)
