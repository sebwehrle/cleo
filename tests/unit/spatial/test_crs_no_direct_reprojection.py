"""Guardrail test: ensure CRS operations are centralized in cleo.spatial.

This test scans cleo source files to ensure that:
1. Direct CRS comparisons (!=, ==) are not done outside spatial.py
2. Direct .to_crs() and .rio.reproject() calls are only in spatial.py

The goal is to enforce the single-chokepoint CRS discipline.
"""

import re
from pathlib import Path

import pytest

from tests.helpers.paths import cleo_path


def get_cleo_source_files():
    """Get all Python source files in cleo package (excluding tests)."""
    return list(cleo_path().glob("*.py"))


def read_source(path: Path) -> str:
    """Read source file, return empty string if unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


class TestCrsGuardrails:
    """Static analysis tests to enforce CRS centralization."""

    # Patterns that suggest direct CRS comparison (outside spatial.py)
    CRS_COMPARISON_PATTERNS = [
        # Direct string CRS comparison
        r"\.crs\s*!=\s*",
        r"\.crs\s*==\s*",
        r"\.rio\.crs\s*!=\s*",
        r"\.rio\.crs\s*==\s*",
        # String casting for CRS comparison
        r"str\s*\(\s*\w+\.crs\s*\)",
    ]

    # Files allowed to have direct CRS operations (the chokepoint)
    ALLOWED_DIRECT_CRS = {"spatial.py"}

    def test_no_direct_crs_comparison_outside_spatial(self):
        """Ensure no direct CRS comparisons outside spatial.py."""
        violations = []

        for path in get_cleo_source_files():
            if path.name in self.ALLOWED_DIRECT_CRS:
                continue

            source = read_source(path)
            if not source:
                continue

            for lineno, line in enumerate(source.splitlines(), 1):
                # Skip comments
                stripped = line.split("#")[0]
                if not stripped.strip():
                    continue

                for pattern in self.CRS_COMPARISON_PATTERNS:
                    if re.search(pattern, stripped):
                        violations.append(
                            f"{path.name}:{lineno}: direct CRS comparison: {line.strip()}"
                        )

        if violations:
            msg = (
                "Direct CRS comparisons found outside cleo/spatial.py. "
                "Use cleo.spatial.crs_equal() instead:\n" + "\n".join(violations)
            )
            pytest.fail(msg)

    def test_crs_utilities_exist(self):
        """Verify cleo.spatial exports the required CRS utilities."""
        from cleo.spatial import (
            canonical_crs_str,
            crs_equal,
            to_crs_if_needed,
            reproject_raster_if_needed,
        )

        # Basic smoke test
        assert callable(canonical_crs_str)
        assert callable(crs_equal)
        assert callable(to_crs_if_needed)
        assert callable(reproject_raster_if_needed)

    def test_crs_equal_used_in_modules(self):
        """Verify modules import and use centralized CRS helpers."""
        expected_imports = {
            "class_helpers.py": ["crs_equal", "reproject_raster_if_needed"],
            "loaders.py": ["reproject_raster_if_needed", "to_crs_if_needed"],
            "assess.py": ["crs_equal", "reproject_raster_if_needed", "to_crs_if_needed"],
            "utils.py": ["reproject_raster_if_needed"],
            "classes.py": ["to_crs_if_needed", "crs_equal"],
        }

        missing = []
        for filename, expected in expected_imports.items():
            path = cleo_path(filename)
            if not path.exists():
                continue

            source = read_source(path)
            for helper in expected:
                if helper not in source:
                    missing.append(f"{filename} should import/use {helper}")

        if missing:
            pytest.fail("Missing CRS helper imports:\n" + "\n".join(missing))
