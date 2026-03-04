"""Boundary regression tests for compute-layer module isolation.

These tests verify that compute-layer modules do not have runtime dependencies
on ``cleo.unification.*`` modules, ensuring a clean architectural boundary.

Uses shared utilities from tools.arch_check.
"""

from __future__ import annotations

from pathlib import Path

from tools.arch_check import (
    COMPUTE_FORBIDDEN_PREFIXES,
    check_compute_layer_boundaries,
    collect_import_violations,
)


def test_vertical_has_no_runtime_import_from_unification() -> None:
    """cleo/vertical.py must not runtime-import from cleo.unification.*."""
    module_path = Path("cleo/vertical.py")
    violations = collect_import_violations(module_path, COMPUTE_FORBIDDEN_PREFIXES)
    assert violations == [], "cleo.vertical has forbidden runtime imports:\n" + "\n".join(violations)


def test_bench_has_no_runtime_import_from_unification() -> None:
    """cleo/bench.py must not runtime-import from cleo.unification.*."""
    module_path = Path("cleo/bench.py")
    violations = collect_import_violations(module_path, COMPUTE_FORBIDDEN_PREFIXES)
    assert violations == [], "cleo.bench has forbidden runtime imports:\n" + "\n".join(violations)


def test_all_compute_layer_modules_isolated() -> None:
    """All compute-layer modules must be isolated from forbidden imports."""
    violations = check_compute_layer_boundaries()
    assert violations == [], "Compute-layer modules have forbidden runtime imports:\n" + "\n".join(violations)
