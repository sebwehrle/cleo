"""Shared architecture boundary checking utilities.

This module provides the single source of truth for architecture boundary
enforcement. Both tool scripts (tools/check_*.py) and unit tests
(tests/unit/compat/) consume this logic to avoid duplication.

All implementations are stdlib-only (ast, pathlib, re) for speed and stability.
"""

from __future__ import annotations

import ast
from pathlib import Path


# =============================================================================
# AST Utilities
# =============================================================================


def dotted_name(node: ast.AST) -> str:
    """Extract dotted module name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return ""


def is_type_checking_test(node: ast.AST) -> bool:
    """Check if an AST node is a TYPE_CHECKING test."""
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "typing" and node.attr == "TYPE_CHECKING"
    return False


def iter_runtime_import_modules(tree: ast.AST) -> list[str]:
    """Iterate over all runtime-imported module names in an AST.

    Excludes imports inside ``if TYPE_CHECKING:`` blocks.

    Args:
        tree: Parsed AST module.

    Returns:
        List of module names imported at runtime.
    """
    modules: list[str] = []

    def visit(node: ast.AST, in_type_checking: bool = False) -> None:
        if isinstance(node, ast.If):
            body_is_type_checking = is_type_checking_test(node.test)
            for child in node.body:
                visit(child, in_type_checking=in_type_checking or body_is_type_checking)
            for child in node.orelse:
                visit(child, in_type_checking=in_type_checking)
            return

        if isinstance(node, ast.Import) and not in_type_checking:
            for alias in node.names:
                modules.append(alias.name)
            return

        if isinstance(node, ast.ImportFrom) and not in_type_checking:
            if node.module:
                modules.append(node.module)
            return

        for child in ast.iter_child_nodes(node):
            visit(child, in_type_checking=in_type_checking)

    visit(tree)
    return modules


def module_matches(module: str, prefix: str) -> bool:
    """Check if a module name matches a prefix (exact or dotted child)."""
    return module == prefix or module.startswith(f"{prefix}.")


def parse_file(path: Path) -> ast.AST:
    """Parse a Python file into an AST."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# =============================================================================
# Boundary Check Functions
# =============================================================================


def collect_import_violations(
    module_path: Path,
    forbidden_prefixes: list[str],
) -> list[str]:
    """Collect runtime imports that violate boundary constraints.

    Args:
        module_path: Path to the Python module to check.
        forbidden_prefixes: List of module prefixes that are forbidden.

    Returns:
        List of violation descriptions.
    """
    violations: list[str] = []
    tree = parse_file(module_path)

    for module in iter_runtime_import_modules(tree):
        for prefix in forbidden_prefixes:
            if module_matches(module, prefix):
                violations.append(f"{module_path}: runtime import forbidden: {module}")
                break

    return violations


def collect_io_call_violations(
    module_path: Path,
) -> list[str]:
    """Collect direct I/O calls that violate policy module constraints.

    Policy modules (cleo/atlas_policies/) must not perform direct I/O.

    Args:
        module_path: Path to the Python module to check.

    Returns:
        List of violation descriptions.
    """
    violations: list[str] = []
    tree = parse_file(module_path)

    forbidden_calls = {"zarr.open_group", "xr.open_zarr", "xarray.open_zarr", "shutil.rmtree"}
    forbidden_attrs = {"to_zarr", "iterdir", "glob", "rglob", "unlink", "rmdir"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = dotted_name(node.func)
        attr = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if dotted in forbidden_calls or attr in forbidden_attrs:
            violations.append(f"{module_path}:{node.lineno} direct I/O call: {dotted or attr}")

    return violations


def collect_underscore_import_violations(
    module_path: Path,
    source_module: str,
) -> list[str]:
    """Collect underscore-prefixed imports from a source module.

    Args:
        module_path: Path to the Python module to check.
        source_module: Module from which underscore imports are forbidden.

    Returns:
        List of violation descriptions.
    """
    violations: list[str] = []
    tree = parse_file(module_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == source_module or mod.startswith(f"{source_module}."):
                for alias in node.names:
                    if alias.name.startswith("_"):
                        violations.append(
                            f"{module_path}:{node.lineno} underscore import: from {mod} import {alias.name}"
                        )

    return violations


# =============================================================================
# Predefined Boundary Rules
# =============================================================================


# Compute layer modules must not import unification
COMPUTE_LAYER_MODULES = [
    Path("cleo/vertical.py"),
    Path("cleo/bench.py"),
    Path("cleo/assess.py"),
    Path("cleo/economics.py"),
]

COMPUTE_FORBIDDEN_PREFIXES = [
    "cleo.unification",
    "cleo.atlas",
    "cleo.domains",
    "cleo.results",
]

# Unification must not import orchestration/facade layers
UNIFICATION_FORBIDDEN_PREFIXES = [
    "cleo.loaders",
    "cleo.atlas",
    "cleo.domains",
    "cleo.results",
    "cleo.class_helpers",
]

# Materializers must not import atlas
MATERIALIZER_FORBIDDEN_PREFIXES = [
    "cleo.atlas",
]

# Orchestration/facade layers must not import materializers directly
ORCHESTRATION_MODULES = [
    Path("cleo/atlas.py"),
    Path("cleo/domains.py"),
    Path("cleo/results.py"),
    Path("cleo/loaders.py"),
]

ORCHESTRATION_FORBIDDEN_PREFIXES = [
    "cleo.unification.materializers",
]


def check_compute_layer_boundaries(root: Path | None = None) -> list[str]:
    """Check that compute layer modules don't import forbidden modules.

    Args:
        root: Repository root directory. Defaults to current directory.

    Returns:
        List of violations.
    """
    root = root or Path(".")
    violations: list[str] = []

    for module_rel in COMPUTE_LAYER_MODULES:
        module_path = root / module_rel
        if not module_path.exists():
            continue
        violations.extend(collect_import_violations(module_path, COMPUTE_FORBIDDEN_PREFIXES))

    return violations


def check_unification_boundaries(root: Path | None = None) -> list[str]:
    """Check that unification modules don't import forbidden modules.

    Args:
        root: Repository root directory. Defaults to current directory.

    Returns:
        List of violations.
    """
    root = root or Path(".")
    violations: list[str] = []
    unification_dir = root / "cleo" / "unification"

    if not unification_dir.exists():
        return violations

    for module_path in sorted(unification_dir.rglob("*.py")):
        violations.extend(collect_import_violations(module_path, UNIFICATION_FORBIDDEN_PREFIXES))

    return violations


def check_materializer_boundaries(root: Path | None = None) -> list[str]:
    """Check that materializers don't runtime-import atlas.

    Args:
        root: Repository root directory. Defaults to current directory.

    Returns:
        List of violations.
    """
    root = root or Path(".")
    violations: list[str] = []
    materializers_dir = root / "cleo" / "unification" / "materializers"

    if not materializers_dir.exists():
        return violations

    for module_path in sorted(materializers_dir.glob("*.py")):
        violations.extend(collect_import_violations(module_path, MATERIALIZER_FORBIDDEN_PREFIXES))

    return violations


def check_orchestration_boundaries(root: Path | None = None) -> list[str]:
    """Check that orchestration modules don't import materializers directly.

    Args:
        root: Repository root directory. Defaults to current directory.

    Returns:
        List of violations.
    """
    root = root or Path(".")
    violations: list[str] = []

    for module_rel in ORCHESTRATION_MODULES:
        module_path = root / module_rel
        if not module_path.exists():
            continue
        violations.extend(collect_import_violations(module_path, ORCHESTRATION_FORBIDDEN_PREFIXES))

    return violations


def check_policy_module_boundaries(root: Path | None = None) -> list[str]:
    """Check that policy modules don't import atlas or perform I/O.

    Args:
        root: Repository root directory. Defaults to current directory.

    Returns:
        List of violations.
    """
    root = root or Path(".")
    violations: list[str] = []
    policy_dir = root / "cleo" / "atlas_policies"

    if not policy_dir.exists():
        return violations

    for module_path in sorted(policy_dir.glob("*.py")):
        # Check for atlas imports
        violations.extend(collect_import_violations(module_path, ["cleo.atlas"]))
        # Check for direct I/O
        violations.extend(collect_io_call_violations(module_path))

    return violations


def check_domains_wind_metrics_boundary(root: Path | None = None) -> list[str]:
    """Check that domains.py doesn't import underscore symbols from wind_metrics.

    Args:
        root: Repository root directory. Defaults to current directory.

    Returns:
        List of violations.
    """
    root = root or Path(".")
    domains_path = root / "cleo" / "domains.py"

    if not domains_path.exists():
        return []

    return collect_underscore_import_violations(domains_path, "cleo.wind_metrics")


def check_loaders_assess_boundary(root: Path | None = None) -> list[str]:
    """Check that loaders.py doesn't runtime-import assess.

    Args:
        root: Repository root directory. Defaults to current directory.

    Returns:
        List of violations.
    """
    root = root or Path(".")
    loaders_path = root / "cleo" / "loaders.py"

    if not loaders_path.exists():
        return []

    return collect_import_violations(loaders_path, ["cleo.assess"])


# =============================================================================
# Aggregate Check
# =============================================================================


def check_all_boundaries(root: Path | None = None) -> list[str]:
    """Run all architecture boundary checks.

    Args:
        root: Repository root directory. Defaults to current directory.

    Returns:
        List of all violations across all checks.
    """
    root = root or Path(".")
    violations: list[str] = []

    violations.extend(check_compute_layer_boundaries(root))
    violations.extend(check_unification_boundaries(root))
    violations.extend(check_materializer_boundaries(root))
    violations.extend(check_orchestration_boundaries(root))
    violations.extend(check_policy_module_boundaries(root))
    violations.extend(check_domains_wind_metrics_boundary(root))
    violations.extend(check_loaders_assess_boundary(root))

    return violations
