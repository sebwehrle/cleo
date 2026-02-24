"""Import smoke tests for the canonical unification package."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


def test_unification_module_imports_smoke() -> None:
    modules = [
        "cleo.unification",
        "cleo.unification.fingerprint",
        "cleo.unification.manifest",
        "cleo.unification.turbines",
        "cleo.unification.raster_io",
        "cleo.unification.nuts_io",
        "cleo.unification.gwa_io",
        "cleo.unification.unifier",
    ]
    for module in modules:
        imported = importlib.import_module(module)
        assert imported is not None


def test_unification_runtime_import_boundaries() -> None:
    root = Path(__file__).resolve().parents[3]
    unification_dir = root / "cleo" / "unification"
    forbidden_prefixes = (
        "cleo.loaders",
        "cleo.atlas",
        "cleo.domains",
        "cleo.results",
        "cleo.class_helpers",
    )

    violations: list[str] = []
    for path in sorted(unification_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden_prefixes:
                        if alias.name == prefix or alias.name.startswith(f"{prefix}."):
                            violations.append(f"{path}:{node.lineno} import {alias.name}")
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden_prefixes:
                    if node.module == prefix or node.module.startswith(f"{prefix}."):
                        violations.append(f"{path}:{node.lineno} from {node.module} import ...")

    assert violations == [], "unification import boundary violated:\n" + "\n".join(violations)
