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


def test_unification_raster_io_no_copdem_runtime_import() -> None:
    raster_io_path = Path(__file__).resolve().parents[3] / "cleo" / "unification" / "raster_io.py"
    tree = ast.parse(raster_io_path.read_text(encoding="utf-8"), filename=str(raster_io_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "cleo.copdem"
                assert not alias.name.startswith("cleo.copdem.")
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "cleo.copdem"
            assert not node.module.startswith("cleo.copdem.")
