"""Static boundary guardrails for atlas policy modules."""

from __future__ import annotations

import ast
from pathlib import Path


POLICY_DIR = Path("cleo/atlas_policies")


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return ""


def test_atlas_policies_no_runtime_import_of_atlas() -> None:
    violations: list[str] = []
    for path in sorted(POLICY_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "cleo.atlas" or alias.name.startswith("cleo.atlas."):
                        violations.append(f"{path}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "cleo.atlas" or mod.startswith("cleo.atlas."):
                    violations.append(f"{path}:{node.lineno} from {mod} import ...")

    assert violations == [], "atlas_policies runtime import boundary violated:\n" + "\n".join(violations)


def test_atlas_policies_have_no_direct_store_io_calls() -> None:
    violations: list[str] = []
    for path in sorted(POLICY_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted_name(node.func)
            attr = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if (
                dotted in {"zarr.open_group", "xr.open_zarr", "xarray.open_zarr", "shutil.rmtree"}
                or attr in {"to_zarr", "iterdir", "glob", "rglob", "unlink", "rmdir"}
            ):
                violations.append(f"{path}:{node.lineno} -> {dotted or attr}")

    assert violations == [], "atlas_policies performed direct store/filesystem I/O:\n" + "\n".join(violations)
