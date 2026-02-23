"""Import smoke tests for the canonical unification package."""

from __future__ import annotations

import importlib


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
