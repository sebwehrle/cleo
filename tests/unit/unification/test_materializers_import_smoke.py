"""Import smoke tests for Phase-3 materializer scaffolding."""

from __future__ import annotations

import importlib


MODULES = [
    "cleo.unification.materializers",
    "cleo.unification.materializers.shared",
    "cleo.unification.materializers.wind",
    "cleo.unification.materializers.landscape",
    "cleo.unification.materializers.region",
]


def test_materializers_modules_import() -> None:
    for mod in MODULES:
        imported = importlib.import_module(mod)
        assert imported is not None
