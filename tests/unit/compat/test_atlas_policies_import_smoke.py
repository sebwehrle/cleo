"""Import smoke tests for Phase 2 atlas_policies scaffold."""

from __future__ import annotations

import importlib


def test_atlas_policies_modules_import_smoke() -> None:
    modules = [
        "cleo.atlas_policies",
        "cleo.atlas_policies.region_selection",
        "cleo.atlas_policies.nuts_catalog",
        "cleo.atlas_policies.cleanup",
    ]
    for module in modules:
        imported = importlib.import_module(module)
        assert imported is not None

