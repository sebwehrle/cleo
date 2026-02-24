"""Import-surface tests.

Cleo is a spatial wind resource assessment package; the geostack is a hard dependency.
"""

from __future__ import annotations

import importlib


def test_import_cleo() -> None:
    import cleo  # noqa: F401


def test_import_public_entrypoint() -> None:
    from cleo import Atlas

    assert Atlas.__name__ == "Atlas"


def test_import_atlas_surface_modules() -> None:
    for mod in [
        "cleo.atlas",
        "cleo.domains",
        "cleo.results",
    ]:
        importlib.import_module(mod)
