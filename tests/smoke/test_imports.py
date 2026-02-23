"""Import-surface tests.

Cleo is a spatial wind resource assessment package; the geostack is a hard dependency.
"""

from __future__ import annotations

import importlib


def test_import_cleo() -> None:
    import cleo  # noqa: F401


def test_import_key_modules() -> None:
    for mod in [
        "cleo.assess",
        "cleo.atlas",
        "cleo.class_helpers",
        "cleo.copdem",
        "cleo.domains",
        "cleo.loaders",
        "cleo.net",
        "cleo.results",
        "cleo.spatial",
        "cleo.store",
        "cleo.utils",
        "cleo.wind_metrics",
    ]:
        importlib.import_module(mod)
