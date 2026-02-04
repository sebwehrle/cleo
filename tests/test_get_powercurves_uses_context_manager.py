"""Guardrail test: YAML files must be opened via context managers."""

from pathlib import Path
from types import SimpleNamespace

import builtins

import cleo.loaders as L


def test_get_powercurves_uses_with_open(tmp_path, monkeypatch):
    resources = tmp_path / "resources"
    resources.mkdir(parents=True, exist_ok=True)

    (resources / "T0.yml").write_text(
        "manufacturer: M\nmodel: X\ncapacity: 1\nV: [0]\ncf: [[0]]\n",
        encoding="utf-8",
    )

    real_open = builtins.open

    class ContextEnforcingFile:
        def __init__(self, real_path: Path, mode: str, encoding: str | None):
            self._f = real_open(real_path, mode, encoding=encoding)
            self._in_ctx = False

        def __enter__(self):
            self._in_ctx = True
            return self

        def __exit__(self, exc_type, exc, tb):
            self._f.close()
            return False

        def read(self, *args, **kwargs):
            if not self._in_ctx:
                raise AssertionError("File read without context manager")
            return self._f.read(*args, **kwargs)

        def close(self):
            self._f.close()

    def enforcing_open(path, mode="r", encoding=None, *args, **kwargs):
        return ContextEnforcingFile(Path(path), mode, encoding)

    monkeypatch.setattr(builtins, "open", enforcing_open)

    self = SimpleNamespace(path=tmp_path, wind_turbines=["T0"])
    L.get_powercurves(self)

    assert hasattr(self, "power_curves")
