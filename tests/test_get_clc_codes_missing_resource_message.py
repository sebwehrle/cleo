"""Guardrail test: get_clc_codes must raise actionable error when resource missing."""

import pytest
from types import SimpleNamespace

import cleo.loaders as L


def test_get_clc_codes_missing_resource_is_actionable(tmp_path):
    parent = SimpleNamespace(path=tmp_path)
    self = SimpleNamespace(parent=parent)

    with pytest.raises(FileNotFoundError) as exc:
        L.get_clc_codes(self)

    msg = str(exc.value)
    assert "deploy_resources" in msg
    assert "clc_codes.yml" in msg
