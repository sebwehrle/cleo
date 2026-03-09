"""
loaders: get_clc_codes tests.
Split from merged test_resources_and_safe_extract.py
"""

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
