"""loaders: test_resources_and_safe_extract.
Merged test file (imports preserved per chunk).
"""

import builtins
import zipfile
import pytest
from pathlib import Path
from types import SimpleNamespace
from importlib import resources as importlib_resources
import cleo.loaders as L

# --- merged from tests/_staging/test_get_clc_codes_missing_resource_message.py ---

def test_get_clc_codes_missing_resource_is_actionable(tmp_path):
    parent = SimpleNamespace(path=tmp_path)
    self = SimpleNamespace(parent=parent)

    with pytest.raises(FileNotFoundError) as exc:
        L.get_clc_codes(self)

    msg = str(exc.value)
    assert "deploy_resources" in msg
    assert "clc_codes.yml" in msg


# --- merged from tests/_staging/test_get_powercurves_uses_context_manager.py ---

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


# --- merged from tests/_staging/test_load_nuts_safe_extract.py ---

def test_load_nuts_rejects_zip_slip(tmp_path, monkeypatch):
    # Craft outer zip containing the expected inner zip name.
    resolution = "03M"
    year = 2021
    crs = 4326
    file_collection = f"ref-nuts-{year}-{resolution}.shp.zip"
    file_name = f"NUTS_RG_{resolution}_{year}_{crs}.shp.zip"

    nuts_path = tmp_path / "data" / "nuts"
    nuts_path.mkdir(parents=True, exist_ok=True)

    outer_zip_path = nuts_path / file_collection

    inner_bytes_path = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner_bytes_path, "w") as inner:
        inner.writestr("../pwned.txt", "x")  # zip-slip attempt

    with zipfile.ZipFile(outer_zip_path, "w") as outer:
        outer.write(inner_bytes_path, arcname=file_name)

    def mock_download_file(url, fpath):
        # download already present
        return True

    monkeypatch.setattr(L, "download_file", mock_download_file)

    parent = SimpleNamespace(path=tmp_path)
    self = SimpleNamespace(parent=parent)

    with pytest.raises(ValueError, match="Unsafe zip member"):
        L.load_nuts(self, resolution=resolution, year=year, crs=crs)

    assert not (nuts_path.parent / "pwned.txt").exists()


# --- merged from tests/_staging/test_resources_packaged.py ---

EXPECTED = {
    "clc_codes.yml",
    "cost_assumptions.yml",
    "Enercon.E101.3050.yml",
    "Enercon.E115.3000.yml",
    "Enercon.E138.3500.yml",
    "Enercon.E160.5560.yml",
    "Enercon.E40.500.yml",
    "Enercon.E82.3000.yml",
    "Vestas.V100.1800.yml",
    "Vestas.V100.2000.yml",
    "Vestas.V112.3075.yml",
    "Vestas.V150.4200.yml",
}

def test_packaged_resources_present():
    root = importlib_resources.files("cleo").joinpath("resources")
    assert root.is_dir(), "Expected package directory cleo/resources/ to exist"

    found = {p.name for p in root.iterdir() if p.is_file() and p.name.endswith(".yml")}
    missing = EXPECTED - found
    assert not missing, f"Missing packaged YAML resources: {sorted(missing)}"
