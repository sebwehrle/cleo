"""class_helpers: test_resources_deploy.

Contracts:
- deploy_resources(self) creates <self.path>/resources and copies default *.yml files there.
- deploy_resources is idempotent and must not overwrite user-provided overrides.
"""

from __future__ import annotations
from pathlib import Path
from cleo.atlas import Atlas

EXPECTED_YMLS = {
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


class _Dummy:
    def __init__(self, path: Path):
        self.path = path


def _resources_dir(base: Path) -> Path:
    return base / "resources"


def _list_yml_names(resdir: Path) -> set[str]:
    return {p.name for p in resdir.glob("*.yml") if p.is_file()}


def test_deploy_resources_copies_defaults(tmp_path: Path) -> None:
    d = _Dummy(tmp_path)

    Atlas._deploy_resources(d)

    resdir = _resources_dir(tmp_path)
    assert resdir.is_dir()

    found = _list_yml_names(resdir)
    missing = EXPECTED_YMLS - found
    assert not missing, f"Missing deployed YAML resources: {sorted(missing)}"


def test_deploy_resources_is_idempotent_and_preserves_overrides(tmp_path: Path) -> None:
    d = _Dummy(tmp_path)
    resdir = _resources_dir(tmp_path)
    resdir.mkdir(parents=True, exist_ok=True)

    # Override file must not be overwritten
    target = resdir / "cost_assumptions.yml"
    sentinel = "SENTINEL_DO_NOT_OVERWRITE\n"
    target.write_text(sentinel, encoding="utf-8")

    Atlas._deploy_resources(d)

    assert target.read_text(encoding="utf-8") == sentinel
