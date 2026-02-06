from pathlib import Path
from cleo.class_helpers import deploy_resources

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

class _Dummy:
    def __init__(self, path: Path):
        self.path = path

def test_deploy_resources_copies_defaults(tmp_path: Path):
    d = _Dummy(tmp_path)
    deploy_resources(d)

    resdir = tmp_path / "resources"
    assert resdir.is_dir()

    found = {p.name for p in resdir.iterdir() if p.is_file() and p.name.endswith(".yml")}
    missing = EXPECTED - found
    assert not missing, f"Missing deployed YAML resources: {sorted(missing)}"

def test_deploy_resources_is_idempotent_and_preserves_overrides(tmp_path: Path):
    d = _Dummy(tmp_path)
    resdir = tmp_path / "resources"
    resdir.mkdir(parents=True, exist_ok=True)

    # Create an override file that must not be overwritten
    target = resdir / "cost_assumptions.yml"
    sentinel = "SENTINEL_DO_NOT_OVERWRITE\n"
    target.write_text(sentinel, encoding="utf-8")

    deploy_resources(d)
    assert target.read_text(encoding="utf-8") == sentinel
