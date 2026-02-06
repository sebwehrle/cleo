from importlib import resources as importlib_resources

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
