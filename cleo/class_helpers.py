# %% imports
import logging
import logging.config
import shutil
from pathlib import Path

from cleo.spatial import crs_equal

logger = logging.getLogger(__name__)


# %% methods
def deploy_resources(self):
    """
    Ensure YAML resource files are present in the workdir at `<atlas.path>/resources/`.

    Contract (A3, always-deploy, idempotent):
    - Packaged defaults live under `cleo/resources/*.yml`.
    - On every Atlas init, ensure workdir has a copy of each packaged YAML.
    - Do NOT overwrite existing workdir YAMLs (workdir is the override surface).
    - Fail loudly if packaged resources cannot be found (broken install).
    """
    import logging
    import shutil
    from pathlib import Path
    from importlib import resources as importlib_resources

    dest_dir = Path(self.path) / "resources"
    dest_dir.mkdir(parents=True, exist_ok=True)

    pkg_root = importlib_resources.files("cleo").joinpath("resources")
    if not pkg_root.is_dir():
        raise FileNotFoundError(
            "Cleo packaged resources are missing (expected package dir `cleo/resources`). "
            "This indicates a broken installation/build. "
            "Reinstall from a proper wheel/sdist, or use the conda environment.yaml install."
        )

    packaged = [
        p for p in pkg_root.iterdir()
        if p.is_file() and p.name.lower().endswith(".yml")
    ]
    if not packaged:
        raise FileNotFoundError(
            "Cleo packaged resources directory exists but contains no *.yml files. "
            "This indicates a broken installation/build."
        )

    copied = 0
    skipped = 0
    for p in packaged:
        dest = dest_dir / p.name
        if dest.exists():
            skipped += 1
            continue

        # `as_file` materializes the resource to a real filesystem path (works for wheels too)
        with importlib_resources.as_file(p) as src_path:
            shutil.copy(src_path, dest)
        copied += 1

    logger.info(
        f"Resource files ensured in {dest_dir} (copied={copied}, skipped_existing={skipped})."
    )


def set_attributes(self):
    self.data.attrs['country'] = self.parent.country
    self.data.attrs['region'] = self.parent.region
    if self.data.rio.crs is None:
        raise AttributeError(f"{self.data} does not have a coordinate reference system.")
    # Semantic CRS comparison using centralized helper
    if not crs_equal(self.data.rio.crs, self.parent.crs):
        raise ValueError(f"Coordinate reference system mismatch: expected={self.parent.crs} got={self.data.rio.crs}")


def setup_logging(self, console_level="INFO", file_level="DEBUG"):
    """
    Configure cleo logger namespace without touching the root logger.
    All cleo modules should log via logging.getLogger(__name__) so messages
    propagate to the 'cleo' logger.
    """
    log_dir = self.path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"cleo_{self.country}.log"

    cleo_logger = logging.getLogger("cleo")
    cleo_logger.setLevel(logging.DEBUG)  # allow handlers to filter
    cleo_logger.propagate = False

    # Avoid duplicate handlers on repeated Atlas creation
    for h in list(cleo_logger.handlers):
        cleo_logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, str(console_level).upper(), logging.INFO))
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(getattr(logging, str(file_level).upper(), logging.DEBUG))
    fh.setFormatter(fmt)

    cleo_logger.addHandler(ch)
    cleo_logger.addHandler(fh)
