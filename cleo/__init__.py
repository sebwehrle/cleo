from importlib.metadata import version, PackageNotFoundError

from cleo.atlas import Atlas

try:
    __version__ = version("cleo")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

__all__ = [
    "Atlas",
    "__version__",
]
