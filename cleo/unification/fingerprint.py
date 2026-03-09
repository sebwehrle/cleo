"""Fingerprint and version helpers for canonical unification ownership."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import numpy as np


def get_git_info(repo_root: Path) -> dict[str, Any]:
    """Get git repository information for versioning.

    :param repo_root: Path to the repository root.
    :returns: Dictionary containing ``unify_version``, ``code_dirty``,
        optional ``git_diff_hash``, and ``package_version``.
    """
    # Get package version
    try:
        from importlib.metadata import version

        package_version = version("cleo")
    except (ImportError, ModuleNotFoundError, PackageNotFoundError):
        package_version = "unknown"

    result: dict[str, Any] = {
        "package_version": package_version,
        "code_dirty": False,
        "unify_version": f"unknown+{package_version}",
    }

    try:
        # Get git hash
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if git_hash.returncode == 0:
            result["unify_version"] = git_hash.stdout.strip()

            # Check if dirty
            git_status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if git_status.returncode == 0 and git_status.stdout.strip():
                result["code_dirty"] = True

                # Get diff hash if dirty
                git_diff = subprocess.run(
                    ["git", "diff"],
                    cwd=repo_root,
                    capture_output=True,
                    timeout=10,
                )

                if git_diff.returncode == 0:
                    diff_hash = hashlib.sha256(git_diff.stdout).hexdigest()[:16]
                    result["git_diff_hash"] = diff_hash

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # Git not available or not a repo - use defaults
        pass

    return result


def hash_grid_id(
    crs_wkt: str,
    transform: tuple[float, ...],
    shape: tuple[int, int],
    y: np.ndarray,
    x: np.ndarray,
    mask_policy: str,
) -> str:
    """Compute a stable hash identifying a grid configuration.

    :param crs_wkt: CRS in WKT format.
    :param transform: Affine transform tuple.
    :param shape: Grid shape ``(height, width)``.
    :param y: Y coordinate array.
    :param x: X coordinate array.
    :param mask_policy: Mask handling policy string.
    :returns: First 16 characters of SHA256 hash.
    """
    h = hashlib.sha256()

    # Hash scalars as JSON for stability
    scalars = {
        "crs_wkt": crs_wkt,
        "transform": list(transform),
        "shape": list(shape),
        "mask_policy": mask_policy,
    }
    h.update(json.dumps(scalars, sort_keys=True).encode("utf-8"))

    # Hash coordinate arrays as bytes
    h.update(np.asarray(y, dtype=np.float64).tobytes())
    h.update(np.asarray(x, dtype=np.float64).tobytes())

    return h.hexdigest()[:16]


def hash_inputs_id(items: list[tuple[str, str]], method: str) -> str:
    """Compute a stable hash identifying input sources.

    :param items: List of ``(name, fingerprint)`` pairs.
    :param method: Fingerprinting method used.
    :returns: First 16 characters of SHA256 hash.
    """
    h = hashlib.sha256()

    # Include method in hash
    h.update(f"method={method}\n".encode("utf-8"))

    # Sort items for stability and hash
    for name, fingerprint in sorted(items):
        h.update(f"{name}:{fingerprint}\n".encode("utf-8"))

    return h.hexdigest()[:16]


def fingerprint_path_mtime_size(path: Path) -> str:
    """Compute fingerprint from path, mtime, and size.

    :param path: Path to file.
    :returns: Fingerprint string.
    """
    stat = path.stat()
    return f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}"


def fingerprint_file(path: Path, method: str = "path_mtime_size") -> str:
    """Compute fingerprint for a file using specified method.

    :param path: Path to file.
    :param method: Fingerprinting method. Currently supports
        ``"path_mtime_size"``.
    :returns: Fingerprint string.
    :raises ValueError: If ``method`` is unknown.
    """
    if method == "path_mtime_size":
        return fingerprint_path_mtime_size(path)
    else:
        raise ValueError(f"Unknown fingerprint method: {method}")
