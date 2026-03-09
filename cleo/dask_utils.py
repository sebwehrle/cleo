from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from importlib.util import find_spec
from typing import Any, Literal, Mapping, TypeAlias, overload

import xarray as xr

ChunksSpec: TypeAlias = Mapping[str, int]
ChunksArg: TypeAlias = ChunksSpec | Literal["auto"] | None
ComputeBackend: TypeAlias = Literal["serial", "threads", "processes", "distributed"]
ComputeWorkers: TypeAlias = int | None


__all__ = [
    "ChunksSpec",
    "ChunksArg",
    "ComputeBackend",
    "ComputeWorkers",
    "dask_is_available",
    "ensure_dask_available",
    "normalize_chunks",
    "normalize_compute_backend",
    "normalize_compute_workers",
    "chunks_for_rasterio",
    "is_dask_backed",
    "maybe_chunk",
    "get_distributed_client_and_dashboard",
    "scheduler_context",
    "compute",
    "persist",
]


def dask_is_available() -> bool:
    """Return ``True`` when ``dask`` can be imported."""
    return find_spec("dask") is not None


def ensure_dask_available(*, feature: str) -> None:
    """Raise a clear error when a dask-required feature is requested.

    :param feature: Short feature label for context in the error message.
    :raises ImportError: If ``dask`` is unavailable.
    """
    if not dask_is_available():
        raise ImportError(f"{feature} requires 'dask' but it is not installed. Install dask in the active environment.")


def normalize_chunks(chunks: ChunksSpec) -> dict[str, int]:
    """Normalize and validate chunk mapping.

    :param chunks: Mapping of dimension name to positive integer chunk size.
    :returns: Plain ``dict[str, int]`` with validated sizes.
    :raises ValueError: If any chunk size is not a positive integer.
    """
    out: dict[str, int] = {}
    for k, v in dict(chunks).items():
        size = int(v)
        if size <= 0:
            raise ValueError(f"Chunk size for dim {k!r} must be > 0, got {v!r}.")
        out[str(k)] = size
    return out


def normalize_compute_backend(backend: str) -> ComputeBackend:
    """Validate and normalize compute backend."""
    normalized = str(backend).strip().lower()
    allowed = {"serial", "threads", "processes", "distributed"}
    if normalized not in allowed:
        raise ValueError(
            f"Unknown compute_backend {backend!r}. Expected one of: 'serial', 'threads', 'processes', 'distributed'."
        )
    return normalized


def normalize_compute_workers(workers: int | None, *, backend: ComputeBackend) -> int | None:
    """Validate and normalize worker-count policy for compute backends."""
    if workers is None:
        return None

    workers_n = int(workers)
    if workers_n <= 0:
        raise ValueError(f"compute_workers must be >= 1 when provided, got {workers!r}.")

    backend_n = normalize_compute_backend(backend)
    if backend_n == "distributed":
        raise ValueError(
            "compute_workers is not supported for compute_backend='distributed'; "
            "configure worker count on the active dask.distributed Client."
        )
    if backend_n == "serial" and workers_n != 1:
        raise ValueError("compute_workers must be None or 1 when compute_backend='serial'.")
    return workers_n


def chunks_for_rasterio(*, chunks: ChunksArg, enabled: bool = True) -> dict[str, int] | Literal["auto"] | None:
    """Prepare chunks mapping for ``rioxarray.open_rasterio``.

    Ensures ``band=1`` exists when not provided.

    :param chunks: Base chunk mapping, ``"auto"``, or ``None``.
    :param enabled: If ``False``, return ``None``.
    :returns: Chunk mapping including ``band``, ``"auto"``, or ``None``.
    """
    if not enabled or chunks is None:
        return None
    if chunks == "auto":
        return "auto"
    chunks_n = normalize_chunks(chunks)
    if "band" not in chunks_n:
        chunks_n = {"band": 1, **chunks_n}
    return chunks_n


def _is_duck_dask_array(arr: Any) -> bool:
    """Return ``True`` if ``arr`` behaves like a dask-backed array."""
    try:
        from xarray.core.utils import is_duck_dask_array
    except (ImportError, ModuleNotFoundError):
        return False
    try:
        return bool(is_duck_dask_array(arr))
    except (TypeError, ValueError, AttributeError):
        return False


def is_dask_backed(obj: Any) -> bool:
    """Return whether an object is dask-backed.

    Supports DataArray, Dataset, and array-like objects.
    """
    if isinstance(obj, xr.DataArray):
        arrays = [obj.data]
    elif isinstance(obj, xr.Dataset):
        arrays = [v.data for v in obj.data_vars.values()]
    elif hasattr(obj, "data"):
        arrays = [getattr(obj, "data")]
    else:
        arrays = [obj]

    for arr in arrays:
        if arr is None:
            continue
        if _is_duck_dask_array(arr):
            return True
        module = getattr(arr, "__module__", "") or ""
        if module.startswith("dask"):
            return True
        if hasattr(arr, "__dask_graph__"):
            return True
    return False


@overload
def maybe_chunk(
    obj: xr.DataArray,
    *,
    chunks: ChunksArg,
    enabled: bool = True,
) -> xr.DataArray: ...


@overload
def maybe_chunk(
    obj: xr.Dataset,
    *,
    chunks: ChunksArg,
    enabled: bool = True,
) -> xr.Dataset: ...


def maybe_chunk(
    obj: xr.DataArray | xr.Dataset,
    *,
    chunks: ChunksArg,
    enabled: bool = True,
) -> xr.DataArray | xr.Dataset:
    """Apply chunking to an xarray object.

    ``band=1`` is auto-added when chunking a DataArray with a singleton ``band``
    dimension and no explicit band chunk is provided.

    :param obj: DataArray or Dataset.
    :param chunks: Chunk mapping, ``"auto"``, or ``None``.
    :param enabled: If ``False``, return ``obj`` unchanged.
    :returns: Chunked object.
    """
    if not enabled or chunks is None:
        return obj

    ensure_dask_available(feature="Chunking")
    if chunks == "auto":
        try:
            return obj.chunk("auto")
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            raise ValueError(f"Failed to chunk object with chunks='auto'. Available dims: {tuple(obj.dims)!r}") from e

    chunks_n = normalize_chunks(chunks)

    if isinstance(obj, xr.DataArray) and "band" in obj.dims and obj.sizes.get("band", 0) == 1:
        if "band" not in chunks_n:
            chunks_n = {"band": 1, **chunks_n}

    try:
        return obj.chunk(chunks_n)
    except (ValueError, TypeError, AttributeError, RuntimeError) as e:
        raise ValueError(f"Failed to chunk object with chunks={chunks_n!r}. Available dims: {tuple(obj.dims)!r}") from e


def get_distributed_client_and_dashboard() -> tuple[Any, str | None]:
    """Return active ``dask.distributed`` client and dashboard link.

    :raises RuntimeError: If distributed is not installed or no active client exists.
    """
    ensure_dask_available(feature="distributed scheduler")
    try:
        from dask.distributed import get_client
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError("compute_backend='distributed' requires 'dask[distributed]' installed.") from e

    try:
        client = get_client()
    except (ValueError, RuntimeError) as e:
        raise RuntimeError(
            "compute_backend='distributed' requires an active dask.distributed Client. "
            "Start one first, e.g. `from dask.distributed import Client; Client()`"
        ) from e

    return client, getattr(client, "dashboard_link", None)


def _local_scheduler_name(backend: ComputeBackend) -> str | None:
    """Map compute backend to dask local scheduler config value."""
    if backend == "serial":
        return "single-threaded"
    if backend == "threads":
        return "threads"
    if backend == "processes":
        return "processes"
    if backend == "distributed":
        return None
    raise ValueError(f"Unknown compute_backend: {backend!r}")


def _ensure_process_backend_entrypoint() -> None:
    """Validate that the current ``__main__`` module is safe for process workers.

    The local ``processes`` scheduler relies on multiprocessing spawn/import
    semantics. Interactive consoles, notebooks, and ``python -`` style entry
    points do not provide a reliably importable ``__main__`` module, which can
    surface as noisy shutdown-time ``sys.excepthook`` output even after a
    successful computation.

    :raises RuntimeError: If the active ``__main__`` module is not file-backed.
    """
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None) if main_module is not None else None
    if isinstance(main_file, str) and main_file and not main_file.startswith("<"):
        return
    raise RuntimeError(
        "compute_backend='processes' requires running CLEO from an importable Python script entrypoint. "
        "Interactive consoles, notebooks, `python -`, and similar sessions should use "
        "compute_backend='threads' or 'serial', or run the script behind "
        '`if __name__ == "__main__":`.'
    )


@contextmanager
def scheduler_context(*, backend: ComputeBackend, num_workers: ComputeWorkers = None):
    """Context manager that applies the requested compute backend.

    :param backend: Requested local or distributed execution backend.
    :param num_workers: Optional worker limit for local schedulers.
    :raises RuntimeError: If ``backend="processes"`` is requested from a
        non-importable console-style entrypoint.
    """
    backend_n = normalize_compute_backend(backend)
    workers_n = normalize_compute_workers(num_workers, backend=backend_n)
    if backend_n == "distributed":
        # Validate an active client exists; execution then follows default distributed behavior.
        get_distributed_client_and_dashboard()
        with nullcontext():
            yield
        return

    # Local scheduler modes use dask config context for deterministic execution policy.
    if backend_n == "processes":
        _ensure_process_backend_entrypoint()
    ensure_dask_available(feature=f"compute_backend={backend_n}")
    import dask

    scheduler = _local_scheduler_name(backend_n)
    config_kwargs: dict[str, Any] = {"scheduler": scheduler}
    if workers_n is not None:
        config_kwargs["num_workers"] = workers_n
    with dask.config.set(**config_kwargs):
        yield


@overload
def compute(obj: xr.DataArray, *, backend: ComputeBackend, num_workers: ComputeWorkers = None) -> xr.DataArray: ...


@overload
def compute(obj: xr.Dataset, *, backend: ComputeBackend, num_workers: ComputeWorkers = None) -> xr.Dataset: ...


def compute(
    obj: xr.DataArray | xr.Dataset,
    *,
    backend: ComputeBackend,
    num_workers: ComputeWorkers = None,
) -> xr.DataArray | xr.Dataset:
    """Compute an xarray object using the requested compute backend."""
    backend_n = normalize_compute_backend(backend)
    with scheduler_context(backend=backend_n, num_workers=num_workers):
        return obj.compute()


@overload
def persist(obj: xr.DataArray, *, backend: ComputeBackend, num_workers: ComputeWorkers = None) -> xr.DataArray: ...


@overload
def persist(obj: xr.Dataset, *, backend: ComputeBackend, num_workers: ComputeWorkers = None) -> xr.Dataset: ...


def persist(
    obj: xr.DataArray | xr.Dataset,
    *,
    backend: ComputeBackend,
    num_workers: ComputeWorkers = None,
) -> xr.DataArray | xr.Dataset:
    """Persist an xarray object using the requested compute backend."""
    backend_n = normalize_compute_backend(backend)
    with scheduler_context(backend=backend_n, num_workers=num_workers):
        return obj.persist()
