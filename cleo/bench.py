"""Minimal benchmarking helpers for CLEO compute flows."""

from __future__ import annotations

import os
import re
import socket
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Any, Callable

import pandas as pd

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    _CLEO_VERSION = version("cleo")
except PackageNotFoundError:
    _CLEO_VERSION = None


def _rss_mb() -> float | None:
    """Return current process RSS in MB, if psutil is available."""
    if psutil is None:
        return None
    process = psutil.Process()
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def _materialize(value: Any) -> Any:
    """Best-effort eager materialization for lazy objects (e.g., dask/xarray)."""
    compute = getattr(value, "compute", None)
    if callable(compute):
        return compute()
    return value


def _default_run_metadata() -> dict[str, Any]:
    """Return standard per-run metadata columns."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "cleo_version": _CLEO_VERSION,
    }


def _normalize_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Return user context as flat ``ctx_*`` columns."""
    if not context:
        return {}
    out: dict[str, Any] = {}
    for key, value in context.items():
        norm = re.sub(r"[^0-9A-Za-z_]+", "_", str(key)).strip("_").lower()
        if not norm:
            continue
        ctx_key = f"ctx_{norm}"
        if ctx_key in out:
            raise ValueError(
                f"Context keys collide after normalization: {key!r} -> {ctx_key!r}"
            )
        out[ctx_key] = value
    return out


def _benchmark_callable(
    fn: Callable[[], Any],
    *,
    repeats: int,
    warmup: int,
    label: str,
    context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if repeats <= 0:
        raise ValueError("repeats must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    rows: list[dict[str, Any]] = []
    context_cols = _normalize_context(context)
    total = warmup + repeats
    for run_idx in range(total):
        start = perf_counter()
        rss_before = _rss_mb()
        ok = True
        error: str | None = None
        try:
            _materialize(fn())
        except Exception as exc:  # pragma: no cover - tested via public wrappers
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed_s = perf_counter() - start
        rss_after = _rss_mb()
        rss_delta_mb = None if rss_before is None or rss_after is None else (rss_after - rss_before)

        if run_idx >= warmup:
            rows.append(
                {
                    "label": label,
                    "run": run_idx - warmup + 1,
                    "seconds": elapsed_s,
                    "ok": ok,
                    "error": error,
                    "rss_delta_mb": rss_delta_mb,
                    **_default_run_metadata(),
                    **context_cols,
                }
            )
    return pd.DataFrame(rows)


def benchmark_case(
    atlas: Any,
    metric: str,
    *,
    kwargs: dict[str, Any] | None = None,
    repeats: int = 3,
    warmup: int = 1,
    cache: bool = False,
    label: str | None = None,
    context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Benchmark one atlas wind metric and return per-run timings."""
    params = dict(kwargs or {})
    resolved_label = label or metric

    def _run_once() -> Any:
        result = atlas.wind.compute(metric, **params)
        if cache:
            # Avoid duplicate graph execution: cache() already materializes to disk.
            result.cache()
            return None
        return _materialize(result.data)

    out = _benchmark_callable(
        _run_once,
        repeats=repeats,
        warmup=warmup,
        label=resolved_label,
        context=context,
    )
    out["metric"] = metric
    out["cache"] = bool(cache)
    return out


def benchmark_compare(
    fn_a: Callable[[], Any],
    fn_b: Callable[[], Any],
    *,
    repeats: int = 3,
    warmup: int = 1,
    label_a: str = "A",
    label_b: str = "B",
    context_a: dict[str, Any] | None = None,
    context_b: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Benchmark and compare two callables. Speedup is median(A)/median(B)."""
    df_a = _benchmark_callable(
        fn_a,
        repeats=repeats,
        warmup=warmup,
        label=label_a,
        context=context_a,
    )
    df_b = _benchmark_callable(
        fn_b,
        repeats=repeats,
        warmup=warmup,
        label=label_b,
        context=context_b,
    )

    median_a = df_a.loc[df_a["ok"], "seconds"].median()
    median_b = df_b.loc[df_b["ok"], "seconds"].median()
    speedup: float | None
    if pd.isna(median_a) or pd.isna(median_b) or float(median_b) == 0.0:
        speedup = None
    else:
        speedup = float(median_a) / float(median_b)

    df = pd.concat([df_a, df_b], ignore_index=True)
    df["speedup_vs_a"] = None
    df.loc[df["label"] == label_a, "speedup_vs_a"] = 1.0
    df.loc[df["label"] == label_b, "speedup_vs_a"] = speedup
    return df
