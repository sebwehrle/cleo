"""Minimal benchmarking helpers for CLEO compute flows."""

from __future__ import annotations

import os
import re
import socket
import sys
import threading
import logging
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter, sleep
from typing import Any, Callable

import numpy as np
import pandas as pd
from cleo.unification.vertical_policy import canonical_json_dumps, sha256_hex_from_json

try:
    import psutil  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    psutil = None

try:
    _CLEO_VERSION = version("cleo")
except PackageNotFoundError:
    _CLEO_VERSION = None


logger = logging.getLogger(__name__)


_BENCH_CAPTURE_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    OSError,
    ArithmeticError,
    ImportError,
    AttributeError,
)


def _rss_mb() -> float | None:
    """Return current process RSS in MB, if psutil is available."""
    if psutil is None:
        return None
    process = psutil.Process()
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def _proc_tree_rss_mb() -> float | None:
    """Return RSS in MB for current process + recursive children, if available."""
    if psutil is None:
        return None
    try:
        proc = psutil.Process()
        rss = float(proc.memory_info().rss)
        for child in proc.children(recursive=True):
            try:
                rss += float(child.memory_info().rss)
            except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
                continue
        return rss / (1024.0 * 1024.0)
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
        return None


def _child_count() -> int | None:
    """Return recursive child process count for current process, if available."""
    if psutil is None:
        return None
    try:
        proc = psutil.Process()
        return int(len(proc.children(recursive=True)))
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
        return None


def _available_mb() -> float | None:
    """Return current system available memory in MB, if available."""
    if psutil is None:
        return None
    try:
        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
        return None


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
            raise ValueError(f"Context keys collide after normalization: {key!r} -> {ctx_key!r}")
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
        tree_rss_before = _proc_tree_rss_mb()
        peak_tree_rss = tree_rss_before
        peak_children = _child_count()
        min_available_mb = _available_mb()
        stop_sampling = False

        def _sample_memory() -> None:
            nonlocal peak_tree_rss, peak_children, min_available_mb
            while not stop_sampling:
                cur_tree = _proc_tree_rss_mb()
                if cur_tree is not None:
                    if peak_tree_rss is None or cur_tree > peak_tree_rss:
                        peak_tree_rss = cur_tree
                cur_children = _child_count()
                if cur_children is not None:
                    if peak_children is None or cur_children > peak_children:
                        peak_children = cur_children
                cur_available = _available_mb()
                if cur_available is not None:
                    if min_available_mb is None or cur_available < min_available_mb:
                        min_available_mb = cur_available
                sleep(0.05)

        sampler = threading.Thread(target=_sample_memory, daemon=True)
        sampler.start()
        ok = True
        error: str | None = None
        try:
            _materialize(fn())
        except _BENCH_CAPTURE_EXCEPTIONS as exc:  # pragma: no cover - tested via public wrappers
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            logger.debug("Benchmark callable raised captured exception.", exc_info=True)
        finally:
            stop_sampling = True
            sampler.join(timeout=1.0)
        elapsed_s = perf_counter() - start
        rss_after = _rss_mb()
        tree_rss_after = _proc_tree_rss_mb()
        rss_delta_mb = None if rss_before is None or rss_after is None else (rss_after - rss_before)
        tree_rss_delta_mb = (
            None if tree_rss_before is None or tree_rss_after is None else (tree_rss_after - tree_rss_before)
        )

        if run_idx >= warmup:
            rows.append(
                {
                    "label": label,
                    "run": run_idx - warmup + 1,
                    "seconds": elapsed_s,
                    "ok": ok,
                    "error": error,
                    "rss_before_mb": rss_before,
                    "rss_after_mb": rss_after,
                    "rss_delta_mb": rss_delta_mb,
                    "proc_tree_rss_before_mb": tree_rss_before,
                    "proc_tree_rss_after_mb": tree_rss_after,
                    "proc_tree_rss_delta_mb": tree_rss_delta_mb,
                    "proc_tree_peak_rss_mb": peak_tree_rss,
                    "peak_child_processes": peak_children,
                    "system_available_min_mb": min_available_mb,
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
            # Avoid duplicate graph execution: materialize() already computes/writes.
            result.materialize()
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


def benchmark_callable_variants(
    *,
    variants: list[dict[str, Any]],
    repeats: int = 3,
    warmup: int = 1,
    baseline_label: str | None = None,
) -> pd.DataFrame:
    """Benchmark callable variants side-by-side.

    Each variant dict must provide:
    - ``label``: unique variant label
    - ``fn``: zero-arg callable to benchmark
    Optional:
    - ``context``: extra context merged into benchmark metadata columns

    Returns per-run rows across all variants plus:
    - ``speedup_vs_baseline`` (median baseline seconds / median variant seconds)
    - ``baseline_label``
    """
    if not variants:
        raise ValueError("variants must be a non-empty list")

    labels = [str(v.get("label", "")) for v in variants]
    if any(not label for label in labels):
        raise ValueError("Each variant must include a non-empty 'label'")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Variant labels must be unique, got: {labels!r}")

    if baseline_label is None:
        baseline_label = labels[0]
    if baseline_label not in labels:
        raise ValueError(f"baseline_label {baseline_label!r} not found in variant labels {labels!r}")

    frames: list[pd.DataFrame] = []
    for variant in variants:
        label = str(variant["label"])
        fn = variant.get("fn")
        if not callable(fn):
            raise ValueError(f"Variant {label!r} must include callable 'fn'")
        context = dict(variant.get("context") or {})
        context["variant_label"] = label
        df = _benchmark_callable(
            fn,
            repeats=repeats,
            warmup=warmup,
            label=label,
            context=context,
        )
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    medians = out.loc[out["ok"]].groupby("label")["seconds"].median()
    baseline_median = medians.get(baseline_label)

    speedup_col: list[float | None] = []
    for label in out["label"]:
        median = medians.get(label)
        if pd.isna(median) or pd.isna(baseline_median) or float(median) == 0.0:
            speedup_col.append(None)
        else:
            speedup_col.append(float(baseline_median) / float(median))

    out["speedup_vs_baseline"] = speedup_col
    out["baseline_label"] = baseline_label
    return out


def benchmark_metric_variants(
    atlas: Any,
    metric: str,
    *,
    variants: list[dict[str, Any]],
    repeats: int = 3,
    warmup: int = 1,
    cache: bool = False,
    baseline_label: str | None = None,
) -> pd.DataFrame:
    """Benchmark multiple kwargs variants of one metric side-by-side.

    Each variant dict must provide:
    - ``label``: unique variant label
    - ``kwargs``: kwargs forwarded to ``atlas.wind.compute(metric, **kwargs)``
    Optional:
    - ``context``: extra context dict merged into benchmark metadata columns

    Returns per-run rows across all variants plus:
    - ``variant_label``
    - ``speedup_vs_baseline`` (median baseline seconds / median variant seconds)
    """
    if not variants:
        raise ValueError("variants must be a non-empty list")

    labels = [str(v.get("label", "")) for v in variants]
    if any(not label for label in labels):
        raise ValueError("Each variant must include a non-empty 'label'")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Variant labels must be unique, got: {labels!r}")

    if baseline_label is None:
        baseline_label = labels[0]
    if baseline_label not in labels:
        raise ValueError(f"baseline_label {baseline_label!r} not found in variant labels {labels!r}")

    frames: list[pd.DataFrame] = []
    for variant in variants:
        label = str(variant["label"])
        kwargs = dict(variant.get("kwargs") or {})
        context = dict(variant.get("context") or {})
        context["variant_label"] = label
        df = benchmark_case(
            atlas,
            metric,
            kwargs=kwargs,
            repeats=repeats,
            warmup=warmup,
            cache=cache,
            label=label,
            context=context,
        )
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    medians = out.loc[out["ok"]].groupby("label")["seconds"].median()
    baseline_median = medians.get(baseline_label)

    speedup_col: list[float | None] = []
    for label in out["label"]:
        median = medians.get(label)
        if pd.isna(median) or pd.isna(baseline_median) or float(median) == 0.0:
            speedup_col.append(None)
        else:
            speedup_col.append(float(baseline_median) / float(median))

    out["speedup_vs_baseline"] = speedup_col
    out["baseline_label"] = baseline_label
    return out


def evaluate_cf_mode_acceptance(
    df: pd.DataFrame,
    *,
    baseline_col: str = "cf_direct",
    candidate_col: str = "cf_candidate",
    turbine_col: str = "turbine_id",
    terrain_bin_col: str = "terrain_bin",
    median_threshold: float = 0.0025,
    p95_threshold: float = 0.0100,
    min_valid_pixels_global: int = 10_000,
    min_valid_pixels_per_turbine: int = 2_000,
    min_valid_pixels_per_bin: int = 2_000,
    insufficient_sample_policy: str = "fail",
) -> pd.DataFrame:
    """Evaluate acceptance thresholds for CF mode deltas across strata.

    Input dataframe contract:
    - must contain columns ``baseline_col`` and ``candidate_col``.
    - may contain ``turbine_col`` and ``terrain_bin_col`` for stratified checks.

    Output columns:
    - ``stratum``: ``global`` | ``per_turbine`` | ``per_bin``
    - ``stratum_key``: ``__all__`` or group key
    - ``n_valid``
    - ``median_abs_delta``
    - ``p95_abs_delta``
    - ``status``: ``pass`` | ``fail`` | ``insufficient_sample``
    """
    if baseline_col not in df.columns or candidate_col not in df.columns:
        raise ValueError(f"Dataframe must contain columns {baseline_col!r} and {candidate_col!r}")
    if insufficient_sample_policy not in {"fail", "skip"}:
        raise ValueError("insufficient_sample_policy must be 'fail' or 'skip'")

    work = df.copy()
    work["delta_abs"] = (work[candidate_col] - work[baseline_col]).abs()
    work = work[np.isfinite(work["delta_abs"])]

    rows: list[dict[str, Any]] = []

    def _evaluate_group(stratum: str, key: str, series: pd.Series, min_valid: int) -> None:
        vals = series.to_numpy(dtype=np.float64)
        n_valid = int(vals.size)
        if n_valid < int(min_valid):
            status = "insufficient_sample"
            if insufficient_sample_policy == "fail":
                status = "fail"
            rows.append(
                dict(
                    stratum=stratum,
                    stratum_key=key,
                    n_valid=n_valid,
                    median_abs_delta=np.nan,
                    p95_abs_delta=np.nan,
                    status=status,
                )
            )
            return

        med = float(np.median(vals))
        p95 = float(np.quantile(vals, 0.95))
        status = "pass" if (med <= median_threshold and p95 <= p95_threshold) else "fail"
        rows.append(
            dict(
                stratum=stratum,
                stratum_key=key,
                n_valid=n_valid,
                median_abs_delta=med,
                p95_abs_delta=p95,
                status=status,
            )
        )

    _evaluate_group("global", "__all__", work["delta_abs"], int(min_valid_pixels_global))

    if turbine_col in work.columns:
        for key, grp in work.groupby(turbine_col, sort=True):
            _evaluate_group("per_turbine", str(key), grp["delta_abs"], int(min_valid_pixels_per_turbine))

    if terrain_bin_col in work.columns:
        for key, grp in work.groupby(terrain_bin_col, sort=True):
            _evaluate_group("per_bin", str(key), grp["delta_abs"], int(min_valid_pixels_per_bin))

    out = pd.DataFrame(rows)
    if not out.empty:
        out["median_threshold"] = float(median_threshold)
        out["p95_threshold"] = float(p95_threshold)
        out["insufficient_sample_policy"] = insufficient_sample_policy
    return out


def build_benchmark_governance_record(
    *,
    benchmark_dataset_id: str,
    dataset_payload: Any,
    benchmark_region_mask_id: str,
    region_mask_payload: Any,
    benchmark_turbine_set_id: str,
    turbine_set_payload: Any,
    policy_snapshot: dict[str, Any],
    benchmark_random_seed: int | None = None,
) -> dict[str, Any]:
    """Build deterministic benchmark-governance IDs/checksums payload."""
    dataset_checksum = sha256_hex_from_json(dataset_payload)
    region_checksum = sha256_hex_from_json(region_mask_payload)
    turbine_checksum = sha256_hex_from_json(turbine_set_payload)
    policy_checksum = sha256_hex_from_json(policy_snapshot)

    record = {
        "benchmark_dataset_id": benchmark_dataset_id,
        "benchmark_dataset_checksum": dataset_checksum,
        "benchmark_region_mask_id": benchmark_region_mask_id,
        "benchmark_region_mask_checksum": region_checksum,
        "benchmark_turbine_set_id": benchmark_turbine_set_id,
        "benchmark_turbine_set_checksum": turbine_checksum,
        "benchmark_policy_snapshot_checksum": policy_checksum,
        "benchmark_random_seed": "none" if benchmark_random_seed is None else int(benchmark_random_seed),
    }
    # Stable string can be stored directly in attrs/manifests when needed.
    record["benchmark_governance_json"] = canonical_json_dumps(record)
    return record
