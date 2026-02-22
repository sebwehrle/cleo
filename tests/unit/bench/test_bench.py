"""Minimal tests for cleo.bench helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from cleo.bench import (
    benchmark_callable_variants,
    benchmark_case,
    benchmark_compare,
    benchmark_metric_variants,
)


class _LazyValue:
    def __init__(self, value):
        self._value = value
        self.compute_calls = 0

    def compute(self):
        self.compute_calls += 1
        return self._value


class _Result:
    def __init__(self):
        self.data = _LazyValue(42)
        self.cache_calls = 0

    def cache(self):
        self.cache_calls += 1
        return self.data


class _Wind:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = 0
        self.results: list[_Result] = []

    def compute(self, metric: str, **kwargs):  # noqa: ARG002
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        res = _Result()
        self.results.append(res)
        return res


class _Atlas:
    def __init__(self, *, fail: bool = False):
        self.wind = _Wind(fail=fail)


def test_benchmark_case_schema_and_cache_flag():
    atlas = _Atlas()
    df = benchmark_case(
        atlas,
        "mean_wind_speed",
        repeats=2,
        warmup=1,
        cache=True,
        label="baseline",
        context={"scheduler": "processes", "workers": 4},
    )

    assert list(df["run"]) == [1, 2]
    assert set(
        [
            "label",
            "run",
            "seconds",
            "ok",
            "error",
            "rss_before_mb",
            "rss_after_mb",
            "rss_delta_mb",
            "proc_tree_rss_before_mb",
            "proc_tree_rss_after_mb",
            "proc_tree_rss_delta_mb",
            "proc_tree_peak_rss_mb",
            "peak_child_processes",
            "system_available_min_mb",
            "metric",
            "cache",
            "timestamp_utc",
            "pid",
            "host",
            "python",
            "cleo_version",
            "ctx_scheduler",
            "ctx_workers",
        ]
    ).issubset(df.columns)
    assert set(df["label"]) == {"baseline"}
    assert set(df["metric"]) == {"mean_wind_speed"}
    assert set(df["cache"]) == {True}
    assert set(df["ctx_scheduler"]) == {"processes"}
    assert set(df["ctx_workers"]) == {4}
    assert df["ok"].all()
    assert atlas.wind.calls == 3  # warmup + repeats
    assert all(res.cache_calls == 1 for res in atlas.wind.results)
    assert all(res.data.compute_calls == 0 for res in atlas.wind.results)


def test_benchmark_case_captures_failures():
    atlas = _Atlas(fail=True)
    df = benchmark_case(atlas, "mean_wind_speed", repeats=1, warmup=0)

    assert len(df) == 1
    assert bool(df.loc[0, "ok"]) is False
    assert "RuntimeError: boom" in str(df.loc[0, "error"])


def test_benchmark_compare_speedup_column(monkeypatch: pytest.MonkeyPatch):
    captured_contexts = []

    def _fake_benchmark_callable(fn, *, repeats, warmup, label, context=None):  # noqa: ANN001, ARG001
        captured_contexts.append((label, context))
        if label == "A":
            return pd.DataFrame(
                [
                    {
                        "label": "A",
                        "run": 1,
                        "seconds": 2.0,
                        "ok": True,
                        "error": None,
                        "rss_before_mb": None,
                        "rss_after_mb": None,
                        "rss_delta_mb": None,
                        "proc_tree_rss_before_mb": None,
                        "proc_tree_rss_after_mb": None,
                        "proc_tree_rss_delta_mb": None,
                        "proc_tree_peak_rss_mb": None,
                        "peak_child_processes": None,
                        "system_available_min_mb": None,
                    }
                ]
            )
        return pd.DataFrame(
            [
                {
                    "label": "B",
                    "run": 1,
                    "seconds": 1.0,
                    "ok": True,
                    "error": None,
                    "rss_before_mb": None,
                    "rss_after_mb": None,
                    "rss_delta_mb": None,
                    "proc_tree_rss_before_mb": None,
                    "proc_tree_rss_after_mb": None,
                    "proc_tree_rss_delta_mb": None,
                    "proc_tree_peak_rss_mb": None,
                    "peak_child_processes": None,
                    "system_available_min_mb": None,
                }
            ]
        )

    monkeypatch.setattr("cleo.bench._benchmark_callable", _fake_benchmark_callable)

    df = benchmark_compare(
        lambda: 1,
        lambda: 2,
        repeats=1,
        warmup=0,
        label_a="A",
        label_b="B",
        context_a={"scheduler": "threads"},
        context_b={"scheduler": "processes"},
    )

    a_speedup = df.loc[df["label"] == "A", "speedup_vs_a"].iloc[0]
    b_speedup = df.loc[df["label"] == "B", "speedup_vs_a"].iloc[0]
    assert float(a_speedup) == 1.0
    assert float(b_speedup) == 2.0
    assert captured_contexts == [
        ("A", {"scheduler": "threads"}),
        ("B", {"scheduler": "processes"}),
    ]


def test_benchmark_case_validates_repeat_and_warmup():
    atlas = _Atlas()
    with pytest.raises(ValueError, match="repeats must be > 0"):
        benchmark_case(atlas, "mean_wind_speed", repeats=0)
    with pytest.raises(ValueError, match="warmup must be >= 0"):
        benchmark_case(atlas, "mean_wind_speed", warmup=-1)


def test_benchmark_compare_validates_repeat_and_warmup():
    with pytest.raises(ValueError, match="repeats must be > 0"):
        benchmark_compare(lambda: 1, lambda: 2, repeats=0)
    with pytest.raises(ValueError, match="warmup must be >= 0"):
        benchmark_compare(lambda: 1, lambda: 2, warmup=-1)


def test_benchmark_case_context_key_collision_raises():
    atlas = _Atlas()
    with pytest.raises(ValueError, match="collide after normalization"):
        benchmark_case(
            atlas,
            "mean_wind_speed",
            context={"workers-per-node": 2, "workers_per_node": 4},
        )


def test_benchmark_case_sets_rss_delta_none_without_psutil(monkeypatch: pytest.MonkeyPatch):
    atlas = _Atlas()
    monkeypatch.setattr("cleo.bench.psutil", None)

    df = benchmark_case(atlas, "mean_wind_speed", repeats=1, warmup=0)

    assert len(df) == 1
    assert df["rss_delta_mb"].isna().all()
    assert df["rss_before_mb"].isna().all()
    assert df["rss_after_mb"].isna().all()
    assert df["proc_tree_rss_before_mb"].isna().all()
    assert df["proc_tree_rss_after_mb"].isna().all()
    assert df["proc_tree_rss_delta_mb"].isna().all()
    assert df["proc_tree_peak_rss_mb"].isna().all()
    assert df["peak_child_processes"].isna().all()
    assert df["system_available_min_mb"].isna().all()


def test_benchmark_metric_variants_schema_and_speedup(monkeypatch: pytest.MonkeyPatch):
    def _fake_benchmark_case(
        atlas,  # noqa: ANN001, ARG001
        metric,  # noqa: ANN001, ARG001
        *,
        kwargs=None,  # noqa: ANN001
        repeats=1,  # noqa: ANN001
        warmup=0,  # noqa: ANN001
        cache=False,  # noqa: ANN001
        label=None,  # noqa: ANN001
        context=None,  # noqa: ANN001
    ):
        sec = 2.0 if label == "baseline" else 1.0
        return pd.DataFrame(
            [
                {
                    "label": label,
                    "run": 1,
                    "seconds": sec,
                    "ok": True,
                    "error": None,
                    "rss_before_mb": None,
                    "rss_after_mb": None,
                    "rss_delta_mb": None,
                    "proc_tree_rss_before_mb": None,
                    "proc_tree_rss_after_mb": None,
                    "proc_tree_rss_delta_mb": None,
                    "proc_tree_peak_rss_mb": None,
                    "peak_child_processes": None,
                    "system_available_min_mb": None,
                    "metric": "capacity_factors",
                    "cache": False,
                    "ctx_variant_label": context["variant_label"],
                }
            ]
        )

    monkeypatch.setattr("cleo.bench.benchmark_case", _fake_benchmark_case)

    df = benchmark_metric_variants(
        atlas=object(),
        metric="capacity_factors",
        variants=[
            {"label": "baseline", "kwargs": {"mode": "hub"}},
            {"label": "candidate", "kwargs": {"mode": "hub", "rews_n": 7}},
        ],
        baseline_label="baseline",
    )

    assert set(df["label"]) == {"baseline", "candidate"}
    assert set(df["ctx_variant_label"]) == {"baseline", "candidate"}
    assert set(["speedup_vs_baseline", "baseline_label"]).issubset(df.columns)
    baseline_speed = float(df.loc[df["label"] == "baseline", "speedup_vs_baseline"].iloc[0])
    candidate_speed = float(df.loc[df["label"] == "candidate", "speedup_vs_baseline"].iloc[0])
    assert baseline_speed == 1.0
    assert candidate_speed == 2.0  # 2.0 / 1.0
    assert set(df["baseline_label"]) == {"baseline"}


def test_benchmark_metric_variants_validates_inputs():
    with pytest.raises(ValueError, match="non-empty list"):
        benchmark_metric_variants(atlas=object(), metric="m", variants=[])

    with pytest.raises(ValueError, match="non-empty 'label'"):
        benchmark_metric_variants(
            atlas=object(),
            metric="m",
            variants=[{"label": "", "kwargs": {}}],
        )

    with pytest.raises(ValueError, match="must be unique"):
        benchmark_metric_variants(
            atlas=object(),
            metric="m",
            variants=[
                {"label": "dup", "kwargs": {}},
                {"label": "dup", "kwargs": {}},
            ],
        )

    with pytest.raises(ValueError, match="baseline_label"):
        benchmark_metric_variants(
            atlas=object(),
            metric="m",
            variants=[{"label": "only", "kwargs": {}}],
            baseline_label="missing",
        )


def test_benchmark_callable_variants_schema_and_speedup():
    def _slow():
        return 1

    def _fast():
        return 2

    df = benchmark_callable_variants(
        variants=[
            {"label": "baseline", "fn": _slow},
            {"label": "candidate", "fn": _fast},
        ],
        repeats=1,
        warmup=0,
        baseline_label="baseline",
    )

    assert set(df["label"]) == {"baseline", "candidate"}
    assert set(df["ctx_variant_label"]) == {"baseline", "candidate"}
    assert set(["speedup_vs_baseline", "baseline_label"]).issubset(df.columns)
    assert set(df["baseline_label"]) == {"baseline"}


def test_benchmark_callable_variants_validates_inputs():
    with pytest.raises(ValueError, match="non-empty list"):
        benchmark_callable_variants(variants=[])

    with pytest.raises(ValueError, match="non-empty 'label'"):
        benchmark_callable_variants(variants=[{"label": "", "fn": lambda: 1}])

    with pytest.raises(ValueError, match="must be unique"):
        benchmark_callable_variants(
            variants=[
                {"label": "dup", "fn": lambda: 1},
                {"label": "dup", "fn": lambda: 2},
            ]
        )

    with pytest.raises(ValueError, match="baseline_label"):
        benchmark_callable_variants(
            variants=[{"label": "only", "fn": lambda: 1}],
            baseline_label="missing",
        )

    with pytest.raises(ValueError, match="callable 'fn'"):
        benchmark_callable_variants(
            variants=[{"label": "bad", "fn": 1}],
        )
