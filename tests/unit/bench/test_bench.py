"""Minimal tests for cleo.bench helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from cleo.bench import benchmark_case, benchmark_compare


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
            "rss_delta_mb",
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
                        "rss_delta_mb": None,
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
                    "rss_delta_mb": None,
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
