"""
tests/unit/test_cross_run.py

Tests for CrossRunComparator — demographic drift detection across runs.
"""

import pytest
import tempfile
import os
import numpy as np
import pandas as pd
import io
import sys

from datalineageml.analysis.cross_run import CrossRunComparator
from datalineageml.storage.sqlite_store import LineageStore


@pytest.fixture
def store(tmp_path):
    s = LineageStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _log_run(store, step, run_id, f_frac, rows=100, ts=None):
    """Log a single snapshot with a given F proportion."""
    store.log_snapshot(
        run_id=run_id, step_name=step, position="after",
        row_count=rows, column_count=2,
        column_names=["gender", "x"],
        null_rates={}, numeric_stats={}, categorical_stats={},
        sensitive_stats={"gender": {
            "F": round(f_frac, 4),
            "M": round(1 - f_frac, 4),
        }},
        recorded_at=ts or f"2026-03-{10 + int(run_id[-1]):02d}T07:00:00",
    )


def _declining_store(store):
    """Three runs with declining F representation."""
    for i, f in enumerate([0.45, 0.38, 0.30]):
        _log_run(store, "clean", f"run-{i}", f, ts=f"2026-03-{20+i:02d}T07:00:00")
    return store


def _stable_store(store):
    """Three runs with stable F representation."""
    for i in range(3):
        _log_run(store, "step", f"s-{i}", 0.50, ts=f"2026-03-{20+i:02d}T07:00:00")
    return store


def _improving_store(store):
    """Three runs with improving F representation."""
    for i, f in enumerate([0.25, 0.35, 0.45]):
        _log_run(store, "step", f"r-{i}", f, ts=f"2026-03-{20+i:02d}T07:00:00")
    return store


def test_compare_step_returns_dict(store):
    _declining_store(store)
    comp   = CrossRunComparator(store=store)
    report = comp.compare_step("clean", "gender")
    assert isinstance(report, dict)


def test_compare_step_required_keys(store):
    _declining_store(store)
    report = CrossRunComparator(store=store).compare_step("clean", "gender")
    for key in ("step_name", "sensitive_col", "position", "n_runs",
                "runs", "deltas", "worst_run", "best_run", "max_drift"):
        assert key in report, f"Missing key: '{key}'"


def test_compare_step_n_runs_correct(store):
    _declining_store(store)
    report = CrossRunComparator(store=store).compare_step("clean", "gender")
    assert report["n_runs"] == 3


def test_compare_step_runs_sorted_chronologically(store):
    _declining_store(store)
    report = CrossRunComparator(store=store).compare_step("clean", "gender")
    timestamps = [r["recorded_at"] for r in report["runs"]]
    assert timestamps == sorted(timestamps)


def test_compare_step_worst_run_is_lowest_f(store):
    """The worst run should be the one with the lowest F proportion."""
    _declining_store(store)
    report = CrossRunComparator(store=store).compare_step("clean", "gender")
    worst_idx = report["worst_run"]
    f_fracs   = [r["distribution"].get("F", 0) for r in report["runs"]]
    assert f_fracs[worst_idx] == min(f_fracs), (
        f"Worst run {worst_idx} has F={f_fracs[worst_idx]:.3f}, "
        f"but min is {min(f_fracs):.3f}"
    )


def test_compare_step_best_run_is_highest_f(store):
    _declining_store(store)
    report = CrossRunComparator(store=store).compare_step("clean", "gender")
    best_idx  = report["best_run"]
    f_fracs   = [r["distribution"].get("F", 0) for r in report["runs"]]
    assert f_fracs[best_idx] == max(f_fracs)


def test_compare_step_max_drift_positive(store):
    _declining_store(store)
    report = CrossRunComparator(store=store).compare_step("clean", "gender")
    assert report["max_drift"] > 0.0


def test_compare_step_max_drift_zero_for_stable(store):
    _stable_store(store)
    report = CrossRunComparator(store=store).compare_step("step", "gender")
    assert report["max_drift"] == pytest.approx(0.0, abs=1e-4)


def test_compare_step_deltas_count(store):
    """n runs → n-1 deltas."""
    _declining_store(store)
    report = CrossRunComparator(store=store).compare_step("clean", "gender")
    assert len(report["deltas"]) == report["n_runs"] - 1


def test_compare_step_delta_direction_declining(store):
    """In a declining scenario, F delta should be negative for each run transition."""
    _declining_store(store)
    report = CrossRunComparator(store=store).compare_step("clean", "gender")
    for delta in report["deltas"]:
        f_delta = delta["delta_by_group"].get("F", 0)
        assert f_delta < 0, (
            f"Expected negative F delta in declining scenario, got {f_delta}"
        )


def test_compare_step_no_snaps_returns_empty(store):
    report = CrossRunComparator(store=store).compare_step("nonexistent", "gender")
    assert report["n_runs"] == 0
    assert report["runs"] == []


def test_compare_step_last_n_runs_filter(store):
    for i, f in enumerate([0.50, 0.45, 0.40, 0.35, 0.30]):
        _log_run(store, "clean2", f"run-{i}", f, ts=f"2026-03-{20+i:02d}T07:00:00")
    report = CrossRunComparator(store=store).compare_step(
        "clean2", "gender", last_n_runs=3
    )
    assert report["n_runs"] == 3


def test_compare_step_run_has_distribution(store):
    _declining_store(store)
    report = CrossRunComparator(store=store).compare_step("clean", "gender")
    for run in report["runs"]:
        assert "distribution" in run
        assert "F" in run["distribution"]
        assert "M" in run["distribution"]


def test_compare_step_row_counts_stored(store):
    _log_run(store, "step_a", "r0", 0.40, rows=300, ts="2026-03-20T07:00:00")
    _log_run(store, "step_a", "r1", 0.35, rows=180, ts="2026-03-21T07:00:00")
    report = CrossRunComparator(store=store).compare_step("step_a", "gender")
    row_counts = [r["row_count"] for r in report["runs"]]
    assert 300 in row_counts
    assert 180 in row_counts


def test_trend_worsening(store):
    _declining_store(store)
    trend = CrossRunComparator(store=store).trend("clean", "gender", "F")
    assert trend["direction"] == "worsening", (
        f"Expected 'worsening', got '{trend['direction']}'"
    )
    assert trend["slope"] < 0


def test_trend_stable(store):
    _stable_store(store)
    trend = CrossRunComparator(store=store).trend("step", "gender", "F")
    assert trend["direction"] == "stable"
    assert abs(trend["slope"]) < 0.005


def test_trend_improving(store):
    _improving_store(store)
    trend = CrossRunComparator(store=store).trend("step", "gender", "F")
    assert trend["direction"] == "improving"
    assert trend["slope"] > 0


def test_trend_insufficient_data(store):
    """Single run cannot determine a trend."""
    _log_run(store, "single", "r0", 0.40, ts="2026-03-20T07:00:00")
    trend = CrossRunComparator(store=store).trend("single", "gender", "F", last_n_runs=1)
    assert trend["direction"] == "insufficient_data"


def test_trend_fractions_list(store):
    _declining_store(store)
    trend = CrossRunComparator(store=store).trend("clean", "gender", "F")
    assert isinstance(trend["fractions"], list)
    assert len(trend["fractions"]) == 3
    # Fractions should be decreasing
    assert trend["fractions"][0] > trend["fractions"][-1]


def test_trend_n_runs(store):
    _declining_store(store)
    trend = CrossRunComparator(store=store).trend("clean", "gender", "F")
    assert trend["n_runs"] == 3


def test_trend_last_n_runs_limit(store):
    for i, f in enumerate([0.50, 0.45, 0.40, 0.35, 0.30]):
        _log_run(store, "multi", f"r{i}", f, ts=f"2026-03-{20+i:02d}T07:00:00")
    trend = CrossRunComparator(store=store).trend("multi", "gender", "F", last_n_runs=3)
    assert trend["n_runs"] == 3


def test_print_report_no_error(store, capsys):
    _declining_store(store)
    report = CrossRunComparator(store=store).compare_step("clean", "gender")
    CrossRunComparator(store=store).print_report(report)
    out = capsys.readouterr().out
    assert "clean" in out
    assert "gender" in out


def test_print_report_shows_worst_run(store, capsys):
    _declining_store(store)
    comp   = CrossRunComparator(store=store)
    report = comp.compare_step("clean", "gender")
    comp.print_report(report)
    out = capsys.readouterr().out
    assert "worst" in out


def test_print_report_empty_no_error(store, capsys):
    report = CrossRunComparator(store=store).compare_step("nonexistent", "gender")
    CrossRunComparator(store=store).print_report(report)
    out = capsys.readouterr().out
    assert "No runs found" in out


# ── integration with DataFrameProfiler

def test_full_integration_with_profiler(store):
    """Simulate 4 real pipeline runs with the profiler, then compare."""
    from datalineageml.analysis.profiler import DataFrameProfiler
    np.random.seed(42)

    profiler = DataFrameProfiler(sensitive_cols=["gender"])

    # Run 0: balanced (50/50)
    df0 = pd.DataFrame({
        "gender": ["F"] * 50 + ["M"] * 50,
        "yield":  np.random.uniform(1.5, 4.0, 100),
    })
    store.log_snapshot(**profiler.profile(
        df0.dropna(), "clean", "after", run_id="run-0"))

    # Run 1: slight decline
    df1 = pd.DataFrame({
        "gender": ["F"] * 40 + ["M"] * 60,
        "yield":  np.random.uniform(1.5, 4.0, 100),
    })
    store.log_snapshot(**profiler.profile(
        df1.dropna(), "clean", "after", run_id="run-1"))

    # Run 2: more decline
    df2 = pd.DataFrame({
        "gender": ["F"] * 28 + ["M"] * 72,
        "yield":  np.random.uniform(1.5, 4.0, 100),
    })
    store.log_snapshot(**profiler.profile(
        df2.dropna(), "clean", "after", run_id="run-2"))

    comp   = CrossRunComparator(store=store)
    report = comp.compare_step("clean", "gender")
    assert report["n_runs"] == 3

    trend = comp.trend("clean", "gender", "F")
    assert trend["direction"] == "worsening"

    f_fracs = [r["distribution"].get("F", 0) for r in report["runs"]]
    assert f_fracs[0] > f_fracs[-1]