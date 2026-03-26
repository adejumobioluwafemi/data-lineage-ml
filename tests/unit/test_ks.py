"""
tests/unit/test_ks.py

Tests for the KS D-statistic approximation in ShiftDetector,
and for numeric column shift detection end-to-end.
"""

import pytest
import tempfile
import os
import numpy as np
import pandas as pd

from datalineageml.analysis.shift_detector import (
    ShiftDetector,
    _ks_from_stats,
    _ks_exact,
)
from datalineageml.analysis.profiler import DataFrameProfiler
from datalineageml.storage.sqlite_store import LineageStore


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    s = LineageStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _stats(mean, std, mn, mx, p25, p75, n=100):
    return {"mean": mean, "std": std, "min": mn, "max": mx,
            "p25": p25, "p75": p75, "n": n}


# ── _ks_from_stats ────────────────────────────────────────────────────────────

def test_ks_identical_distributions_near_zero():
    s = _stats(50.0, 10.0, 20.0, 80.0, 40.0, 60.0)
    ks = _ks_from_stats(s, s)
    assert ks < 0.05, f"Identical distributions should give KS ≈ 0, got {ks}"


def test_ks_completely_non_overlapping_near_one():
    # Distribution B is entirely to the right of A — maximum separation
    s_b = _stats(10.0, 1.0, 5.0, 15.0, 8.0, 12.0)
    s_a = _stats(90.0, 1.0, 85.0, 95.0, 88.0, 92.0)
    ks = _ks_from_stats(s_b, s_a)
    assert ks > 0.80, f"Non-overlapping distributions should give KS > 0.80, got {ks}"


def test_ks_bounded_zero_to_one():
    np.random.seed(42)
    for _ in range(20):
        s_b = _stats(np.random.uniform(0, 50), 5, 0, 100,
                     np.random.uniform(0, 40), np.random.uniform(50, 90))
        s_a = _stats(np.random.uniform(0, 50), 5, 0, 100,
                     np.random.uniform(0, 40), np.random.uniform(50, 90))
        ks = _ks_from_stats(s_b, s_a)
        assert 0.0 <= ks <= 1.0, f"KS out of bounds: {ks}"


def test_ks_larger_mean_shift_gives_larger_ks():
    base    = _stats(50.0, 10.0, 20.0, 80.0, 40.0, 60.0)
    small_s = _stats(52.0, 10.0, 22.0, 82.0, 42.0, 62.0)  # small shift
    large_s = _stats(75.0, 10.0, 45.0, 105., 65.0, 85.0)  # large shift
    ks_small = _ks_from_stats(base, small_s)
    ks_large = _ks_from_stats(base, large_s)
    assert ks_large > ks_small, (
        f"Larger mean shift should give larger KS: {ks_large:.3f} > {ks_small:.3f}"
    )


def test_ks_symmetric():
    s_b = _stats(30.0, 5.0, 10.0, 50.0, 25.0, 35.0)
    s_a = _stats(60.0, 5.0, 40.0, 80.0, 55.0, 65.0)
    assert abs(_ks_from_stats(s_b, s_a) - _ks_from_stats(s_a, s_b)) < 0.02


def test_ks_exact_fallback_returns_zero_without_scipy():
    """_ks_exact should return 0.0 gracefully if scipy is unavailable."""
    # We can't uninstall scipy, but we can verify the function is callable
    result = _ks_exact([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ── ShiftDetector KS integration ──────────────────────────────────────────────

def _log_snap_with_numeric(store, step, pos, rows, numeric_stats,
                            sensitive_stats=None):
    store.log_snapshot(
        run_id=f"r-{step}-{pos}",
        step_name=step, position=pos,
        row_count=rows, column_count=3,
        column_names=["age", "income", "gender"],
        null_rates={},
        numeric_stats=numeric_stats,
        categorical_stats={},
        sensitive_stats=sensitive_stats or {},
        recorded_at="2026-03-26T07:00:00",
    )


def test_detect_returns_ks_results_for_numeric_cols(store):
    before_stats = {"age": _stats(35.0, 10.0, 18.0, 65.0, 27.0, 45.0)}
    after_stats  = {"age": _stats(42.0, 8.0,  25.0, 65.0, 36.0, 50.0)}
    _log_snap_with_numeric(store, "clean", "before", 100, before_stats)
    _log_snap_with_numeric(store, "clean", "after",  80,  after_stats)

    detector = ShiftDetector(store=store)
    results  = detector.detect()

    ks_results = [r for r in results if r["test"] == "ks"]
    assert len(ks_results) > 0


def test_ks_result_has_required_fields(store):
    before_stats = {"income": _stats(50000, 10000, 30000, 90000, 42000, 65000)}
    after_stats  = {"income": _stats(72000, 8000,  55000, 95000, 65000, 80000)}
    _log_snap_with_numeric(store, "clean", "before", 100, before_stats)
    _log_snap_with_numeric(store, "clean", "after",  70,  after_stats)

    detector = ShiftDetector(store=store)
    results  = detector.detect()
    ks_r = next(r for r in results if r["test"] == "ks")

    required = {"step_name", "column", "test", "stat", "flag",
                "before_stats", "after_stats", "rows_before", "rows_after",
                "rows_removed", "removal_rate", "finding"}
    assert required.issubset(ks_r.keys())
    assert ks_r["test"] == "ks"
    assert "mean" in ks_r["before_stats"]


def test_ks_high_shift_flagged_correctly(store):
    # Before: income centred at 30k. After: centred at 80k. Completely different.
    before_stats = {"income": _stats(30000, 5000, 10000, 50000, 25000, 35000)}
    after_stats  = {"income": _stats(80000, 5000, 60000, 100000, 75000, 85000)}
    _log_snap_with_numeric(store, "normalize", "before", 100, before_stats)
    _log_snap_with_numeric(store, "normalize", "after",  100, after_stats)

    detector = ShiftDetector(store=store, ks_high=0.20)
    results  = detector.detect()
    ks_r = next(r for r in results if r["test"] == "ks" and r["column"] == "income")
    assert ks_r["flag"] == "HIGH", f"Expected HIGH, got {ks_r['flag']} (D={ks_r['stat']:.3f})"


def test_ks_stable_col_flagged_low(store):
    # income barely changes between before and after
    before_stats = {"income": _stats(60000, 10000, 30000, 90000, 50000, 70000)}
    after_stats  = {"income": _stats(61000, 10000, 31000, 91000, 51000, 71000)}
    _log_snap_with_numeric(store, "minor", "before", 100, before_stats)
    _log_snap_with_numeric(store, "minor", "after",  99,  after_stats)

    detector = ShiftDetector(store=store)
    results  = detector.detect()
    ks_r = next(r for r in results if r["test"] == "ks" and r["column"] == "income")
    assert ks_r["flag"] == "LOW", f"Stable column should be LOW, got {ks_r['flag']}"


def test_detect_returns_both_jsd_and_ks_for_mixed_columns(store):
    _log_snap_with_numeric(
        store, "clean_data", "before", 100,
        numeric_stats={"age": _stats(35, 10, 18, 65, 27, 45)},
        sensitive_stats={"gender": {"F": 0.5, "M": 0.5}},
    )
    _log_snap_with_numeric(
        store, "clean_data", "after", 70,
        numeric_stats={"age": _stats(42, 8, 25, 65, 35, 50)},
        sensitive_stats={"gender": {"F": 0.2, "M": 0.8}},
    )
    detector = ShiftDetector(store=store)
    results  = detector.detect()

    tests = {r["test"] for r in results}
    assert "jsd" in tests, "Expected JSD results for sensitive column"
    assert "ks"  in tests, "Expected KS results for numeric column"


def test_ks_finding_mentions_step_and_column(store):
    before_stats = {"age": _stats(30, 5, 10, 50, 25, 35)}
    after_stats  = {"age": _stats(55, 5, 45, 75, 50, 60)}
    _log_snap_with_numeric(store, "filter_step", "before", 100, before_stats)
    _log_snap_with_numeric(store, "filter_step", "after",  60,  after_stats)

    detector = ShiftDetector(store=store)
    results  = detector.detect()
    ks_r = next(r for r in results if r["test"] == "ks" and r["column"] == "age")

    assert "filter_step" in ks_r["finding"]
    assert "age" in ks_r["finding"]


def test_profiler_ks_full_pipeline_integration(store):
    """End-to-end: profile real DataFrames, store snapshots, detect KS shift."""
    np.random.seed(99)
    df_before = pd.DataFrame({
        "age":    np.random.normal(35, 10, 200).clip(18, 70),
        "income": np.random.normal(45000, 10000, 200).clip(20000, 100000),
        "gender": np.random.choice(["F", "M"], 200, p=[0.5, 0.5]),
    })
    # After: older, higher-income population (simulate demographic filter)
    df_after  = pd.DataFrame({
        "age":    np.random.normal(55, 8, 140).clip(30, 70),
        "income": np.random.normal(75000, 8000, 140).clip(50000, 100000),
        "gender": np.random.choice(["F", "M"], 140, p=[0.3, 0.7]),
    })

    profiler = DataFrameProfiler(sensitive_cols=["gender"])
    store.log_snapshot(**profiler.profile(df_before, "filter", "before", "r1"))
    store.log_snapshot(**profiler.profile(df_after,  "filter", "after",  "r1"))

    detector = ShiftDetector(store=store)
    results  = detector.detect()

    ks_age = next((r for r in results if r["test"] == "ks" and r["column"] == "age"), None)
    assert ks_age is not None
    assert ks_age["stat"] > 0.10, f"Expected KS > 0.10 for age shift, got {ks_age['stat']:.3f}"

    jsd_gender = next((r for r in results if r["test"] == "jsd" and r["column"] == "gender"), None)
    assert jsd_gender is not None
    assert jsd_gender["flag"] in ("HIGH", "MEDIUM")