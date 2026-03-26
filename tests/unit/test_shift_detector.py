"""
tests/unit/test_shift_detector.py

Tests for ShiftDetector and _jensen_shannon_divergence.
Run with: python run_tests.py  OR  pytest tests/unit/test_shift_detector.py -v
"""

import pytest
import math
import io
import sys
import tempfile
import os
import pandas as pd
import numpy as np

from datalineageml.analysis.shift_detector import (
    ShiftDetector,
    _jensen_shannon_divergence,
)
from datalineageml.storage.sqlite_store import LineageStore
from datalineageml.analysis.profiler import DataFrameProfiler


@pytest.fixture
def store(tmp_path):
    s = LineageStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _log_snap(store, step_name, position, gender_dist, rows):
    """Helper: log a minimal snapshot with just sensitive_stats for gender."""
    store.log_snapshot(
        run_id=f"run-{step_name}-{position}",
        step_name=step_name,
        position=position,
        row_count=rows,
        column_count=2,
        column_names=["gender", "income"],
        null_rates={"gender": 0.0, "income": 0.0},
        numeric_stats={},
        categorical_stats={},
        sensitive_stats={"gender": gender_dist},
        recorded_at="2026-03-26T07:00:00",
    )


def _log_oyo_pair(store):
    """Log a before/after pair that mirrors the Oyo State scenario."""
    _log_snap(store, "clean_data", "before",
              {"F": 0.40, "M": 0.60}, 500)
    _log_snap(store, "clean_data", "after",
              {"F": 0.22, "M": 0.78}, 290)


# Jensen-Shannon divergence

def test_jsd_identical_distributions_is_zero():
    dist = {"F": 0.4, "M": 0.6}
    assert _jensen_shannon_divergence(dist, dist) == pytest.approx(0.0, abs=1e-9)


def test_jsd_completely_different_distributions_near_one():
    p = {"A": 1.0}
    q = {"B": 1.0}
    # JSD of disjoint dists = 1.0 (log base 2)
    assert _jensen_shannon_divergence(p, q) == pytest.approx(1.0, abs=1e-6)


def test_jsd_symmetric():
    p = {"F": 0.4, "M": 0.6}
    q = {"F": 0.2, "M": 0.8}
    assert (_jensen_shannon_divergence(p, q) ==
            pytest.approx(_jensen_shannon_divergence(q, p), abs=1e-9))


def test_jsd_bounded_zero_to_one():
    for _ in range(20):
        np.random.seed(_ * 7)
        vals = np.random.dirichlet([1, 1, 1])
        p = {"A": vals[0], "B": vals[1], "C": vals[2]}
        vals2 = np.random.dirichlet([1, 1, 1])
        q = {"A": vals2[0], "B": vals2[1], "C": vals2[2]}
        jsd = _jensen_shannon_divergence(p, q)
        assert 0.0 <= jsd <= 1.0, f"JSD out of bounds: {jsd}"


def test_jsd_small_shift_gives_low_value():
    p = {"F": 0.50, "M": 0.50}
    q = {"F": 0.48, "M": 0.52}
    jsd = _jensen_shannon_divergence(p, q)
    assert jsd < 0.01


def test_jsd_oyo_state_shift_is_high():
    # 40% → 22% female proportion: JSD = ~0.028, above HIGH threshold of 0.02
    before = {"F": 0.40, "M": 0.60}
    after  = {"F": 0.22, "M": 0.78}
    jsd = _jensen_shannon_divergence(before, after)
    assert jsd >= 0.02, f"Expected JSD >= 0.02, got {jsd:.4f}"


def test_jsd_empty_dict_returns_zero():
    assert _jensen_shannon_divergence({}, {"F": 1.0}) == 0.0
    assert _jensen_shannon_divergence({"F": 1.0}, {}) == 0.0


def test_jsd_handles_missing_keys_in_one_dist():
    # Q has a value P doesn't have
    p = {"F": 0.5, "M": 0.5}
    q = {"F": 0.3, "M": 0.3, "X": 0.4}
    jsd = _jensen_shan_divergence_safe(p, q)
    assert 0.0 <= jsd <= 1.0


def _jensen_shan_divergence_safe(p, q):
    return _jensen_shannon_divergence(p, q)


# ShiftDetector.detect() 

def test_detect_returns_list(store):
    _log_oyo_pair(store)
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    assert isinstance(results, list)


def test_detect_finds_clean_data_step(store):
    _log_oyo_pair(store)
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    step_names = [r["step_name"] for r in results]
    assert "clean_data" in step_names


def test_detect_result_has_required_fields(store):
    _log_oyo_pair(store)
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    assert len(results) > 0
    r = results[0]
    required = {"step_name", "column", "js_divergence", "flag",
                "before_dist", "after_dist", "rows_before", "rows_after",
                "rows_removed", "removal_rate", "finding"}
    assert required.issubset(r.keys())


def test_detect_sorted_by_jsd_descending(store):
    # Add two steps with different shift magnitudes
    _log_snap(store, "big_shift",   "before", {"F": 0.50, "M": 0.50}, 100)
    _log_snap(store, "big_shift",   "after",  {"F": 0.10, "M": 0.90}, 80)
    _log_snap(store, "small_shift", "before", {"F": 0.50, "M": 0.50}, 100)
    _log_snap(store, "small_shift", "after",  {"F": 0.48, "M": 0.52}, 98)

    detector = ShiftDetector(store=store)
    results  = detector.detect()

    jsds = [r["js_divergence"] for r in results]
    assert jsds == sorted(jsds, reverse=True), "Results not sorted by JSD descending"


def test_detect_oyo_flagged_as_high(store):
    _log_oyo_pair(store)
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    oyo_results = [r for r in results if r["step_name"] == "clean_data"]
    assert len(oyo_results) > 0
    assert oyo_results[0]["flag"] == "HIGH"


def test_detect_no_snapshots_returns_empty(store):
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    assert results == []


def test_detect_unpaired_snapshot_ignored(store):
    # Only 'before', no 'after' — should be ignored
    _log_snap(store, "orphan_step", "before", {"F": 0.5, "M": 0.5}, 100)
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    step_names = [r["step_name"] for r in results]
    assert "orphan_step" not in step_names


def test_detect_multiple_sensitive_cols(store):
    store.log_snapshot(
        run_id="r1", step_name="step_multi", position="before",
        row_count=100, column_count=3,
        column_names=["gender", "zone", "income"],
        null_rates={}, numeric_stats={}, categorical_stats={},
        sensitive_stats={"gender": {"F": 0.5, "M": 0.5},
                         "zone":   {"A": 0.4, "B": 0.6}},
        recorded_at="2026-03-26T07:00:00",
    )
    store.log_snapshot(
        run_id="r2", step_name="step_multi", position="after",
        row_count=80, column_count=3,
        column_names=["gender", "zone", "income"],
        null_rates={}, numeric_stats={}, categorical_stats={},
        sensitive_stats={"gender": {"F": 0.2, "M": 0.8},
                         "zone":   {"A": 0.38, "B": 0.62}},
        recorded_at="2026-03-26T07:00:01",
    )
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    columns  = [r["column"] for r in results]
    assert "gender" in columns
    assert "zone"   in columns


def test_detect_row_removal_stats(store):
    _log_oyo_pair(store)
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    r = next(x for x in results if x["step_name"] == "clean_data")
    assert r["rows_before"] == 500
    assert r["rows_after"]  == 290
    assert r["rows_removed"] == 210
    assert pytest.approx(r["removal_rate"], abs=1e-4) == 210 / 500


def test_detect_step_names_filter(store):
    _log_oyo_pair(store)
    _log_snap(store, "other_step", "before", {"F": 0.5, "M": 0.5}, 100)
    _log_snap(store, "other_step", "after",  {"F": 0.1, "M": 0.9}, 80)

    detector = ShiftDetector(store=store)
    results  = detector.detect(step_names=["clean_data"])
    step_names = {r["step_name"] for r in results}
    assert step_names == {"clean_data"}


# ShiftDetector flagging

def test_flag_high_above_threshold(store):
    _log_snap(store, "s", "before", {"F": 0.5, "M": 0.5}, 100)
    _log_snap(store, "s", "after",  {"F": 0.1, "M": 0.9}, 80)
    detector = ShiftDetector(store=store, high_threshold=0.05) # type: ignore
    results  = detector.detect()
    assert results[0]["flag"] == "HIGH"


def test_flag_low_for_near_identical(store):
    _log_snap(store, "s", "before", {"F": 0.50, "M": 0.50}, 100)
    _log_snap(store, "s", "after",  {"F": 0.49, "M": 0.51}, 98)
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    assert results[0]["flag"] == "LOW"


def test_custom_thresholds_respected(store):
    # With very high thresholds, even Oyo-level shift should be LOW
    _log_oyo_pair(store)
    detector = ShiftDetector(store=store, high_threshold=0.99, # type: ignore
                             medium_threshold=0.50) # type: ignore
    results  = detector.detect()
    assert results[0]["flag"] == "LOW"


# finding text

def test_finding_mentions_step_name(store):
    _log_oyo_pair(store)
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    assert "clean_data" in results[0]["finding"]


def test_finding_mentions_column(store):
    _log_oyo_pair(store)
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    assert "gender" in results[0]["finding"]


def test_finding_mentions_candidate_source(store):
    _log_oyo_pair(store)
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    assert "candidate causal" in results[0]["finding"].lower()


def test_finding_low_does_not_flag_candidate(store):
    _log_snap(store, "s", "before", {"F": 0.50, "M": 0.50}, 100)
    _log_snap(store, "s", "after",  {"F": 0.49, "M": 0.51}, 98)
    detector = ShiftDetector(store=store)
    results  = detector.detect()
    assert "candidate" not in results[0]["finding"].lower()


# top_candidate 

def test_top_candidate_returns_highest_jsd(store):
    _log_oyo_pair(store)
    _log_snap(store, "minor", "before", {"F": 0.5, "M": 0.5}, 100)
    _log_snap(store, "minor", "after",  {"F": 0.49, "M": 0.51}, 99)

    detector = ShiftDetector(store=store)
    results  = detector.detect()
    top      = detector.top_candidate(results)

    assert top is not None
    assert top["step_name"] == "clean_data"


def test_top_candidate_empty_results_returns_none(store):
    detector = ShiftDetector(store=store)
    assert detector.top_candidate([]) is None


# print_report

def test_print_report_no_error(store):
    _log_oyo_pair(store)
    detector = ShiftDetector(store=store)
    results  = detector.detect()

    cap = io.StringIO()
    sys.stdout = cap
    try:
        detector.print_report(results, title="Test Report")
    finally:
        sys.stdout = sys.__stdout__

    out = cap.getvalue()
    assert "Test Report" in out
    assert "clean_data"  in out
    assert "HIGH"        in out


def test_print_report_empty_results_no_error(store):
    detector = ShiftDetector(store=store)
    cap = io.StringIO()
    sys.stdout = cap
    try:
        detector.print_report([], title="Empty")
    finally:
        sys.stdout = sys.__stdout__
    assert "No paired snapshots" in cap.getvalue()


# integration: profiler + store + detector

def test_full_integration_oyo_scenario(store):
    """End-to-end: profile a real DataFrame, store snapshots, detect shift."""
    np.random.seed(42)
    n = 300
    gender = np.random.choice(["F", "M"], n, p=[0.40, 0.60])
    is_f   = gender == "F"
    land   = np.where(
        is_f,
        np.random.choice(["registered", None], n, p=[0.11, 0.89]), # type: ignore
        np.random.choice(["registered", None], n, p=[0.67, 0.33]), # type: ignore
    ).tolist()
    df = pd.DataFrame({
        "gender": gender,
        "land_title": land,
        "yield": np.random.uniform(1.5, 4.0, n).round(2),
    })

    profiler = DataFrameProfiler(sensitive_cols=["gender"])

    # Profile before
    snap_before = profiler.profile(df, step_name="clean_data",
                                   position="before", run_id="run-001")
    store.log_snapshot(**snap_before)

    # Apply the biased transformation
    df_clean = df.dropna()

    # Profile after
    snap_after = profiler.profile(df_clean, step_name="clean_data",
                                  position="after", run_id="run-001")
    store.log_snapshot(**snap_after)

    # Run detector
    detector = ShiftDetector(store=store)
    results  = detector.detect()

    assert len(results) > 0
    top = detector.top_candidate(results)
    assert top["step_name"] == "clean_data" # type: ignore
    assert top["column"]    == "gender" # type: ignore
    assert top["flag"]      == "HIGH" # type: ignore
    assert top["js_divergence"] > 0.02 # type: ignore
    assert "candidate causal" in top["finding"].lower() # type: ignore