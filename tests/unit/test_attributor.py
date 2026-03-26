"""
tests/unit/test_attributor.py

Tests for CausalAttributor — the metric correlator (Layer 3.2).
"""

import pytest
import tempfile
import os
import numpy as np
import pandas as pd

from datalineageml.analysis.attributor import CausalAttributor, _recommendation
from datalineageml.analysis.profiler import DataFrameProfiler
from datalineageml.analysis.shift_detector import ShiftDetector
from datalineageml.storage.sqlite_store import LineageStore

@pytest.fixture
def store(tmp_path):
    s = LineageStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _oyo_store(tmp_path):
    """Convenience: return a store pre-loaded with the Oyo State scenario."""
    s = LineageStore(db_path=str(tmp_path / "oyo.db"))
    # load_data: no change
    s.log_snapshot(
        run_id="r-load-before", step_name="load_data", position="before",
        row_count=500, column_count=4, column_names=["gender","land","yield","eligible"],
        null_rates={}, numeric_stats={}, categorical_stats={},
        sensitive_stats={"gender": {"F": 0.40, "M": 0.60}},
        recorded_at="2026-03-26T07:00:00",
    )
    s.log_snapshot(
        run_id="r-load-after", step_name="load_data", position="after",
        row_count=500, column_count=4, column_names=["gender","land","yield","eligible"],
        null_rates={}, numeric_stats={}, categorical_stats={},
        sensitive_stats={"gender": {"F": 0.40, "M": 0.60}},
        recorded_at="2026-03-26T07:00:01",
    )
    # clean_data: LARGE shift — this is the biased step
    s.log_snapshot(
        run_id="r-clean-before", step_name="clean_data", position="before",
        row_count=500, column_count=4, column_names=["gender","land","yield","eligible"],
        null_rates={}, numeric_stats={}, categorical_stats={},
        sensitive_stats={"gender": {"F": 0.40, "M": 0.60}},
        recorded_at="2026-03-26T07:00:02",
    )
    s.log_snapshot(
        run_id="r-clean-after", step_name="clean_data", position="after",
        row_count=290, column_count=4, column_names=["gender","land","yield","eligible"],
        null_rates={}, numeric_stats={}, categorical_stats={},
        sensitive_stats={"gender": {"F": 0.22, "M": 0.78}},
        recorded_at="2026-03-26T07:00:03",
    )
    # normalize: tiny shift (just from floating point)
    s.log_snapshot(
        run_id="r-norm-before", step_name="normalize", position="before",
        row_count=290, column_count=4, column_names=["gender","land","yield","eligible"],
        null_rates={}, numeric_stats={}, categorical_stats={},
        sensitive_stats={"gender": {"F": 0.22, "M": 0.78}},
        recorded_at="2026-03-26T07:00:04",
    )
    s.log_snapshot(
        run_id="r-norm-after", step_name="normalize", position="after",
        row_count=290, column_count=4, column_names=["gender","land","yield","eligible"],
        null_rates={}, numeric_stats={}, categorical_stats={},
        sensitive_stats={"gender": {"F": 0.221, "M": 0.779}},
        recorded_at="2026-03-26T07:00:05",
    )
    return s


def test_attribute_returns_dict(tmp_path):
    store = _oyo_store(tmp_path)
    att   = CausalAttributor(store=store)
    result = att.attribute(sensitive_col="gender")
    assert isinstance(result, dict)
    store.close()


def test_attribute_required_keys(tmp_path):
    store = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    required = {"attributed_step", "column", "test", "stat", "confidence",
                "flag", "rows_removed", "removal_rate", "bias_metric",
                "all_scores", "evidence", "recommendation"}
    assert required.issubset(result.keys())
    store.close()


def test_attribute_identifies_clean_data(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    assert result["attributed_step"] == "clean_data", (
        f"Expected 'clean_data', got '{result['attributed_step']}'"
    )
    store.close()


def test_attribute_column_correct(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    assert result["column"] == "gender"
    store.close()


def test_attribute_flag_is_high(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    assert result["flag"] == "HIGH"
    store.close()


def test_attribute_confidence_between_0_and_1(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    assert 0.0 <= result["confidence"] <= 1.0
    store.close()


def test_attribute_confidence_higher_when_only_one_candidate(store):
    store.log_snapshot(
        run_id="rb", step_name="only_step", position="before",
        row_count=500, column_count=2, column_names=["gender", "x"],
        null_rates={}, numeric_stats={}, categorical_stats={},
        sensitive_stats={"gender": {"F": 0.5, "M": 0.5}},
        recorded_at="2026-03-26T07:00:00",
    )
    store.log_snapshot(
        run_id="ra", step_name="only_step", position="after",
        row_count=300, column_count=2, column_names=["gender", "x"],
        null_rates={}, numeric_stats={}, categorical_stats={},
        sensitive_stats={"gender": {"F": 0.2, "M": 0.8}},
        recorded_at="2026-03-26T07:00:01",
    )
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    assert result["confidence"] == pytest.approx(1.0)


def test_attribute_rows_removed_correct(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    assert result["rows_removed"] == 210  # 500 - 290
    assert pytest.approx(result["removal_rate"], abs=1e-4) == 210 / 500
    store.close()


def test_attribute_all_scores_sorted_descending(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    scores = [s["score"] for s in result["all_scores"]]
    assert scores == sorted(scores, reverse=True)
    store.close()


def test_attribute_inconclusive_when_no_snapshots(store):
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    assert result["attributed_step"] is None
    assert result["confidence"] == 0.0


def test_attribute_inconclusive_wrong_column(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="zone")
    assert result["attributed_step"] is None
    store.close()

def test_evidence_mentions_attributed_step(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    assert "clean_data" in result["evidence"]
    store.close()


def test_evidence_mentions_jsd(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    assert "JSD" in result["evidence"]
    store.close()


def test_recommendation_mentions_step_name(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    assert "clean_data" in result["recommendation"]
    store.close()


def test_recommendation_mentions_dropna_for_clean_step(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    rec_lower = result["recommendation"].lower()
    assert "clean" in rec_lower or "dropna" in rec_lower or "stratified" in rec_lower or "imputation" in rec_lower or "disproportionat" in rec_lower
    store.close()


def test_recommendation_pattern_matching():
    """_recommendation produces appropriate advice for known step name patterns."""
    mock_top = {"step_name": "dropna_records", "score": 0.5,
                "stat": 0.03, "flag": "HIGH", "test": "jsd",
                "rows_removed": 100, "removal_rate": 0.2,
                "before_dist": {}, "after_dist": {}}
    rec = _recommendation(mock_top, "gender")
    assert "stratified" in rec.lower() or "imputation" in rec.lower()


def test_recommendation_filter_step():
    mock_top = {"step_name": "filter_records", "score": 0.4,
                "stat": 0.025, "flag": "HIGH", "test": "jsd",
                "rows_removed": 80, "removal_rate": 0.16,
                "before_dist": {}, "after_dist": {}}
    rec = _recommendation(mock_top, "gender")
    assert "filter" in rec.lower() or "demographic" in rec.lower()


# ── logged metrics integration ────────────────────────────────────────────────

def test_attribute_uses_logged_metric(tmp_path):
    store  = _oyo_store(tmp_path)
    # Log a bias metric
    store.log_metrics(
        run_id="r-clean-before",
        metrics={"gender_bias_score": 0.34},
        metric_source="manual_audit",
        step_name="clean_data",
    )
    result = CausalAttributor(store=store).attribute(
        sensitive_col="gender",
        metric_name="gender_bias_score",
    )
    assert result["bias_metric"] != {}
    assert result["bias_metric"]["name"] == "gender_bias_score"
    assert pytest.approx(result["bias_metric"]["value"], abs=1e-4) == 0.34
    store.close()


def test_attribute_bias_metric_empty_when_not_logged(tmp_path):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    # No metric logged, no outcome_col provided
    assert result["bias_metric"] == {}
    store.close()


# ── step_names filter ─────────────────────────────────────────────────────────

def test_attribute_step_names_filter(tmp_path):
    store  = _oyo_store(tmp_path)
    # Restrict to normalize only — should give a very small shift → lower confidence
    result = CausalAttributor(store=store).attribute(
        sensitive_col="gender",
        step_names=["normalize"],
    )
    assert result["attributed_step"] == "normalize"
    store.close()


# ── print_attribution ─────────────────────────────────────────────────────────

def test_print_attribution_no_error(tmp_path, capsys):
    store  = _oyo_store(tmp_path)
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    CausalAttributor(store=store).print_attribution(result)
    out = capsys.readouterr().out
    assert "clean_data"   in out
    assert "gender"       in out
    assert "Attributed"   in out
    assert "Recommendation" in out
    store.close()


def test_print_attribution_inconclusive_no_error(store, capsys):
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")
    CausalAttributor(store=store).print_attribution(result)
    out = capsys.readouterr().out
    assert "INCONCLUSIVE" in out


# ── full end-to-end with profiler ─────────────────────────────────────────────

def test_full_loop_oyo_scenario(tmp_path):
    """Full loop: profile DataFrames → store snapshots → detect → attribute."""
    np.random.seed(42)
    n = 400
    gender = np.random.choice(["F", "M"], n, p=[0.40, 0.60])
    is_f   = gender == "F"
    land   = np.where(
        is_f,
        np.random.choice(["registered", None], n, p=[0.11, 0.89]), # type: ignore
        np.random.choice(["registered", None], n, p=[0.67, 0.33]), # type: ignore
    ).tolist()
    df = pd.DataFrame({
        "gender":     gender,
        "land_title": land,
        "yield_t_ha": np.random.uniform(1.5, 4.0, n).round(2),
    })

    profiler = DataFrameProfiler(sensitive_cols=["gender"])
    store    = LineageStore(db_path=str(tmp_path / "full_loop.db"))

    # Instrument the pipeline manually
    store.log_snapshot(**profiler.profile(df, "load_data",   "before", "r1"))
    store.log_snapshot(**profiler.profile(df, "load_data",   "after",  "r1"))

    store.log_snapshot(**profiler.profile(df,           "clean_data", "before", "r2"))
    store.log_snapshot(**profiler.profile(df.dropna(),  "clean_data", "after",  "r2"))

    df_clean = df.dropna().copy()
    store.log_snapshot(**profiler.profile(df_clean, "normalize", "before", "r3"))
    df_clean["yield_t_ha"] = (df_clean["yield_t_ha"] - df_clean["yield_t_ha"].min()) / \
                              (df_clean["yield_t_ha"].max() - df_clean["yield_t_ha"].min())
    store.log_snapshot(**profiler.profile(df_clean, "normalize", "after",  "r3"))

    # Attribute
    result = CausalAttributor(store=store).attribute(sensitive_col="gender")

    assert result["attributed_step"] == "clean_data"
    assert result["flag"]            == "HIGH"
    assert result["confidence"]      > 0.5
    assert "clean_data" in result["evidence"]
    assert result["recommendation"]  != ""

    store.close()