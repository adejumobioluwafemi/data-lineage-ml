"""
tests/unit/test_metrics.py

Tests for all fairness metrics:
  - DemographicParityGap (binary + multiclass)
  - EqualizedOdds (binary + multiclass)
  - PredictiveParity (binary + multiclass)
  - RegressionFairnessAuditor (ME, MAE, calibration, lazy solution)
  - FairnessResult / RegressionFairnessReport containers
  - compute_metric dispatcher
"""

import pytest
import math
import io
import sys
import numpy as np
import pandas as pd

from datalineageml.analysis.metrics import (
    DemographicParityGap,
    EqualizedOdds,
    PredictiveParity,
    RegressionFairnessAuditor,
    FairnessResult,
    compute_metric,
    _max_pairwise_gap,
)
from datalineageml.storage.sqlite_store import LineageStore

@pytest.fixture
def binary_balanced():
    return pd.DataFrame({
        "gender":   ["F"] * 100 + ["M"] * 100,
        "eligible": [1, 0] * 100,
        "pred":     [1, 0] * 100,
    })


@pytest.fixture
def binary_biased():
    """Female farmers systematically under-predicted."""
    np.random.seed(42)
    n = 500
    g   = np.random.choice(["F", "M"], n, p=[0.4, 0.6])
    y   = np.random.binomial(1, 0.60, n)
    yh  = np.random.binomial(1, np.where(g == "F", 0.30, 0.75), n)
    return pd.DataFrame({"gender": g, "eligible": y, "pred": yh})


@pytest.fixture
def multiclass_df():
    """3-class problem: crop quality = 0 (poor), 1 (medium), 2 (high)."""
    np.random.seed(7)
    n = 300
    g   = np.random.choice(["F", "M"], n, p=[0.4, 0.6])
    # Female farmers systematically rated lower
    true_probs_f = [0.50, 0.35, 0.15]
    true_probs_m = [0.20, 0.40, 0.40]
    y = np.array([
        np.random.choice([0, 1, 2], p=true_probs_f if gi == "F"
                         else true_probs_m)
        for gi in g
    ])
    # Biased predictor: over-predicts class 0 for F, over-predicts class 2 for M
    pred_probs_f = [0.60, 0.30, 0.10]
    pred_probs_m = [0.15, 0.35, 0.50]
    yh = np.array([
        np.random.choice([0, 1, 2], p=pred_probs_f if gi == "F"
                         else pred_probs_m)
        for gi in g
    ])
    return pd.DataFrame({"gender": g, "quality": y, "pred": yh})


@pytest.fixture
def regression_df():
    """Crop yield regression: female farms systematically under-predicted."""
    np.random.seed(99)
    n = 400
    g = np.random.choice(["F", "M"], n, p=[0.4, 0.6])
    actual  = np.where(g == "F",
                       np.random.normal(2.5, 0.6, n),
                       np.random.normal(3.0, 0.6, n))
    # Biased predictor: under-predicts female yields
    pred    = np.where(g == "F",
                       actual - 0.4 + np.random.normal(0, 0.1, n),
                       actual + 0.0 + np.random.normal(0, 0.1, n))
    return pd.DataFrame({"gender": g, "yield": actual, "pred": pred})


@pytest.fixture
def store(tmp_path):
    s = LineageStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


# ═══════════════════════════════════════════════════════════
# FairnessResult container
# ═══════════════════════════════════════════════════════════

def test_fairness_result_repr():
    r = FairnessResult("test", 0.34, {"F": 0.42, "M": 0.76}, 0.34,
                       "gender", "eligible", None, 2, "test", {})
    assert "test" in repr(r) and "0.34" in repr(r)


def test_to_store_kwargs_keys():
    r = FairnessResult("demographic_parity_gap", 0.34,
                       {"F": 0.42, "M": 0.76}, 0.34,
                       "gender", "eligible", None, 2, "test", {})
    kw = r.to_store_kwargs(run_id="r001", step_name="clean")
    assert kw["run_id"] == "r001"
    assert kw["step_name"] == "clean"
    assert "demographic_parity_gap" in kw["metrics"]


def test_to_store_kwargs_integrates_with_store(store):
    r = FairnessResult("demographic_parity_gap", 0.34,
                       {"F": 0.42, "M": 0.76}, 0.34,
                       "gender", "eligible", None, 2, "test", {})
    store.log_metrics(**r.to_store_kwargs(run_id="r001"))
    m = store.get_metrics(metric_name="demographic_parity_gap")
    assert len(m) == 1
    assert abs(m[0]["metric_value"] - 0.34) < 1e-5


# ═══════════════════════════════════════════════════════════
# DemographicParityGap — binary
# ═══════════════════════════════════════════════════════════

def test_dpg_binary_perfect_fairness(binary_balanced):
    r = DemographicParityGap.compute(binary_balanced, "gender", "eligible")
    assert r.primary_value == pytest.approx(0.0, abs=0.01)
    assert r.n_classes == 2


def test_dpg_binary_known_values():
    df = pd.DataFrame({
        "g": ["F"] * 100 + ["M"] * 100,
        "y": [1]*30 + [0]*70 + [1]*70 + [0]*30,
    })
    r = DemographicParityGap.compute(df, "g", "y")
    assert r.primary_value == pytest.approx(0.40, abs=1e-4)
    assert r.group_values["F"] == pytest.approx(0.30, abs=1e-4)
    assert r.group_values["M"] == pytest.approx(0.70, abs=1e-4)


def test_dpg_binary_fourfifths_flagged():
    df = pd.DataFrame({
        "g": ["F"] * 100 + ["M"] * 100,
        "y": [1]*20 + [0]*80 + [1]*80 + [0]*20,
    })
    r = DemographicParityGap.compute(df, "g", "y")
    assert r.details["fourfifths"] is False
    assert "4/5" in r.interpretation


def test_dpg_binary_no_model_needed():
    df = pd.DataFrame({"g": ["F","M","F","M"], "y": [0,1,0,1]})
    r = DemographicParityGap.compute(df, "g", "y")
    assert r is not None


def test_dpg_binary_missing_col_raises():
    df = pd.DataFrame({"g": ["F","M"]})
    with pytest.raises(ValueError, match="Columns not found"):
        DemographicParityGap.compute(df, "g", "missing")


def test_dpg_binary_single_group_raises():
    df = pd.DataFrame({"g": ["F"]*10, "y": [1]*10})
    with pytest.raises(ValueError, match="at least 2 groups"):
        DemographicParityGap.compute(df, "g", "y")


def test_dpg_binary_ignores_null_sensitive_col():
    df = pd.DataFrame({"g": ["F","M",None,"F","M"], "y": [1,0,1,0,1]})
    r = DemographicParityGap.compute(df, "g", "y")
    assert "None" not in r.group_values


# ═══════════════════════════════════════════════════════════
# DemographicParityGap — multiclass
# ═══════════════════════════════════════════════════════════

def test_dpg_multiclass_detects_n_classes(multiclass_df):
    r = DemographicParityGap.compute(multiclass_df, "gender", "quality")
    assert r.n_classes == 3


def test_dpg_multiclass_metric_name(multiclass_df):
    r = DemographicParityGap.compute(multiclass_df, "gender", "quality")
    assert r.metric_name == "demographic_parity_gap"


def test_dpg_multiclass_has_class_gaps(multiclass_df):
    r = DemographicParityGap.compute(multiclass_df, "gender", "quality")
    assert "class_gaps" in r.details
    assert len(r.details["class_gaps"]) == 3


def test_dpg_multiclass_gap_is_positive(multiclass_df):
    r = DemographicParityGap.compute(multiclass_df, "gender", "quality")
    assert r.primary_value >= 0.0


def test_dpg_multiclass_interpretation_mentions_classes(multiclass_df):
    r = DemographicParityGap.compute(multiclass_df, "gender", "quality")
    assert "class" in r.interpretation.lower() or "macro" in r.interpretation.lower()


def test_dpg_multiclass_biased_finds_gap(multiclass_df):
    # Female farmers clustered at class 0, male at class 2 → large class gaps
    r = DemographicParityGap.compute(multiclass_df, "gender", "quality")
    assert r.primary_value > 0.05, (
        f"Expected macro DPG > 0.05 for biased multiclass, got {r.primary_value:.4f}"
    )


# ═══════════════════════════════════════════════════════════
# EqualizedOdds — binary
# ═══════════════════════════════════════════════════════════

def test_eo_binary_perfect_fairness(binary_balanced):
    r = EqualizedOdds.compute(binary_balanced, "gender", "eligible", "pred")
    assert r.primary_value == pytest.approx(0.0, abs=0.01)
    assert r.n_classes == 2


def test_eo_binary_biased_tpr_gap(binary_biased):
    r = EqualizedOdds.compute(binary_biased, "gender", "eligible", "pred")
    assert r.details["tpr_by_group"]["M"] > r.details["tpr_by_group"]["F"]


def test_eo_binary_metric_name(binary_balanced):
    r = EqualizedOdds.compute(binary_balanced, "gender", "eligible", "pred")
    assert r.metric_name == "equalized_odds_gap"


def test_eo_binary_has_tpr_fpr_details(binary_balanced):
    r = EqualizedOdds.compute(binary_balanced, "gender", "eligible", "pred")
    assert "tpr_by_group" in r.details and "fpr_by_group" in r.details


def test_eo_binary_requires_prediction_col():
    df = pd.DataFrame({"g": ["F","M"], "y": [1,0]})
    with pytest.raises(ValueError, match="prediction_col"):
        compute_metric("eo", df, "g", "y")


def test_eo_binary_missing_pred_col_raises():
    df = pd.DataFrame({"g": ["F","M"], "y": [1,0]})
    with pytest.raises(ValueError):
        EqualizedOdds.compute(df, "g", "y", "nonexistent_pred")


# ═══════════════════════════════════════════════════════════
# EqualizedOdds — multiclass
# ═══════════════════════════════════════════════════════════

def test_eo_multiclass_n_classes(multiclass_df):
    r = EqualizedOdds.compute(multiclass_df, "gender", "quality", "pred")
    assert r.n_classes == 3


def test_eo_multiclass_metric_name(multiclass_df):
    r = EqualizedOdds.compute(multiclass_df, "gender", "quality", "pred")
    assert r.metric_name == "equalized_odds_gap"


def test_eo_multiclass_has_class_eo_gaps(multiclass_df):
    r = EqualizedOdds.compute(multiclass_df, "gender", "quality", "pred")
    assert "class_eo_gaps" in r.details
    assert len(r.details["class_eo_gaps"]) == 3


def test_eo_multiclass_biased_finds_gap(multiclass_df):
    r = EqualizedOdds.compute(multiclass_df, "gender", "quality", "pred")
    assert r.primary_value > 0.05


def test_eo_multiclass_interpretation_mentions_classes(multiclass_df):
    r = EqualizedOdds.compute(multiclass_df, "gender", "quality", "pred")
    assert "class" in r.interpretation.lower() or "macro" in r.interpretation.lower()


# ═══════════════════════════════════════════════════════════
# PredictiveParity — binary
# ═══════════════════════════════════════════════════════════

def test_pp_binary_perfect_fairness(binary_balanced):
    r = PredictiveParity.compute(binary_balanced, "gender", "eligible", "pred")
    assert r.primary_value == pytest.approx(0.0, abs=0.01)
    assert r.n_classes == 2


def test_pp_binary_known_precision():
    # F: 3 TP, 1 FP → PPV = 0.75.  M: 4 TP, 0 FP → PPV = 1.0
    df = pd.DataFrame({
        "g": ["F","F","F","F","M","M","M","M"],
        "y": [1,1,1,0,1,1,1,1],
        "p": [1,1,1,1,1,1,1,1],
    })
    r = PredictiveParity.compute(df, "g", "y", "p")
    assert r.details["ppv_by_group"]["F"] == pytest.approx(0.75, abs=1e-4)
    assert r.details["ppv_by_group"]["M"] == pytest.approx(1.00, abs=1e-4)
    assert r.primary_value == pytest.approx(0.25, abs=1e-4)


def test_pp_binary_metric_name(binary_balanced):
    r = PredictiveParity.compute(binary_balanced, "gender", "eligible", "pred")
    assert r.metric_name == "predictive_parity_gap"


def test_pp_binary_no_positive_preds_raises():
    df = pd.DataFrame({
        "g": ["F","F","M","M"],
        "y": [1,0,1,0],
        "p": [0,0,1,0],  # F never predicted positive
    })
    with pytest.raises(ValueError, match="at least 2 groups"):
        PredictiveParity.compute(df, "g", "y", "p")


def test_pp_binary_requires_pred_col():
    df = pd.DataFrame({"g": ["F","M"], "y": [1,0]})
    with pytest.raises(ValueError, match="prediction_col"):
        compute_metric("pp", df, "g", "y")


# ═══════════════════════════════════════════════════════════
# PredictiveParity — multiclass
# ═══════════════════════════════════════════════════════════

def test_pp_multiclass_n_classes(multiclass_df):
    r = PredictiveParity.compute(multiclass_df, "gender", "quality", "pred")
    assert r.n_classes == 3


def test_pp_multiclass_metric_name(multiclass_df):
    r = PredictiveParity.compute(multiclass_df, "gender", "quality", "pred")
    assert r.metric_name == "predictive_parity_gap"


def test_pp_multiclass_has_class_pp_gaps(multiclass_df):
    r = PredictiveParity.compute(multiclass_df, "gender", "quality", "pred")
    assert "class_pp_gaps" in r.details


def test_pp_multiclass_gap_non_negative(multiclass_df):
    r = PredictiveParity.compute(multiclass_df, "gender", "quality", "pred")
    assert r.primary_value >= 0.0


# ═══════════════════════════════════════════════════════════
# RegressionFairnessAuditor
# ═══════════════════════════════════════════════════════════

def test_regression_audit_returns_report(regression_df):
    aud = RegressionFairnessAuditor()
    rep = aud.audit(regression_df, "gender", "yield", "pred")
    from datalineageml.analysis.metrics import RegressionFairnessReport
    assert isinstance(rep, RegressionFairnessReport)


def test_regression_audit_detects_systematic_underprediction(regression_df):
    """Female farmers are under-predicted → negative ME for F, near-zero for M."""
    aud = RegressionFairnessAuditor()
    rep = aud.audit(regression_df, "gender", "yield", "pred")
    me_f = rep.group_stats["F"]["mean_error"]
    me_m = rep.group_stats["M"]["mean_error"]
    assert me_f < 0, f"Expected F under-predicted (ME < 0), got {me_f:.4f}"
    assert me_f < me_m, f"Expected F ME < M ME, got F:{me_f:.4f} M:{me_m:.4f}"


def test_regression_audit_me_gap_positive(regression_df):
    aud = RegressionFairnessAuditor()
    rep = aud.audit(regression_df, "gender", "yield", "pred")
    assert rep.me_gap > 0.0


def test_regression_audit_mae_gap_positive(regression_df):
    aud = RegressionFairnessAuditor()
    rep = aud.audit(regression_df, "gender", "yield", "pred")
    assert rep.mae_gap >= 0.0


def test_regression_audit_calibration_gap_positive(regression_df):
    aud = RegressionFairnessAuditor()
    rep = aud.audit(regression_df, "gender", "yield", "pred")
    assert rep.calibration_gap >= 0.0


def test_regression_audit_group_stats_keys(regression_df):
    rep = RegressionFairnessAuditor().audit(regression_df, "gender", "yield", "pred")
    for g, s in rep.group_stats.items():
        for key in ("n", "mean_error", "mae", "residual_std",
                    "actual_mean", "pred_mean", "calib_error", "lazy_flag"):
            assert key in s, f"Missing key '{key}' in group_stats['{g}']"


def test_regression_perfect_predictions_no_gap():
    """Perfect predictions → zero gaps, no lazy flags."""
    df = pd.DataFrame({
        "g":    ["F"] * 50 + ["M"] * 50,
        "y":    np.random.RandomState(1).normal(3.0, 0.5, 100),
        "pred": None,
    })
    df["pred"] = df["y"]  # perfect predictions
    rep = RegressionFairnessAuditor().audit(df, "g", "y", "pred")
    assert rep.me_gap   == pytest.approx(0.0, abs=1e-6)
    assert rep.mae_gap  == pytest.approx(0.0, abs=1e-6)
    assert rep.lazy_flags == []


def test_regression_lazy_solution_detected():
    """A model predicting the group mean should be flagged as lazy.

    The lazy solution signature: residual_std / outcome_std ≈ 1.0
    (the model explains almost none of the within-group variance).
    Default threshold flags groups where ratio >= 0.95.
    """
    np.random.seed(42)
    n = 200
    g = np.array(["F"] * 100 + ["M"] * 100)
    y = np.concatenate([
        np.random.normal(2.5, 0.8, 100),
        np.random.normal(3.5, 0.8, 100),
    ])
    # Lazy prediction: predict exactly the group mean for every observation.
    # residual = pred - actual = group_mean - actual → std(resid) ≈ std(actual)
    group_means = {"F": float(y[:100].mean()), "M": float(y[100:].mean())}
    pred = np.array([group_means[gi] for gi in g])

    df = pd.DataFrame({"g": g, "y": y, "pred": pred})
    # Use default threshold (0.05) — ratio ≈ 1.0 should be flagged
    rep = RegressionFairnessAuditor(lazy_threshold=0.05).audit(df, "g", "y", "pred")
    assert len(rep.lazy_flags) > 0, (
        "Expected lazy solution to be detected when predicting group means. "
        f"Got ratios: { {g: round(s['residual_std'] / (s['actual_mean'] or 1), 3) for g, s in rep.group_stats.items()} }"
    )


def test_regression_non_lazy_not_flagged(regression_df):
    """A model with informative predictions should not be flagged as lazy."""
    rep = RegressionFairnessAuditor().audit(regression_df, "gender", "yield", "pred")
    assert rep.lazy_flags == [], (
        f"Non-lazy model incorrectly flagged: {rep.lazy_flags}"
    )


def test_regression_non_numeric_outcome_raises():
    df = pd.DataFrame({
        "g": ["F","M"],
        "y": ["high","low"],
        "pred": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="numeric"):
        RegressionFairnessAuditor().audit(df, "g", "y", "pred")


def test_regression_missing_col_raises():
    df = pd.DataFrame({"g": ["F","M"], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Columns not found"):
        RegressionFairnessAuditor().audit(df, "g", "y", "nonexistent")


def test_regression_to_store_kwargs_keys(regression_df, store):
    rep = RegressionFairnessAuditor().audit(regression_df, "gender", "yield", "pred")
    kw  = rep.to_store_kwargs(run_id="r001", step_name="normalize")
    assert kw["run_id"] == "r001"
    assert "regression_me_gap"          in kw["metrics"]
    assert "regression_mae_gap"         in kw["metrics"]
    assert "regression_calibration_gap" in kw["metrics"]


def test_regression_store_kwargs_integrates(regression_df, store):
    rep = RegressionFairnessAuditor().audit(regression_df, "gender", "yield", "pred")
    store.log_metrics(**rep.to_store_kwargs(run_id="r001"))
    me_metrics = store.get_metrics(metric_name="regression_me_gap")
    assert len(me_metrics) == 1
    assert me_metrics[0]["metric_value"] == pytest.approx(rep.me_gap, abs=1e-5)


def test_regression_print_report_no_error(regression_df, capsys):
    rep = RegressionFairnessAuditor().audit(regression_df, "gender", "yield", "pred")
    rep.print_report()
    out = capsys.readouterr().out
    assert "ME gap" in out
    assert "MAE gap" in out
    assert "Calibration" in out


# ═══════════════════════════════════════════════════════════
# compute_metric dispatcher
# ═══════════════════════════════════════════════════════════

def test_compute_metric_dpg(binary_balanced):
    r = compute_metric("dpg", binary_balanced, "gender", "eligible")
    assert r.metric_name == "demographic_parity_gap"


def test_compute_metric_eo(binary_balanced):
    r = compute_metric("eo", binary_balanced, "gender", "eligible", "pred")
    assert r.metric_name == "equalized_odds_gap"


def test_compute_metric_pp(binary_balanced):
    r = compute_metric("pp", binary_balanced, "gender", "eligible", "pred")
    assert r.metric_name == "predictive_parity_gap"


def test_compute_metric_full_names(binary_balanced):
    for name, pred in [
        ("demographic_parity_gap",  None),
        ("equalized_odds_gap",      "pred"),
        ("predictive_parity_gap",   "pred"),
    ]:
        r = compute_metric(name, binary_balanced, "gender", "eligible", pred)
        assert r.metric_name == name


def test_compute_metric_unknown_raises():
    df = pd.DataFrame({"g": ["A","B"], "y": [1,0]})
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_metric("invalid_metric_name", df, "g", "y")


def test_compute_metric_eo_without_pred_raises():
    df = pd.DataFrame({"g": ["A","B"], "y": [1,0]})
    with pytest.raises(ValueError, match="prediction_col"):
        compute_metric("eo", df, "g", "y")


# ═══════════════════════════════════════════════════════════
# Chouldechova impossibility theorem
# ═══════════════════════════════════════════════════════════

def test_chouldechova_tension_documented():
    """When base rates differ by group, EO and PP cannot both hold."""
    np.random.seed(13)
    df = pd.DataFrame({
        "g":    ["A"] * 200 + ["B"] * 200,
        "y":    [1]*120 + [0]*80 + [1]*60 + [0]*140,  # different base rates
        "pred": [1]*100 + [0]*100 + [1]*100 + [0]*100,
    })
    eo = EqualizedOdds.compute(df, "g", "y", "pred")
    pp = PredictiveParity.compute(df, "g", "y", "pred")
    # Both should compute without error — the tension is real but the
    # functions should handle it gracefully
    assert isinstance(eo.primary_value, float)
    assert isinstance(pp.primary_value, float)


# ═══════════════════════════════════════════════════════════
# _max_pairwise_gap helper
# ═══════════════════════════════════════════════════════════

def test_max_pairwise_gap_two():
    assert _max_pairwise_gap({"A": 0.8, "B": 0.4}) == pytest.approx(0.4)


def test_max_pairwise_gap_three():
    assert _max_pairwise_gap({"A": 0.9, "B": 0.5, "C": 0.3}) == pytest.approx(0.6)


def test_max_pairwise_gap_identical():
    assert _max_pairwise_gap({"A": 0.5, "B": 0.5}) == pytest.approx(0.0)


def test_max_pairwise_gap_excludes_nan():
    assert _max_pairwise_gap({"A": 0.8, "B": float("nan"), "C": 0.3}) == pytest.approx(0.5)