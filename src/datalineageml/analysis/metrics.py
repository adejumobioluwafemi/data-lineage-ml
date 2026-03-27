"""
src/datalineageml/analysis/metrics.py

Fairness metric computation for DataLineageML.

Classification metrics (binary and multiclass):
    DemographicParityGap    — equal selection rates across groups
    EqualizedOdds           — equal TPR/FPR (binary) or macro-averaged
                              per-class rates (multiclass)
    PredictiveParity        — equal precision across groups

Regression fairness diagnostics:
    RegressionFairnessAuditor — group-level residual analysis, calibration
                                check, and "lazy solution" guard

All return a FairnessResult (classification) or RegressionFairnessReport
(regression) that feeds directly into store.log_metrics() and CausalAttributor.

──────────────────────────────────────────────────────────────────────────────
WHICH METRIC TO USE
──────────────────────────────────────────────────────────────────────────────

Binary classification:
  DPG  — who gets selected at all (access/allocation). No model needed.
  EO   — false negatives and false positives both carry harm.
  PP   — being predicted positive should mean the same thing for all groups.

Multiclass classification:
  DPG  — selection rate per class per group (one-vs-rest for each class).
  EO   — macro-averaged TPR/FPR gap across all classes.
  PP   — macro-averaged precision gap across all classes.

Regression:
  Use RegressionFairnessAuditor. There is no single "correct" metric.
  The auditor runs four diagnostics:
    1. Mean Error (ME) gap — is the model systematically over/under-predicting
       for any group? (signed — direction matters)
    2. Mean Absolute Error (MAE) gap — magnitude of error difference.
    3. Calibration check — do group-level predicted means match actual means?
    4. Lazy solution guard — is the model just predicting the group mean?
       If residual std ≈ 0, the model has learned the group mean, not the
       outcome. This is the failure mode of naive fairness constraints.

──────────────────────────────────────────────────────────────────────────────
KNOWN THEORETICAL LIMITS (Chouldechova 2017, Kleinberg et al. 2017)
──────────────────────────────────────────────────────────────────────────────

When base rates differ by group, it is mathematically impossible to satisfy
both EqualizedOdds and PredictiveParity simultaneously. This is not a bug in
the implementation — it is a theorem. Choosing a metric is a policy decision
about which kind of error is costlier in your application.

Usage:
    from datalineageml.analysis.metrics import (
        DemographicParityGap,
        EqualizedOdds,
        PredictiveParity,
        RegressionFairnessAuditor,
        compute_metric,
    )

    # Binary or multiclass — no model needed for DPG
    result = DemographicParityGap.compute(df, "gender", "subsidy_eligible")
    store.log_metrics(**result.to_store_kwargs(run_id="r-001"))

    # After training a classifier
    df["pred"] = model.predict(X)
    result = EqualizedOdds.compute(df, "gender", "eligible", "pred")

    # For regression models
    df["pred"] = regressor.predict(X)
    report = RegressionFairnessAuditor.audit(df, "gender", "yield_t_ha", "pred")
    report.print_report()
    store.log_metrics(**report.to_store_kwargs(run_id="r-001"))
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple


# ── FairnessResult ────────────────────────────────────────────────────────────

class FairnessResult:
    """Container for a computed classification fairness metric.

    Attributes:
        metric_name:    Canonical name used as key in store.log_metrics().
        primary_value:  Headline scalar — the gap between best/worst group.
        group_values:   Per-group metric values.
        gap:            Max pairwise absolute difference between groups.
        sensitive_col:  Demographic column name.
        outcome_col:    True label column name.
        prediction_col: Prediction column name (None for DPG).
        n_classes:      Number of outcome classes (2 = binary).
        interpretation: One-sentence plain-language explanation.
        details:        Full breakdown dict.
    """

    def __init__(self, metric_name, primary_value, group_values, gap,
                 sensitive_col, outcome_col, prediction_col,
                 n_classes, interpretation, details):
        self.metric_name    = metric_name
        self.primary_value  = primary_value
        self.group_values   = group_values
        self.gap            = gap
        self.sensitive_col  = sensitive_col
        self.outcome_col    = outcome_col
        self.prediction_col = prediction_col
        self.n_classes      = n_classes
        self.interpretation = interpretation
        self.details        = details

    def to_store_kwargs(self, run_id: str,
                        step_name: Optional[str] = None) -> Dict:
        """Return kwargs for store.log_metrics(**kwargs)."""
        return dict(
            run_id        = run_id,
            metrics       = {self.metric_name: round(self.primary_value, 6)},
            metric_source = "DataLineageML.metrics",
            step_name     = step_name,
            tags          = {
                "sensitive_col": self.sensitive_col,
                "outcome_col":   self.outcome_col,
                "n_classes":     str(self.n_classes),
            },
        )

    def print_report(self) -> None:
        _print_classification_report(self)

    def __repr__(self):
        return (f"FairnessResult({self.metric_name}={self.primary_value:.4f}, "
                f"gap={self.gap:.4f}, n_classes={self.n_classes})")


# Demographic Parity Gap 
class DemographicParityGap:
    """Demographic Parity Gap — equal selection rates across groups.

    Binary: gap = |rate_A - rate_B|
    Multiclass: for each class c, compute selection rate per group (one-vs-rest),
                then report the macro-average gap across all classes.

    Does NOT require a trained model. Computable from outcome labels alone,
    making it suitable for pre-training bias attribution.

    The 4/5ths rule (US EEOC) flags DPG as potentially discriminatory when
    the minimum group's rate is below 80% of the maximum group's rate.
    """

    @staticmethod
    def compute(df, sensitive_col: str, outcome_col: str,
                positive_label: Any = 1) -> FairnessResult:
        """
        Args:
            df:             pandas DataFrame.
            sensitive_col:  Demographic group column.
            outcome_col:    Binary or multiclass outcome column.
            positive_label: For binary only — the positive class label.
                            Ignored for multiclass (all classes are evaluated).
        """
        _require_cols(df, [sensitive_col, outcome_col])
        classes = _detect_classes(df, outcome_col)

        if len(classes) == 2:
            return _dpg_binary(df, sensitive_col, outcome_col, positive_label)
        else:
            return _dpg_multiclass(df, sensitive_col, outcome_col, classes)


def _dpg_binary(df, sensitive_col, outcome_col, positive_label):
    groups, group_rates = _selection_rates(df, sensitive_col, outcome_col,
                                           positive_label)
    if len(group_rates) < 2:
        raise ValueError(_not_enough_groups(sensitive_col, group_rates))

    max_r = max(group_rates.values())
    min_r = min(group_rates.values())
    gap   = round(max_r - min_r, 6)
    max_g = max(group_rates, key=group_rates.get) # type: ignore
    min_g = min(group_rates, key=group_rates.get) # type: ignore

    fourfifths = group_rates[min_g] >= 0.8 * group_rates[max_g]
    interp = (
        f"Group '{min_g}' has a {gap:.1%} lower selection rate than "
        f"group '{max_g}' ({group_rates[min_g]:.1%} vs {group_rates[max_g]:.1%}). "
        + ("This exceeds the 4/5ths rule threshold (potential discrimination)."
           if not fourfifths
           else "This is within the 4/5ths rule tolerance.")
    )
    return FairnessResult(
        metric_name    = "demographic_parity_gap",
        primary_value  = gap,
        group_values   = group_rates,
        gap            = gap,
        sensitive_col  = sensitive_col,
        outcome_col    = outcome_col,
        prediction_col = None,
        n_classes      = 2,
        interpretation = interp,
        details        = {
            "max_group": max_g, "min_group": min_g,
            "max_rate": max_r,  "min_rate": min_r,
            "fourfifths": fourfifths,
            "class_gaps": {str(positive_label): gap},
        },
    )


def _dpg_multiclass(df, sensitive_col, outcome_col, classes):
    """One-vs-rest DPG per class, then macro-average the gaps."""
    class_gaps: Dict[str, float] = {}
    class_rates: Dict[str, Dict[str, float]] = {}

    for cls in classes:
        _, rates = _selection_rates(df, sensitive_col, outcome_col, cls)
        if len(rates) < 2:
            continue
        gap = round(max(rates.values()) - min(rates.values()), 6)
        class_gaps[str(cls)]  = gap
        class_rates[str(cls)] = rates

    if not class_gaps:
        raise ValueError(_not_enough_groups(sensitive_col, {}))

    macro_gap = round(sum(class_gaps.values()) / len(class_gaps), 6)
    worst_cls = max(class_gaps, key=class_gaps.get) # type: ignore

    interp = (
        f"Multiclass DPG (macro-average across {len(classes)} classes): "
        f"{macro_gap:.4f}. "
        f"Largest gap in class '{worst_cls}': {class_gaps[worst_cls]:.4f}."
    )
    return FairnessResult(
        metric_name    = "demographic_parity_gap",
        primary_value  = macro_gap,
        group_values   = {cls: class_rates[cls] for cls in class_rates},
        gap            = macro_gap,
        sensitive_col  = sensitive_col,
        outcome_col    = outcome_col,
        prediction_col = None,
        n_classes      = len(classes),
        interpretation = interp,
        details        = {
            "class_gaps":  class_gaps,
            "class_rates": class_rates,
            "macro_gap":   macro_gap,
            "worst_class": worst_cls,
        },
    )


# Equalized Odds 
class EqualizedOdds:
    """Equalized Odds — equal TPR and FPR across groups.

    Binary: gap = max(|ΔTPR|, |ΔFPR|)
    Multiclass: per-class TPR/FPR (one-vs-rest), then macro-average.

    Requires model predictions (prediction_col).

    When base rates differ across groups, EO and Predictive Parity cannot
    both hold simultaneously (Chouldechova 2017).
    """

    @staticmethod
    def compute(df, sensitive_col: str, outcome_col: str,
                prediction_col: str, positive_label: Any = 1) -> FairnessResult:
        _require_cols(df, [sensitive_col, outcome_col, prediction_col])
        classes = _detect_classes(df, outcome_col)

        if len(classes) == 2:
            return _eo_binary(df, sensitive_col, outcome_col,
                              prediction_col, positive_label)
        else:
            return _eo_multiclass(df, sensitive_col, outcome_col,
                                  prediction_col, classes)


def _eo_binary(df, sensitive_col, outcome_col, prediction_col, positive_label):
    groups = _groups(df, sensitive_col)
    tpr_by_g: Dict[str, float] = {}
    fpr_by_g: Dict[str, float] = {}

    for g in groups:
        sub = df[df[sensitive_col] == g].dropna(
            subset=[outcome_col, prediction_col])
        if len(sub) == 0:
            continue
        y  = sub[outcome_col]   == positive_label
        yh = sub[prediction_col] == positive_label
        tp = int((y &  yh).sum()); fn = int((y & ~yh).sum())
        fp = int((~y &  yh).sum()); tn = int((~y & ~yh).sum())
        tpr_by_g[str(g)] = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 6)
        fpr_by_g[str(g)] = round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 6)

    _check_enough_groups(tpr_by_g, sensitive_col)
    tpr_gap = _max_pairwise_gap(tpr_by_g)
    fpr_gap = _max_pairwise_gap(fpr_by_g)
    eo_gap  = round(max(tpr_gap, fpr_gap), 6)

    max_tpr_g = max(tpr_by_g, key=tpr_by_g.get) # type: ignore
    min_tpr_g = min(tpr_by_g, key=tpr_by_g.get) # type: ignore
    interp = (
        f"TPR gap: {tpr_gap:.1%} ('{max_tpr_g}': {tpr_by_g[max_tpr_g]:.1%} "
        f"vs '{min_tpr_g}': {tpr_by_g[min_tpr_g]:.1%}). "
        f"FPR gap: {fpr_gap:.1%}. EO gap: {eo_gap:.1%}."
    )
    return FairnessResult(
        metric_name    = "equalized_odds_gap",
        primary_value  = eo_gap,
        group_values   = tpr_by_g,
        gap            = eo_gap,
        sensitive_col  = sensitive_col,
        outcome_col    = outcome_col,
        prediction_col = prediction_col,
        n_classes      = 2,
        interpretation = interp,
        details        = {
            "tpr_by_group": tpr_by_g, "fpr_by_group": fpr_by_g,
            "tpr_gap": tpr_gap, "fpr_gap": fpr_gap, "eo_gap": eo_gap,
        },
    )


def _eo_multiclass(df, sensitive_col, outcome_col, prediction_col, classes):
    """One-vs-rest per class, then macro-average the EO gaps."""
    class_eo_gaps: Dict[str, float] = {}
    class_tpr:     Dict[str, Dict] = {}

    for cls in classes:
        sub  = df.dropna(subset=[sensitive_col, outcome_col, prediction_col])
        y    = sub[outcome_col]    == cls
        yh   = sub[prediction_col] == cls
        grps = _groups(sub, sensitive_col)

        tpr_g: Dict[str, float] = {}
        fpr_g: Dict[str, float] = {}
        for g in grps:
            m  = sub[sensitive_col] == g
            tp = int((y[m] & yh[m]).sum()); fn = int((y[m] & ~yh[m]).sum())
            fp = int((~y[m] & yh[m]).sum()); tn = int((~y[m] & ~yh[m]).sum())
            tpr_g[str(g)] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr_g[str(g)] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_gap = _max_pairwise_gap(tpr_g)
        fpr_gap = _max_pairwise_gap(fpr_g)
        class_eo_gaps[str(cls)] = round(max(tpr_gap, fpr_gap), 6)
        class_tpr[str(cls)]     = tpr_g

    if not class_eo_gaps:
        raise ValueError(_not_enough_groups(sensitive_col, {}))

    macro_gap = round(sum(class_eo_gaps.values()) / len(class_eo_gaps), 6)
    worst_cls = max(class_eo_gaps, key=class_eo_gaps.get) # type: ignore

    interp = (
        f"Multiclass EO (macro-average across {len(classes)} classes): "
        f"{macro_gap:.4f}. "
        f"Largest gap in class '{worst_cls}': {class_eo_gaps[worst_cls]:.4f}."
    )
    return FairnessResult(
        metric_name    = "equalized_odds_gap",
        primary_value  = macro_gap,
        group_values   = class_tpr,
        gap            = macro_gap,
        sensitive_col  = sensitive_col,
        outcome_col    = outcome_col,
        prediction_col = prediction_col,
        n_classes      = len(classes),
        interpretation = interp,
        details        = {
            "class_eo_gaps": class_eo_gaps,
            "class_tpr":     class_tpr,
            "macro_gap":     macro_gap,
            "worst_class":   worst_cls,
        },
    )


# Predictive Parity 
class PredictiveParity:
    """Predictive Parity — equal precision across groups.

    Binary: gap = |precision_A - precision_B|
    Multiclass: per-class precision (one-vs-rest), then macro-average.

    Requires model predictions (prediction_col).
    """

    @staticmethod
    def compute(df, sensitive_col: str, outcome_col: str,
                prediction_col: str, positive_label: Any = 1) -> FairnessResult:
        _require_cols(df, [sensitive_col, outcome_col, prediction_col])
        classes = _detect_classes(df, outcome_col)

        if len(classes) == 2:
            return _pp_binary(df, sensitive_col, outcome_col,
                              prediction_col, positive_label)
        else:
            return _pp_multiclass(df, sensitive_col, outcome_col,
                                  prediction_col, classes)


def _pp_binary(df, sensitive_col, outcome_col, prediction_col, positive_label):
    groups = _groups(df, sensitive_col)
    ppv_by_g: Dict[str, float] = {}

    for g in groups:
        sub = df[df[sensitive_col] == g].dropna(
            subset=[outcome_col, prediction_col])
        pos_pred = sub[sub[prediction_col] == positive_label]
        if len(pos_pred) == 0:
            ppv_by_g[str(g)] = float("nan")
            continue
        ppv = (pos_pred[outcome_col] == positive_label).mean()
        ppv_by_g[str(g)] = round(float(ppv), 6)

    valid = {k: v for k, v in ppv_by_g.items() if v == v}  # exclude NaN
    if len(valid) < 2:
        raise ValueError(
            "Need at least 2 groups with positive predictions for Predictive Parity. "
            f"Groups with predictions: {list(valid.keys())}"
        )

    pp_gap = round(_max_pairwise_gap(valid), 6)
    max_g  = max(valid, key=valid.get) # type: ignore
    min_g  = min(valid, key=valid.get) # type: ignore
    interp = (
        f"Precision gap: {pp_gap:.1%} "
        f"('{max_g}': {valid[max_g]:.1%} vs '{min_g}': {valid[min_g]:.1%}). "
        f"Being predicted positive is "
        f"{'equally' if pp_gap < 0.05 else 'not equally'} "
        f"meaningful across groups."
    )
    return FairnessResult(
        metric_name    = "predictive_parity_gap",
        primary_value  = pp_gap,
        group_values   = ppv_by_g,
        gap            = pp_gap,
        sensitive_col  = sensitive_col,
        outcome_col    = outcome_col,
        prediction_col = prediction_col,
        n_classes      = 2,
        interpretation = interp,
        details        = {"ppv_by_group": ppv_by_g, "valid_groups": valid,
                          "pp_gap": pp_gap},
    )


def _pp_multiclass(df, sensitive_col, outcome_col, prediction_col, classes):
    """One-vs-rest precision per class, then macro-average."""
    class_pp_gaps: Dict[str, float] = {}
    class_ppv:     Dict[str, Dict]  = {}

    for cls in classes:
        sub  = df.dropna(subset=[sensitive_col, outcome_col, prediction_col])
        grps = _groups(sub, sensitive_col)
        ppv_g: Dict[str, float] = {}
        for g in grps:
            m        = sub[sensitive_col] == g
            pos_pred = sub[m][sub[m][prediction_col] == cls]
            if len(pos_pred) == 0:
                ppv_g[str(g)] = float("nan")
                continue
            ppv = (pos_pred[outcome_col] == cls).mean()
            ppv_g[str(g)] = round(float(ppv), 6)

        valid = {k: v for k, v in ppv_g.items() if v == v}
        if len(valid) >= 2:
            class_pp_gaps[str(cls)] = round(_max_pairwise_gap(valid), 6)
            class_ppv[str(cls)]     = ppv_g

    if not class_pp_gaps:
        raise ValueError(
            "Insufficient positive predictions per group for any class."
        )

    macro_gap = round(sum(class_pp_gaps.values()) / len(class_pp_gaps), 6)
    worst_cls = max(class_pp_gaps, key=class_pp_gaps.get) # type: ignore

    interp = (
        f"Multiclass PP (macro-average across {len(class_pp_gaps)} classes "
        f"with sufficient predictions): {macro_gap:.4f}. "
        f"Largest gap in class '{worst_cls}': {class_pp_gaps[worst_cls]:.4f}."
    )
    return FairnessResult(
        metric_name    = "predictive_parity_gap",
        primary_value  = macro_gap,
        group_values   = class_ppv,
        gap            = macro_gap,
        sensitive_col  = sensitive_col,
        outcome_col    = outcome_col,
        prediction_col = prediction_col,
        n_classes      = len(classes),
        interpretation = interp,
        details        = {
            "class_pp_gaps": class_pp_gaps,
            "class_ppv":     class_ppv,
            "macro_gap":     macro_gap,
            "worst_class":   worst_cls,
        },
    )


# Regression Fairness Auditor
class RegressionFairnessReport:
    """Result of RegressionFairnessAuditor.audit().

    Attributes:
        sensitive_col:   Demographic column.
        outcome_col:     True value column.
        prediction_col:  Predicted value column.
        group_stats:     Per-group {mean_error, mae, residual_std,
                         actual_mean, pred_mean, n, lazy_flag}.
        me_gap:          Max absolute Mean Error difference across groups.
                         Signed errors (over/under-prediction direction matters).
        mae_gap:         Max absolute MAE difference across groups.
        calibration_gap: Max |actual_mean - pred_mean| across groups.
        lazy_flags:      Groups where residual_std ≈ 0 (lazy solution detected).
        interpretation:  Plain-language summary.
        warnings:        List of issues found.
    """

    def __init__(self, sensitive_col, outcome_col, prediction_col,
                 group_stats, me_gap, mae_gap, calibration_gap,
                 lazy_flags, interpretation, warnings_list):
        self.sensitive_col   = sensitive_col
        self.outcome_col     = outcome_col
        self.prediction_col  = prediction_col
        self.group_stats     = group_stats
        self.me_gap          = me_gap
        self.mae_gap         = mae_gap
        self.calibration_gap = calibration_gap
        self.lazy_flags      = lazy_flags
        self.interpretation  = interpretation
        self.warnings        = warnings_list

    def to_store_kwargs(self, run_id: str,
                        step_name: Optional[str] = None) -> Dict:
        """Return kwargs for store.log_metrics(**kwargs).

        Logs three metrics: me_gap, mae_gap, calibration_gap.
        """
        return dict(
            run_id        = run_id,
            metrics       = {
                "regression_me_gap":          round(self.me_gap, 6),
                "regression_mae_gap":         round(self.mae_gap, 6),
                "regression_calibration_gap": round(self.calibration_gap, 6),
            },
            metric_source = "RegressionFairnessAuditor",
            step_name     = step_name,
            tags          = {
                "sensitive_col":  self.sensitive_col,
                "outcome_col":    self.outcome_col,
                "lazy_solution":  str(bool(self.lazy_flags)),
                "lazy_groups":    str(self.lazy_flags),
            },
        )

    def print_report(self) -> None:
        _print_regression_report(self)

    def __repr__(self):
        return (
            f"RegressionFairnessReport("
            f"me_gap={self.me_gap:.4f}, mae_gap={self.mae_gap:.4f}, "
            f"calibration_gap={self.calibration_gap:.4f}, "
            f"lazy={self.lazy_flags})"
        )


class RegressionFairnessAuditor:
    """Regression fairness diagnostics — four-part audit.

    Standard classification fairness metrics (DPG, EO, PP) are inappropriate
    for regression because:

    1. Converting to binary by thresholding loses information and introduces
       a discretization artefact: the threshold itself can be biased.
    2. MAE gap alone is insufficient — equal MAE across groups is compatible
       with one group being systematically over-predicted and another
       under-predicted (errors cancel if you measure in the wrong direction).
    3. A naive fairness constraint (minimize MAE gap) can be satisfied by
       predicting the group mean for all samples — zero within-group
       discrimination but a useless model. This is the "lazy solution."

    This auditor runs four diagnostics:

    Diagnostic 1 — Mean Error (ME) gap
        ME = mean(predicted - actual). Signed. A positive ME means the model
        over-predicts for that group; negative means under-prediction.
        ME gap = max(ME) - min(ME) across groups.
        Interpretation: if ME gap is large, the model is systematically biased
        in opposite directions for different groups.

    Diagnostic 2 — Mean Absolute Error (MAE) gap
        MAE = mean(|predicted - actual|). Unsigned magnitude.
        MAE gap = max(MAE) - min(MAE) across groups.
        Interpretation: one group's predictions are less accurate than
        another's, regardless of direction.

    Diagnostic 3 — Calibration gap
        Calibration = |mean(predicted) - mean(actual)| per group.
        A well-calibrated model should have mean(predicted) ≈ mean(actual)
        for every group. Calibration gap = max calibration error across groups.

    Diagnostic 4 — Lazy solution guard
        Residual std = std(predicted - actual) per group.
        If residual_std ≈ 0 for a group, the model is predicting a near-
        constant value for all members of that group — likely the group mean.
        This satisfies many fairness constraints while learning nothing useful.
        Threshold: residual_std < 5% of the outcome's overall std → LAZY flag.

    Args:
        lazy_threshold: If residual_std/outcome_std >= (1.0 - lazy_threshold), the group is
                        as a lazy solution. Default: 0.05 (5%).
    """

    def __init__(self, lazy_threshold: float = 0.05):
        self.lazy_threshold = lazy_threshold

    def audit(
        self,
        df,
        sensitive_col:  str,
        outcome_col:    str,
        prediction_col: str,
    ) -> RegressionFairnessReport:
        """Run the four-part fairness audit.

        Args:
            df:             pandas DataFrame with true values and predictions.
            sensitive_col:  Demographic group column.
            outcome_col:    True continuous outcome column.
            prediction_col: Predicted continuous value column.

        Returns:
            RegressionFairnessReport.
        """
        _require_cols(df, [sensitive_col, outcome_col, prediction_col])
        _assert_numeric(df, outcome_col)
        _assert_numeric(df, prediction_col)

        sub = df.dropna(subset=[sensitive_col, outcome_col, prediction_col])
        groups = list(sub[sensitive_col].unique())

        overall_std = float(sub[outcome_col].std())

        group_stats: Dict[str, Dict] = {}
        lazy_flags:  List[str] = []
        warns:       List[str] = []

        for g in groups:
            mask   = sub[sensitive_col] == g
            actual = sub.loc[mask, outcome_col].astype(float)
            pred   = sub.loc[mask, prediction_col].astype(float)
            resid  = pred - actual

            n          = int(len(actual))
            me         = float(resid.mean())
            mae        = float(resid.abs().mean())
            res_std    = float(resid.std()) if n > 1 else 0.0
            act_std    = float(actual.std()) if n > 1 else 0.0
            act_mean   = float(actual.mean())
            pred_mean  = float(pred.mean())
            calib_err  = abs(pred_mean - act_mean)

            # Lazy solution: residual_std / outcome_std ≥ threshold
            # A model predicting the group mean has residual_std ≈ outcome_std
            # (ratio → 1.0). A useful model has ratio ≪ 1.0.
            # Default threshold 0.95: flags models where predictions explain
            # less than 5% of the within-group variance.
            ratio     = res_std / act_std if act_std > 0 else 0.0
            is_lazy   = ratio >= (1.0 - self.lazy_threshold) and n > 10

            if is_lazy:
                lazy_flags.append(str(g))
                warns.append(
                    f"Group '{g}': residual_std/outcome_std={ratio:.3f} ≥ "
                    f"{1.0 - self.lazy_threshold:.2f} (lazy solution threshold). "
                    f"The model explains <{self.lazy_threshold:.0%} of within-group "
                    f"variance — it may be predicting the group mean, not "
                    f"individual outcomes."
                )

            group_stats[str(g)] = {
                "n":           n,
                "mean_error":  round(me, 6),        # signed: +over, -under
                "mae":         round(mae, 6),
                "residual_std": round(res_std, 6),
                "actual_mean": round(act_mean, 6),
                "pred_mean":   round(pred_mean, 6),
                "calib_error": round(calib_err, 6),
                "lazy_flag":   is_lazy,
            }

        # Compute gaps
        me_vals   = {g: s["mean_error"]  for g, s in group_stats.items()}
        mae_vals  = {g: s["mae"]         for g, s in group_stats.items()}
        cal_vals  = {g: s["calib_error"] for g, s in group_stats.items()}

        me_gap   = round(_max_pairwise_gap(me_vals),  6)
        mae_gap  = round(_max_pairwise_gap(mae_vals), 6)
        calib_gap = round(max(cal_vals.values()), 6)

        # Interpretation
        worst_me_g  = max(me_vals,  key=lambda g: abs(me_vals[g]))
        worst_mae_g = max(mae_vals, key=mae_vals.get) # type: ignore
        interp_parts = [
            f"ME gap: {me_gap:.4f} (worst group: '{worst_me_g}', "
            f"ME={me_vals[worst_me_g]:+.4f}).",
            f"MAE gap: {mae_gap:.4f} (worst: '{worst_mae_g}').",
            f"Calibration gap: {calib_gap:.4f}.",
        ]
        if lazy_flags:
            interp_parts.append(
                f"⚠ Lazy solution detected in group(s): "
                f"{', '.join(lazy_flags)}."
            )
        interp = " ".join(interp_parts)

        if me_gap > 0.1 * overall_std:
            warns.append(
                f"ME gap ({me_gap:.4f}) exceeds 10% of outcome std "
                f"({overall_std:.4f}). The model has systematic directional "
                f"bias: it over-predicts for some groups and under-predicts "
                f"for others."
            )
        if calib_gap > 0.05 * overall_std:
            warns.append(
                f"Calibration gap ({calib_gap:.4f}) exceeds 5% of outcome "
                f"std. Group-level predicted means differ significantly from "
                f"actual means."
            )

        return RegressionFairnessReport(
            sensitive_col   = sensitive_col,
            outcome_col     = outcome_col,
            prediction_col  = prediction_col,
            group_stats     = group_stats,
            me_gap          = me_gap,
            mae_gap         = mae_gap,
            calibration_gap = calib_gap,
            lazy_flags      = lazy_flags,
            interpretation  = interp,
            warnings_list   = warns,
        )

    # Convenience classmethod so you can call RegressionFairnessAuditor.audit(...)
    # without instantiating. Mirrors the classification API.
    @classmethod
    def audit_static(cls, df, sensitive_col, outcome_col, prediction_col,
                     lazy_threshold=0.05):
        return cls(lazy_threshold=lazy_threshold).audit(
            df, sensitive_col, outcome_col, prediction_col)


# compute_metric dispatcher
_METRIC_MAP = {
    "dpg":                    DemographicParityGap,
    "demographic_parity":     DemographicParityGap,
    "demographic_parity_gap": DemographicParityGap,
    "eo":                     EqualizedOdds,
    "equalized_odds":         EqualizedOdds,
    "equalized_odds_gap":     EqualizedOdds,
    "pp":                     PredictiveParity,
    "predictive_parity":      PredictiveParity,
    "predictive_parity_gap":  PredictiveParity,
}


def compute_metric(name: str, df, sensitive_col: str, outcome_col: str,
                   prediction_col: Optional[str] = None,
                   positive_label: Any = 1) -> FairnessResult:
    """Compute any supported classification fairness metric by name.

    For regression, use ``RegressionFairnessAuditor`` directly.

    Args:
        name:           Short or full metric name: 'dpg', 'eo', 'pp',
                        'demographic_parity_gap', etc.
        df:             pandas DataFrame.
        sensitive_col:  Demographic column.
        outcome_col:    True label column.
        prediction_col: Required for 'eo' and 'pp'.
        positive_label: Positive class for binary problems. Default: 1.
    """
    key = name.lower().strip()
    if key not in _METRIC_MAP:
        raise ValueError(
            f"Unknown metric: '{name}'. "
            f"Supported classification metrics: "
            f"{sorted(set(_METRIC_MAP.keys()))}. "
            f"For regression, use RegressionFairnessAuditor."
        )

    cls = _METRIC_MAP[key]
    if cls in (EqualizedOdds, PredictiveParity):
        if prediction_col is None:
            raise ValueError(
                f"'{name}' requires prediction_col — a column of model "
                f"predictions. Train a model, add predictions to the "
                f"DataFrame, then pass the column name here."
            )
        return cls.compute(df, sensitive_col, outcome_col,
                           prediction_col, positive_label)
    return cls.compute(df, sensitive_col, outcome_col, positive_label)

def _require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found in DataFrame: {missing}. "
            f"Available: {list(df.columns)}"
        )


def _assert_numeric(df, col):
    try:
        df[col].astype(float)
    except (ValueError, TypeError):
        raise ValueError(
            f"Column '{col}' must be numeric for regression fairness. "
            f"Found dtype: {df[col].dtype}"
        )


def _detect_classes(df, outcome_col) -> List:
    """Return sorted unique classes in outcome_col."""
    classes = sorted(df[outcome_col].dropna().unique().tolist())
    return classes


def _groups(df, sensitive_col) -> List:
    return df[sensitive_col].dropna().unique().tolist()


def _selection_rates(df, sensitive_col, outcome_col,
                     positive_label) -> Tuple[List, Dict[str, float]]:
    groups = _groups(df, sensitive_col)
    rates: Dict[str, float] = {}
    for g in groups:
        mask   = df[sensitive_col] == g
        subset = df.loc[mask, outcome_col].dropna()
        if len(subset) == 0:
            continue
        rates[str(g)] = round(float((subset == positive_label).mean()), 6)
    return groups, rates


def _check_enough_groups(group_dict, sensitive_col):
    if len(group_dict) < 2:
        raise ValueError(_not_enough_groups(sensitive_col, group_dict))


def _not_enough_groups(sensitive_col, found):
    return (
        f"Need at least 2 groups in '{sensitive_col}', "
        f"found: {list(found.keys())}"
    )


def _max_pairwise_gap(values: Dict[str, float]) -> float:
    """Maximum absolute difference between any two group values."""
    vals = [v for v in values.values() if v == v]  # exclude NaN
    if len(vals) < 2:
        return 0.0
    return max(abs(a - b) for i, a in enumerate(vals) for b in vals[i+1:])


def _bar(frac: float, width: int = 25) -> str:
    filled = round(max(0.0, min(1.0, frac)) * width)
    return "█" * filled + "░" * (width - filled)


def _print_classification_report(result: FairnessResult) -> None:
    w = 68
    suffix = f" ({result.n_classes}-class)" if result.n_classes > 2 else ""
    print(f"\n{'═' * w}")
    print(f"  Fairness Report: {result.metric_name}{suffix}")
    print(f"{'═' * w}")
    print(f"  Sensitive column: {result.sensitive_col}")
    print(f"  Outcome column:   {result.outcome_col}")
    if result.prediction_col:
        print(f"  Prediction col:   {result.prediction_col}")
    print()

    if result.n_classes == 2:
        print(f"  {'Group':18s}  {'Rate / Value':>14}  Bar")
        print(f"  {'─'*18}  {'─'*14}  {'─'*25}")
        for grp, val in sorted(result.group_values.items(),
                                key=lambda x: -(x[1] if x[1]==x[1] else 0)):
            if val != val:
                print(f"  {grp:18s}  {'(undefined)':>14}")
                continue
            print(f"  {grp:18s}  {val:>14.1%}  {_bar(val)}")
    else:
        # Multiclass: show class-level breakdown
        d = result.details
        if "class_gaps" in d:
            print(f"  Per-class gaps (one-vs-rest):")
            for cls, gap in sorted(d["class_gaps"].items(),
                                   key=lambda x: -x[1]):
                marker = "  ⚠" if gap > 0.10 else ""
                print(f"    Class '{cls}':  gap = {gap:.4f}{marker}")

    print()
    print(f"  Gap:  {result.gap:.4f}  ({result.gap:.1%})")
    print(f"\n  {result.interpretation}")

    d = result.details
    if "tpr_by_group" in d and result.n_classes == 2:
        print(f"\n  {'Group':18s}  {'TPR':>8}  {'FPR':>8}")
        print(f"  {'─'*18}  {'─'*8}  {'─'*8}")
        for g in sorted(d["tpr_by_group"]):
            print(f"  {g:18s}  {d['tpr_by_group'][g]:>8.1%}  "
                  f"{d['fpr_by_group'].get(g, 0):>8.1%}")
        print(f"\n  TPR gap: {d['tpr_gap']:.4f}   FPR gap: {d['fpr_gap']:.4f}")

    if "pp_gap" in d and "ppv_by_group" in d and result.n_classes == 2:
        print(f"\n  Precision (PPV) by group:")
        for g, v in sorted(d["ppv_by_group"].items()):
            if v == v:
                print(f"    {g:18s}  {v:.1%}")
            else:
                print(f"    {g:18s}  (no positive predictions)")

    print(f"\n{'═' * w}\n")


def _print_regression_report(report: RegressionFairnessReport) -> None:
    w = 72
    print(f"\n{'═' * w}")
    print(f"  Regression Fairness Audit Report")
    print(f"{'═' * w}")
    print(f"  Sensitive column: {report.sensitive_col}")
    print(f"  Outcome column:   {report.outcome_col}")
    print(f"  Prediction col:   {report.prediction_col}")
    print()

    print(f"  {'Group':16s}  {'N':>6}  {'ME':>9}  {'MAE':>8}  "
          f"{'Res.Std':>8}  {'Calib.Err':>10}  {'Lazy?':>6}")
    print(f"  {'─'*16}  {'─'*6}  {'─'*9}  {'─'*8}  "
          f"{'─'*8}  {'─'*10}  {'─'*6}")
    for g, s in sorted(report.group_stats.items()):
        lazy = "⚠ YES" if s["lazy_flag"] else "no"
        print(f"  {g:16s}  {s['n']:>6}  {s['mean_error']:>+9.4f}  "
              f"{s['mae']:>8.4f}  {s['residual_std']:>8.4f}  "
              f"{s['calib_error']:>10.4f}  {lazy:>6}")

    print()
    print(f"  ME gap (max pairwise):          {report.me_gap:.4f}")
    print(f"  MAE gap (max pairwise):         {report.mae_gap:.4f}")
    print(f"  Calibration gap (max error):    {report.calibration_gap:.4f}")
    if report.lazy_flags:
        print(f"\n  ⚠ Lazy solution detected in: {', '.join(report.lazy_flags)}")
    print(f"\n  {report.interpretation}")

    if report.warnings:
        print(f"\n  Warnings:")
        for w_msg in report.warnings:
            print(f"    • {w_msg}")

    print(f"\n{'═' * w}\n")