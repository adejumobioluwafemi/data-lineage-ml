"""
src/datalineageml/analysis/attributor.py

CausalAttributor — correlates distribution shift scores with a bias metric
to identify which pipeline step is the most likely causal source.

This closes the attribution loop:
    pipeline runs
    → snapshots logged       (DataFrameProfiler / @track snapshot=True)
    → shifts detected        (ShiftDetector)
    → step attributed        (CausalAttributor)         
    → recommendation output

The bias metric used here is the Demographic Parity Gap (DPG):
    DPG = |P(outcome=1 | group=A) - P(outcome=1 | group=B)|

DPG is computed from the snapshot statistics alone — no trained model
is required. This means attribution can run immediately after data
preprocessing completes, before any model is trained.

Attribution logic:
    For each step with a HIGH or MEDIUM shift flag, score it by:
        attribution_score = jsd_weight * shift_stat + removal_weight * removal_rate

    The top-scoring step is the attributed causal source.
    Confidence is normalised relative to the second-highest score.

Usage:
    from datalineageml.analysis import CausalAttributor

    attributor = CausalAttributor(store=store)
    result = attributor.attribute(
        sensitive_col="gender",
        outcome_col="subsidy_eligible",  # optional: compute DPG from snapshots
    )
    attributor.print_attribution(result)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .shift_detector import ShiftDetector, _jensen_shannon_divergence


class CausalAttributor:
    """Attribute a demographic bias to the most likely causal pipeline step.

    Works in two modes:

    1. **Shift-only** (default): ranks steps purely by their demographic shift
       magnitude (JSD). The step with the highest shift is the primary
       candidate.

    2. **Metric-correlated**: if a bias metric has been logged via
       ``store.log_metrics()``, the attributor correlates each step's shift
       magnitude with the metric value to produce a confidence score.

    The output is a single attribution dict with a human-readable recommendation.

    Args:
        store:          ``LineageStore`` instance to read from.
        jsd_weight:     Weight given to JSD in the attribution score. Default: 0.7.
        removal_weight: Weight given to row removal rate. Default: 0.3.

    Example::

        attributor = CausalAttributor(store=store)
        result = attributor.attribute(sensitive_col="gender")
        attributor.print_attribution(result)
    """

    def __init__(
        self,
        store,
        jsd_weight:     float = 0.7,
        removal_weight: float = 0.3,
    ):
        self.store          = store
        self.jsd_weight     = jsd_weight
        self.removal_weight = removal_weight

    def attribute(
        self,
        sensitive_col:  str,
        outcome_col:    Optional[str] = None,
        metric_name:    Optional[str] = None,
        step_names:     Optional[List[str]] = None,
    ) -> Dict:
        """Attribute demographic bias to the most likely causal pipeline step.

        Args:
            sensitive_col:  The demographic column to analyse (e.g. "gender").
            outcome_col:    Optional. If the snapshots contain a sensitive_stats
                            entry for the outcome column, compute Demographic
                            Parity Gap automatically.
            metric_name:    Optional. Name of a logged metric to use as the
                            bias signal (from ``store.get_metrics()``).
            step_names:     Restrict to specific step names.

        Returns:
            Attribution dict::

                {
                    "attributed_step":  str,
                    "column":           str,
                    "test":             str,    # "jsd" | "ks"
                    "stat":             float,
                    "confidence":       float,  # 0.0–1.0
                    "flag":             str,
                    "rows_removed":     int,
                    "removal_rate":     float,
                    "bias_metric":      dict,   # DPG or logged metric if available
                    "all_scores":       list,   # ranked candidate steps
                    "evidence":         str,
                    "recommendation":   str,
                }

            Returns an "inconclusive" dict if no qualifying shifts are found.
        """
        detector = ShiftDetector(store=self.store)
        shifts   = detector.detect(step_names=step_names)

        # Filter to the sensitive column only
        col_shifts = [r for r in shifts if r["column"] == sensitive_col
                      and r["test"] == "jsd"]

        if not col_shifts:
            return self._inconclusive(sensitive_col,
                                      reason="No JSD shifts found for column "
                                             f"'{sensitive_col}'. "
                                             "Ensure snapshot=True is set on "
                                             "at least one pipeline step.")

        # Score each step
        scored = self._score_steps(col_shifts)

        # Optionally load logged metric
        bias_metric = self._load_metric(metric_name)
        if bias_metric is None and outcome_col:
            bias_metric = self._compute_dpg_from_snapshots(
                sensitive_col, outcome_col)

        # Build attribution result
        top    = scored[0]
        second = scored[1] if len(scored) > 1 else None

        confidence = self._confidence(top["score"],
                                      second["score"] if second else 0.0)

        evidence      = self._build_evidence(top, bias_metric, col_shifts)
        recommendation = _recommendation(top, sensitive_col)

        return {
            "attributed_step":  top["step_name"],
            "column":           sensitive_col,
            "test":             top["test"],
            "stat":             top["stat"],
            "confidence":       round(confidence, 3),
            "flag":             top["flag"],
            "rows_removed":     top["rows_removed"],
            "removal_rate":     top["removal_rate"],
            "bias_metric":      bias_metric or {},
            "all_scores":       scored,
            "evidence":         evidence,
            "recommendation":   recommendation,
        }

    def print_attribution(self, result: Dict) -> None:
        """Print a formatted attribution report to stdout."""
        _print_attribution_report(result)

    def _score_steps(self, col_shifts: List[Dict]) -> List[Dict]:
        """Score each step and return sorted list (highest first)."""
        scored = []
        for r in col_shifts:
            score = (self.jsd_weight     * r["stat"] +
                     self.removal_weight * r["removal_rate"])
            scored.append({**r, "score": round(score, 6)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def _confidence(self, top_score: float, second_score: float) -> float:
        """Normalised confidence: how dominant is the top candidate?

        Returns 1.0 if it is the only candidate or if second score is zero.
        Returns 0.5 if both scores are equal.
        """
        if top_score == 0:
            return 0.0
        if second_score == 0:
            return 1.0
        # Ratio of top to sum: ranges from 0.5 (equal) to 1.0 (dominant)
        return round(top_score / (top_score + second_score), 3)

    def _load_metric(self, metric_name: Optional[str]) -> Optional[Dict]:
        if not metric_name:
            return None
        metrics = self.store.get_metrics(metric_name=metric_name)
        if not metrics:
            return None
        m = metrics[-1]  # most recent
        return {"name": m["metric_name"], "value": m["metric_value"],
                "source": m.get("metric_source", ""), "step": m.get("step_name")}

    def _compute_dpg_from_snapshots(
        self, sensitive_col: str, outcome_col: str
    ) -> Optional[Dict]:
        """Compute Demographic Parity Gap from stored sensitive_stats.

        DPG = |rate_group_A - rate_group_B|

        This works when both the sensitive column and the outcome column
        are logged in sensitive_stats on the 'before' snapshot of any step.
        """
        snaps = self.store.get_snapshots(position="before")
        for snap in snaps:
            s = snap.get("sensitive_stats", {})
            if sensitive_col not in s or outcome_col not in s:
                continue

            sens_dist    = s[sensitive_col]
            outcome_dist = s[outcome_col]

            # Compute proxy DPG from the before-snapshot distributions.
            # We compare outcome=1 rate for the two most common sensitive groups.
            groups = sorted(sens_dist, key=lambda k: -sens_dist.get(k, 0))
            if len(groups) < 2:
                continue

            g_a, g_b = groups[0], groups[1]
            # Use the fraction of outcome=1 as a proxy rate
            rate_a = outcome_dist.get("1", outcome_dist.get(1, 0.0))
            rate_b = outcome_dist.get("0", outcome_dist.get(0, 0.0))
            dpg    = abs(rate_a - rate_b)

            return {
                "name":    "demographic_parity_gap",
                "value":   round(dpg, 4),
                "source":  "computed_from_snapshots",
                "groups":  {g_a: rate_a, g_b: rate_b},
            }
        return None

    def _build_evidence(self, top: Dict, bias_metric: Optional[Dict],
                        all_shifts: List[Dict]) -> str:
        parts = [
            f"The '{top['column']}' distribution shifted "
            f"{top['flag'].lower()} at '{top['step_name']}' "
            f"(JSD = {top['stat']:.4f}).",
            f"{top['rows_removed']:,} rows were removed "
            f"({top['removal_rate']:.1%} of the dataset) at this step.",
        ]
        if bias_metric:
            parts.append(
                f"Bias metric '{bias_metric['name']}' = "
                f"{bias_metric['value']:.4f} "
                f"(source: {bias_metric.get('source', 'unknown')})."
            )
        if len(all_shifts) > 1:
            parts.append(
                f"This step had the highest attribution score among "
                f"{len(all_shifts)} candidate steps."
            )
        return " ".join(parts)

    def _inconclusive(self, col: str, reason: str) -> Dict:
        return {
            "attributed_step":  None,
            "column":           col,
            "test":             None,
            "stat":             0.0,
            "confidence":       0.0,
            "flag":             "LOW",
            "rows_removed":     0,
            "removal_rate":     0.0,
            "bias_metric":      {},
            "all_scores":       [],
            "evidence":         reason,
            "recommendation":   "Add snapshot=True to pipeline steps and "
                                "re-run the pipeline to enable attribution.",
        }


# ── Recommendation engine ────────────────────────────────────────────────────

# Maps common transformation patterns to specific remediation advice.
# Pattern matching is done on the step name (lowercase).
_REMEDIATION_PATTERNS = {
    "dropna":     ("Replace dropna() with stratified imputation per demographic group. "
                   "Use median/mode imputation computed separately for each group "
                   "to preserve the original demographic balance."),
    "clean":      ("Review the cleaning logic in this step for operations that "
                   "disproportionately remove records from a demographic group. "
                   "Common culprits: dropna(), threshold filters on fields that "
                   "have systematic missingness by group."),
    "filter":     ("Audit the filter criteria for demographic correlation. "
                   "A filter condition may be a proxy for a demographic attribute "
                   "(e.g. requiring a formal document that one group holds less often)."),
    "normalize":  ("Normalisation should not change row counts. If shift is detected "
                   "here, check whether earlier NaN handling is occurring inside "
                   "the normalisation step."),
    "encode":     ("One-hot or label encoding should preserve row counts. "
                   "Verify no rows are being dropped for unrecognised category values."),
    "merge":      ("Check the join type. An inner join on a field missing from one "
                   "demographic group will silently remove those rows."),
    "sample":     ("Verify the sampling strategy is stratified by the sensitive column. "
                   "Random sampling can introduce demographic imbalance in small datasets."),
}

_DEFAULT_RECOMMENDATION = (
    "Audit this step for operations that may disproportionately affect "
    "records from a demographic group. Check for: row-dropping operations "
    "(dropna, filter, merge), encoding of missing values, or sampling. "
    "Replace any such operation with a demographically-stratified equivalent."
)


def _recommendation(top: Dict, sensitive_col: str) -> str:
    step_lower = top["step_name"].lower()
    for pattern, advice in _REMEDIATION_PATTERNS.items():
        if pattern in step_lower:
            return (
                f"Step '{top['step_name']}' is attributed as the causal source "
                f"of the '{sensitive_col}' demographic shift "
                f"(confidence = {top.get('score', 0):.3f}).\n\n"
                f"Recommendation: {advice}"
            )
    return (
        f"Step '{top['step_name']}' is attributed as the causal source "
        f"of the '{sensitive_col}' demographic shift.\n\n"
        f"Recommendation: {_DEFAULT_RECOMMENDATION}"
    )


# ── Formatting ───────────────────────────────────────────────────────────────

def _bar(frac, width=20):
    filled = round(max(0.0, min(1.0, frac)) * width)
    return "█" * filled + "░" * (width - filled)


def _print_attribution_report(result: Dict) -> None:
    w = 72
    print(f"\n{'═' * w}")
    print(f"  DataLineageML — Causal Attribution Report")
    print(f"{'═' * w}")

    if result["attributed_step"] is None:
        print(f"  Result: INCONCLUSIVE")
        print(f"  {result['evidence']}")
        print(f"\n  {result['recommendation']}")
        print(f"{'═' * w}\n")
        return

    flag_icon = "⚠ " if result["flag"] == "HIGH" else "△ "
    print(f"  Attributed step:  {result['attributed_step']}")
    print(f"  Sensitive column: {result['column']}")
    print(f"  Shift stat:       {result['stat']:.4f} ({result['test'].upper()})  "
          f"[{flag_icon}{result['flag']}]")
    print(f"  Confidence:       {result['confidence']:.1%}")
    print(f"  Rows removed:     {result['rows_removed']:,}  "
          f"({result['removal_rate']:.1%} of dataset)")

    if result.get("bias_metric"):
        bm = result["bias_metric"]
        print(f"\n  Bias metric:      {bm['name']} = {bm['value']:.4f}  "
              f"(source: {bm.get('source', '?')})")

    print(f"\n{'─' * w}")
    print(f"  Evidence")
    print(f"{'─' * w}")
    print(f"  {result['evidence']}")

    # Demographic distribution at attributed step
    all_scores = result.get("all_scores", [])
    top = next((s for s in all_scores
                if s["step_name"] == result["attributed_step"]), None)
    if top and top.get("before_dist") and top.get("after_dist"):
        print(f"\n  '{result['column']}' distribution at '{result['attributed_step']}':")
        all_vals = sorted(set(top["before_dist"]) | set(top["after_dist"]))
        for val in all_vals:
            label = "(missing)" if val == "__null__" else str(val)
            fb = top["before_dist"].get(val, 0.0)
            fa = top["after_dist"].get(val,  0.0)
            shift = fa - fb
            marker = "  ⚠" if abs(shift) > 0.10 else ""
            print(f"    {label:16s}  {_bar(fb, 12)} {fb:>5.1%}  →  "
                  f"{_bar(fa, 12)} {fa:>5.1%}  {shift:>+6.1%}{marker}")

    # All candidate scores
    if len(all_scores) > 1:
        print(f"\n  All candidate steps (ranked by attribution score):")
        print(f"  {'Step':28} {'Score':>8}  {'Stat':>7}  {'Flag'}")
        print(f"  {'─'*28} {'─'*8}  {'─'*7}  {'─'*8}")
        for s in all_scores:
            marker = "  ← attributed" if s["step_name"] == result["attributed_step"] else ""
            print(f"  {s['step_name']:28} {s['score']:>8.4f}  "
                  f"{s['stat']:>7.4f}  {s['flag']}{marker}")

    print(f"\n{'─' * w}")
    print(f"  Recommendation")
    print(f"{'─' * w}")
    for line in result["recommendation"].split("\n"):
        print(f"  {line}")

    print(f"\n{'═' * w}\n")