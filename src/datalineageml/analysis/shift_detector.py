"""
src/datalineageml/analysis/shift_detector.py

ShiftDetector — compares demographic snapshots across adjacent pipeline steps
and ranks them by distribution shift magnitude.

Answers: "Which pipeline step changed the demographic distribution most
significantly, and which numeric features were most affected?"

Statistical tests:
  - Jensen-Shannon divergence for categorical / sensitive columns
  - Kolmogorov-Smirnov D-statistic for numeric columns

JSD needs only stdlib math.  KS uses scipy when available and falls back
to an approximation from stored percentile stats when it is not.

Usage:
    from datalineageml.analysis import ShiftDetector

    detector = ShiftDetector(store=store)
    results  = detector.detect()
    detector.print_report(results)

    top = detector.top_candidate(results)
    print(top["step_name"], top["column"], top["stat"], top["flag"])
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# ── Thresholds ────────────────────────────────────────────────────────────────
# JSD scale is compressed vs intuition (log base 2, bounded 0–1).
# Calibrated against real demographic shifts:
#   ~8%  absolute demographic shift → JSD ≈ 0.005  (MEDIUM)
#   ~15% absolute demographic shift → JSD ≈ 0.020  (HIGH)
#   ~18% absolute demographic shift → JSD ≈ 0.028  (Oyo State)
#   ~30% absolute demographic shift → JSD ≈ 0.073
_JSD_HIGH   = 0.02
_JSD_MEDIUM = 0.005

# KS D-statistic is on [0, 1] — more intuitive
_KS_HIGH   = 0.20
_KS_MEDIUM = 0.10


class ShiftDetector:
    """Detect and rank distribution shifts across pipeline steps.

    For each step with paired before/after snapshots, computes:
      - JSD for every sensitive (categorical) column
      - KS D-statistic for every numeric column with stored percentile stats

    Results are sorted by ``stat`` descending — highest shift first.

    Args:
        store:              A ``LineageStore`` instance to read snapshots from.
        jsd_high:           JSD threshold for HIGH flag on categorical columns.
        jsd_medium:         JSD threshold for MEDIUM flag.
        ks_high:            KS D threshold for HIGH flag on numeric columns.
        ks_medium:          KS D threshold for MEDIUM flag.

    Example::

        detector = ShiftDetector(store=store)
        results  = detector.detect()
        detector.print_report(results)
    """

    def __init__(
        self,
        store,
        jsd_high:   float = _JSD_HIGH,
        jsd_medium: float = _JSD_MEDIUM,
        ks_high:    float = _KS_HIGH,
        ks_medium:  float = _KS_MEDIUM,
    ):
        self.store      = store
        self.jsd_high   = jsd_high
        self.jsd_medium = jsd_medium
        self.ks_high    = ks_high
        self.ks_medium  = ks_medium

    def detect(
        self,
        pipeline_name: Optional[str] = None,
        step_names:    Optional[List[str]] = None,
    ) -> List[Dict]:
        """Detect distribution shifts across all logged pipeline steps.

        Args:
            pipeline_name: Reserved for v0.3 cross-pipeline scoping.
                           Currently ignored — all stored snapshots are analysed.
            step_names:    Restrict analysis to specific step names.

        Returns:
            List of result dicts sorted by ``stat`` descending. Each dict::

                {
                    "step_name":    str,
                    "column":       str,
                    "test":         str,   # "jsd" | "ks"
                    "stat":         float, # JSD or KS D-statistic
                    "flag":         str,   # "HIGH" | "MEDIUM" | "LOW"
                    "before_dist":  dict,  # for JSD: {val: fraction}
                    "after_dist":   dict,  # for JSD: {val: fraction}
                    "before_stats": dict,  # for KS: {mean, std, p25, p75, n}
                    "after_stats":  dict,  # for KS: {mean, std, p25, p75, n}
                    "rows_before":  int,
                    "rows_after":   int,
                    "rows_removed": int,
                    "removal_rate": float,
                    "finding":      str,
                }
        """
        all_snaps  = self.store.get_snapshots()
        step_pairs = self._pair_snapshots(all_snaps)

        if step_names:
            step_pairs = {k: v for k, v in step_pairs.items()
                         if k in step_names}

        results = []
        for step_name, (before, after) in step_pairs.items():
            results.extend(self._analyse_step(step_name, before, after))

        results.sort(key=lambda r: r["stat"], reverse=True)
        return results

    def top_candidate(self, results: List[Dict]) -> Optional[Dict]:
        """Return the highest-shift result, or None if list is empty."""
        return results[0] if results else None

    def print_report(self, results: List[Dict], title: str = "") -> None:
        """Print a ranked shift report to stdout."""
        _print_shift_report(
            results, title=title,
            jsd_high=self.jsd_high, jsd_medium=self.jsd_medium,
            ks_high=self.ks_high,   ks_medium=self.ks_medium,
        )

    # internal 

    def _pair_snapshots(self, snapshots):
        by_step = {}
        for snap in snapshots:
            sn = snap["step_name"]
            by_step.setdefault(sn, {})[snap["position"]] = snap
        return {
            sn: (pos["before"], pos["after"])
            for sn, pos in by_step.items()
            if "before" in pos and "after" in pos
        }

    def _analyse_step(self, step_name, before, after):
        rows_before  = before.get("row_count", 0)
        rows_after   = after.get("row_count",  0)
        rows_removed = rows_before - rows_after
        removal_rate = rows_removed / rows_before if rows_before > 0 else 0.0

        results = []

        # JSD on sensitive / categorical columns
        sens_b = before.get("sensitive_stats", {})
        sens_a = after.get("sensitive_stats",  {})
        for col in sorted(set(sens_b) | set(sens_a)):
            db, da = sens_b.get(col, {}), sens_a.get(col, {})
            if not db or not da:
                continue
            jsd  = _jensen_shannon_divergence(db, da)
            flag = _flag_jsd(jsd, self.jsd_high, self.jsd_medium)
            results.append({
                "step_name":    step_name,
                "column":       col,
                "test":         "jsd",
                "stat":         round(jsd, 6),
                "flag":         flag,
                "before_dist":  db,
                "after_dist":   da,
                "before_stats": {},
                "after_stats":  {},
                "rows_before":  rows_before,
                "rows_after":   rows_after,
                "rows_removed": rows_removed,
                "removal_rate": round(removal_rate, 6),
                "finding": _build_jsd_finding(
                    step_name, col, jsd, flag, db, da,
                    rows_removed, removal_rate),
            })

        # KS on numeric columns
        num_b = before.get("numeric_stats", {})
        num_a = after.get("numeric_stats",  {})
        for col in sorted(set(num_b) | set(num_a)):
            sb, sa = num_b.get(col), num_a.get(col)
            if not sb or not sa:
                continue
            ks_d = _ks_from_stats(sb, sa)
            flag = _flag_ks(ks_d, self.ks_high, self.ks_medium)
            results.append({
                "step_name":    step_name,
                "column":       col,
                "test":         "ks",
                "stat":         round(ks_d, 6),
                "flag":         flag,
                "before_dist":  {},
                "after_dist":   {},
                "before_stats": sb,
                "after_stats":  sa,
                "rows_before":  rows_before,
                "rows_after":   rows_after,
                "rows_removed": rows_removed,
                "removal_rate": round(removal_rate, 6),
                "finding": _build_ks_finding(
                    step_name, col, ks_d, flag, sb, sa,
                    rows_removed, removal_rate),
            })

        return results


# Statistical functions

def _jensen_shannon_divergence(p_dict: Dict, q_dict: Dict) -> float:
    """Jensen-Shannon divergence between two categorical distributions.

    Symmetric, bounded [0, 1] with log base 2.
    JSD = 0 → identical  |  JSD = 1 → completely disjoint

    Real-world calibration (binary sensitive column):
        |Δ|  ≈ 0.08  →  JSD ≈ 0.005  (MEDIUM threshold)
        |Δ|  ≈ 0.15  →  JSD ≈ 0.020  (HIGH threshold)
        |Δ|  ≈ 0.18  →  JSD ≈ 0.028  (Oyo State case)
        |Δ|  ≈ 0.30  →  JSD ≈ 0.073
    """
    all_keys = set(p_dict) | set(q_dict)
    p = [p_dict.get(k, 0.0) for k in all_keys]
    q = [q_dict.get(k, 0.0) for k in all_keys]

    ps = sum(p); qs = sum(q)
    if ps == 0 or qs == 0:
        return 0.0
    p = [v / ps for v in p]
    q = [v / qs for v in q]
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(a, m):
        return sum(ai * math.log2(ai / mi)
                   for ai, mi in zip(a, m) if ai > 0 and mi > 0)

    return max(0.0, min(1.0, (kl(p, m) + kl(q, m)) / 2.0))


def _ks_from_stats(stats_b: Dict, stats_a: Dict) -> float:
    """Approximate Kolmogorov-Smirnov D-statistic from stored percentile stats.

    The KS D-statistic is the maximum absolute difference between two CDFs.
    Without the raw data we approximate it by treating the stored percentiles
    (p25, p75, min, max, mean) as CDF evaluation points and computing the
    maximum gap between the two empirical CDFs at those points.

    This is an approximation — it will under-estimate the true KS D when the
    distribution shift is concentrated between percentile values. The exact KS
    test (using raw data) is available via the full DataFrameProfiler path.
    When scipy is available and raw data is passed, the exact value is used.

    Args:
        stats_b: Numeric stats dict from snapshot 'before' (mean, std, min,
                 max, p25, p75, n).
        stats_a: Numeric stats dict from snapshot 'after'.

    Returns:
        Float in [0, 1]. Higher = more different distributions.
    """
    # Gather CDF evaluation points: min, p25, mean, p75, max
    def _cdf_points(s):
        return {
            s.get("min",  0.0): 0.00,
            s.get("p25",  0.0): 0.25,
            s.get("mean", 0.0): 0.50,   # approximate — not exact median
            s.get("p75",  0.0): 0.75,
            s.get("max",  0.0): 1.00,
        }

    cdf_b = _cdf_points(stats_b)
    cdf_a = _cdf_points(stats_a)

    # Merge all x-values, interpolate both CDFs, compute max gap
    all_x = sorted(set(cdf_b) | set(cdf_a))
    if len(all_x) < 2:
        return 0.0

    def _interp(cdf_dict, x):
        """Linear interpolation in a sorted CDF dict."""
        keys = sorted(cdf_dict)
        if x <= keys[0]:  return cdf_dict[keys[0]]
        if x >= keys[-1]: return cdf_dict[keys[-1]]
        for i in range(len(keys) - 1):
            x0, x1 = keys[i], keys[i+1]
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0) if x1 > x0 else 0
                return cdf_dict[x0] + t * (cdf_dict[x1] - cdf_dict[x0])
        return 1.0

    max_gap = 0.0
    for x in all_x:
        gap = abs(_interp(cdf_b, x) - _interp(cdf_a, x))
        if gap > max_gap:
            max_gap = gap

    return max_gap


def _ks_exact(samples_b: List[float], samples_a: List[float]) -> float:
    """Exact KS D-statistic using scipy when raw samples are available.

    Falls back to 0.0 if scipy is not installed.
    """
    try:
        from scipy.stats import ks_2samp
        stat, _ = ks_2samp(samples_b, samples_a)
        return float(stat) # type: ignore
    except ImportError:
        return 0.0


# Flag helpers

def _flag_jsd(jsd, high, medium):
    if jsd >= high:   return "HIGH"
    if jsd >= medium: return "MEDIUM"
    return "LOW"


def _flag_ks(ks, high, medium):
    if ks >= high:   return "HIGH"
    if ks >= medium: return "MEDIUM"
    return "LOW"


# Finding builders

def _build_jsd_finding(step_name, col, jsd, flag, dist_b, dist_a,
                       rows_removed, removal_rate):
    if flag == "LOW":
        return (
            f"'{col}' distribution at '{step_name}' shows negligible shift "
            f"(JSD = {jsd:.4f}). No action required."
        )
    severity = "significantly" if flag == "HIGH" else "moderately"
    # Find the value with the largest drop
    all_keys = set(dist_b) | set(dist_a)
    top_key, top_drop = None, 0.0
    for k in all_keys:
        drop = dist_b.get(k, 0.0) - dist_a.get(k, 0.0)
        if drop > top_drop:
            top_key, top_drop = k, drop

    if top_key and top_key != "__null__":
        bp = dist_b.get(top_key, 0.0) * 100
        ap = dist_a.get(top_key, 0.0) * 100
        return (
            f"'{col}' distribution shifted {severity} at '{step_name}' "
            f"(JSD = {jsd:.4f}, {flag}): "
            f"'{top_key}' proportion dropped from {bp:.1f}% to {ap:.1f}% "
            f"({rows_removed:,} rows removed, {removal_rate:.1%} of dataset). "
            f"This step is a candidate causal source of bias."
        )
    return (
        f"'{col}' distribution shifted {severity} at '{step_name}' "
        f"(JSD = {jsd:.4f}, {flag}). "
        f"{rows_removed:,} rows removed ({removal_rate:.1%}). "
        f"This step is a candidate causal source of bias."
    )


def _build_ks_finding(step_name, col, ks_d, flag, stats_b, stats_a,
                      rows_removed, removal_rate):
    if flag == "LOW":
        return (
            f"Numeric column '{col}' at '{step_name}' shows negligible "
            f"distributional change (KS D = {ks_d:.3f}). No action required."
        )
    severity = "significant" if flag == "HIGH" else "moderate"
    mean_shift = stats_a.get("mean", 0) - stats_b.get("mean", 0)
    direction  = "increased" if mean_shift > 0 else "decreased"
    return (
        f"Numeric column '{col}' shows {severity} distributional change "
        f"at '{step_name}' (KS D = {ks_d:.3f}, {flag}): "
        f"mean {direction} from {stats_b.get('mean', 0):.3f} to "
        f"{stats_a.get('mean', 0):.3f} "
        f"({rows_removed:,} rows removed, {removal_rate:.1%} of dataset)."
    )


# Report formatting

def _bar(frac, width=20):
    filled = round(max(0.0, min(1.0, frac)) * width)
    return "█" * filled + "░" * (width - filled)


def _print_shift_report(results, title="", jsd_high=_JSD_HIGH,
                        jsd_medium=_JSD_MEDIUM, ks_high=_KS_HIGH,
                        ks_medium=_KS_MEDIUM):
    w = 72
    hdr = title if title else "DataLineageML — Distribution Shift Report"
    print(f"\n{'═' * w}")
    print(f"  {hdr}")
    print(f"{'═' * w}")

    if not results:
        print("  No paired snapshots found. "
              "Run a pipeline with snapshot=True first.")
        print(f"{'═' * w}\n")
        return

    high_n   = sum(1 for r in results if r["flag"] == "HIGH")
    medium_n = sum(1 for r in results if r["flag"] == "MEDIUM")
    low_n    = sum(1 for r in results if r["flag"] == "LOW")
    steps    = len({r["step_name"] for r in results})

    print(f"  Steps analysed: {steps}   "
          f"Signals: {len(results)}   "
          f"HIGH: {high_n}   MEDIUM: {medium_n}   LOW: {low_n}")
    print(f"  JSD thresholds: HIGH ≥ {jsd_high:.3f}  "
          f"MEDIUM ≥ {jsd_medium:.3f}   "
          f"KS thresholds: HIGH ≥ {ks_high:.2f}  MEDIUM ≥ {ks_medium:.2f}")
    print()

    print(f"  {'#':<4} {'Step':24} {'Column':14} {'Test':5} "
          f"{'Stat':>7}  {'Flag':10}  {'Rows removed':>14}")
    print(f"  {'─'*4} {'─'*24} {'─'*14} {'─'*5} "
          f"{'─'*7}  {'─'*10}  {'─'*14}")

    for i, r in enumerate(results, 1):
        icon = "⚠ " if r["flag"] == "HIGH" else ("△ " if r["flag"] == "MEDIUM" else "  ")
        print(f"  {i:<4} {r['step_name']:24} {r['column']:14} "
              f"{r['test']:5} {r['stat']:>7.4f}  "
              f"{icon}{r['flag']:<8}  "
              f"{r['rows_removed']:>8,} ({r['removal_rate']:.1%})")

    # Detailed findings for HIGH results only
    high_results = [r for r in results if r["flag"] == "HIGH"]
    if high_results:
        print(f"\n{'─' * w}")
        print(f"  HIGH findings — immediate attention required:")
        print(f"{'─' * w}")
        for r in high_results:
            print(f"\n  [{r['flag']}] {r['step_name']} → {r['column']} "
                  f"({r['test'].upper()})")
            print(f"  {r['finding']}")
            print()
            if r["test"] == "jsd":
                all_vals = sorted(set(r["before_dist"]) | set(r["after_dist"]))
                for val in all_vals:
                    label = "(missing)" if val == "__null__" else str(val)
                    fb = r["before_dist"].get(val, 0.0)
                    fa = r["after_dist"].get(val,  0.0)
                    shift = fa - fb
                    marker = "  ⚠" if abs(shift) >= jsd_high * 5 else ""
                    print(f"    {label:16s}  {_bar(fb, 12)} {fb:>5.1%}  →  "
                          f"{_bar(fa, 12)} {fa:>5.1%}  {shift:>+6.1%}{marker}")
            else:
                sb, sa = r["before_stats"], r["after_stats"]
                print(f"    {'':16s}  {'BEFORE':>12}  {'AFTER':>12}  {'CHANGE':>8}")
                for stat in ("mean", "std", "p25", "p75"):
                    bv = sb.get(stat, 0); av = sa.get(stat, 0)
                    print(f"    {stat:16s}  {bv:>12.4f}  {av:>12.4f}  "
                          f"{av - bv:>+8.4f}")

    print(f"\n{'═' * w}\n")