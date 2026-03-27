"""
src/datalineageml/analysis/cross_run.py

Cross-run comparison — detect demographic drift across pipeline runs over time.

Answers: "Did the gender balance at clean_data get worse between
last Tuesday's run and today's run?"

This is the drift detection use case. The store already contains
every run ever logged. This module provides the API to query it.

Usage:
    from datalineageml.analysis.cross_run import CrossRunComparator

    comparator = CrossRunComparator(store=store)

    # Compare all runs of 'clean_data' step over time
    report = comparator.compare_step(
        step_name="clean_data",
        sensitive_col="gender",
    )
    comparator.print_report(report)

    # Check if bias is trending worse over recent runs
    trend = comparator.trend(
        step_name="clean_data",
        sensitive_col="gender",
        group="F",
    )
    print(trend["direction"])  # "worsening" | "stable" | "improving"
"""

from __future__ import annotations

from typing import Dict, List, Optional


class CrossRunComparator:
    """Compare demographic distributions across multiple pipeline runs.

    Reads snapshot history from the store and computes:
    - Per-run demographic distributions at a given step
    - Run-to-run delta (how much did F% change between run N and run N+1)
    - Trend direction over the most recent N runs
    - Worst run vs best run comparison

    Args:
        store: ``LineageStore`` instance.

    Example::

        comparator = CrossRunComparator(store=store)
        report = comparator.compare_step("clean_data", "gender")
        comparator.print_report(report)
    """

    def __init__(self, store):
        self.store = store

    def compare_step(
        self,
        step_name:     str,
        sensitive_col: str,
        position:      str = "after",
        last_n_runs:   Optional[int] = None,
    ) -> Dict:
        """Compare the sensitive column distribution across all runs of a step.

        Args:
            step_name:     Pipeline step to analyse.
            sensitive_col: Demographic column to track.
            position:      Snapshot position to read. Default: "after"
                           (the output of the step).
            last_n_runs:   If set, only compare the most recent N runs.

        Returns:
            CrossRunReport dict::

                {
                    "step_name":     str,
                    "sensitive_col": str,
                    "position":      str,
                    "n_runs":        int,
                    "runs": [
                        {
                            "run_index":    int,      # 0 = oldest
                            "run_id":       str,
                            "recorded_at":  str,
                            "row_count":    int,
                            "distribution": dict,     # {value: fraction}
                        },
                        ...
                    ],
                    "deltas": [                       # run[i+1] - run[i]
                        {"from_run": int, "to_run": int,
                         "delta_by_group": {value: delta_fraction}},
                    ],
                    "worst_run":  int,                # index of most biased run
                    "best_run":   int,                # index of least biased run
                    "max_drift":  float,              # largest single-run delta
                }
        """
        snaps = self.store.get_snapshots(step_name=step_name, position=position)

        # Filter to snaps that have sensitive_stats for the column
        snaps = [s for s in snaps
                 if sensitive_col in s.get("sensitive_stats", {})]

        if not snaps:
            return _empty_report(step_name, sensitive_col, position)

        # Sort chronologically by recorded_at
        snaps = sorted(snaps, key=lambda s: s.get("recorded_at", ""))

        if last_n_runs:
            snaps = snaps[-last_n_runs:]

        runs = []
        for i, snap in enumerate(snaps):
            dist = snap["sensitive_stats"][sensitive_col]
            runs.append({
                "run_index":    i,
                "run_id":       snap.get("run_id", ""),
                "recorded_at":  snap.get("recorded_at", ""),
                "row_count":    snap.get("row_count", 0),
                "distribution": dist,
            })

        # Compute run-to-run deltas
        deltas = []
        for i in range(len(runs) - 1):
            d_from = runs[i]["distribution"]
            d_to   = runs[i + 1]["distribution"]
            all_vals = set(d_from) | set(d_to)
            delta_by_group = {
                v: round(d_to.get(v, 0.0) - d_from.get(v, 0.0), 6)
                for v in all_vals
            }
            deltas.append({
                "from_run":      i,
                "to_run":        i + 1,
                "delta_by_group": delta_by_group,
                "max_abs_delta": round(max(abs(d) for d in delta_by_group.values()), 6),
            })

        # Identify worst and best runs (by minority group proportion — lowest = worst)
        minority_fracs = []
        for run in runs:
            d = run["distribution"]
            # Minority = lowest-fraction group (excluding __null__)
            valid = {k: v for k, v in d.items() if k != "__null__"}
            if valid:
                minority_fracs.append(min(valid.values()))
            else:
                minority_fracs.append(0.0)

        worst_run = int(minority_fracs.index(min(minority_fracs)))
        best_run  = int(minority_fracs.index(max(minority_fracs)))
        max_drift = max((d["max_abs_delta"] for d in deltas), default=0.0)

        return {
            "step_name":     step_name,
            "sensitive_col": sensitive_col,
            "position":      position,
            "n_runs":        len(runs),
            "runs":          runs,
            "deltas":        deltas,
            "worst_run":     worst_run,
            "best_run":      best_run,
            "max_drift":     round(max_drift, 6),
        }

    def trend(
        self,
        step_name:     str,
        sensitive_col: str,
        group:         str,
        last_n_runs:   int = 5,
        position:      str = "after",
    ) -> Dict:
        """Assess whether a group's representation is trending up, down, or stable.

        Uses simple linear regression on the group's fraction across recent runs.

        Args:
            step_name:     Pipeline step to analyse.
            sensitive_col: Demographic column.
            group:         Specific group value to track (e.g. "F").
            last_n_runs:   Number of recent runs to include. Default: 5.
            position:      Snapshot position. Default: "after".

        Returns:
            Dict with keys:
                - ``direction``: "worsening" | "stable" | "improving"
                - ``slope``:     Rate of change per run (positive = improving)
                - ``fractions``: List of fractions per run
                - ``n_runs``:    Number of runs analysed
        """
        report = self.compare_step(step_name, sensitive_col, position, last_n_runs)
        if report["n_runs"] < 2:
            return {"direction": "insufficient_data", "slope": 0.0,
                    "fractions": [], "n_runs": report["n_runs"]}

        fractions = [
            run["distribution"].get(group, 0.0)
            for run in report["runs"]
        ]
        slope = _linear_slope(fractions)
        if abs(slope) < 0.005:
            direction = "stable"
        elif slope > 0:
            direction = "improving"
        else:
            direction = "worsening"

        return {
            "direction": direction,
            "slope":     round(slope, 6),
            "fractions": fractions,
            "n_runs":    len(fractions),
        }

    def print_report(self, report: Dict) -> None:
        """Print a formatted cross-run comparison report."""
        _print_cross_run_report(report)

def _empty_report(step_name, sensitive_col, position) -> Dict:
    return {
        "step_name":     step_name,
        "sensitive_col": sensitive_col,
        "position":      position,
        "n_runs":        0,
        "runs":          [],
        "deltas":        [],
        "worst_run":     None,
        "best_run":      None,
        "max_drift":     0.0,
    }


def _linear_slope(values: List[float]) -> float:
    """Ordinary least squares slope of a sequence of values."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


def _bar(frac: float, width: int = 20) -> str:
    filled = round(max(0.0, min(1.0, frac)) * width)
    return "█" * filled + "░" * (width - filled)


def _print_cross_run_report(report: Dict) -> None:
    w = 72
    print(f"\n{'═' * w}")
    print(f"  Cross-Run Comparison: '{report['step_name']}' "
          f"→ '{report['sensitive_col']}' ({report['position']})")
    print(f"{'═' * w}")

    if report["n_runs"] == 0:
        print(f"  No runs found for step '{report['step_name']}' with "
              f"sensitive column '{report['sensitive_col']}'.")
        print(f"  Ensure snapshot=True and sensitive_cols=['{report['sensitive_col']}']"
              f" are set on @track.")
        print(f"{'═' * w}\n")
        return

    print(f"  Runs found: {report['n_runs']}   "
          f"Max drift between runs: {report['max_drift']:.4f}")
    print()

    # All values seen across all runs
    all_vals = set()
    for run in report["runs"]:
        all_vals.update(run["distribution"].keys())
    all_vals = sorted(v for v in all_vals if v != "__null__")

    # Header
    val_headers = "  ".join(f"{v[:6]:>8}" for v in all_vals)
    print(f"  {'Run':>4}  {'Recorded':>19}  {'Rows':>6}  {val_headers}")
    print(f"  {'─'*4}  {'─'*19}  {'─'*6}  " +
          "  ".join("─"*8 for _ in all_vals))

    for run in report["runs"]:
        d      = run["distribution"]
        ts     = run["recorded_at"][:19] if run["recorded_at"] else "unknown"
        rows   = run["row_count"]
        marker = ""
        if run["run_index"] == report["worst_run"]:
            marker = " ⚠ worst"
        elif run["run_index"] == report["best_run"]:
            marker = " ✓ best"
        val_str = "  ".join(f"{d.get(v, 0.0):>7.1%}" for v in all_vals)
        print(f"  {run['run_index']:>4}  {ts}  {rows:>6,}  {val_str}{marker}")

    # Deltas
    if report["deltas"]:
        print(f"\n  Run-to-run deltas:")
        for delta in report["deltas"]:
            d_str = "  ".join(
                f"{v[:6]:>6}: {delta['delta_by_group'].get(v, 0.0):>+6.1%}"
                for v in all_vals
            )
            flag = "  ⚠" if delta["max_abs_delta"] > 0.05 else ""
            print(f"    Run {delta['from_run']} → {delta['to_run']}:  "
                  f"{d_str}{flag}")

    print(f"\n{'═' * w}\n")