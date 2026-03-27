"""
datalineageml.analysis

Causal attribution modules (v0.2).

  DataFrameProfiler         — compute statistical + demographic snapshots
  ShiftDetector             — rank pipeline steps by distribution shift (JSD + KS)
  CausalAttributor          — attribute a bias metric to the causal step
  CrossRunComparator        — compare demographic distributions across runs over time
  FairnessResult            — container returned by all classification fairness metrics
  DemographicParityGap      — equal selection rates (no model required)
  EqualizedOdds             — equal TPR + FPR (requires model predictions)
  PredictiveParity          — equal precision (requires model predictions)
  RegressionFairnessAuditor — four-part regression fairness audit
  compute_metric            — dispatch any metric by name
  discover_sensitive_cols   — auto-discover candidate sensitive columns
  suggest_sensitive_cols    — return column names above confidence threshold
  print_snapshot_comparison — human-readable before/after snapshot comparison
"""

from .profiler       import DataFrameProfiler, print_snapshot_comparison
from .shift_detector import ShiftDetector
from .attributor     import CausalAttributor
from .cross_run      import CrossRunComparator
from .metrics        import (
    FairnessResult,
    DemographicParityGap,
    EqualizedOdds,
    PredictiveParity,
    RegressionFairnessAuditor,
    compute_metric,
)
from .sensitive_cols import (
    discover_sensitive_cols,
    suggest_sensitive_cols,
    print_sensitive_col_report,
    SensitiveColCandidate,
)

__all__ = [
    "DataFrameProfiler",
    "print_snapshot_comparison",
    "ShiftDetector",
    "CausalAttributor",
    "CrossRunComparator",
    "FairnessResult",
    "DemographicParityGap",
    "EqualizedOdds",
    "PredictiveParity",
    "RegressionFairnessAuditor",
    "compute_metric",
    "discover_sensitive_cols",
    "suggest_sensitive_cols",
    "print_sensitive_col_report",
    "SensitiveColCandidate",
]