"""
datalineageml.analysis

Causal attribution modules (v0.2).

  DataFrameProfiler         — statistical + demographic snapshot computation
  ShiftDetector             — rank pipeline steps by distribution shift (JSD + KS)
  CausalAttributor          — attribute a bias metric to the causal step
  print_snapshot_comparison — human-readable before/after snapshot comparison
"""

from .profiler    import DataFrameProfiler, print_snapshot_comparison
from .shift_detector import ShiftDetector
from .attributor  import CausalAttributor

__all__ = [
    "DataFrameProfiler",
    "print_snapshot_comparison",
    "ShiftDetector",
    "CausalAttributor",
]