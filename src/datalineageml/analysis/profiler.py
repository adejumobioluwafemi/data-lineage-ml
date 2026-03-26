"""
src/datalineageml/analysis/profiler.py

DataFrameProfiler — computes a statistical snapshot of a pandas DataFrame
at a pipeline step and persists it to the lineage store.

This is the foundation of causal attribution: by comparing profiles across
adjacent steps, the ShiftDetector (Week 3) can identify which transformation
changed a demographic distribution — and by how much.

Usage (automatic via @track):
    @track(name="clean", snapshot=True, sensitive_cols=["gender"])
    def clean(df):
        return df.dropna()

Usage (manual):
    from datalineageml.analysis.profiler import DataFrameProfiler
    from datalineageml import LineageStore

    store = LineageStore(db_path="pipeline.db")
    profiler = DataFrameProfiler(sensitive_cols=["gender", "zone"])

    profile = profiler.profile(df, step_name="clean_data", run_id="uuid")
    store.log_snapshot(**profile)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any


class DataFrameProfiler:
    """Compute and store statistical profiles of DataFrames at pipeline steps.

    The profiler captures five categories of statistics:

    1. Shape         — row count, column count, column names
    2. Null rates    — fraction of null values per column (0.0–1.0)
    3. Numeric stats — mean, std, min, max, p25, p75 per numeric column
    4. Categorical   — value counts (top N) per non-numeric column
    5. Sensitive     — normalised frequency distribution per sensitive column
                       (the key input for bias/shift detection)

    Args:
        sensitive_cols:  Column names to track demographic distributions for.
                         These appear in ``sensitive_stats`` and are the primary
                         signal for the ShiftDetector.
        max_categories:  Maximum number of category values to store per column.
                         Default: 10.
        sample_size:     Row limit for profiling large DataFrames. When the
                         DataFrame exceeds this, a stratified random sample is
                         used for numeric stats (null rates and sensitive stats
                         always use the full DataFrame). None = no limit.
                         Default: 50_000.

    Example::

        profiler = DataFrameProfiler(sensitive_cols=["gender", "zone"])
        profile  = profiler.profile(df, step_name="clean_data", run_id="abc")
        store.log_snapshot(**profile)
    """

    def __init__(
        self,
        sensitive_cols: Optional[List[str]] = None,
        max_categories: int = 10,
        sample_size: Optional[int] = 50_000,
    ):
        self.sensitive_cols  = list(sensitive_cols or [])
        self.max_categories  = max_categories
        self.sample_size     = sample_size

    # public API 

    def profile(
        self,
        df: Any,
        step_name: str,
        position: str,
        run_id: Optional[str] = None,
    ) -> Dict:
        """Profile a DataFrame and return a dict ready for ``store.log_snapshot()``.

        Args:
            df:         The pandas DataFrame to profile.
            step_name:  Name of the pipeline step this snapshot belongs to.
            position:   ``'before'`` (input to the step) or ``'after'`` (output).
            run_id:     UUID linking this snapshot to a ``@track`` call.
                        Auto-generated if not provided.

        Returns:
            Dict with all fields required by ``LineageStore.log_snapshot()``.

        Raises:
            ImportError: If pandas is not installed.
            ValueError:  If ``position`` is not ``'before'`` or ``'after'``.
        """
        import pandas as pd  # deferred — keep pandas optional

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"DataFrameProfiler.profile() expects a pandas DataFrame, "
                f"got {type(df).__name__}"
            )
        if position not in ("before", "after"):
            raise ValueError(
                f"position must be 'before' or 'after', got: {position!r}"
            )

        run_id      = run_id or str(uuid.uuid4())
        recorded_at = datetime.utcnow().isoformat()

        # Work on a sample for large DataFrames (numeric stats only)
        df_sample = self._sample(df)

        null_rates        = self._null_rates(df)
        numeric_stats     = self._numeric_stats(df_sample)
        categorical_stats = self._categorical_stats(df)
        sensitive_stats   = self._sensitive_stats(df)

        return dict(
            run_id            = run_id,
            step_name         = step_name,
            position          = position,
            row_count         = len(df),
            column_count      = len(df.columns),
            column_names      = list(df.columns),
            null_rates        = null_rates,
            numeric_stats     = numeric_stats,
            categorical_stats = categorical_stats,
            sensitive_stats   = sensitive_stats,
            recorded_at       = recorded_at,
        )

    def print_profile(self, df: Any, step_name: str = "step") -> None:
        """Print a formatted human-readable profile to stdout.

        Useful for interactive debugging in notebooks or scripts.

        Example::

            profiler = DataFrameProfiler(sensitive_cols=["gender"])
            profiler.print_profile(df_before_cleaning, "raw_data")
        """
        p = self.profile(df, step_name=step_name, position="before")
        _print_profile(p, title=f"Profile: {step_name}")

    # private helpers 

    def _sample(self, df):
        """Return a random sample if the DataFrame exceeds sample_size."""
        if self.sample_size is None or len(df) <= self.sample_size:
            return df
        return df.sample(n=self.sample_size, random_state=42)

    def _null_rates(self, df) -> Dict[str, float]:
        """Fraction of null values per column (uses full DataFrame)."""
        return {
            col: round(float(df[col].isna().mean()), 6)
            for col in df.columns
        }

    def _numeric_stats(self, df) -> Dict[str, Dict[str, float]]:
        """mean / std / min / max / p25 / p75 per numeric column."""
        stats: Dict[str, Dict] = {}
        for col in df.select_dtypes(include="number").columns:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            stats[col] = {
                "mean": round(float(s.mean()), 6),
                "std":  round(float(s.std()),  6),
                "min":  round(float(s.min()),  6),
                "max":  round(float(s.max()),  6),
                "p25":  round(float(s.quantile(0.25)), 6),
                "p75":  round(float(s.quantile(0.75)), 6),
                "n":    int(len(s)),
            }
        return stats

    def _categorical_stats(self, df) -> Dict[str, Dict[str, int]]:
        """Top-N value counts per non-numeric column."""
        stats: Dict[str, Dict] = {}
        for col in df.select_dtypes(exclude="number").columns:
            if col in self.sensitive_cols:
                continue  # sensitive cols get their own normalised stats
            vc = df[col].value_counts().head(self.max_categories)
            stats[col] = {str(k): int(v) for k, v in vc.items()}
        return stats

    def _sensitive_stats(self, df) -> Dict[str, Dict[str, float]]:
        """Normalised frequency distribution per sensitive column.

        Values sum to 1.0 (within floating-point tolerance).
        Missing values in a sensitive column are tracked under the key
        ``"__null__"`` so they are not silently excluded.
        """
        stats: Dict[str, Dict] = {}
        for col in self.sensitive_cols:
            if col not in df.columns:
                continue  # silently skip missing columns
            # Include nulls as a separate category
            series = df[col].fillna("__null__")
            vc = series.value_counts(normalize=True)
            stats[col] = {str(k): round(float(v), 6) for k, v in vc.items()}
        return stats


# formatting helpers 

def _bar(fraction: float, width: int = 30) -> str:
    """Return an ASCII progress bar for a fraction (0–1)."""
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def _print_profile(profile: Dict, title: str = "DataFrame Profile") -> None:
    """Print a formatted, human-readable profile to stdout."""
    w = 65
    print(f"\n{'─' * w}")
    print(f"  {title}")
    print(f"{'─' * w}")
    print(f"  Rows: {profile['row_count']:,}   Columns: {profile['column_count']}")
    print()

    # Null rates
    null_rates = profile.get("null_rates", {})
    notnull = {c: v for c, v in null_rates.items() if v > 0}
    if notnull:
        print("  Null rates (columns with missing data):")
        for col, rate in sorted(notnull.items(), key=lambda x: -x[1]):
            bar = _bar(rate, 20)
            print(f"    {col:25s}  {bar}  {rate:.1%}")
        print()

    # Numeric stats — show mean ± std
    numeric = profile.get("numeric_stats", {})
    if numeric:
        print("  Numeric features (mean ± std):")
        for col, s in numeric.items():
            print(f"    {col:25s}  {s['mean']:>10.3f} ± {s['std']:.3f}"
                  f"   [{s['min']:.2f}, {s['max']:.2f}]")
        print()

    # Sensitive column distributions — the key output
    sensitive = profile.get("sensitive_stats", {})
    if sensitive:
        print("  Sensitive column distributions:")
        for col, dist in sensitive.items():
            print(f"    {col}:")
            for val, frac in sorted(dist.items(), key=lambda x: -x[1]):
                if val == "__null__":
                    label = f"(missing)"
                else:
                    label = str(val)
                bar = _bar(frac, 25)
                print(f"      {label:20s}  {bar}  {frac:.1%}")
        print()

    print(f"{'─' * w}\n")


def print_snapshot_comparison(before: Dict, after: Dict,
                               step_name: str = "step") -> None:
    """Print a side-by-side demographic comparison of before/after snapshots.

    This is the human-readable version of what ShiftDetector will compute
    automatically in Week 3.

    Args:
        before: Snapshot dict returned by ``store.get_snapshots(position='before')[0]``
        after:  Snapshot dict returned by ``store.get_snapshots(position='after')[0]``
        step_name: Name shown in the header.

    Example::

        snaps  = store.get_snapshots("clean_data")
        before = next(s for s in snaps if s["position"] == "before")
        after  = next(s for s in snaps if s["position"] == "after")
        print_snapshot_comparison(before, after, step_name="clean_data")
    """
    w = 65
    print(f"\n{'═' * w}")
    print(f"  Snapshot comparison: {step_name}")
    print(f"{'═' * w}")
    print(f"  {'':25s}  {'BEFORE':>10}  {'AFTER':>10}  {'SHIFT':>8}")
    print(f"  {'Rows':25s}  {before['row_count']:>10,}  "
          f"{after['row_count']:>10,}  "
          f"{after['row_count'] - before['row_count']:>+8,}")
    print()

    sensitive_before = before.get("sensitive_stats", {})
    sensitive_after  = after.get("sensitive_stats",  {})

    all_cols = set(sensitive_before) | set(sensitive_after)
    for col in sorted(all_cols):
        dist_b = sensitive_before.get(col, {})
        dist_a = sensitive_after.get(col,  {})
        all_vals = set(dist_b) | set(dist_a)

        print(f"  {col} distribution:")
        has_high_shift = False

        for val in sorted(all_vals):
            fb = dist_b.get(val, 0.0)
            fa = dist_a.get(val, 0.0)
            shift = fa - fb

            label = "(missing)" if val == "__null__" else val
            flag  = ""
            if abs(shift) > 0.10:
                flag = "  ⚠  HIGH SHIFT"
                has_high_shift = True
            elif abs(shift) > 0.05:
                flag = "  △  moderate"

            bar_b = _bar(fb, 12)
            bar_a = _bar(fa, 12)
            print(f"    {label:18s}  {bar_b} {fb:>5.1%}  →  "
                  f"{bar_a} {fa:>5.1%}  {shift:>+6.1%}{flag}")

        if has_high_shift:
            print(f"\n  ⚠  Significant demographic shift detected at '{col}'.")
            print(f"     This step is a candidate causal source of bias.")
        print()

    print(f"{'═' * w}\n")