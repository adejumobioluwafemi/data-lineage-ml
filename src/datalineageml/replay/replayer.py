"""
src/datalineageml/replay/replayer.py

CounterfactualReplayer — verifies bias attribution by re-running the
pipeline with a replacement function at the attributed step, then
measuring the before/after delta on the bias metric.

This is the proof step. Attribution tells you which step caused the bias.
The counterfactual replayer answers: "If we had used a different function
at that step, how much of the bias would have been eliminated?"

Design constraints (v0.2 minimal version):
  ✓ Supports pipelines of @track-decorated functions that take a DataFrame
    and return a DataFrame — covers 80%+ of real preprocessing pipelines.
  ✓ Schema validation: replacement function must return a DataFrame with
    the same columns as the original step's output.
  ✓ Full downstream replay: all steps after the replaced step are re-run
    on the fixed data to produce a realistic outcome comparison.
  ✗ Does not support steps with non-DataFrame arguments (models, configs).
  ✗ Does not support branching pipelines.
  ✗ Does not support async steps.
  ✗ Does not support replacement functions that change the column schema.
    (If you need to add a column, add it before registering the step.)

Usage:
    from datalineageml.replay import CounterfactualReplayer

    replayer = CounterfactualReplayer(store=store)

    # Register each step in order
    replayer.register("load_data",         load_data)
    replayer.register("clean_data",        clean_data,    snapshot=True, sensitive_cols=["gender"])
    replayer.register("engineer_features", engineer_features)
    replayer.register("normalize",         normalize)

    # Define a replacement for the attributed step
    def impute_data(df):
        \"\"\"Stratified imputation instead of dropna — the fairness fix.\"\"\"
        df = df.copy()
        for col in df.select_dtypes(include="number").columns:
            for g in df["gender"].unique():
                mask = (df["gender"] == g) & df[col].isna()
                fill = df.loc[df["gender"] == g, col].median()
                df.loc[mask, col] = fill
        return df

    # Run the counterfactual
    result = replayer.replay(
        raw_data       = df_raw,
        replace_step   = "clean_data",
        replacement_fn = impute_data,
        sensitive_col  = "gender",
    )

    replayer.print_result(result)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import time


class CounterfactualReplayer:
    """Re-run a pipeline with a replacement function at the attributed step.

    Registers pipeline steps in order, then replays the full pipeline twice:
    once with the original functions (biased run) and once with a replacement
    at the attributed step (fixed run). Compares demographic snapshots and
    any provided bias metric between the two runs.

    Args:
        store:   ``LineageStore`` instance. Used to log counterfactual
                 snapshots and retrieve the original run for comparison.
                 Pass None to run without persistence (comparison only).

    Workflow:
        1. ``register()`` each step in pipeline order.
        2. ``replay()`` with the raw input data, the step to replace, and
           the replacement function.
        3. ``print_result()`` to show the before/after comparison.
    """

    def __init__(self, store=None):
        self.store   = store
        self._steps: List[Dict] = []   # ordered list of registered steps

    def register(
        self,
        step_name:      str,
        fn:             Callable,
        snapshot:       bool = False,
        sensitive_cols: Optional[List[str]] = None,
    ) -> "CounterfactualReplayer":
        """Register a pipeline step for replay.

        Steps must be registered in the same order they run in the original
        pipeline. Every step function must accept a pandas DataFrame as its
        first (and usually only) positional argument and return a DataFrame.

        Args:
            step_name:      Name of the step (must match the name used in
                            ``@track`` decoration for snapshot lookup).
            fn:             The original step function.
            snapshot:       If True, log demographic snapshots at this step
                            during replay.
            sensitive_cols: Columns to track for demographic fairness.

        Returns:
            self — for chaining:
                replayer.register("a", fn_a).register("b", fn_b)
        """
        self._steps.append({
            "name":           step_name,
            "fn":             fn,
            "snapshot":       snapshot,
            "sensitive_cols": sensitive_cols or [],
        })
        return self

    def register_tracked(self, fn: Callable) -> "CounterfactualReplayer":
        """Register a ``@track``-decorated function, extracting its metadata.

        Reads the ``name``, ``snapshot``, and ``sensitive_cols`` from the
        ``@track`` decorator — no need to re-specify them.

        Args:
            fn:  A function decorated with ``@track``. The decorator must have
                 been called with at least ``name=`` for this to work reliably.
                 If metadata cannot be read (e.g. the function is not tracked),
                 the function name is used as the step name with no snapshot.

        Returns:
            self — for chaining.

        Example::

            @track(name="clean_data", snapshot=True, sensitive_cols=["gender"])
            def clean_data(df):
                return df.dropna()

            # Instead of:
            # replayer.register("clean_data", clean_data, snapshot=True,
            #                    sensitive_cols=["gender"])
            # Just:
            replayer.register_tracked(clean_data)
        """
        meta = _extract_track_metadata(fn)
        self._steps.append(meta)
        return self

    def replay(
        self,
        raw_data:       Any,
        replace_step:   str,
        replacement_fn: Callable,
        sensitive_col:  str,
        bias_metric_fn: Optional[Callable] = None,
    ) -> Dict:
        """Run the biased pipeline and the fixed pipeline, then compare.

        The biased run uses all original registered functions.
        The fixed run replaces ``replace_step`` with ``replacement_fn``
        and re-runs all subsequent steps on the fixed output.

        Args:
            raw_data:       The raw input DataFrame (before any pipeline step).
            replace_step:   Name of the step to replace (must be registered).
            replacement_fn: Replacement function. Must accept a DataFrame and
                            return a DataFrame with the same columns.
            sensitive_col:  Demographic column to compare across runs.
            bias_metric_fn: Optional callable(df) → float that computes a
                            bias metric from the final output DataFrame.
                            Used to measure before/after improvement.
                            Example: lambda df: abs(df[df.gender=="F"]["eligible"].mean()
                                                    - df[df.gender=="M"]["eligible"].mean())

        Returns:
            CounterfactualResult dict — see ``print_result()`` for layout.

        Raises:
            ValueError: If ``replace_step`` is not registered.
            ValueError: If ``replacement_fn`` returns a DataFrame with
                        different columns than the original step.
        """
        import pandas as pd

        _validate_registered(self._steps, replace_step)

        # 1. Biased run (original pipeline)
        biased_out, biased_snaps, biased_timing = self._run_pipeline(
            raw_data, replace_step=None, replacement_fn=None,
            sensitive_col=sensitive_col,
        )

        # 2. Fixed run (replacement at attributed step) 
        fixed_out, fixed_snaps, fixed_timing = self._run_pipeline(
            raw_data, replace_step=replace_step, replacement_fn=replacement_fn,
            sensitive_col=sensitive_col,
        )

        # 3. Schema validation
        biased_step_out = biased_snaps.get(replace_step, {})
        fixed_step_out  = fixed_snaps.get(replace_step, {})

        if (biased_step_out.get("after_cols") and
                fixed_step_out.get("after_cols") and
                set(biased_step_out["after_cols"]) != set(fixed_step_out["after_cols"])):
            raise ValueError(
                f"Replacement function for '{replace_step}' returned "
                f"different columns than the original.\n"
                f"  Original:    {sorted(biased_step_out['after_cols'])}\n"
                f"  Replacement: {sorted(fixed_step_out['after_cols'])}\n"
                f"The replacement function must preserve the column schema."
            )

        # 4. Compute bias metrics
        bias_before = None
        bias_after  = None
        if bias_metric_fn is not None:
            try:
                bias_before = float(bias_metric_fn(biased_out))
                bias_after  = float(bias_metric_fn(fixed_out))
            except Exception as exc:
                bias_before = None
                bias_after  = None

        # 5. Demographic comparison at the replaced step
        demographic_comparison = _compare_demographics(
            biased_snaps, fixed_snaps, replace_step, sensitive_col
        )

        # 6. Persist counterfactual snapshots
        if self.store is not None:
            _persist_counterfactual_snapshots(
                self.store, fixed_snaps, replace_step, sensitive_col
            )

        # 7. Build result 
        result = _build_result(
            replace_step         = replace_step,
            sensitive_col        = sensitive_col,
            biased_snaps         = biased_snaps,
            fixed_snaps          = fixed_snaps,
            demographic_comparison = demographic_comparison,
            bias_before          = bias_before,
            bias_after           = bias_after,
            biased_timing        = biased_timing,
            fixed_timing         = fixed_timing,
            biased_final         = biased_out,
            fixed_final          = fixed_out,
        )

        return result

    def print_result(self, result: Dict) -> None:
        """Print the counterfactual comparison report."""
        _print_counterfactual_report(result)

    def _run_pipeline(
        self,
        raw_data,
        replace_step:   Optional[str],
        replacement_fn: Optional[Callable],
        sensitive_col:  str,
    ) -> Tuple[Any, Dict, Dict]:
        """Run the full registered pipeline, optionally replacing one step.

        Returns:
            (final_output, snapshots_by_step, timing_by_step)
        """
        from datalineageml.analysis.profiler import DataFrameProfiler

        current = raw_data
        snapshots: Dict[str, Dict] = {}
        timing:    Dict[str, float] = {}

        for step in self._steps:
            name      = step["name"]
            fn        = replacement_fn if (name == replace_step and
                                           replacement_fn is not None) else step["fn"]
            do_snap   = step["snapshot"]
            sens_cols = step["sensitive_cols"]

            profiler  = DataFrameProfiler(sensitive_cols=sens_cols)

            # Snapshot before
            snap_before = None
            if do_snap:
                try:
                    snap_before = profiler.profile(current, name, "before")
                except Exception:
                    pass

            # Run the step
            t0     = time.perf_counter()
            output = _call_step(fn, current, name)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            timing[name] = elapsed

            # Snapshot after
            snap_after = None
            if do_snap and output is not None:
                try:
                    snap_after = profiler.profile(output, name, "after")
                except Exception:
                    pass

            snapshots[name] = {
                "before":      snap_before,
                "after":       snap_after,
                "after_cols":  list(output.columns) if hasattr(output, "columns") else [],
                "rows_in":     len(current) if hasattr(current, "__len__") else None,
                "rows_out":    len(output)  if hasattr(output,  "__len__") else None,
            }

            current = output

        return current, snapshots, timing


# Helpers 

def _extract_track_metadata(fn) -> dict:
    """Extract step name, snapshot, and sensitive_cols from a @track wrapper."""
    meta = getattr(fn, "__track_meta__", None)
    if meta:
        return {
            "name":           meta.get("name", fn.__name__),
            "fn":             fn,
            "snapshot":       meta.get("snapshot", False),
            "sensitive_cols": meta.get("sensitive_cols", []) or [],
        }
    return {"name": fn.__name__, "fn": fn, "snapshot": False, "sensitive_cols": []}


def _call_step(fn: Callable, df: Any, step_name: str) -> Any:
    """Call a pipeline step function, unwrapping @track wrappers if needed."""
    try:
        return fn(df)
    except Exception as exc:
        raise RuntimeError(
            f"Step '{step_name}' raised an exception during replay: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _validate_registered(steps: List[Dict], replace_step: str) -> None:
    names = [s["name"] for s in steps]
    if replace_step not in names:
        raise ValueError(
            f"Step '{replace_step}' is not registered. "
            f"Registered steps: {names}. "
            f"Call replayer.register('{replace_step}', fn) first."
        )


def _compare_demographics(
    biased_snaps: Dict,
    fixed_snaps:  Dict,
    replace_step: str,
    sensitive_col: str,
) -> Dict:
    """Extract before/after demographic distributions at the replaced step."""
    result = {
        "step_name":     replace_step,
        "sensitive_col": sensitive_col,
        "biased": {"before": {}, "after": {}, "rows_in": None, "rows_out": None},
        "fixed":  {"before": {}, "after": {}, "rows_in": None, "rows_out": None},
    }

    for run_label, snaps in [("biased", biased_snaps), ("fixed", fixed_snaps)]:
        step_info = snaps.get(replace_step, {})
        result[run_label]["rows_in"]  = step_info.get("rows_in")
        result[run_label]["rows_out"] = step_info.get("rows_out")

        for pos in ("before", "after"):
            snap = step_info.get(pos)
            if snap and "sensitive_stats" in snap:
                dist = snap["sensitive_stats"].get(sensitive_col, {})
                result[run_label][pos] = dist

    return result


def _persist_counterfactual_snapshots(
    store, fixed_snaps: Dict, replace_step: str, sensitive_col: str
) -> None:
    """Log fixed-run snapshots to the store with counterfactual tag."""
    import uuid
    from datetime import datetime

    for step_name, info in fixed_snaps.items():
        for pos in ("before", "after"):
            snap = info.get(pos)
            if snap is None:
                continue
            try:
                store.log_snapshot(
                    run_id      = f"counterfactual-{step_name}-{str(uuid.uuid4())[:8]}",
                    step_name   = f"{step_name}__counterfactual",
                    position    = pos,
                    row_count       = snap.get("row_count", 0),
                    column_count    = snap.get("column_count", 0),
                    column_names    = snap.get("column_names", []),
                    null_rates      = snap.get("null_rates", {}),
                    numeric_stats   = snap.get("numeric_stats", {}),
                    categorical_stats = snap.get("categorical_stats", {}),
                    sensitive_stats = snap.get("sensitive_stats", {}),
                    recorded_at     = datetime.utcnow().isoformat(),
                )
            except Exception:
                pass  # persistence is best-effort


def _build_result(
    replace_step, sensitive_col,
    biased_snaps, fixed_snaps,
    demographic_comparison,
    bias_before, bias_after,
    biased_timing, fixed_timing,
    biased_final, fixed_final,
) -> Dict:
    """Assemble the full counterfactual result dict."""
    dem = demographic_comparison

    # Row counts
    biased_rows_out = dem["biased"]["rows_out"] or 0
    fixed_rows_out  = dem["fixed"]["rows_out"]  or 0
    rows_recovered  = fixed_rows_out - biased_rows_out

    # Demographic shift at the replaced step
    biased_dist_after = dem["biased"]["after"]
    fixed_dist_after  = dem["fixed"]["after"]
    biased_dist_before = dem["biased"]["before"]

    # Bias improvement
    bias_reduction = None
    bias_reduction_pct = None
    if bias_before is not None and bias_after is not None and bias_before > 0:
        bias_reduction     = round(bias_before - bias_after, 6)
        bias_reduction_pct = round(bias_reduction / bias_before * 100, 2)

    # JSD between biased-after and fixed-after distributions
    jsd_improvement = None
    if biased_dist_after and fixed_dist_after:
        try:
            from datalineageml.analysis.shift_detector import _jensen_shannon_divergence
            jsd_biased = _jensen_shannon_divergence(
                biased_dist_before, biased_dist_after)
            jsd_fixed  = _jensen_shannon_divergence(
                biased_dist_before, fixed_dist_after)
            jsd_improvement = round(jsd_biased - jsd_fixed, 6)
        except Exception:
            pass

    # Verdict
    verdict, verdict_detail = _verdict(
        bias_before, bias_after, bias_reduction_pct,
        biased_dist_after, fixed_dist_after, sensitive_col
    )

    return {
        "replace_step":        replace_step,
        "sensitive_col":       sensitive_col,

        # Demographics at replaced step
        "dist_before_fix":     biased_dist_after,   # original biased output
        "dist_after_fix":      fixed_dist_after,    # replacement output
        "dist_original_input": biased_dist_before,  # input to step (same both runs)

        # Row counts
        "biased_rows_out":     biased_rows_out,
        "fixed_rows_out":      fixed_rows_out,
        "rows_recovered":      rows_recovered,

        # Bias metric comparison
        "bias_metric_before":     bias_before,
        "bias_metric_after":      bias_after,
        "bias_reduction":         bias_reduction,
        "bias_reduction_pct":     bias_reduction_pct,

        # JSD shift improvement
        "jsd_improvement":     jsd_improvement,

        # Timing (ms)
        "biased_timing":       biased_timing,
        "fixed_timing":        fixed_timing,

        # Verdict
        "verdict":             verdict,
        "verdict_detail":      verdict_detail,

        # Full snapshots for debugging
        "_biased_snaps":       biased_snaps,
        "_fixed_snaps":        fixed_snaps,
    }


def _verdict(
    bias_before, bias_after, bias_reduction_pct,
    dist_biased, dist_fixed, sensitive_col
) -> Tuple[str, str]:
    """Produce a verdict string from the comparison results."""
    if bias_before is not None and bias_after is not None:
        if bias_reduction_pct >= 50:
            return ("STRONG", f"Bias metric reduced by {bias_reduction_pct:.1f}%. "
                    f"Attribution confirmed — the replacement function substantially "
                    f"mitigates the bias.")
        elif bias_reduction_pct >= 20:
            return ("MODERATE", f"Bias metric reduced by {bias_reduction_pct:.1f}%. "
                    f"Attribution supported — partial mitigation. Consider "
                    f"additional fixes.")
        elif bias_reduction_pct > 0:
            return ("WEAK", f"Bias metric reduced by only {bias_reduction_pct:.1f}%. "
                    f"The replacement helps marginally. The attributed step may "
                    f"not be the sole causal source.")
        else:
            return ("INCONCLUSIVE", f"Bias metric did not improve ({bias_reduction_pct:.1f}%). "
                    f"The replacement function may not address the root cause, or "
                    f"the attribution may be incorrect.")

    # Fallback: compare demographic distributions
    if dist_biased and dist_fixed:
        vals = sorted(set(dist_biased) | set(dist_fixed))
        max_improvement = max(
            abs(dist_fixed.get(v, 0) - dist_biased.get(v, 0))
            for v in vals
        ) if vals else 0

        if max_improvement > 0.10:
            return ("STRONG", f"Demographic distribution at '{sensitive_col}' "
                    f"substantially improved (max shift recovered: "
                    f"{max_improvement:.1%}).")
        elif max_improvement > 0.05:
            return ("MODERATE", f"Demographic distribution partially improved "
                    f"({max_improvement:.1%} shift recovered).")
        else:
            return ("WEAK", f"Minimal demographic improvement detected "
                    f"({max_improvement:.1%}).")

    return ("UNKNOWN", "Insufficient data to assess counterfactual impact.")


# Report formatting 

def _bar(frac: float, width: int = 20) -> str:
    filled = round(max(0.0, min(1.0, frac)) * width)
    return "█" * filled + "░" * (width - filled)


def _print_counterfactual_report(result: Dict) -> None:
    w = 72
    verdict_icons = {
        "STRONG": "✅", "MODERATE": "🔶", "WEAK": "⚠️", "INCONCLUSIVE": "❌",
        "UNKNOWN": "❓"
    }
    icon = verdict_icons.get(result["verdict"], "❓")

    print(f"\n{'═' * w}")
    print(f"  DataLineageML — Counterfactual Replay Report")
    print(f"{'═' * w}")
    print(f"  Replaced step:    {result['replace_step']}")
    print(f"  Sensitive column: {result['sensitive_col']}")
    print()

    # Row recovery
    print(f"  Row counts at '{result['replace_step']}':")
    print(f"    Biased pipeline:   {result['biased_rows_out']:>6,} rows")
    print(f"    Fixed pipeline:    {result['fixed_rows_out']:>6,} rows")
    rec = result['rows_recovered']
    print(f"    Rows recovered:    {rec:>+6,}  "
          f"({'%.1f' % (rec / max(result['biased_rows_out'], 1) * 100)}%)")

    # Demographic distribution comparison
    print(f"\n  '{result['sensitive_col']}' distribution at '{result['replace_step']}':")
    dist_orig  = result["dist_original_input"]
    dist_bias  = result["dist_before_fix"]
    dist_fix   = result["dist_after_fix"]

    all_vals = sorted(set(dist_orig) | set(dist_bias) | set(dist_fix))
    print(f"  {'Value':14s}  {'Input':>7}  {'Biased out':>10}  "
          f"{'Fixed out':>10}  {'Δ recovered':>12}")
    print(f"  {'─'*14}  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*12}")
    for val in all_vals:
        label = "(missing)" if val == "__null__" else str(val)
        f_in  = dist_orig.get(val, 0.0)
        f_b   = dist_bias.get(val, 0.0)
        f_f   = dist_fix.get(val,  0.0)
        delta = f_f - f_b
        marker = "  ✓" if abs(delta) > 0.05 else ""
        print(f"  {label:14s}  {f_in:>7.1%}  "
              f"{_bar(f_b, 8)} {f_b:>5.1%}  "
              f"{_bar(f_f, 8)} {f_f:>5.1%}  "
              f"{delta:>+11.1%}{marker}")

    # Bias metric comparison
    if result["bias_metric_before"] is not None:
        print(f"\n  Bias metric comparison:")
        bb = result["bias_metric_before"]
        ba = result["bias_metric_after"]
        br = result["bias_reduction"]
        bp = result["bias_reduction_pct"]
        print(f"    Before fix:  {bb:.4f}")
        print(f"    After fix:   {ba:.4f}")
        print(f"    Reduction:   {br:+.4f}  ({bp:+.1f}%)")

    if result["jsd_improvement"] is not None:
        print(f"\n  JSD shift improvement: {result['jsd_improvement']:+.4f}")
        print(f"  (positive = demographic distribution moved closer to input)")

    # Verdict
    print(f"\n{'─' * w}")
    print(f"  Verdict: {icon}  {result['verdict']}")
    print(f"  {result['verdict_detail']}")
    print(f"{'═' * w}\n")