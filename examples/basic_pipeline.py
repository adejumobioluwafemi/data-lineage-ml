"""
examples/basic_pipeline.py

DataLineageML v0.2 — complete causal attribution + counterfactual proof demo.

Full loop:
  1. Pipeline runs with snapshot=True
  2. ShiftDetector: gender distribution shifted significantly at clean_data
  3. CausalAttributor: clean_data attributed (confidence 100%)
  4. CounterfactualReplayer: bias drops from X to Y after imputation fix
  5. Verdict: STRONG / MODERATE — attribution confirmed

Run:
    pip install -e ".[all]"
    python examples/basic_pipeline.py
"""

import sys, os
try:
    import datalineageml
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd
import numpy as np
import datalineageml as dlm
from datalineageml import track, LineageContext, CounterfactualReplayer
from datalineageml.analysis import (
    DataFrameProfiler,
    ShiftDetector,
    CausalAttributor,
)
from datalineageml.analysis.metrics import DemographicParityGap

# ── Global store ──────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "../lineage_demo.db")
dlm.init(db_path=DB_PATH)

# ── Synthetic data: structured missingness mirrors Oyo State patterns ─────────
np.random.seed(42)
N = 200
_gender  = np.random.choice(["F", "M"], N, p=[0.5, 0.5])
_is_f    = _gender == "F"
_land    = np.where(_is_f,
               np.random.choice(["registered", None], N, p=[0.11, 0.89]), # type: ignore
               np.random.choice(["registered", None], N, p=[0.67, 0.33])).tolist() # type: ignore
_income  = np.where(_is_f,
               np.random.normal(45000, 8000, N).clip(20000, 90000),
               np.random.normal(60000, 10000, N).clip(20000, 100000))
_score   = np.random.uniform(0.4, 0.95, N).round(3)
_target  = ((_score > 0.65) & (_income > 42000)).astype(int)

RAW_DATA = pd.DataFrame({
    "gender":     _gender,
    "land_title": _land,
    "income":     _income.round(0),
    "score":      _score,
    "target":     _target,
})


def load_data(df):
    return df.copy()


def clean_data(df):
    """Biased: dropna disproportionately removes female farmers."""
    return df.dropna().reset_index(drop=True)


def engineer_features(df):
    df = df.copy()
    df["score_income"] = df["score"] * (df["income"] / df["income"].max())
    return df


def normalize(df):
    df = df.copy()
    for col in ["income", "score", "score_income"]:
        lo, hi = df[col].min(), df[col].max()
        if hi > lo:
            df[col] = (df[col] - lo) / (hi - lo)
    return df


# Tracked versions for the main pipeline run 

@track(name="load_data",         tags={"stage": "ingestion"})
def t_load(df):  return load_data(df)

@track(name="clean_data",        tags={"stage": "preprocessing"},
       snapshot=True, sensitive_cols=["gender"])
def t_clean(df): return clean_data(df)

@track(name="engineer_features", tags={"stage": "feature_engineering"})
def t_feats(df): return engineer_features(df)

@track(name="normalize",         tags={"stage": "preprocessing"})
def t_norm(df):  return normalize(df)


# ── Replacement function 

def impute_data(df):
    """Fixed: stratified mode imputation preserves demographic balance."""
    df = df.copy()
    for g in df["gender"].dropna().unique():
        mask = (df["gender"] == g) & df["land_title"].isna()
        fill = df.loc[df["gender"] == g, "land_title"].mode()
        df.loc[mask, "land_title"] = fill.iloc[0] if len(fill) > 0 else "unregistered"
    return df


# ── Bias metric (no model needed)

def dpg_bias_metric(df):
    """Demographic Parity Gap on target — computable from data alone."""
    if "target" not in df.columns or "gender" not in df.columns:
        return 0.0
    f_rate = df[df["gender"] == "F"]["target"].mean()
    m_rate = df[df["gender"] == "M"]["target"].mean()
    if np.isnan(f_rate) or np.isnan(m_rate):
        return 0.0
    return abs(float(f_rate) - float(m_rate))


def main():
    store = dlm.get_default_store()
    store.clear()

    w = 72
    print(f"\n{'═' * w}")
    print(f"  DataLineageML v0.2  ·  Full loop: detect → attribute → prove")
    print(f"{'═' * w}\n")

    # ── 1. Profile raw data 
    profiler = DataFrameProfiler(sensitive_cols=["gender"])
    profiler.print_profile(RAW_DATA, step_name="raw_data")

    # ── 2. Run biased pipeline
    print("── Step 1: Run pipeline ──\n")
    with LineageContext(name="demo_pipeline_v1"):
        d1 = t_load(RAW_DATA)
        d2 = t_clean(d1)
        d3 = t_feats(d2)
        d4 = t_norm(d3)

    for s in store.get_steps():
        icon = "✓" if s["status"] == "success" else "✗"
        snap = " [snapshot]" if store.get_snapshots(s["step_name"]) else ""
        print(f"  {icon}  {s['step_name']:24s}  {s['duration_ms']:6.1f}ms{snap}")

    # ── 3. Detect shift
    print(f"\n── Step 2: Detect distribution shifts ──")
    detector = ShiftDetector(store=store)
    shifts   = detector.detect()
    detector.print_report(shifts, title="Shift Report")

    # ── 4. Attribute 
    print(f"── Step 3: Attribute the causal step ──")
    attributor = CausalAttributor(store=store)
    attribution = attributor.attribute(sensitive_col="gender")
    attributor.print_attribution(attribution)

    attributed_step = attribution["attributed_step"]
    if attributed_step is None:
        print("  Attribution inconclusive — cannot run counterfactual.\n")
        return

    # Log DPG metric against the attributed step
    dpg = DemographicParityGap.compute(d2, "gender", "target")
    clean_run = store.get_steps("clean_data")
    if clean_run:
        store.log_metrics(**dpg.to_store_kwargs(
            run_id=clean_run[0]["run_id"], step_name="clean_data"))

    # ── 5. Counterfactual proof
    print(f"── Step 4: Counterfactual proof ──")
    replayer = (CounterfactualReplayer(store=store)
                .register("load_data",         load_data)
                .register("clean_data",        clean_data,
                          snapshot=True, sensitive_cols=["gender"])
                .register("engineer_features", engineer_features)
                .register("normalize",         normalize))

    result = replayer.replay(
        raw_data       = RAW_DATA,
        replace_step   = attributed_step,
        replacement_fn = impute_data,
        sensitive_col  = "gender",
        bias_metric_fn = dpg_bias_metric,
    )
    replayer.print_result(result)

    # ── 6. Summary 
    print(f"── Summary ──")
    if result["bias_metric_before"] is not None:
        bb  = result["bias_metric_before"]
        ba  = result["bias_metric_after"]
        pct = result["bias_reduction_pct"]
        rec = result["rows_recovered"]
        print(f"  Causal step:     '{attributed_step}'  "
              f"(confidence {attribution['confidence']:.0%})")
        print(f"  Bias before fix: {bb:.4f}  (DPG = {bb:.1%})")
        print(f"  Bias after fix:  {ba:.4f}  (DPG = {ba:.1%})")
        print(f"  Reduction:       {pct:+.1f}%")
        print(f"  Rows recovered:  {rec:+,}")
        print(f"  Verdict:         {result['verdict']} — {result['verdict_detail'][:80]}")
    print()


if __name__ == "__main__":
    main()