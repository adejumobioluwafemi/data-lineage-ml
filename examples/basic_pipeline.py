"""
examples/basic_pipeline.py

DataLineageML v0.2 — full causal attribution demo.

Complete loop:
  1. Pipeline runs with snapshot=True
  2. Snapshots logged automatically (gender distribution at each step)
  3. ShiftDetector ranks steps by JSD + KS shift magnitude
  4. CausalAttributor identifies the causal step and generates a recommendation
  5. Bias metric logged against the attributed run_id

Run:
    pip install -e ".[all]"
    python examples/basic_pipeline.py
"""

import sys, os

try:
    import datalineageml  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd
import datalineageml as dlm
from datalineageml import track, LineageContext, LineageGraph
from datalineageml.analysis import (
    DataFrameProfiler,
    print_snapshot_comparison,
    ShiftDetector,
    CausalAttributor,
)

# ── Global store ──────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "../lineage_demo.db")
dlm.init(db_path=DB_PATH)

# ── Synthetic dataset ─────────────────────────────────────────────────────────
# Null pattern mirrors the Oyo State structural inequality:
# female rows are more likely to be missing 'income' and 'age'
# because of under-registration in formal employment records.
RAW_DATA = {
    "age":    [25, 32, None, 45, 28, 38, None, 52, 29, 41,
               34, 27, None, 48, 31, 39, 26, None, 44, 35],
    "income": [50000, 72000, 65000, None, 48000, None, 55000, 88000,
               None,  76000, 58000, None, None,  84000, 52000, 69000,
               None,  78000, None,  62000],
    "score":  [0.72, 0.85, 0.61, 0.90, 0.55, 0.88, 0.70, 0.92,
               0.58, 0.81, 0.67, 0.79, 0.53, 0.95, 0.62, 0.84,
               0.59, 0.76, 0.88, 0.71],
    "gender": ["F", "M", "F", "M", "F", "F", "F", "M",
               "F", "M", "M", "F", "F", "M", "M", "M",
               "F", "M", "F", "M"],
    "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
               0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
}


# ── Pipeline functions ────────────────────────────────────────────────────────

@track(name="load_data", tags={"stage": "ingestion"})
def load_data():
    return pd.DataFrame(RAW_DATA)


@track(
    name="clean_data",
    tags={"stage": "preprocessing"},
    snapshot=True,              # ← log demographic snapshot before AND after
    sensitive_cols=["gender"],  # ← track gender distribution at this step
)
def clean_data(df):
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  clean_data: {before} rows → {len(df)} rows "
          f"(dropped {before - len(df)})")
    return df


@track(name="engineer_features", tags={"stage": "feature_engineering"})
def engineer_features(df):
    df = df.copy()
    df["income_per_age"] = df["income"] / df["age"]
    df["high_score"]     = (df["score"] > 0.75).astype(int)
    print(f"  engineer_features: shape {df.shape}")
    return df


@track(name="normalize", tags={"stage": "preprocessing"})
def normalize(df):
    df = df.copy()
    for col in ["age", "income", "score", "income_per_age"]:
        lo, hi = df[col].min(), df[col].max()
        if hi > lo:
            df[col] = (df[col] - lo) / (hi - lo)
    print(f"  normalize: scaled 4 columns")
    return df


@track(name="train_model", tags={"stage": "training", "model": "logistic_regression"})
def train_model(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X = df[["age", "income", "score", "income_per_age", "high_score"]]
    y = df["target"]
    model = LogisticRegression(random_state=42, max_iter=500)
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    model.fit(X, y)
    print(f"  train_model: CV accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    store = dlm.get_default_store()
    store.clear()

    print("\n" + "═" * 72)
    print("  DataLineageML v0.2  ·  Full causal attribution demo")
    print("═" * 72)

    # ── Step 1: Profile raw data ──────────────────────────────────────────
    raw_df   = pd.DataFrame(RAW_DATA)
    profiler = DataFrameProfiler(sensitive_cols=["gender"])
    print()
    profiler.print_profile(raw_df, step_name="raw_data")

    # ── Step 2: Run the pipeline ──────────────────────────────────────────
    print("── Running pipeline ─────────────────────────────────────────────────\n")
    with LineageContext(name="churn_pipeline_v2"):
        d1 = load_data()
        d2 = clean_data(d1)
        d3 = engineer_features(d2)
        d4 = normalize(d3)
        model = train_model(d4)

    # ── Step 3: Logged steps ──────────────────────────────────────────────
    print("\n── Logged steps ─────────────────────────────────────────────────────")
    for s in store.get_steps():
        icon = "✓" if s["status"] == "success" else "✗"
        snap = " [snapshot]" if store.get_snapshots(s["step_name"]) else ""
        print(f"  {icon}  {s['step_name']:28s}  {s['duration_ms']:6.1f}ms"
              f"  hash={s['output_hash'][:10]}...{snap}")

    # ── Step 4: Manual snapshot comparison (Week 2) ───────────────────────
    snaps = store.get_snapshots("clean_data")
    if len(snaps) == 2:
        before = next(s for s in snaps if s["position"] == "before")
        after  = next(s for s in snaps if s["position"] == "after")
        print_snapshot_comparison(before, after, step_name="clean_data")

    # ── Step 5: ShiftDetector — ranked shift report (Week 3) ─────────────
    detector = ShiftDetector(store=store)
    shifts   = detector.detect(pipeline_name="churn_pipeline_v2")
    detector.print_report(shifts, title="Shift Report — churn_pipeline_v2")

    # ── Step 6: CausalAttributor — attribution + recommendation (Week 4) ─
    print("── Causal Attribution ───────────────────────────────────────────────")
    attributor = CausalAttributor(store=store)
    result     = attributor.attribute(sensitive_col="gender")
    attributor.print_attribution(result)

    # ── Step 7: Log the bias metric ───────────────────────────────────────
    if result["attributed_step"]:
        clean_step = store.get_steps("clean_data")
        if clean_step:
            top_jsd = result["stat"]
            snaps2  = store.get_snapshots("clean_data")
            f_before = next((s["sensitive_stats"]["gender"].get("F", 0)
                             for s in snaps2 if s["position"] == "before"), 0)
            f_after  = next((s["sensitive_stats"]["gender"].get("F", 0)
                             for s in snaps2 if s["position"] == "after"), 0)
            store.log_metrics(
                run_id=clean_step[0]["run_id"],
                metrics={
                    "gender_jsd":                      round(top_jsd, 4),
                    "gender_representation_shift":     round(f_before - f_after, 4),
                    "attribution_confidence":          round(result["confidence"], 4),
                },
                metric_source="CausalAttributor",
                step_name=result["attributed_step"],
                tags={
                    "sensitive_col": "gender",
                    "flag":          result["flag"],
                    "pipeline":      "churn_pipeline_v2",
                },
            )
            print(f"  ✓ Bias metrics logged:")
            print(f"      gender_jsd                    = {top_jsd:.4f}")
            print(f"      gender_representation_shift   = {f_before - f_after:.4f}")
            print(f"      attribution_confidence        = {result['confidence']:.4f}")
            print(f"      attributed to step:             '{result['attributed_step']}'")
            print()

    # ── Step 8: Pipeline status ───────────────────────────────────────────
    print("── Pipeline run ─────────────────────────────────────────────────────")
    for p in store.get_pipelines():
        print(f"  {p['name']}  status={p['status']}  "
              f"started={p['started_at'][:19]}")

    # ── Step 9: Lineage graph ─────────────────────────────────────────────
    html = os.path.join(os.path.dirname(__file__), "../lineage_demo.html")
    print(f"\n── Lineage graph → {os.path.basename(html)} ──────────────────────")
    try:
        LineageGraph().show(output_html=html)
        print(f"  Open with: open lineage_demo.html\n")
    except ImportError as e:
        print(f"  Skipped: {e}\n")


if __name__ == "__main__":
    main()