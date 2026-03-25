"""
examples/basic_pipeline.py

A self-contained demo of DataLineageML showing the full v0.2 API:
  - dlm.init() global store (no store= needed on every function)
  - snapshot=True with sensitive_cols demographic tracking
  - store.log_metrics() for safety/fairness measurement
  - LineageGraph visual DAG

Run with:
    pip install -e ".[all]"
    python examples/basic_pipeline.py
"""

import sys
import os

# Only inject src/ if the package is NOT already installed
try:
    import datalineageml  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd
import datalineageml as dlm
from datalineageml import track, LineageContext, LineageGraph

# 1. Initialise global store 
# One call here — no store= needed anywhere below.
DB_PATH = os.path.join(os.path.dirname(__file__), "../lineage_demo.db")
dlm.init(db_path=DB_PATH)

# 2. Pipeline steps — no store= argument needed 

@track(name="load_data", tags={"stage": "ingestion", "source": "synthetic"})
def load_data():
    """Simulate loading a mixed dataset with demographic variation."""
    return pd.DataFrame({
        "age":    [25, 32, None, 45, 28, 38, None, 52, 29, 41,
                   34, 27, None, 48, 31, 39, 26, None, 44, 35],
        "income": [50000, 72000, 65000, None, 48000, 91000, 55000, 88000,
                   47000, 76000, 58000, 43000, None, 84000, 52000, 69000,
                   45000, 78000, None, 62000],
        "score":  [0.72, 0.85, 0.61, 0.90, 0.55, 0.88, 0.70, 0.92,
                   0.58, 0.81, 0.67, 0.79, 0.53, 0.95, 0.62, 0.84,
                   0.59, 0.76, 0.88, 0.71],
        "gender": ["F", "M", "F", "M", "F", "M", "F", "M",
                   "F", "M", "F", "M", "F", "M", "F", "M",
                   "F", "F", "M", "M"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                   0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    })


@track(
    name="clean_data",
    tags={"stage": "preprocessing"},
    snapshot=True,
    sensitive_cols=["gender"],   # ← track gender distribution before/after
)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing values."""
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    print(f"  clean_data: {before} rows → {len(df)} rows  ({dropped} dropped)")
    return df


@track(name="engineer_features", tags={"stage": "feature_engineering"})
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features useful for prediction."""
    df = df.copy()
    df["income_per_age"] = df["income"] / df["age"]
    df["high_score"]     = (df["score"] > 0.75).astype(int)
    print(f"  engineer_features: added 2 features → shape {df.shape}")
    return df


@track(name="normalize", tags={"stage": "preprocessing"})
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize all numeric feature columns."""
    df = df.copy()
    for col in ["age", "income", "score", "income_per_age"]:
        lo, hi = df[col].min(), df[col].max()
        df[col] = (df[col] - lo) / (hi - lo)
    print(f"  normalize: scaled 4 columns")
    return df


@track(name="train_model", tags={"stage": "training", "model": "logistic_regression"})
def train_model(df: pd.DataFrame):
    """Train a logistic regression classifier and report CV accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    features = ["age", "income", "score", "income_per_age", "high_score"]
    X = df[features]
    y = df["target"]

    model = LogisticRegression(random_state=42, max_iter=500)
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    model.fit(X, y)
    print(f"  train_model: CV accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
    return model


# 3. Run the pipeline

def main():
    print("\n── DataLineageML v0.2 demo ──────────────────────────────────────\n")

    store = dlm.get_default_store()
    store.clear()  # fresh start each run

    with LineageContext(name="churn_pipeline_v2"):
        raw   = load_data()
        clean = clean_data(raw)         # ← snapshots logged here automatically
        feats = engineer_features(clean)
        normd = normalize(feats)
        model = train_model(normd)

    # 4. Inspect steps 
    print("\n── Logged steps ──────────────────────────────────────────────────")
    for s in store.get_steps():
        icon = "✓" if s["status"] == "success" else "✗"
        print(f"  {icon}  {s['step_name']:28s}  {s['duration_ms']:6.1f}ms  "
              f"hash={s['output_hash'][:10]}...")

    # 5. Show demographic snapshots from clean_data 
    print("\n── Demographic snapshots (clean_data) ───────────────────────────")
    snaps = store.get_snapshots("clean_data")
    for snap in snaps:
        gender = snap["sensitive_stats"].get("gender", {})
        fracs  = "  ".join(f"{k}: {v:.1%}" for k, v in sorted(gender.items()))
        print(f"  [{snap['position']:6s}]  rows={snap['row_count']:4d}  gender → {fracs}")

    if len(snaps) == 2:
        before_f = snaps[0]["sensitive_stats"]["gender"].get("F", 0)
        after_f  = snaps[1]["sensitive_stats"]["gender"].get("F", 0)
        shift    = before_f - after_f
        flag     = "HIGH ⚠" if shift > 0.05 else "LOW ✓"
        print(f"\n  Gender shift (F): {shift:+.1%}  [{flag}]")

    # 6. Log a bias metric 
    # In a real pipeline this would come from EquiTrace or a fairness toolkit.
    # Here we use the gender shift as a proxy metric for demonstration.
    clean_step = store.get_steps("clean_data")
    if clean_step:
        run_id = clean_step[0]["run_id"]
        store.log_metrics(
            run_id=run_id,
            metrics={"gender_representation_shift": round(shift, 4)},
            metric_source="datalineageml_demo",
            step_name="clean_data",
            tags={"sensitive_col": "gender", "pipeline": "churn_pipeline_v2"},
        )
        print(f"\n  Logged metric: gender_representation_shift = {shift:.4f}")

    # 7. Pipeline status
    print("\n── Pipeline runs ─────────────────────────────────────────────────")
    for p in store.get_pipelines():
        print(f"  {p['name']}  status={p['status']}  started={p['started_at'][:19]}")

    # 8. Lineage graph
    html_path = os.path.join(os.path.dirname(__file__), "../lineage_demo.html")
    print(f"\n── Lineage graph → {html_path} ──")
    try:
        LineageGraph().show(output_html=html_path)
        print(f"  Open with: open {os.path.basename(html_path)}\n")
    except ImportError as e:
        print(f"  Skipped: {e}\n")


if __name__ == "__main__":
    main()