"""
examples/basic_pipeline.py

A self-contained demo of DataLineageML.

Run with:
    pip install -e ".[all]"
    python examples/basic_pipeline.py
"""

import sys
import os

# Only inject src path if the package is NOT already installed
try:
    import datalineageml  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd
from datalineageml import track, LineageContext, LineageStore, LineageGraph

# ── Shared store — ALL @track calls and LineageContext use this same store ──
# This is the single source of truth. Every step writes here.
DB_PATH = os.path.join(os.path.dirname(__file__), "../lineage_demo.db")
STORE = LineageStore(db_path=DB_PATH)

# ── Pipeline steps — store= passed explicitly so all writes go to STORE ──────

@track(name="load_data", tags={"stage": "ingestion", "source": "synthetic"}, store=STORE)
def load_data():
    """Simulate loading a dataset."""
    return pd.DataFrame({
        "age":    [25, 32, None, 45, 28, 38, None, 52, 29, 41],
        "income": [50000, 72000, 65000, None, 48000, 91000, 55000, 88000, 47000, 76000],
        "score":  [0.72, 0.85, 0.61, 0.90, 0.55, 0.88, 0.70, 0.92, 0.58, 0.81],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@track(name="clean_data", tags={"stage": "preprocessing"}, store=STORE)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values."""
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  clean_data: {before} rows → {len(df)} rows (dropped {before - len(df)} nulls)")
    return df


@track(name="engineer_features", tags={"stage": "feature_engineering"}, store=STORE)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features."""
    df = df.copy()
    df["income_per_age"] = df["income"] / df["age"]
    df["high_score"] = (df["score"] > 0.75).astype(int)
    print(f"  engineer_features: added 2 features → shape {df.shape}")
    return df


@track(name="normalize", tags={"stage": "preprocessing"}, store=STORE)
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize numeric features."""
    df = df.copy()
    for col in ["age", "income", "score", "income_per_age"]:
        col_min, col_max = df[col].min(), df[col].max()
        df[col] = (df[col] - col_min) / (col_max - col_min)
    print(f"  normalize: scaled 4 columns")
    return df


@track(name="train_model", tags={"stage": "training", "model": "logistic_regression"}, store=STORE)
def train_model(df: pd.DataFrame):
    """Train a simple classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    feature_cols = ["age", "income", "score", "income_per_age", "high_score"]
    X = df[feature_cols]
    y = df["target"]

    model = LogisticRegression(random_state=42)
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    model.fit(X, y)
    print(f"  train_model: CV accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
    return model


# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    print("\n── Running DataLineageML demo pipeline ──\n")

    STORE.clear()  # fresh start each run

    with LineageContext(name="demo_churn_pipeline", store=STORE):
        raw   = load_data()
        clean = clean_data(raw)
        feats = engineer_features(clean)
        normd = normalize(feats)
        model = train_model(normd)

    # ── Inspect logged steps ──────────────────────────────────────────────────
    print("\n── Logged steps ──")
    steps = STORE.get_steps()
    for s in steps:
        icon = "✓" if s["status"] == "success" else "✗"
        print(f"  {icon} {s['step_name']:30s}  {s['duration_ms']:6.1f}ms  "
              f"out_hash={s['output_hash'][:12]}...")

    print("\n── Logged pipelines ──")
    for p in STORE.get_pipelines():
        print(f"  {p['name']}  status={p['status']}  started={p['started_at'][:19]}")

    # ── Lineage graph ─────────────────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(__file__), "../lineage_demo.html")
    print(f"\n── Generating lineage graph → {output_path} ──")
    try:
        graph = LineageGraph(store=STORE)
        graph.show(output_html=output_path)
        print(f"  Done. Run: open {output_path}\n")
    except ImportError as e:
        print(f"  Skipped — {e}\n")


if __name__ == "__main__":
    main()
