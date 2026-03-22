# DataLineageML

> Lightweight data provenance tracker for ML pipelines.  
> Wrap any Python function with `@track` — get automatic lineage logging, dataset hashing, and visual pipeline graphs. Zero mandatory dependencies.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-orange.svg)](https://pypi.org/project/datalineageml/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

---

## Why DataLineageML?

Most ML teams lose track of exactly what data went where. When a model starts performing worse, the first question is always: *did the data change?* DataLineageML answers that question automatically.

It tracks:
- **What** function ran, on **what** data (via cryptographic hashing)
- **When** it ran, how long it took, and whether it succeeded or failed
- **How** data flowed through your pipeline (visual DAG)
- **Which version** of a dataset produced which model artefact

Unlike DVC or MLflow, DataLineageML requires no server, no configuration file, and no cloud account. It writes to a local SQLite file and stays out of your way.

---

## Quick start

```bash
pip install datalineageml
```

```python
from datalineageml import track, LineageContext, LineageGraph

# 1. Decorate any function
@track(name="clean_data", tags={"stage": "preprocessing"})
def clean_data(df):
    return df.dropna().reset_index(drop=True)

@track(name="engineer_features", tags={"stage": "features"})
def engineer_features(df):
    df["age_squared"] = df["age"] ** 2
    return df

@track(name="train_model", tags={"stage": "training"})
def train_model(df):
    from sklearn.ensemble import RandomForestClassifier
    X, y = df.drop("target", axis=1), df["target"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# 2. Group steps into a named pipeline
with LineageContext(name="churn_prediction_v1"):
    df = clean_data(raw_df)
    df = engineer_features(df)
    model = train_model(df)

# 3. Visualise the lineage graph
graph = LineageGraph()
graph.show()                          # opens interactive Plotly graph
graph.show(output_html="lineage.html")  # or save to file
```

---

## Installation

### Minimal (core only — no visualization)
```bash
pip install datalineageml
```

### With visualization
```bash
pip install "datalineageml[viz]"
```

### With pandas helpers
```bash
pip install "datalineageml[pandas]"
```

### Everything
```bash
pip install "datalineageml[all]"
```

### Development
```bash
git clone https://github.com/adejumobioluwafemi/data-lineage-ml.git
cd data-lineage-ml
pip install -e ".[dev]"
pytest tests/unit/ -v
```

---

## Core concepts

### `@track` — the decorator

```python
from datalineageml import track

@track(
    name="my_step",          # optional: human-readable name (defaults to function name)
    tags={"version": "v2"},  # optional: arbitrary key-value metadata
)
def my_transform(df):
    return df.dropna()
```

Every call logs:

| Field | What it captures |
|---|---|
| `step_name` | Name of the transformation |
| `input_hashes` | MD5 hash of every argument (DataFrame, array, dict, etc.) |
| `output_hash` | MD5 hash of the return value |
| `duration_ms` | Wall-clock execution time in milliseconds |
| `started_at` | UTC timestamp |
| `status` | `"success"` or `"failed"` |
| `error` | Exception message if failed |
| `tags` | Your custom metadata |

### `LineageContext` — pipeline grouping

```python
from datalineageml import LineageContext

with LineageContext(name="my_pipeline") as ctx:
    result = step_one(data)
    result = step_two(result)
    # If any step raises, the pipeline is marked "failed"
```

Groups all tracked steps under a named pipeline run. Makes it easy to compare runs across experiments.

### `LineageStore` — direct access to records

```python
from datalineageml import LineageStore

store = LineageStore(db_path="lineage.db")  # default path

# Get all logged steps
steps = store.get_steps()

# Filter by step name
cleans = store.get_steps("clean_data")

# Get all pipeline runs
pipelines = store.get_pipelines()
```

### `LineageGraph` — visual DAG

```python
from datalineageml import LineageGraph

graph = LineageGraph()

# Get the raw NetworkX graph for custom analysis
G = graph.build()
print(list(G.nodes()))
print(list(G.edges()))

# Render interactive Plotly graph
graph.show()

# Save to HTML (great for sharing / GitHub Pages)
graph.show(output_html="lineage.html")
```

Green nodes = success. Red nodes = failed.

---

## Pandas integration

```python
from datalineageml.integrations.pandas_integration import tracked_read_csv, tracked_merge

# Automatically tracked — no decorator needed
df = tracked_read_csv("data/raw/customers.csv")
merged = tracked_merge(df_customers, df_orders, on="customer_id")
```

---

## Custom store path

By default, lineage is written to `lineage.db` in your working directory.
Override per-function or globally:

```python
from datalineageml import LineageStore, track

store = LineageStore(db_path="experiments/run_42/lineage.db")

@track(name="preprocess", store=store)
def preprocess(df):
    return df.dropna()
```

---

## Real-world example: crop yield pipeline

```python
import pandas as pd
from datalineageml import track, LineageContext, LineageGraph

@track(name="load_satellite_data", tags={"source": "sentinel-2"})
def load_satellite_data(path):
    return pd.read_parquet(path)

@track(name="merge_weather", tags={"source": "openweather-api"})
def merge_weather(satellite_df, weather_df):
    return pd.merge(satellite_df, weather_df, on=["lat", "lon", "date"])

@track(name="normalize_features", tags={"stage": "preprocessing"})
def normalize_features(df):
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
    return df

@track(name="train_yield_model", tags={"model": "LSTM", "crop": "soybean"})
def train_yield_model(df):
    # ... your training code
    return model

with LineageContext(name="soybean_yield_v3"):
    sat   = load_satellite_data("data/satellite_2024.parquet")
    wx    = pd.read_parquet("data/weather_2024.parquet")
    df    = merge_weather(sat, wx)
    df    = normalize_features(df)
    model = train_yield_model(df)

LineageGraph().show(output_html="soybean_lineage.html")
```

---

## Roadmap

### v0.1 (current)
- [x] `@track` decorator with input/output hashing
- [x] SQLite persistence (zero-dependency core)
- [x] `LineageContext` pipeline grouping
- [x] Plotly + NetworkX lineage graph
- [x] Pandas integration helpers

### v0.2 (planned)
- [ ] scikit-learn Pipeline wrapper
- [ ] Streamlit dashboard for lineage exploration
- [ ] Export lineage to JSON / YAML
- [ ] Hash comparison CLI (`datalineageml diff run_1 run_2`)

### v0.3 (future)
- [ ] MLflow backend adapter
- [ ] Async step tracking
- [ ] PyTorch DataLoader integration

---

## Testing

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ -v --cov=src/datalineageml --cov-report=term-missing

# Run a single test file
pytest tests/unit/test_decorator.py -v
```

Tests use `tmp_path` (pytest built-in fixture) so every test gets its own isolated SQLite file. No test ever affects another.

---

## Project structure

```
data-lineage-ml/
├── src/
│   └── datalineageml/
│       ├── __init__.py              # public API: track, LineageContext, LineageStore, LineageGraph
│       ├── trackers/
│       │   ├── decorator.py         # @track implementation
│       │   └── context.py           # LineageContext
│       ├── storage/
│       │   └── sqlite_store.py      # SQLite persistence layer
│       ├── visualization/
│       │   └── graph.py             # NetworkX + Plotly DAG
│       └── integrations/
│           └── pandas_integration.py
├── tests/
│   ├── unit/
│   │   ├── test_store.py
│   │   ├── test_decorator.py
│   │   └── test_context.py
│   └── integration/                 # coming in v0.2
├── examples/
│   └── basic_pipeline.py
├── docs/
├── conftest.py
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Contributing

Contributions welcome — especially new integrations (sklearn, PyTorch, Spark).

```bash
git clone https://github.com/adejumobioluwafemi/data-lineage-ml.git
cd data-lineage-ml
pip install -e ".[dev]"

# Make your changes, then run:
black src/ tests/
ruff check src/ tests/
pytest tests/unit/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Oluwafemi Adejumobi** — AI/ML Engineer, Ibadan, Nigeria  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_HANDLE) · [HuggingFace](https://huggingface.co/YOUR_HANDLE)

---

*Built as part of a portfolio of AI safety and ML infrastructure tools. See also: [EquiTrace](https://github.com/YOUR_USERNAME/equitrace) (LLM bias detection) · [PrivacyAudit](https://github.com/YOUR_USERNAME/privacy-audit) · [AgentTrace](https://github.com/YOUR_USERNAME/agent-trace)*
