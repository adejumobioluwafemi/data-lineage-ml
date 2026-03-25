# DataLineageML

> **Causal data provenance for AI safety.**
> Find out *which pipeline step caused your model's bias or safety failure* — automatically, verifiably, and without a cloud account.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-orange.svg)](https://pypi.org/project/datalineageml/)
[![Tests](https://img.shields.io/badge/tests-19%20passing-brightgreen.svg)]()
[![Zero deps](https://img.shields.io/badge/core_deps-zero-blue.svg)]()

---

## The problem this solves

Your model is biased. Or it started failing after a data update. Or an audit found a fairness gap. You know *what* went wrong, but not *where* in the pipeline it went wrong.

Existing tools answer the wrong question. MLflow, W&B, and LangSmith tell you **what happened** inside a pipeline — inputs, outputs, latencies. They do not tell you **which data transformation caused the outcome you are trying to fix**.

DataLineageML is built for that second question.

It tracks every transformation step in your pipeline with cryptographic data hashes and demographic snapshots. When a safety or fairness metric degrades, it computes distribution shift scores across each step, attributes the most likely causal step, and lets you verify the fix with a counterfactual replay — before the model reaches production.

---

## A real example

In Oyo State, Nigeria, a crop yield model was allocating agricultural subsidies. An audit found female-headed farms were 34% less likely to receive fertiliser support than male-headed farms with equivalent yield histories. Three months of manual investigation found nothing.

DataLineageML found the cause in one run:

```
Step: clean_data (df.dropna())

  Input:   female-headed farms = 38.4%  (n = 12,847)
  Output:  female-headed farms = 22.1%  (n =  7,403)

  Jensen-Shannon divergence on gender: 0.81  [HIGH]

  Finding: dropna() removed 42% of female-headed records.
  Reason:  formal land title required — held by 11% of women vs 67% of men.
```

The counterfactual replay replaced `dropna()` with stratified imputation:

```
  Original bias score:  0.34
  Post-fix bias score:  0.09   (-74%)
  Accuracy change:      -0.3%  (negligible)
```

The fix took one afternoon. This is what DataLineageML is for.

---

## Quick start

```bash
pip install datalineageml
```

```python
import datalineageml as dlm
from datalineageml import track, LineageContext, LineageGraph

# One init call — all @track decorators use this store automatically
dlm.init(db_path="my_pipeline.db")

@track(name="load_data", tags={"source": "farm_registry"})
def load_data(path):
    return pd.read_csv(path)

@track(name="clean_data", tags={"stage": "preprocessing"})
def clean_data(df):
    return df.dropna()                    # ← DataLineageML will catch this

@track(name="engineer_features")
def engineer_features(df):
    df["income_per_ha"] = df["income"] / df["farm_size_ha"]
    return df

@track(name="train_model", tags={"model": "random_forest"})
def train_model(df):
    from sklearn.ensemble import RandomForestClassifier
    X, y = df.drop("subsidy_eligible", axis=1), df["subsidy_eligible"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# Group all steps under a named pipeline run
with LineageContext(name="subsidy_model_v2"):
    raw   = load_data("data/farms_oyo_2025.csv")
    clean = clean_data(raw)
    feats = engineer_features(clean)
    model = train_model(feats)

# Visualise the lineage graph
LineageGraph().show(output_html="lineage.html")
```

---

## How it works

Every `@track`-decorated function automatically logs:

| Field | What it captures |
|---|---|
| `input_hashes` | Cryptographic hash of every argument (DataFrame, array, dict) |
| `output_hash` | Hash of the return value |
| `duration_ms` | Wall-clock execution time in milliseconds |
| `started_at` | UTC timestamp |
| `status` | `"success"` or `"failed"` |
| `error` | Exception message if failed |
| `tags` | Your custom metadata |

In v0.2, every step will also log a **demographic snapshot** — the statistical distribution of sensitive attributes at that point in the pipeline. That snapshot chain is what enables causal attribution.

---

## Installation

```bash
# Core only — zero mandatory dependencies
pip install datalineageml

# With visualisation (NetworkX + Plotly lineage graph)
pip install "datalineageml[viz]"

# With pandas integration helpers
pip install "datalineageml[pandas]"

# Everything
pip install "datalineageml[all]"

# Development
git clone https://github.com/adejumobioluwafemi/data-lineage-ml.git
cd data-lineage-ml
pip install -e ".[dev]"
python run_tests.py          # 19/19 tests, no pytest needed
```

---

## Core API

### `@track` — instrument any function

```python
from datalineageml import track

@track(
    name="normalize",           # human-readable name (defaults to function name)
    tags={"stage": "prep"},     # arbitrary key-value metadata
    snapshot=True,              # (v0.2) log demographic distributions at this step
    sensitive_cols=["gender"],  # (v0.2) columns to track for fairness
)
def normalize(df):
    return (df - df.mean()) / df.std()
```

### `LineageContext` — group steps into a named pipeline run

```python
from datalineageml import LineageContext

with LineageContext(name="training_pipeline_v3"):
    df    = load_data(path)
    df    = clean_data(df)
    model = train_model(df)
# If any step raises, the pipeline is marked "failed" automatically
```

### `LineageStore` — query the lineage database directly

```python
from datalineageml import LineageStore

store = LineageStore(db_path="my_pipeline.db")

steps     = store.get_steps()                # all logged steps
cleans    = store.get_steps("clean_data")    # filter by step name
pipelines = store.get_pipelines()           # all pipeline runs
```

### `LineageGraph` — interactive visual DAG

```python
from datalineageml import LineageGraph

graph = LineageGraph()
graph.show()                             # interactive Plotly graph in browser
graph.show(output_html="lineage.html")  # save to shareable HTML

G = graph.build()                        # raw NetworkX graph for custom analysis
```

Green nodes = success. Red nodes = failed.

---

## Pandas integration

```python
from datalineageml.integrations.pandas_integration import tracked_read_csv, tracked_merge

df     = tracked_read_csv("data/farms.csv")
merged = tracked_merge(df_farms, df_weather, on=["lat", "lon", "date"])
# Both calls are automatically logged — no @track needed
```

---

## Using an explicit store (multi-pipeline or testing)

```python
from datalineageml import LineageStore, track, LineageContext

STORE = LineageStore(db_path="experiments/run_42/lineage.db")

@track(name="preprocess", store=STORE)
def preprocess(df):
    return df.dropna()

with LineageContext(name="experiment_42", store=STORE):
    result = preprocess(raw_df)
```

---

## How DataLineageML differs from MLflow, W&B, and LangSmith

All three tools do **observability** — recording what happened. None do **causal attribution** — identifying which step caused a safety failure.

| Capability | MLflow | W&B Weave | LangSmith | DataLineageML |
|---|:---:|:---:|:---:|:---:|
| Pipeline tracing | ✓ | ✓ | ✓ | ✓ |
| Dataset versioning | ✓ | ✓ | ✓ | ✓ |
| Demographic snapshots per step | ✗ | ✗ | ✗ | v0.2 |
| Distribution shift detection | ✗ | ✗ | ✗ | v0.2 |
| Causal step attribution | ✗ | ✗ | ✗ | v0.2 |
| Counterfactual pipeline replay | ✗ | ✗ | ✗ | v0.2 |
| Zero-dependency offline core | ✗ | ✗ | ✗ | ✓ |
| No cloud account required | ✗ | ✗ | ✗ | ✓ |
| Cross-agent provenance | ✗ | ✗ | ✗ | v0.3 |

DataLineageML is the only tool designed for environments where cloud data transfer is legally restricted — government systems, healthcare, regulated finance, and public sector AI in Africa and the EU.

---

## Roadmap

### v0.1 — current
- [x] `@track` decorator with input/output hashing
- [x] SQLite persistence (zero mandatory dependencies)
- [x] `LineageContext` pipeline grouping
- [x] Plotly + NetworkX lineage graph
- [x] Pandas integration helpers
- [x] 19 unit tests, stdlib runner (no pytest needed)

### v0.2 — causal attribution (in development)
- [ ] Global `dlm.init()` default store
- [ ] Demographic snapshot logging per step (`snapshot=True`, `sensitive_cols=`)
- [ ] Distribution shift detector (Jensen-Shannon divergence + KS test)
- [ ] Causal step attributor with confidence scoring
- [ ] Counterfactual pipeline replayer
- [ ] Evaluation metric logging (`store.log_metrics(...)`)

### v0.3 — multi-agent and LLM pipelines
- [ ] Cross-agent provenance tracking via `chain_id`
- [ ] Prompt versioning and lineage
- [ ] RAG corpus snapshot and drift detection
- [ ] Streamlit dashboard for lineage exploration

---

## Testing

```bash
# No pytest needed — stdlib runner included
python run_tests.py

# With pytest (if installed)
pytest tests/unit/ -v
pytest tests/unit/ --cov=src/datalineageml --cov-report=term-missing
```

All 19 tests use `tempfile` so every test gets its own isolated SQLite file. No test ever affects another.

---

## Project structure

```
data-lineage-ml/
├── src/datalineageml/
│   ├── __init__.py              ← public API: track, LineageContext, LineageStore, LineageGraph
│   ├── trackers/
│   │   ├── decorator.py         ← @track implementation
│   │   └── context.py           ← LineageContext
│   ├── storage/
│   │   └── sqlite_store.py      ← SQLite persistence (zero deps)
│   ├── visualization/
│   │   └── graph.py             ← NetworkX + Plotly DAG
│   └── integrations/
│       └── pandas_integration.py
├── tests/unit/
│   ├── test_store.py            ← 5 store tests
│   ├── test_decorator.py        ← 11 decorator tests
│   └── test_context.py          ← 3 context tests
├── examples/
│   └── basic_pipeline.py        ← runnable end-to-end demo
├── run_tests.py                 ← stdlib test runner (no pytest needed)
├── pyproject.toml
├── LICENSE (MIT)
├── CONTRIBUTING.md
└── README.md
```

---

## Contributing

Contributions welcome — especially the v0.2 causal attribution modules and new framework integrations (sklearn Pipeline, PyTorch DataLoader).

```bash
git clone https://github.com/adejumobioluwafemi/data-lineage-ml.git
cd data-lineage-ml
pip install -e ".[dev]"

black src/ tests/
pytest tests/unit/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## Research

DataLineageML is part of an active research programme on **causal data provenance for AI safety**.

Core research question: *Can automated pipeline-level data provenance graphs reliably identify the specific data transformation responsible for a measurable demographic bias in a trained model — and can counterfactual replay of that step verify the remediation?*

If you are working on fairness attribution, causal ML, data governance, or responsible AI in low-resource settings, collaborations are welcome.

📄 [Read the research framing paper](docs/DataLineageML_Research_Framing.pdf)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Oluwafemi Adejumobi** — AI/ML Engineer & Researcher, Ibadan, Nigeria  
[GitHub](https://github.com/adejumobioluwafemi) · [LinkedIn](https://linkedin.com/in/YOUR_HANDLE) · [HuggingFace](https://huggingface.co/YOUR_HANDLE)

---

*Part of our portfolio of AI safety and trustworthy ML tools.*  
*See also: [EquiTrace](https://github.com/adejumobioluwafemi/equitrace) (LLM bias detection) · [PrivacyAudit](https://github.com/adejumobioluwafemi/privacy-audit) · [AgentTrace](https://github.com/adejumobioluwafemi/agent-trace)*