# DataLineageML

> **Causal data provenance for AI safety.**
> Find out *which pipeline step caused your model's bias or safety failure* — automatically, verifiably, and without a cloud account.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-orange.svg)](https://pypi.org/project/datalineageml/)
[![Tests](https://img.shields.io/badge/tests-318%20passing-brightgreen.svg)]()
[![Zero deps](https://img.shields.io/badge/core_deps-zero-blue.svg)]()

---

## The problem this solves

Your model is biased. Or it started failing after a data update. Or an audit found a fairness gap. You know *what* went wrong — but not *where* in the pipeline it went wrong.

Existing tools answer the wrong question. MLflow, W&B, and LangSmith tell you **what happened** inside a pipeline — inputs, outputs, latencies. They do not tell you **which data transformation caused the outcome you are trying to fix**.

DataLineageML is built for that second question.

---

## A real example

A crop yield model in Oyo State, Nigeria was producing a 34% gender bias in agricultural subsidy allocation. Three months of manual audit found nothing. DataLineageML found the cause in one run:

```
Step: clean_data (df.dropna())

  Input:   female-headed farms = 38.4%  (n = 12,847)
  Output:  female-headed farms = 22.1%  (n =  7,403)

  Jensen-Shannon divergence on gender: 0.079  [HIGH]

  Finding: 'gender' distribution shifted significantly at 'clean_data'.
           'F' proportion dropped from 38.4% to 22.1%
           (5,444 rows removed, 42.4% of dataset).
           This step is a candidate causal source of bias.

Counterfactual replay — replace dropna() with stratified imputation:
  Original bias score:  0.34  →  After fix: 0.09   (−74%)
  Accuracy change:      −0.3%  (negligible)
```

The fix took one afternoon.

---

## Quick start

```bash
pip install datalineageml
```

```python
import datalineageml as dlm
from datalineageml import track, LineageContext
from datalineageml.analysis import (
    ShiftDetector, CausalAttributor,
    discover_sensitive_cols,
)
from datalineageml.replay import CounterfactualReplayer

# 1. Initialise — one call, no store= needed anywhere else
dlm.init(db_path="pipeline.db")

# 2. Auto-discover which columns to track
sensitive = discover_sensitive_cols(df)
cols      = [c.column for c in sensitive if c.confidence >= 0.7]
# → ["gender", "age_group", "zone"]

# 3. Instrument your pipeline
@track(name="clean_data", snapshot=True, sensitive_cols=cols)
def clean_data(df):
    return df.dropna()

@track(name="engineer_features")
def engineer_features(df):
    df["score_per_ha"] = df["score"] / df["farm_size_ha"]
    return df

# 4. Run it
with LineageContext(name="subsidy_model_v2"):
    d2 = clean_data(raw_df)
    d3 = engineer_features(d2)

# 5. Detect the shift
store    = dlm.get_default_store()
detector = ShiftDetector(store=store)
shifts   = detector.detect()
detector.print_report(shifts)
# → [HIGH] clean_data → gender  JSD=0.079

# 6. Attribute the causal step
attr = CausalAttributor(store=store).attribute(sensitive_col="gender")
# → {"attributed_step": "clean_data", "confidence": 1.0, ...}

# 7. Prove the fix
def impute_data(df):
    for g in df["gender"].unique():
        mask = (df["gender"] == g) & df["land_title"].isna()
        df.loc[mask, "land_title"] = df.loc[df["gender"]==g, "land_title"].mode()[0]
    return df

result = (CounterfactualReplayer(store=store)
    .register_tracked(clean_data)       # reads @track metadata automatically
    .register("engineer_features", engineer_features)
    .replay(raw_df, "clean_data", impute_data, "gender",
            bias_metric_fn=lambda df: abs(
                df[df.gender=="F"]["eligible"].mean() -
                df[df.gender=="M"]["eligible"].mean())))

# → Verdict: STRONG — Bias reduced by 74%

# 8. Export a shareable HTML audit report
from datalineageml import generate_report
generate_report(store=store, output_path="audit.html",
                attribution_result=attr, counterfactual_result=result)
```

---

## How it differs from MLflow, W&B, and LangSmith

All three do **observability** — recording what happened. None do **causal attribution** — identifying which step caused a safety failure.

| Capability | MLflow | W&B Weave | LangSmith | DataLineageML |
|---|:---:|:---:|:---:|:---:|
| Pipeline tracing | ✓ | ✓ | ✓ | ✓ |
| Dataset versioning | ✓ | ✓ | ✓ | ✓ |
| Sensitive column auto-discovery | ✗ | ✗ | ✗ | ✓ |
| Demographic snapshots per step | ✗ | ✗ | ✗ | ✓ |
| Distribution shift detection (JSD + KS) | ✗ | ✗ | ✗ | ✓ |
| Causal step attribution | ✗ | ✗ | ✗ | ✓ |
| Counterfactual pipeline replay | ✗ | ✗ | ✗ | ✓ |
| Cross-run drift detection | ✗ | ✗ | ✗ | ✓ |
| Fairness metrics (DPG / EO / PP / Regression) | ✗ | ✗ | ✗ | ✓ |
| HTML audit report export | ✗ | ✗ | ✗ | ✓ |
| Zero-dependency offline core | ✗ | ✗ | ✗ | ✓ |

---

## Full API reference

### Tracking

```python
import datalineageml as dlm
from datalineageml import track, LineageContext, LineageStore, LineageGraph

dlm.init(db_path="pipeline.db")        # global store — one call, optional after

@track(
    name="clean_data",                 # step name (defaults to function name)
    tags={"stage": "prep"},            # arbitrary metadata
    snapshot=True,                     # log demographic snapshots before/after
    sensitive_cols=["gender", "zone"], # columns to track distributions for
)
def clean_data(df): return df.dropna()

with LineageContext(name="pipeline_v1"):
    result = clean_data(raw_df)
```

### Analysis

```python
from datalineageml.analysis import (
    DataFrameProfiler,          # compute + print statistical snapshots
    ShiftDetector,              # detect JSD + KS shifts, ranked by severity
    CausalAttributor,           # attribute a bias to the most likely causal step
    CrossRunComparator,         # detect drift across pipeline runs over time
    DemographicParityGap,       # binary/multiclass: selection rate gap (no model)
    EqualizedOdds,              # binary/multiclass: TPR/FPR gap (requires model)
    PredictiveParity,           # binary/multiclass: precision gap (requires model)
    RegressionFairnessAuditor,  # ME gap, MAE gap, calibration, lazy-solution guard
    compute_metric,             # dispatch any metric by name: 'dpg', 'eo', 'pp'
    discover_sensitive_cols,    # auto-discover demographic columns from data
    suggest_sensitive_cols,     # return column names above confidence threshold
    print_snapshot_comparison,  # human-readable before/after distribution view
)
```

### Counterfactual replay

```python
from datalineageml.replay import CounterfactualReplayer

replayer = (CounterfactualReplayer(store=store)
    .register_tracked(clean_data)          # reads @track metadata automatically
    .register("normalize", normalize_fn))  # or register manually

result = replayer.replay(
    raw_data       = df_raw,
    replace_step   = "clean_data",
    replacement_fn = impute_data,
    sensitive_col  = "gender",
    bias_metric_fn = lambda df: ...,       # optional — any callable(df) → float
)
replayer.print_result(result)
```

### Report export

```python
from datalineageml import generate_report

generate_report(
    store                 = store,
    output_path           = "audit.html",
    pipeline_name         = "oyo_subsidy_v1",
    sensitive_col         = "gender",
    attribution_result    = attr,         # optional
    counterfactual_result = cf_result,    # optional
    title                 = "Fairness Audit — Q1 2026",
)
```

---

## Installation

```bash
pip install datalineageml                    # core — zero mandatory dependencies
pip install "datalineageml[viz]"             # + NetworkX + Plotly lineage graph
pip install "datalineageml[pandas]"          # + pandas integration helpers
pip install "datalineageml[all]"             # everything

# Development
git clone https://github.com/adejumobioluwafemi/data-lineage-ml.git
cd data-lineage-ml
pip install -e ".[dev]"
pytest tests/unit/ -v                        # 318 tests
python run_tests.py                          # stdlib runner, no pytest needed
```

---

## Fairness metrics — which to use

| Metric | Requires model? | Use when |
|---|---|---|
| `DemographicParityGap` | No | Who gets selected at all (access/allocation). Run before training. |
| `EqualizedOdds` | Yes | False negatives and false positives both carry harm (medical, credit). |
| `PredictiveParity` | Yes | Being predicted positive should mean the same thing for all groups. |
| `RegressionFairnessAuditor` | Yes | Continuous output; detects ME gap, MAE gap, calibration error, lazy solutions. |

When base rates differ across groups, Equalized Odds and Predictive Parity cannot both hold simultaneously (Chouldechova 2017). Choosing a metric is a policy decision about which error type is costlier.

---

## Project structure

```
data-lineage-ml/
├── src/datalineageml/
│   ├── __init__.py              ← public API: track, LineageContext, LineageStore,
│   │                               LineageGraph, CounterfactualReplayer, generate_report
│   ├── trackers/
│   │   ├── decorator.py         ← @track (stores __track_meta__ for register_tracked)
│   │   └── context.py           ← LineageContext
│   ├── storage/
│   │   └── sqlite_store.py      ← SQLite (zero deps): steps, pipelines, snapshots, metrics
│   ├── visualization/
│   │   └── graph.py             ← Plotly + NetworkX DAG
│   ├── integrations/
│   │   └── pandas_integration.py
│   ├── analysis/
│   │   ├── profiler.py          ← DataFrameProfiler
│   │   ├── shift_detector.py    ← ShiftDetector (JSD + KS)
│   │   ├── attributor.py        ← CausalAttributor
│   │   ├── cross_run.py         ← CrossRunComparator
│   │   ├── metrics.py           ← DPG, EO, PP, RegressionFairnessAuditor
│   │   └── sensitive_cols.py    ← discover_sensitive_cols, suggest_sensitive_cols
│   ├── replay/
│   │   └── replayer.py          ← CounterfactualReplayer
│   └── report.py                ← generate_report (HTML export)
├── tests/unit/                  ← 318 tests, all isolated
├── examples/
│   ├── basic_pipeline.py        ← full loop demo (5 mins to run)
│   └── agro.ipynb               ← Oyo State case study (complete v0.2 walkthrough)
├── run_tests.py                 ← stdlib runner (no pytest needed)
├── pyproject.toml
├── CHANGELOG.md
├── CONTRIBUTING.md
└── README.md
```

---

## Roadmap

### v0.2 — **released**
- [x] Global `dlm.init()` default store
- [x] Demographic snapshots per step (`snapshot=True`, `sensitive_cols=`)
- [x] Sensitive column auto-discovery (`discover_sensitive_cols`)
- [x] Distribution shift detection — JSD (categorical) + KS approximation (numeric)
- [x] Causal step attribution with confidence scoring
- [x] Fairness metrics — DPG, EO, PP (binary + multiclass), regression audit
- [x] Counterfactual pipeline replayer with `register_tracked()`
- [x] Cross-run drift detection (`CrossRunComparator`)
- [x] HTML audit report export (`generate_report`)
- [x] 318 unit tests

### v0.3 — planned
- [ ] CLI: `datalineageml audit pipeline.db --sensitive gender`
- [ ] sklearn Pipeline wrapper (auto-track every `fit_transform`)
- [ ] Cross-agent provenance tracking via `chain_id`
- [ ] Prompt versioning and lineage for LLM pipelines
- [ ] Wasserstein distance as alternative shift metric
- [ ] Multi-step attribution (currently attributes to a single step)

---

## Research

DataLineageML is part of an active research programme on **causal data provenance for AI safety**.

Research question: *Can automated pipeline-level provenance graphs reliably identify the specific data transformation responsible for a measurable demographic bias — and can counterfactual replay verify the remediation?*

The Oyo State case study demonstrates this end-to-end: a `dropna()` call attributed automatically (JSD = 0.079, confidence 100%), remediated with stratified imputation, bias reduced 74%, verified by counterfactual replay — without training a model.

📄 [Read the research framing paper](docs/DataLineageML_Research_Framing.pdf)

Collaborations welcome — especially on fairness attribution, causal ML, data governance, and responsible AI in low-resource settings.

---

## License

MIT — see [LICENSE](LICENSE).

## Author

**Oluwafemi Adejumobi** — AI/ML Engineer & Researcher, Ibadan, Nigeria
[GitHub](https://github.com/adejumobioluwafemi) · [LinkedIn](https://linkedin.com/in/YOUR_HANDLE) · [HuggingFace](https://huggingface.co/YOUR_HANDLE)

---

*Part of a portfolio of AI safety tools. See also: [EquiTrace](https://github.com/adejumobioluwafemi/equitrace) · [PrivacyAudit](https://github.com/adejumobioluwafemi/privacy-audit) · [AgentTrace](https://github.com/adejumobioluwafemi/agent-trace)*