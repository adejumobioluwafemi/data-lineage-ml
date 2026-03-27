# Contributing to DataLineageML

DataLineageML is an active research project. v0.2 is complete. Contributions are welcome at every level.

---

## Table of contents

- [Project direction](#project-direction)
- [Getting started](#getting-started)
- [What we need most](#what-we-need-most)
- [Research vs polish — honest assessment](#research-vs-polish--honest-assessment)
- [What to leave for the maintainer](#what-to-leave-for-the-maintainer)
- [Module guide](#module-guide)
- [Code style](#code-style)
- [Writing tests](#writing-tests)
- [Opening issues](#opening-issues)
- [Opening pull requests](#opening-pull-requests)

---

## Project direction

> DataLineageML is not an observability tool. It is a causal attribution engine. Given a fairness failure in an AI system, it identifies the exact data transformation step responsible and lets you replay the pipeline without it to verify the fix.

Three questions every new feature should answer before being built:

1. Does it help a user find *which pipeline step* caused a safety failure?
2. Does it work offline with zero cloud dependencies?
3. Does it keep the core API simple enough that someone can instrument a pipeline in under five minutes?

---

## Getting started

```bash
git clone https://github.com/adejumobioluwafemi/data-lineage-ml.git
cd data-lineage-ml
pip install -e ".[dev]"

# Verify
python run_tests.py      # stdlib runner, no pytest needed
pytest tests/unit/ -v    # if you have pytest installed
```

Expected: 318/318 tests passing.

---

## What we need most

### Priority 1 — CLI (very high value, 2 days)

The single highest-impact contribution. A user who wants to audit an existing `pipeline.db` file should not have to write any Python code.

```bash
datalineageml audit pipeline.db --sensitive gender
datalineageml audit pipeline.db --sensitive gender --output report.html
datalineageml compare run_1.db run_2.db --sensitive gender
```

Implementation: `src/datalineageml/cli.py` using `argparse`. Entry point in `pyproject.toml`. The audit command runs `ShiftDetector`, `CausalAttributor`, and `generate_report` on the provided store path.

### Priority 2 — sklearn Pipeline wrapper (medium value, 3 days)

```python
from datalineageml.integrations.sklearn_integration import tracked_pipeline

pipeline = tracked_pipeline(
    sklearn.pipeline.Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier()),
    ]),
    sensitive_cols=["gender"],
)
pipeline.fit(X, y)
# Every fit_transform step is automatically tracked
```

### Priority 3 — Wasserstein distance (research contribution, 2 days)

An alternative to Jensen-Shannon divergence with better properties for continuous distributions and ordinal categories. Add as `test="wasserstein"` option in `ShiftDetector.detect()`.

### Priority 4 — Multi-step attribution (research contribution, 4 days)

Currently `CausalAttributor` attributes bias to a single step. Real pipelines often have multiple contributing steps — e.g. a `dropna()` that removes 20% of female records, followed by a `filter()` that removes another 10%. The attributor should return a ranked list with partial attribution weights.

### Priority 5 — Bug reports

A minimal reproducible example in an issue is worth more than a PR without one.

---

## Research vs polish — honest assessment

This is the full picture of what is genuinely novel vs what is useful tooling:

| Feature | Type | Why it matters |
|---|---|---|
| CLI | Polish/DX | Makes the tool usable without writing code. Very high user impact. |
| sklearn wrapper | Integration | Broadens audience to the largest ML ecosystem. |
| Wasserstein distance | Research | Better metric for ordinal/continuous sensitive columns (age groups, income bands). |
| Multi-step attribution | Research | Currently attributes to one step — real bias often has multiple sources. This is the hardest open problem. |
| Causal graph discovery | Research | Automatically infer which upstream steps influence each downstream step using structural causal models |
| Cross-agent provenance | Research | Track demographic shifts through LLM fine-tuning pipelines where data flows across multiple agents. |
| Formal verification of counterfactual claims | Research | Use do-calculus to prove attribution claims rather than empirically verify them. |
| Streamlit dashboard | Polish | Nice but not novel — other tools already do this well. |
| Async step support in replayer | Feature | Adds complexity without enabling new use cases at v0.2 scale. |
| Cloud backends | Out of scope | The offline-first design is a core feature, not a limitation. |

---

## What to leave for the maintainer

Do not open PRs for the following without prior discussion:

- **SQLite schema changes** — every schema change needs a migration strategy for existing `.db` files.
- **Renaming public API** — `track`, `LineageContext`, `LineageStore`, `LineageGraph`, `CounterfactualReplayer`, `generate_report`. New parameters may be added; existing ones cannot be removed without a deprecation cycle.
- **The core attribution algorithms** — JSD thresholds, confidence scoring weights, attribution ranking. These are research parameters documented in the framing paper. Algorithmic changes should start as a Discussion.
- **New visualization backends** — Plotly is the only supported backend.
- **Cloud/server backends** — will not be merged until v1.0 at the earliest.

---

## Module guide

```
src/datalineageml/
│
├── trackers/              ← HOW steps are captured
│   ├── decorator.py       ← @track + __track_meta__
│   └── context.py         ← LineageContext
│
├── storage/               ← WHERE data is persisted
│   └── sqlite_store.py    ← SQLite: steps, pipelines, snapshots, metrics
│
├── visualization/         ← HOW lineage is displayed
│   └── graph.py           ← Plotly + NetworkX DAG
│
├── integrations/          ← HOW third-party libs plug in
│   └── pandas_integration.py
│
├── analysis/              ← WHY failures happened
│   ├── profiler.py        ← DataFrameProfiler
│   ├── shift_detector.py  ← ShiftDetector (JSD + KS)
│   ├── attributor.py      ← CausalAttributor
│   ├── cross_run.py       ← CrossRunComparator
│   ├── metrics.py         ← DPG, EO, PP, RegressionFairnessAuditor
│   └── sensitive_cols.py  ← discover_sensitive_cols
│
├── replay/                ← PROVING the fix works
│   └── replayer.py        ← CounterfactualReplayer
│
└── report.py              ← HTML audit report export
```

---

## Code style

```bash
black src/ tests/ --line-length 100
ruff check src/ tests/
```

Rules:
- Line length: 100 characters
- Type hints on all public function signatures
- Docstring on every public class and method (Google style)
- No `print()` in library code — use `warnings.warn()` for user-facing messages
- No hardcoded file paths — always use `tmp_path` (pytest) or `tempfile` (stdlib)
- No `datetime.utcnow()` — use `datetime.now(timezone.utc)` (Python 3.12+ compatibility)

---

## Writing tests

Every feature needs tests. Every bug fix needs a test that would have caught the bug.

```python
# Always use tmp_path (pytest) or tempfile — never hardcode a path
@pytest.fixture
def store(tmp_path):
    s = LineageStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()
```

Test naming:

```
test_<what_it_does>            ← preferred
test_<component>_<behaviour>   ← also good
test_thing                     ← too vague
test_case_1                    ← never
```

What every test for a new public function must cover:
1. Happy path
2. Failure path (correct exception, correct message)
3. Edge case (empty DataFrame, None value, single row)
4. Idempotency where applicable

What tests must never do:
- Touch the network
- Write to a fixed path
- Depend on execution order
- Import optional dependencies without a try/except guard

Run both the stdlib runner and pytest before every PR — they test different things:

```bash
python run_tests.py      # fast, no deps, catches regressions in the original 64 tests
pytest tests/unit/ -v    # full suite including parametrized and fixture-based tests
```

---

## Opening issues

```
Python version:   3.x.x
OS:               macOS / Linux / Windows
Package version:  0.x.x

What I expected:
One sentence.

What happened instead:
One sentence. Include the full traceback.

Minimal reproducible example:
```python
import datalineageml as dlm
from datalineageml import track
dlm.init(db_path=":memory:")

@track(name="test")
def fn(x): return x

fn(1)  # ← error happens here
```

Additional context:
```

---

## Opening pull requests

Checklist before opening:

- [ ] `python run_tests.py` passes
- [ ] `pytest tests/unit/ -v` passes
- [ ] `black src/ tests/ --line-length 100` run
- [ ] `ruff check src/ tests/` clean
- [ ] Every new public function has type hints and a docstring
- [ ] `README.md` updated if the public API changed
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`
- [ ] PR description explains *why* the change is needed

PR title format:

```
feat: add CLI audit command
fix: KS threshold miscalibrated for small datasets
docs: update CONTRIBUTING with v0.2 module guide
test: add edge cases for empty DataFrame in profiler
refactor: extract _hash_input into standalone module
```

One feature or fix per PR.

---

## Questions?

Open a GitHub Discussion for design direction, research framing, or "is this a good idea?" questions. Issues are for bugs and confirmed feature requests only.

---

*DataLineageML — causal data provenance for AI safety.*
*Built by Oluwafemi Philip Adejumobi · Ibadan, Nigeria · MIT License*