# Contributing to DataLineageML

DataLineageML is an active research project building toward **causal data provenance for AI safety**. Contributions are welcome at every level — from fixing a typo to implementing a causal attribution module.

---

## Table of contents

- [Understanding the project direction](#understanding-the-project-direction)
- [Getting started](#getting-started)
- [Running tests](#running-tests)
- [What we need most](#what-we-need-most)
- [What to leave for the maintainer](#what-to-leave-for-the-maintainer)
- [Module guide](#module-guide)
- [Adding a new feature](#adding-a-new-feature)
- [Code style](#code-style)
- [Writing tests](#writing-tests)
- [Opening issues](#opening-issues)
- [Opening pull requests](#opening-pull-requests)

---

## Understanding the project direction

Before contributing, read the one-paragraph positioning in the README:

> DataLineageML is not an observability tool. It is a causal attribution engine. Given a safety or fairness failure in an AI system, it identifies the exact data transformation step responsible and lets you replay the pipeline without it to verify the fix.

This shapes every contribution decision. If a feature makes the tool better at **observability** (recording what happened) but not at **causal attribution** (explaining why it happened), it is probably not the right thing to build here. MLflow already does observability excellently.

The three questions every new feature should answer:
1. Does this help a user find *which pipeline step* caused a safety failure?
2. Does this work offline with zero cloud dependencies?
3. Does this keep the core API simple enough that someone can instrument a pipeline in under five minutes?

---

## Getting started

```bash
git clone https://github.com/adejumobioluwafemi/data-lineage-ml.git
cd data-lineage-ml
pip install -e ".[dev]"
```

Verify the install:

```bash
python run_tests.py
# Expected: 19/19 tests passed ✓
```

If you have pytest installed:

```bash
pytest tests/unit/ -v
```

---

## Running tests

```bash
# Stdlib runner — no dependencies needed
python run_tests.py

# pytest (if installed)
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=src/datalineageml --cov-report=term-missing

# Single test file
pytest tests/unit/test_decorator.py -v
```

All tests use `tempfile.NamedTemporaryFile` so each test gets its own isolated SQLite file. No global state, no file cleanup needed, no test order dependencies.

---

## What we need most

Contributions are prioritised in this order. Start from the top.

### 1. v0.2 causal attribution modules (highest priority)

These are the novel research features — nothing like this exists in MLflow, W&B, or LangSmith.

**`src/datalineageml/analysis/profiler.py`** — `DataFrameProfiler`
Computes a statistical snapshot of a DataFrame: shape, null rates, numeric distributions (mean/std/min/max/p25/p75), categorical value counts (top 10), and critically — the distribution of any tagged `sensitive_cols` (gender, age group, region, etc.). Called automatically inside `@track` when `snapshot=True`.

```python
@track(name="clean_data", snapshot=True, sensitive_cols=["gender", "zone"])
def clean_data(df):
    return df.dropna()
# Automatically stores demographic snapshot before and after
```

**`src/datalineageml/analysis/shift_detector.py`** — `ShiftDetector`
Compares demographic snapshots between adjacent pipeline steps. Uses Jensen-Shannon divergence for categorical columns and the Kolmogorov-Smirnov test for numeric ones (both in `scipy.stats`). Returns shift scores per column per step, with HIGH/MEDIUM/LOW flags.

**`src/datalineageml/analysis/attributor.py`** — `CausalAttributor`
Takes a set of fairness/safety metrics and the shift scores, ranks steps by attribution likelihood, and produces a finding: which step most likely caused the metric degradation, with what confidence, and why.

**`src/datalineageml/replay/replayer.py`** — `CounterfactualReplayer`
Re-executes the pipeline from a flagged step with a replacement function, re-measures the metric, and reports the before/after delta. This is the "proof" step — attribution is only reported when counterfactual evidence confirms it.

### 2. Global default store (`dlm.init()`)

Currently every `@track` call needs `store=STORE` explicitly. A module-level default store fixes this:

```python
import datalineageml as dlm
dlm.init(db_path="pipeline.db")  # called once at top of script

@dlm.track(name="clean")         # no store= needed
def clean(df): ...
```

Implementation: a module-level `_DEFAULT_STORE` variable in `src/datalineageml/__init__.py`, set by `dlm.init()`. The `@track` decorator falls back to `_DEFAULT_STORE` if no explicit `store=` is passed. Must maintain full backwards compatibility with explicit `store=` passing.

### 3. New framework integrations

- **scikit-learn Pipeline wrapper** — wrap `sklearn.pipeline.Pipeline` so every `fit_transform` step is automatically tracked
- **PyTorch DataLoader** — track dataset versions passed into training loops
- **HuggingFace `datasets`** — track `Dataset` objects (hash the dataset fingerprint, not the full data)

### 4. Additional hashing strategies

The current `_hash_input()` in `decorator.py` handles pandas DataFrames, NumPy arrays, dicts, and primitives. Common edge cases that need coverage:

- `scipy.sparse` matrices
- `polars` DataFrames
- Python dataclasses and Pydantic models
- File paths — hash the file content, not the path string

### 5. Bug reports

If something is broken, a minimal reproducible example in an issue is more valuable than a PR without one. See [Opening issues](#opening-issues).

---

## What to leave for the maintainer

Do not open PRs for the following without prior discussion in an issue:

- **SQLite schema changes** — backwards compatibility is critical. Every schema change requires a migration strategy.
- **Changing the public API** — `track`, `LineageContext`, `LineageStore`, `LineageGraph` — the signatures of these are stable. New parameters may be added but existing ones cannot be removed or renamed without a deprecation cycle.
- **New visualization backends** — Plotly is the only supported backend. Adding matplotlib, Bokeh, etc. is scope creep unless there is strong evidence of user demand.
- **Cloud/server backends** — the zero-dependency, offline-first design is a core feature, not a limitation. PRs adding cloud storage backends will not be merged until v1.0 at the earliest.
- **The causal attribution algorithms themselves** — `ShiftDetector`, `CausalAttributor`, and `CounterfactualReplayer` are research contributions. The algorithmic choices (Jensen-Shannon vs Wasserstein, confidence thresholds, attribution ranking) will be decided by the maintainer based on the research framing paper. Contributions to the scaffolding, tests, and API design are welcome; contributions to the core algorithms should start as a discussion.

---

## Module guide

Use this map when deciding where to put new code:

```
src/datalineageml/
│
├── trackers/              ← HOW steps are captured
│   ├── decorator.py       ← @track — the core instrument
│   └── context.py         ← LineageContext — pipeline grouping
│
├── storage/               ← WHERE data is persisted
│   └── sqlite_store.py    ← SQLite backend (zero deps)
│
├── visualization/         ← HOW lineage is displayed
│   └── graph.py           ← NetworkX + Plotly DAG
│
├── integrations/          ← HOW third-party libs plug in
│   └── pandas_integration.py
│
├── analysis/              ← WHY failures happened (v0.2)
│   ├── profiler.py        ← Statistical snapshot per step
│   ├── shift_detector.py  ← Distribution shift scoring
│   └── attributor.py      ← Causal step attribution
│
└── replay/                ← PROVING the fix works (v0.2)
    └── replayer.py        ← Counterfactual pipeline replay
```

New code that does not fit cleanly into one of these modules is a signal that it may be out of scope.

---

## Adding a new feature

### Step 1 — open an issue first

For anything beyond a bug fix or typo, open an issue describing what you want to build and why. This avoids spending a week on something that will not be merged.

### Step 2 — write the test first

DataLineageML uses a test-first approach for new features. Before writing implementation code, write a test that fails because the feature does not exist yet. This forces you to design the API from the user's perspective.

```python
# tests/unit/test_profiler.py  — write this BEFORE profiler.py exists

def test_profiler_captures_gender_distribution(store, tmp_path):
    import pandas as pd
    from datalineageml.analysis.profiler import DataFrameProfiler

    df = pd.DataFrame({
        "age": [25, 32, 45, 28, 38],
        "gender": ["F", "M", "M", "F", "M"],
        "income": [50000, 72000, 65000, 48000, 91000],
    })

    profiler = DataFrameProfiler(sensitive_cols=["gender"])
    snapshot = profiler.profile(df, step_name="test_step", run_id="run-001")

    assert snapshot["sensitive"]["gender"]["F"] == pytest.approx(0.4)
    assert snapshot["sensitive"]["gender"]["M"] == pytest.approx(0.6)
```

### Step 3 — implement the minimum that makes the test pass

Do not implement features beyond what the test requires. Add tests incrementally as you extend the feature.

### Step 4 — check the full suite still passes

```bash
python run_tests.py
# All existing 19 tests must still pass, plus your new ones
```

### Step 5 — update documentation

- Add a docstring to every public class and method
- Update `README.md` if the feature changes the public API
- Update the relevant section of `CONTRIBUTING.md` if you add a new module

---

## Code style

```bash
# Format (required before every PR)
black src/ tests/ --line-length 100

# Lint (required — fix all warnings)
ruff check src/ tests/
```

Rules:
- Line length: 100 characters
- Type hints on all public function signatures
- Docstring on every public class and method (Google style)
- No `print()` in library code — use `warnings.warn()` for user-facing messages
- No hardcoded file paths anywhere — always use `tmp_path` or `tempfile` in tests

Example of acceptable docstring style:

```python
def log_step(self, *, run_id: str, step_name: str, ...) -> None:
    """Persist a completed pipeline step to the lineage store.

    Args:
        run_id: UUID identifying this specific function call.
        step_name: Human-readable name of the transformation step.
        ...

    Raises:
        sqlite3.OperationalError: If the database is locked or corrupted.
    """
```

---

## Writing tests

Every new feature needs tests. Every bug fix needs a test that would have caught the bug.

**Required patterns:**

```python
# Always use tempfile — never hardcode a path
import tempfile
import os

def setUp(self):
    self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    self.store = LineageStore(db_path=self.tmp.name)

def tearDown(self):
    self.store.close()
    os.unlink(self.tmp.name)
```

**Test naming convention:**

```
test_<what it does>           ← good
test_<component>_<behaviour>  ← also good

test_thing                    ← too vague
test_case_1                   ← never
```

**What every test must cover for a new public function:**

1. The happy path — does it work correctly with valid input?
2. The failure path — does it raise the right exception with invalid input?
3. The edge case — empty DataFrame, None values, single-row dataset, etc.
4. Idempotency where applicable — running it twice produces the same result

**What tests must NOT do:**

- Touch the network
- Write to a fixed path (always use `tempfile`)
- Depend on test execution order
- Import optional dependencies without a try/except guard

---

## Opening issues

Use this template:

```
**Python version:** 3.x.x
**OS:** macOS / Linux / Windows
**DataLineageML version:** 0.x.x

**What I expected:**
One sentence.

**What happened instead:**
One sentence. Include the full traceback if there is one.

**Minimal reproducible example:**

```python
# The smallest possible code that reproduces the issue
from datalineageml import track, LineageStore

store = LineageStore(db_path=":memory:")

@track(name="test", store=store)
def fn(x):
    return x

fn(1)  # ← error happens here
```

**Additional context:**
Anything else that might be relevant.
```

---

## Opening pull requests

Before opening a PR:

- [ ] `python run_tests.py` passes (all existing + new tests)
- [ ] `black src/ tests/ --line-length 100` has been run
- [ ] `ruff check src/ tests/` passes with no warnings
- [ ] Every new public function has a type-hinted signature and docstring
- [ ] `README.md` is updated if the public API changed
- [ ] The PR description explains *why* the change is needed, not just *what* it does

PR title format:

```
feat: add DataFrameProfiler with sensitive column tracking
fix: store fragmentation when store= omitted from @track
docs: extend CONTRIBUTING with v0.2 module guide
test: add edge cases for empty DataFrame hashing
refactor: extract _hash_input into standalone module
```

Keep PRs focused. One feature or fix per PR. A PR that adds a profiler AND fixes a visualization bug AND updates the README should be three PRs.

---

## Questions?

Open a GitHub Discussion (not an issue) for questions about design direction, research framing, or "is this a good idea?" conversations. Issues are for bugs and confirmed feature requests only.

---

*DataLineageML — causal data provenance for AI safety.*  
*Built by Oluwafemi Adejumobi · Ibadan, Nigeria · MIT License*