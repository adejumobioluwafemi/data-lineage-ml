# Changelog

All notable changes to DataLineageML are documented here.
Format follows [Semantic Versioning](https://semver.org/).

---

## [0.2.0] — 2026-03-26

### Summary

v0.2 completes the causal attribution loop: pipeline runs → demographic snapshots → shift detection → step attribution → counterfactual proof → report export. Every component works offline with zero mandatory cloud dependencies.

### Added — core tracking

- `dlm.init(db_path=...)` — global default store. One call at the top of your script; no `store=` argument needed anywhere else.
- `dlm.reset()` — clear the global store (for tests and scripting).
- `@track(snapshot=True, sensitive_cols=[...])` — automatic demographic snapshot logging before and after any decorated step.
- `__track_meta__` on every `@track` wrapper — stores name, snapshot, and sensitive_cols so `register_tracked()` can read them without re-specification.
- SQLite schema extended with two new tables: `snapshots` and `metrics`.
- `store.log_snapshot(...)` / `store.get_snapshots(step_name=, position=)`.
- `store.log_metrics(run_id, metrics={...}, metric_source=, step_name=, tags=)` / `store.get_metrics(run_id=, metric_name=)`.
- `store.clear()` now wipes snapshots and metrics tables as well as steps and pipelines.

### Added — analysis

- `DataFrameProfiler` — computes shape, null rates, numeric stats (mean/std/p25/p75), categorical value counts, and sensitive column distributions. Called automatically by `@track(snapshot=True)` or directly.
- `print_snapshot_comparison(before, after)` — human-readable before/after demographic distribution view.
- `ShiftDetector` — ranks pipeline steps by distribution shift magnitude. Two statistical tests:
  - Jensen-Shannon divergence for categorical/sensitive columns (bounded [0,1], log base 2).
  - Kolmogorov-Smirnov D-statistic approximation from stored percentile stats for numeric columns.
  - Thresholds calibrated against real demographic shifts: HIGH ≥ 0.02 JSD, MEDIUM ≥ 0.005 JSD.
- `CausalAttributor` — correlates shift scores with a bias metric to produce a ranked attribution. Scoring: 70% JSD weight + 30% removal rate weight. Confidence = top_score / (top_score + second_score). Recommendation engine maps step name patterns (dropna, filter, merge, sample...) to specific remediation advice.
- `CrossRunComparator` — reads snapshot history to detect demographic drift across multiple pipeline runs over time. `compare_step()` and `trend()` (worsening / stable / improving via OLS slope).
- `DemographicParityGap` — binary and multiclass (one-vs-rest, macro-averaged). Does not require a trained model. Includes 4/5ths rule flag.
- `EqualizedOdds` — binary and multiclass (TPR/FPR gap, macro-averaged). Requires model predictions.
- `PredictiveParity` — binary and multiclass (precision gap, macro-averaged). Requires model predictions.
- `RegressionFairnessAuditor` — four diagnostics: Mean Error gap, MAE gap, calibration gap, and lazy solution guard (residual_std / outcome_std ≥ threshold).
- `FairnessResult.to_store_kwargs(run_id, step_name)` — direct logging to the lineage store.
- `RegressionFairnessReport.to_store_kwargs(run_id, step_name)` — logs me_gap, mae_gap, calibration_gap.
- `compute_metric(name, df, ...)` — dispatch any classification metric by short name ('dpg', 'eo', 'pp').
- `discover_sensitive_cols(df)` — heuristic sensitive column discovery using name matching (including Nigerian/West African administrative terminology), cardinality checking, and distribution unevenness. Returns `SensitiveColCandidate` list sorted by confidence.
- `suggest_sensitive_cols(df, min_confidence=0.6)` — convenience wrapper returning column name strings.
- `print_sensitive_col_report(df)` — formatted discovery report with suggested `sensitive_cols=` value.

### Added — counterfactual replay

- `CounterfactualReplayer` — re-runs the full pipeline twice (biased + fixed), compares demographic distributions and bias metrics, persists counterfactual snapshots, issues STRONG / MODERATE / WEAK / INCONCLUSIVE verdict.
- `replayer.register(name, fn, snapshot=, sensitive_cols=)` — explicit registration.
- `replayer.register_tracked(fn)` — reads `__track_meta__` from `@track` wrapper, eliminating double-specification.
- Chaining API: `CounterfactualReplayer().register_tracked(a).register_tracked(b)`.
- Schema validation: raises `ValueError` if replacement function changes the column schema before any downstream step runs.
- Counterfactual snapshots persisted to store with `__counterfactual` step name suffix.

### Added — report export

- `generate_report(store, output_path, ...)` — self-contained HTML audit report. Contains pipeline summary, shift table, demographic distribution charts (pure HTML/CSS, no external resources), attribution finding, counterfactual comparison, and logged metrics. Safe to email or attach to regulatory submissions. XSS-safe (all user strings escaped).

### Changed

- `ShiftDetector.__init__` parameters renamed for clarity: `high_threshold` → `jsd_high`, `medium_threshold` → `jsd_medium`. New parameters: `ks_high`, `ks_medium`.
- `ShiftDetector.detect()` results now include `test` ("jsd" or "ks") and `stat` (the relevant statistic) fields. The old `js_divergence` field is removed — use `stat`.
- `_log_snapshot_safe()` in `decorator.py` now delegates to `DataFrameProfiler` instead of duplicating computation inline.

### Fixed

- `PredictiveParity` error message inconsistency: "Need ≥2 groups" → "Need at least 2 groups" (aligns with all other error messages).
- Lazy solution detection in `RegressionFairnessAuditor` — previously used absolute residual std threshold (wrong); now uses `residual_std / outcome_std` ratio (correct).
- `_build_finding()` in `shift_detector.py` — `NameError` on `rows_removed` variable.

### Tests

- 318 unit tests (was 19 in v0.1). Zero test order dependencies. All tests use `tempfile` for isolation.
- New test files: `test_profiler.py`, `test_shift_detector.py`, `test_ks.py`, `test_attributor.py`, `test_global_store.py`, `test_replayer.py`, `test_metrics.py`, `test_sensitive_cols.py`, `test_cross_run.py`, `test_report.py`.

---

## [0.1.0] — 2026-03-18

### Added

- `@track` decorator with automatic input/output hashing for DataFrames, NumPy arrays, dicts, and primitives.
- `LineageStore` — SQLite-backed persistence with zero mandatory dependencies. Tables: `steps`, `pipelines`.
- `LineageContext` — context manager for grouping steps into named pipeline runs. Marks pipeline as "failed" if any step raises.
- `LineageGraph` — interactive Plotly + NetworkX lineage DAG visualisation. Green = success, red = failed.
- `tracked_read_csv`, `tracked_merge` — pandas integration helpers.
- 19 unit tests covering all core components. Stdlib runner (`run_tests.py`) — no pytest needed.

### Dependencies

- Core: zero mandatory (uses only `sqlite3`, `hashlib`, `uuid`, `json`, `functools`).
- Optional viz: `networkx>=3.0`, `plotly>=5.0`.
- Optional pandas: `pandas>=1.5`.