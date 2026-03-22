# Changelog

All notable changes to DataLineageML will be documented here.

Format: [Semantic Versioning](https://semver.org/)

---

## [0.1.0] — 2026-03-18

### Added
- `@track` decorator with automatic input/output hashing for DataFrames, NumPy arrays, dicts, and primitives
- `LineageStore` — SQLite-backed persistence with zero mandatory dependencies
- `LineageContext` — context manager for grouping steps into named pipeline runs
- `LineageGraph` — interactive Plotly + NetworkX lineage DAG visualization
- Pandas integration helpers: `tracked_read_csv`, `tracked_merge`
- 16 unit tests covering all core components
- Full documentation and examples

### Dependencies
- Core: zero (stdlib only — `sqlite3`, `hashlib`, `uuid`, `json`)
- Optional viz: `networkx>=3.0`, `plotly>=5.0`
- Optional pandas: `pandas>=1.5`
