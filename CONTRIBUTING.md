# Contributing to DataLineageML

Thank you for your interest in contributing!

## Getting started

```bash
git clone https://github.com/adejumobioluwafemi/data-lineage-ml.git
cd data-lineage-ml
pip install -e ".[dev]"
```

## Before submitting a PR

```bash
# Format
black src/ tests/

# Lint
ruff check src/ tests/

# Tests — all must pass
pytest tests/unit/ -v

# Coverage — aim for >90% on new code
pytest tests/unit/ --cov=src/datalineageml --cov-report=term-missing
```

## What we need most

- New integrations (scikit-learn Pipeline, PyTorch DataLoader, Spark)
- Additional hashing strategies for edge-case types
- Bug reports with a minimal reproducible example

## What to leave for the maintainer

- Changes to the SQLite schema (backwards compatibility is critical)
- New visualization backends
- Breaking changes to the public API (`track`, `LineageContext`, `LineageStore`, `LineageGraph`)

## Code style

- Black formatting, line length 100
- Type hints on all public functions
- Docstring on every public class and method
- Tests for every new feature — use `tmp_path` fixture, never a hardcoded path

## Opening issues

Please include:
1. Python version and OS
2. Minimal code that reproduces the issue
3. Expected vs actual behaviour
