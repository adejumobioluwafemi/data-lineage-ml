"""
Unit tests for LineageStore.
Run with: pytest tests/unit/test_store.py -v
"""

import pytest
import os
from datalineageml.storage.sqlite_store import LineageStore


@pytest.fixture
def store(tmp_path):
    """Fresh in-memory store for each test."""
    db = str(tmp_path / "test_lineage.db")
    s = LineageStore(db_path=db)
    yield s
    s.close()


def test_store_creates_tables(store):
    steps = store.get_steps()
    assert isinstance(steps, list)


def test_log_and_retrieve_step(store):
    store.log_step(
        run_id="run-001",
        step_name="test_step",
        fn_module="test_module",
        fn_qualname="test_fn",
        input_hashes={"arg_0": "abc123"},
        output_hash="def456",
        duration_ms=12.5,
        started_at="2026-03-16T07:00:00",
        status="success",
        error=None,
        tags={"stage": "test"},
    )
    steps = store.get_steps()
    assert len(steps) == 1
    assert steps[0]["step_name"] == "test_step"
    assert steps[0]["status"] == "success"
    assert steps[0]["output_hash"] == "def456"


def test_filter_steps_by_name(store):
    for name in ["step_a", "step_b", "step_a"]:
        store.log_step(
            run_id="x", step_name=name, fn_module="m", fn_qualname="f",
            input_hashes={}, output_hash="h", duration_ms=1.0,
            started_at="2026-03-16T07:00:00", status="success",
            error=None, tags={},
        )
    assert len(store.get_steps("step_a")) == 2
    assert len(store.get_steps("step_b")) == 1


def test_clear_wipes_all_records(store):
    store.log_step(
        run_id="r", step_name="s", fn_module="m", fn_qualname="f",
        input_hashes={}, output_hash="h", duration_ms=1.0,
        started_at="2026-03-16T07:00:00", status="success",
        error=None, tags={},
    )
    store.clear()
    assert store.get_steps() == []


def test_pipeline_lifecycle(store):
    store.log_pipeline_start(
        pipeline_id="pipe-001", name="my_pipeline",
        started_at="2026-03-16T07:00:00",
    )
    pipelines = store.get_pipelines()
    assert len(pipelines) == 1
    assert pipelines[0]["status"] == "running"

    store.log_pipeline_end(
        pipeline_id="pipe-001", status="success",
        ended_at="2026-03-16T07:05:00",
    )
    pipelines = store.get_pipelines()
    assert pipelines[0]["status"] == "success"
    assert pipelines[0]["ended_at"] is not None
