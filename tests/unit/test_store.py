"""
Tests for LineageStore — original suite + snapshots + metrics.
Run with: python run_tests.py  OR  pytest tests/unit/test_store.py -v
"""

import pytest # type: ignore
import os
import json
import tempfile
from datalineageml.storage.sqlite_store import LineageStore


# fixture 

@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test.db")
    s = LineageStore(db_path=db)
    yield s
    s.close()


# helpers 

def _log_step(store, step_name="test_step", status="success", error=None):
    store.log_step(
        run_id="run-001", step_name=step_name,
        fn_module="m", fn_qualname="f",
        input_hashes={"arg_0": "abc"}, output_hash="def",
        duration_ms=10.0, started_at="2026-03-16T07:00:00",
        status=status, error=error, tags={}
    )


def _log_snapshot(store, run_id="run-001", step_name="clean_data",
                  position="before", row_count=100,
                  sensitive_stats=None):
    store.log_snapshot(
        run_id=run_id,
        step_name=step_name,
        position=position,
        row_count=row_count,
        column_count=3,
        column_names=["age", "gender", "income"],
        null_rates={"age": 0.0, "gender": 0.05, "income": 0.1},
        numeric_stats={"age": {"mean": 35.0, "std": 10.0, "min": 18.0,
                               "max": 65.0, "p25": 27.0, "p75": 45.0}},
        categorical_stats={"gender": {"F": 48, "M": 52}},
        sensitive_stats=sensitive_stats or {"gender": {"F": 0.48, "M": 0.52}},
        recorded_at="2026-03-16T07:00:00",
    )


# STORE TESTS 

def test_store_creates_empty_tables(store):
    assert store.get_steps() == []


def test_log_and_retrieve_step(store):
    _log_step(store)
    steps = store.get_steps()
    assert len(steps) == 1
    assert steps[0]["step_name"] == "test_step"
    assert steps[0]["status"] == "success"
    assert steps[0]["output_hash"] == "def"


def test_filter_steps_by_name(store):
    _log_step(store, "step_a")
    _log_step(store, "step_b")
    _log_step(store, "step_a")
    assert len(store.get_steps("step_a")) == 2
    assert len(store.get_steps("step_b")) == 1


def test_clear_wipes_all_records(store):
    _log_step(store)
    store.clear()
    assert store.get_steps() == []


def test_pipeline_lifecycle(store):
    store.log_pipeline_start(
        pipeline_id="pipe-001", name="my_pipeline",
        started_at="2026-03-16T07:00:00"
    )
    pipes = store.get_pipelines()
    assert len(pipes) == 1
    assert pipes[0]["status"] == "running"
    store.log_pipeline_end(
        pipeline_id="pipe-001", status="success",
        ended_at="2026-03-16T07:05:00"
    )
    pipes = store.get_pipelines()
    assert pipes[0]["status"] == "success"
    assert pipes[0]["ended_at"] is not None


# SNAPSHOT TESTS 

def test_snapshot_log_and_retrieve(store):
    _log_snapshot(store)
    snaps = store.get_snapshots()
    assert len(snaps) == 1
    assert snaps[0]["step_name"] == "clean_data"
    assert snaps[0]["position"] == "before"
    assert snaps[0]["row_count"] == 100


def test_snapshot_json_fields_are_parsed(store):
    _log_snapshot(store)
    snap = store.get_snapshots()[0]
    # JSON fields should come back as dicts/lists, not strings
    assert isinstance(snap["column_names"], list)
    assert isinstance(snap["null_rates"], dict)
    assert isinstance(snap["numeric_stats"], dict)
    assert isinstance(snap["categorical_stats"], dict)
    assert isinstance(snap["sensitive_stats"], dict)


def test_snapshot_sensitive_stats_values(store):
    _log_snapshot(store, sensitive_stats={"gender": {"F": 0.38, "M": 0.62}})
    snap = store.get_snapshots()[0]
    assert pytest.approx(snap["sensitive_stats"]["gender"]["F"], abs=1e-6) == 0.38
    assert pytest.approx(snap["sensitive_stats"]["gender"]["M"], abs=1e-6) == 0.62


def test_snapshot_filter_by_step_name(store):
    _log_snapshot(store, step_name="clean_data")
    _log_snapshot(store, step_name="normalize")
    assert len(store.get_snapshots("clean_data")) == 1
    assert len(store.get_snapshots("normalize")) == 1
    assert len(store.get_snapshots()) == 2


def test_snapshot_filter_by_position(store):
    _log_snapshot(store, position="before", row_count=100)
    _log_snapshot(store, position="after", row_count=74)
    befores = store.get_snapshots(position="before")
    afters = store.get_snapshots(position="after")
    assert len(befores) == 1 and befores[0]["row_count"] == 100
    assert len(afters) == 1 and afters[0]["row_count"] == 74


def test_snapshot_before_after_pair(store):
    _log_snapshot(store, position="before", row_count=12847,
                  sensitive_stats={"gender": {"F": 0.384, "M": 0.616}})
    _log_snapshot(store, position="after", row_count=7403,
                  sensitive_stats={"gender": {"F": 0.221, "M": 0.779}})
    snaps = store.get_snapshots()
    assert len(snaps) == 2
    before = next(s for s in snaps if s["position"] == "before")
    after  = next(s for s in snaps if s["position"] == "after")
    assert before["row_count"] == 12847
    assert after["row_count"] == 7403
    # Demographic shift is visible
    assert after["sensitive_stats"]["gender"]["F"] < before["sensitive_stats"]["gender"]["F"]


def test_snapshot_invalid_position_raises(store):
    with pytest.raises(AssertionError):
        store.log_snapshot(
            run_id="r", step_name="s", position="middle",
            row_count=10, column_count=2, column_names=["a"],
            null_rates={}, numeric_stats={}, categorical_stats={},
            sensitive_stats={}, recorded_at="2026-03-16T07:00:00"
        )


def test_snapshot_clear_wipes_snapshots(store):
    _log_snapshot(store)
    store.clear()
    assert store.get_snapshots() == []


def test_metrics_log_and_retrieve(store):
    store.log_metrics(
        run_id="run-001",
        metrics={"gender_bias_score": 0.34, "accuracy": 0.91},
        metric_source="equitrace",
    )
    metrics = store.get_metrics()
    assert len(metrics) == 2
    names = {m["metric_name"] for m in metrics}
    assert names == {"gender_bias_score", "accuracy"}


def test_metrics_values_stored_correctly(store):
    store.log_metrics(run_id="run-001", metrics={"bias": 0.34})
    m = store.get_metrics()[0]
    assert pytest.approx(m["metric_value"], abs=1e-6) == 0.34


def test_metrics_filter_by_run_id(store):
    store.log_metrics(run_id="run-001", metrics={"score": 0.5})
    store.log_metrics(run_id="run-002", metrics={"score": 0.7})
    assert len(store.get_metrics(run_id="run-001")) == 1
    assert len(store.get_metrics(run_id="run-002")) == 1
    assert len(store.get_metrics()) == 2


def test_metrics_filter_by_metric_name(store):
    store.log_metrics(
        run_id="run-001",
        metrics={"gender_bias": 0.34, "accuracy": 0.91, "f1": 0.88}
    )
    result = store.get_metrics(metric_name="gender_bias")
    assert len(result) == 1
    assert result[0]["metric_name"] == "gender_bias"


def test_metrics_source_and_tags_stored(store):
    store.log_metrics(
        run_id="run-001",
        metrics={"bias": 0.34},
        metric_source="equitrace",
        step_name="train_model",
        tags={"model_version": "v3", "dataset": "oyo_2025"}
    )
    m = store.get_metrics()[0]
    assert m["metric_source"] == "equitrace"
    assert m["step_name"] == "train_model"
    assert isinstance(m["tags"], dict)
    assert m["tags"]["model_version"] == "v3"


def test_metrics_clear_wipes_metrics(store):
    store.log_metrics(run_id="run-001", metrics={"bias": 0.34})
    store.clear()
    assert store.get_metrics() == []
