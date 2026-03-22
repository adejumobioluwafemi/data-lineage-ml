"""
Unit tests for LineageContext.
Run with: pytest tests/unit/test_context.py -v
"""

import pytest
from datalineageml.trackers.context import LineageContext
from datalineageml.trackers.decorator import track
from datalineageml.storage.sqlite_store import LineageStore


@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test.db")
    s = LineageStore(db_path=db)
    yield s
    s.close()


def test_context_logs_pipeline_start_and_end(store):
    with LineageContext(name="my_pipeline", store=store):
        pass

    pipelines = store.get_pipelines()
    assert len(pipelines) == 1
    assert pipelines[0]["name"] == "my_pipeline"
    assert pipelines[0]["status"] == "success"
    assert pipelines[0]["ended_at"] is not None


def test_context_marks_failed_on_exception(store):
    with pytest.raises(RuntimeError):
        with LineageContext(name="failing_pipeline", store=store):
            raise RuntimeError("pipeline blew up")

    pipelines = store.get_pipelines()
    assert pipelines[0]["status"] == "failed"


def test_context_with_tracked_steps(store):
    @track(name="step_one", store=store)
    def step_one(x):
        return x + 1

    @track(name="step_two", store=store)
    def step_two(x):
        return x * 2

    with LineageContext(name="full_pipeline", store=store):
        result = step_one(5)
        result = step_two(result)

    assert result == 12
    steps = store.get_steps()
    assert len(steps) == 2
    assert steps[0]["step_name"] == "step_one"
    assert steps[1]["step_name"] == "step_two"
