"""
Unit tests for the @track decorator.
Run with: pytest tests/unit/test_decorator.py -v
"""

import pytest
from datalineageml.trackers.decorator import track
from datalineageml.storage.sqlite_store import LineageStore


@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test.db")
    s = LineageStore(db_path=db)
    yield s
    s.clear()
    s.close()


def test_track_logs_successful_step(store):
    @track(name="add_numbers", store=store)
    def add(a, b):
        return a + b

    result = add(2, 3)
    assert result == 5

    steps = store.get_steps()
    assert len(steps) == 1
    assert steps[0]["step_name"] == "add_numbers"
    assert steps[0]["status"] == "success"


def test_track_uses_function_name_by_default(store):
    @track(store=store)
    def my_transform(x):
        return x * 2

    my_transform(10)
    steps = store.get_steps()
    assert steps[0]["step_name"] == "my_transform"


def test_track_logs_failed_step(store):
    @track(name="failing_step", store=store)
    def bad_fn():
        raise ValueError("intentional error")

    with pytest.raises(ValueError):
        bad_fn()

    steps = store.get_steps()
    assert steps[0]["status"] == "failed"
    assert "intentional error" in steps[0]["error"]


def test_track_preserves_return_value(store):
    @track(store=store)
    def identity(x):
        return x

    assert identity({"key": "value"}) == {"key": "value"}
    assert identity([1, 2, 3]) == [1, 2, 3]
    assert identity(None) is None


def test_track_records_tags(store):
    import json

    @track(name="tagged_step", tags={"stage": "preprocessing", "version": "v1"}, store=store)
    def tagged_fn(x):
        return x

    tagged_fn(42)
    steps = store.get_steps()
    tags = json.loads(steps[0]["tags"])
    assert tags["stage"] == "preprocessing"
    assert tags["version"] == "v1"


def test_track_records_duration(store):
    import time

    @track(name="slow_step", store=store)
    def slow_fn():
        time.sleep(0.05)
        return True

    slow_fn()
    steps = store.get_steps()
    assert steps[0]["duration_ms"] >= 40  # at least 40ms


def test_track_hashes_inputs(store):
    import json

    @track(name="hash_test", store=store)
    def fn(x, y):
        return x + y

    fn(1, 2)
    steps = store.get_steps()
    hashes = json.loads(steps[0]["input_hashes"])
    assert "arg_0" in hashes
    assert "arg_1" in hashes


def test_track_multiple_calls_logged_separately(store):
    @track(name="multi", store=store)
    def fn(x):
        return x

    fn(1)
    fn(2)
    fn(3)
    steps = store.get_steps()
    assert len(steps) == 3
