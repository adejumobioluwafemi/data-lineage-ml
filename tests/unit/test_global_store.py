"""
Tests for the global default store (dlm.init() / Layer 5).
Run with: python run_tests.py  OR  pytest tests/unit/test_global_store.py -v

These tests are careful to reset the global store before and after each test
so they never interfere with each other or with the other test files.
"""

import pytest # type: ignore
import tempfile
import os
import datalineageml as dlm
from datalineageml import track, LineageContext, LineageStore


@pytest.fixture(autouse=True)
def reset_global_store():
    """Always reset the global store before and after every test in this file."""
    dlm.reset()
    yield
    dlm.reset()


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "global_test.db")

def test_init_returns_lineage_store(tmp_db):
    store = dlm.init(db_path=tmp_db)
    assert isinstance(store, LineageStore)


def test_init_sets_global_default_store(tmp_db):
    dlm.init(db_path=tmp_db)
    assert dlm.get_default_store() is not None


def test_init_creates_db_file(tmp_db):
    dlm.init(db_path=tmp_db)
    assert os.path.exists(tmp_db)


def test_get_default_store_returns_none_before_init():
    # No init called — should be None (reset by fixture)
    assert dlm.get_default_store() is None


def test_reset_clears_global_store(tmp_db):
    dlm.init(db_path=tmp_db)
    assert dlm.get_default_store() is not None
    dlm.reset()
    assert dlm.get_default_store() is None


def test_init_twice_replaces_store(tmp_path):
    db1 = str(tmp_path / "db1.db")
    db2 = str(tmp_path / "db2.db")
    dlm.init(db_path=db1)
    s1 = dlm.get_default_store()
    dlm.init(db_path=db2)
    s2 = dlm.get_default_store()
    assert s1 is not s2
    assert s2.db_path == db2


# @track with global store 
def test_track_uses_global_store_when_no_store_arg(tmp_db):
    dlm.init(db_path=tmp_db)
    global_store = dlm.get_default_store()

    @track(name="global_step")   # no store= argument
    def fn(x): return x * 2

    fn(5)

    steps = global_store.get_steps()
    assert len(steps) == 1
    assert steps[0]["step_name"] == "global_step"


def test_track_explicit_store_overrides_global(tmp_path):
    global_db = str(tmp_path / "global.db")
    explicit_db = str(tmp_path / "explicit.db")

    dlm.init(db_path=global_db)
    explicit_store = LineageStore(db_path=explicit_db)

    @track(name="explicit_step", store=explicit_store)
    def fn(x): return x

    fn(99)

    # Should be in explicit store, NOT in global store
    assert len(explicit_store.get_steps()) == 1
    assert len(dlm.get_default_store().get_steps()) == 0


def test_track_multiple_functions_all_use_global_store(tmp_db):
    dlm.init(db_path=tmp_db)
    store = dlm.get_default_store()

    @track(name="step_a")
    def step_a(x): return x + 1

    @track(name="step_b")
    def step_b(x): return x * 2

    @track(name="step_c")
    def step_c(x): return x - 1

    step_c(step_b(step_a(1)))

    steps = store.get_steps()
    assert len(steps) == 3
    names = [s["step_name"] for s in steps]
    assert "step_a" in names
    assert "step_b" in names
    assert "step_c" in names


# LineageContext with global store

def test_lineage_context_uses_global_store(tmp_db):
    dlm.init(db_path=tmp_db)
    store = dlm.get_default_store()

    with LineageContext(name="global_pipeline"):
        pass

    pipelines = store.get_pipelines()
    assert len(pipelines) == 1
    assert pipelines[0]["name"] == "global_pipeline"


def test_lineage_context_explicit_store_overrides_global(tmp_path):
    global_db  = str(tmp_path / "global.db")
    explicit_db = str(tmp_path / "explicit.db")

    dlm.init(db_path=global_db)
    explicit_store = LineageStore(db_path=explicit_db)

    with LineageContext(name="explicit_pipeline", store=explicit_store):
        pass

    assert len(explicit_store.get_pipelines()) == 1
    assert len(dlm.get_default_store().get_pipelines()) == 0


# end-to-end: global store used by both @track and LineageContext 

def test_end_to_end_global_store_pipeline(tmp_db):
    dlm.init(db_path=tmp_db)
    store = dlm.get_default_store()

    @track(name="load")
    def load(): return [1, 2, 3, 4, 5]

    @track(name="process")
    def process(data): return [x * 2 for x in data]

    with LineageContext(name="e2e_pipeline"):
        data = load()
        result = process(data)

    assert result == [2, 4, 6, 8, 10]
    assert len(store.get_steps()) == 2
    assert len(store.get_pipelines()) == 1
    assert store.get_pipelines()[0]["status"] == "success"


def test_global_store_persists_across_function_calls(tmp_db):
    """Verify that multiple function calls all accumulate in the same store."""
    dlm.init(db_path=tmp_db)

    @track(name="repeat_fn")
    def fn(x): return x

    for i in range(5):
        fn(i)

    assert len(dlm.get_default_store().get_steps()) == 5
