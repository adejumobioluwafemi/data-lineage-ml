"""
Tests for the @track decorator.
Run with: python run_tests.py  OR  pytest tests/unit/test_decorator.py -v
"""

import pytest # type: ignore
import json
import time
import tempfile
import os
from datalineageml.trackers.decorator import track
from datalineageml.storage.sqlite_store import LineageStore


@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test.db")
    s = LineageStore(db_path=db)
    yield s
    s.close()


# ── ORIGINAL 11 DECORATOR TESTS (unchanged) ───────────────────────────────────

def test_track_logs_successful_step(store):
    @track(name="add_numbers", store=store)
    def add(a, b): return a + b
    result = add(2, 3)
    assert result == 5
    steps = store.get_steps()
    assert len(steps) == 1
    assert steps[0]["step_name"] == "add_numbers"
    assert steps[0]["status"] == "success"


def test_track_uses_function_name_by_default(store):
    @track(store=store)
    def my_transform(x): return x * 2
    my_transform(10)
    assert store.get_steps()[0]["step_name"] == "my_transform"


def test_track_logs_failed_step(store):
    @track(name="failing_step", store=store)
    def bad_fn(): raise ValueError("intentional error")
    with pytest.raises(ValueError):
        bad_fn()
    steps = store.get_steps()
    assert steps[0]["status"] == "failed"
    assert "intentional error" in steps[0]["error"]


def test_track_preserves_return_value(store):
    @track(store=store)
    def identity(x): return x
    assert identity({"key": "value"}) == {"key": "value"}
    assert identity([1, 2, 3]) == [1, 2, 3]
    assert identity(None) is None


def test_track_records_tags(store):
    @track(name="tagged", tags={"stage": "preprocessing", "version": "v1"}, store=store)
    def tagged_fn(x): return x
    tagged_fn(42)
    tags = json.loads(store.get_steps()[0]["tags"])
    assert tags["stage"] == "preprocessing"
    assert tags["version"] == "v1"


def test_track_records_duration(store):
    @track(name="slow_step", store=store)
    def slow_fn():
        time.sleep(0.05)
        return True
    slow_fn()
    assert store.get_steps()[0]["duration_ms"] >= 40


def test_track_hashes_inputs(store):
    @track(name="hash_test", store=store)
    def fn(x, y): return x + y
    fn(1, 2)
    hashes = json.loads(store.get_steps()[0]["input_hashes"])
    assert "arg_0" in hashes
    assert "arg_1" in hashes


def test_track_multiple_calls_logged_separately(store):
    @track(name="multi", store=store)
    def fn(x): return x
    fn(1); fn(2); fn(3)
    assert len(store.get_steps()) == 3


def test_track_hashes_dataframe_inputs(store):
    import pandas as pd
    @track(name="df_test", store=store)
    def fn(df): return df.shape[0]
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    fn(df)
    hashes = json.loads(store.get_steps()[0]["input_hashes"])
    assert "arg_0" in hashes
    assert hashes["arg_0"] != ""


def test_same_data_produces_same_hash(store):
    import pandas as pd
    @track(name="hash_stability", store=store)
    def fn(df): return df
    df = pd.DataFrame({"x": [1, 2, 3]})
    fn(df); fn(df)
    steps = store.get_steps()
    h1 = json.loads(steps[0]["input_hashes"])["arg_0"]
    h2 = json.loads(steps[1]["input_hashes"])["arg_0"]
    assert h1 == h2


def test_different_data_produces_different_hash(store):
    import pandas as pd
    @track(name="hash_diff", store=store)
    def fn(df): return df
    fn(pd.DataFrame({"x": [1, 2, 3]}))
    fn(pd.DataFrame({"x": [4, 5, 6]}))
    steps = store.get_steps()
    h1 = json.loads(steps[0]["input_hashes"])["arg_0"]
    h2 = json.loads(steps[1]["input_hashes"])["arg_0"]
    assert h1 != h2


# ── NEW: SNAPSHOT INTEGRATION TESTS ──────────────────────────────────────────

def test_snapshot_true_logs_before_and_after(store):
    import pandas as pd
    df_in = pd.DataFrame({"age": [25, None, 38], "gender": ["F", "M", "F"]})

    @track(name="clean", store=store, snapshot=True, sensitive_cols=["gender"])
    def clean(df): return df.dropna()

    clean(df_in)

    snaps = store.get_snapshots()
    assert len(snaps) == 2
    positions = {s["position"] for s in snaps}
    assert positions == {"before", "after"}


def test_snapshot_captures_row_count_change(store):
    import pandas as pd
    df = pd.DataFrame({
        "age": [25, None, 38, 45, None],
        "income": [50000, 60000, None, 70000, 80000]
    })

    @track(name="dropna_step", store=store, snapshot=True)
    def dropna_step(df): return df.dropna()

    dropna_step(df)

    before = store.get_snapshots(position="before")[0]
    after  = store.get_snapshots(position="after")[0]
    assert before["row_count"] == 5
    assert after["row_count"] == 2  # only rows with no nulls


def test_snapshot_captures_gender_distribution_shift(store):
    import pandas as pd
    # 4 females: 3 without land_title (removed by dropna), 1 with (survives)
    # 5 males: all have land_title (all survive)
    # Female proportion drops from ~44% to ~17% — mirrors the Oyo State pattern
    df = pd.DataFrame({
        "gender":     ["F", "M", "M", "F", "M", "M", "F", "M", "F"],
        "land_title": [None, "yes", "yes", None, "yes", "yes", "yes", "yes", None],
        "income":     [40, 70, 65, 45, 80, 75, 42, 90, 38],
    })

    @track(name="clean", store=store, snapshot=True, sensitive_cols=["gender"])
    def clean(df): return df.dropna()

    clean(df)

    before = store.get_snapshots(position="before")[0]
    after  = store.get_snapshots(position="after")[0]

    female_before = before["sensitive_stats"]["gender"].get("F", 0.0)
    female_after  = after["sensitive_stats"]["gender"].get("F", 0.0)

    # Female proportion should drop after dropna removes null-land_title rows
    assert female_before > female_after


def test_snapshot_false_logs_no_snapshots(store):
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2, 3]})

    @track(name="no_snapshot", store=store, snapshot=False)
    def fn(df): return df

    fn(df)

    assert store.get_steps()[0]["step_name"] == "no_snapshot"
    assert store.get_snapshots() == []


def test_snapshot_default_is_false(store):
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2, 3]})

    @track(name="default_snap", store=store)  # no snapshot= arg
    def fn(df): return df

    fn(df)
    assert store.get_snapshots() == []


def test_snapshot_does_not_break_on_non_dataframe(store):
    """Snapshot=True on a non-DataFrame function should not raise."""
    @track(name="scalar_fn", store=store, snapshot=True)
    def fn(x, y): return x + y

    result = fn(3, 4)
    assert result == 7
    assert store.get_steps()[0]["status"] == "success"
    assert store.get_snapshots() == []  # no DataFrame → no snapshot


def test_snapshot_linked_to_correct_run_id(store):
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2, None]})

    @track(name="track_run_id", store=store, snapshot=True)
    def fn(df): return df.dropna()

    fn(df)

    step = store.get_steps()[0]
    snaps = store.get_snapshots()
    for snap in snaps:
        assert snap["run_id"] == step["run_id"]
