"""
run_tests.py — runs all unit tests using stdlib unittest.
No pytest needed.

Usage:
    python run_tests.py
"""

import sys
import os
import unittest
import tempfile
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import datalineageml as dlm
from datalineageml.storage.sqlite_store import LineageStore
from datalineageml.trackers.decorator import track
from datalineageml.trackers.context import LineageContext

# Helpers

def make_store(suffix=".db"):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    return LineageStore(db_path=tmp.name), tmp.name


def cleanup(store, path):
    store.close()
    try: os.unlink(path)
    except: pass


def log_step(store, step_name="test_step", status="success"):
    store.log_step(
        run_id="run-001", step_name=step_name,
        fn_module="m", fn_qualname="f",
        input_hashes={"arg_0": "abc"}, output_hash="def",
        duration_ms=10.0, started_at="2026-03-16T07:00:00",
        status=status, error=None, tags={}
    )


def log_snapshot(store, position="before", row_count=100,
                 sensitive_stats=None):
    store.log_snapshot(
        run_id="run-001", step_name="clean_data",
        position=position, row_count=row_count, column_count=3,
        column_names=["age", "gender", "income"],
        null_rates={"age": 0.0, "gender": 0.05},
        numeric_stats={"age": {"mean": 35.0, "std": 10.0, "min": 18.0,
                               "max": 65.0, "p25": 27.0, "p75": 45.0}},
        categorical_stats={"gender": {"F": 48, "M": 52}},
        sensitive_stats=sensitive_stats or {"gender": {"F": 0.48, "M": 0.52}},
        recorded_at="2026-03-16T07:00:00",
    )

# Store Tests 
class TestLineageStore(unittest.TestCase):

    def setUp(self):
        self.store, self.path = make_store()

    def tearDown(self):
        cleanup(self.store, self.path)

    def test_creates_empty_tables(self):
        self.assertEqual(self.store.get_steps(), [])

    def test_log_and_retrieve_step(self):
        log_step(self.store)
        steps = self.store.get_steps()
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["step_name"], "test_step")
        self.assertEqual(steps[0]["status"], "success")

    def test_filter_by_name(self):
        log_step(self.store, "step_a")
        log_step(self.store, "step_b")
        log_step(self.store, "step_a")
        self.assertEqual(len(self.store.get_steps("step_a")), 2)
        self.assertEqual(len(self.store.get_steps("step_b")), 1)

    def test_clear_wipes_all(self):
        log_step(self.store)
        self.store.clear()
        self.assertEqual(self.store.get_steps(), [])

    def test_pipeline_lifecycle(self):
        self.store.log_pipeline_start(
            pipeline_id="pipe-001", name="my_pipeline",
            started_at="2026-03-16T07:00:00"
        )
        pipes = self.store.get_pipelines()
        self.assertEqual(len(pipes), 1)
        self.assertEqual(pipes[0]["status"], "running")
        self.store.log_pipeline_end(
            pipeline_id="pipe-001", status="success",
            ended_at="2026-03-16T07:05:00"
        )
        self.assertEqual(self.store.get_pipelines()[0]["status"], "success")



# Snapshot Tests 
class TestSnapshots(unittest.TestCase):

    def setUp(self):
        self.store, self.path = make_store()

    def tearDown(self):
        cleanup(self.store, self.path)

    def test_log_and_retrieve(self):
        log_snapshot(self.store)
        snaps = self.store.get_snapshots()
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0]["step_name"], "clean_data")
        self.assertEqual(snaps[0]["position"], "before")

    def test_json_fields_are_parsed(self):
        log_snapshot(self.store)
        snap = self.store.get_snapshots()[0]
        self.assertIsInstance(snap["column_names"], list)
        self.assertIsInstance(snap["null_rates"], dict)
        self.assertIsInstance(snap["sensitive_stats"], dict)

    def test_sensitive_stats_values(self):
        log_snapshot(self.store, sensitive_stats={"gender": {"F": 0.38, "M": 0.62}})
        snap = self.store.get_snapshots()[0]
        self.assertAlmostEqual(snap["sensitive_stats"]["gender"]["F"], 0.38, places=5)

    def test_filter_by_position(self):
        log_snapshot(self.store, position="before", row_count=100)
        log_snapshot(self.store, position="after", row_count=74)
        befores = self.store.get_snapshots(position="before")
        afters  = self.store.get_snapshots(position="after")
        self.assertEqual(befores[0]["row_count"], 100)
        self.assertEqual(afters[0]["row_count"], 74)

    def test_before_after_shows_demographic_shift(self):
        log_snapshot(self.store, position="before",
                     sensitive_stats={"gender": {"F": 0.384, "M": 0.616}})
        log_snapshot(self.store, position="after",
                     sensitive_stats={"gender": {"F": 0.221, "M": 0.779}})
        snaps = self.store.get_snapshots()
        before = next(s for s in snaps if s["position"] == "before")
        after  = next(s for s in snaps if s["position"] == "after")
        self.assertGreater(before["sensitive_stats"]["gender"]["F"],
                           after["sensitive_stats"]["gender"]["F"])

    def test_invalid_position_raises(self):
        with self.assertRaises(AssertionError):
            self.store.log_snapshot(
                run_id="r", step_name="s", position="middle",
                row_count=10, column_count=2, column_names=["a"],
                null_rates={}, numeric_stats={}, categorical_stats={},
                sensitive_stats={}, recorded_at="2026-03-16T07:00:00"
            )

    def test_clear_wipes_snapshots(self):
        log_snapshot(self.store)
        self.store.clear()
        self.assertEqual(self.store.get_snapshots(), [])

    def test_filter_by_step_name(self):
        log_snapshot(self.store)
        snaps = self.store.get_snapshots("clean_data")
        self.assertEqual(len(snaps), 1)
        self.assertEqual(self.store.get_snapshots("nonexistent"), [])


# Metrics Tests
class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.store, self.path = make_store()

    def tearDown(self):
        cleanup(self.store, self.path)

    def test_log_and_retrieve(self):
        self.store.log_metrics(
            run_id="run-001",
            metrics={"gender_bias": 0.34, "accuracy": 0.91}
        )
        metrics = self.store.get_metrics()
        self.assertEqual(len(metrics), 2)
        names = {m["metric_name"] for m in metrics}
        self.assertIn("gender_bias", names)
        self.assertIn("accuracy", names)

    def test_value_stored_correctly(self):
        self.store.log_metrics(run_id="r", metrics={"bias": 0.34})
        self.assertAlmostEqual(self.store.get_metrics()[0]["metric_value"],
                               0.34, places=5)

    def test_filter_by_metric_name(self):
        self.store.log_metrics(
            run_id="r",
            metrics={"gender_bias": 0.34, "accuracy": 0.91}
        )
        result = self.store.get_metrics(metric_name="gender_bias")
        self.assertEqual(len(result), 1)

    def test_source_and_tags_stored(self):
        self.store.log_metrics(
            run_id="r", metrics={"bias": 0.34},
            metric_source="equitrace",
            step_name="train_model",
            tags={"model": "rf", "version": "v3"}
        )
        m = self.store.get_metrics()[0]
        self.assertEqual(m["metric_source"], "equitrace")
        self.assertEqual(m["step_name"], "train_model")
        self.assertIsInstance(m["tags"], dict)
        self.assertEqual(m["tags"]["model"], "rf")

    def test_clear_wipes_metrics(self):
        self.store.log_metrics(run_id="r", metrics={"bias": 0.34})
        self.store.clear()
        self.assertEqual(self.store.get_metrics(), [])

# Decorator Tests
class TestTrackDecorator(unittest.TestCase):

    def setUp(self):
        self.store, self.path = make_store()

    def tearDown(self):
        cleanup(self.store, self.path)

    def test_logs_successful_step(self):
        @track(name="add_numbers", store=self.store)
        def add(a, b): return a + b
        result = add(2, 3)
        self.assertEqual(result, 5)
        steps = self.store.get_steps()
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["status"], "success")

    def test_uses_function_name_by_default(self):
        @track(store=self.store)
        def my_transform(x): return x * 2
        my_transform(10)
        self.assertEqual(self.store.get_steps()[0]["step_name"], "my_transform")

    def test_logs_failed_step(self):
        @track(name="failing", store=self.store)
        def bad(): raise ValueError("intentional")
        with self.assertRaises(ValueError):
            bad()
        steps = self.store.get_steps()
        self.assertEqual(steps[0]["status"], "failed")
        self.assertIn("intentional", steps[0]["error"])

    def test_preserves_return_value(self):
        @track(store=self.store)
        def identity(x): return x
        self.assertEqual(identity({"k": "v"}), {"k": "v"})
        self.assertEqual(identity([1, 2, 3]), [1, 2, 3])
        self.assertIsNone(identity(None))

    def test_records_tags(self):
        @track(name="tagged", tags={"stage": "test", "ver": "1"}, store=self.store)
        def fn(x): return x
        fn(1)
        tags = json.loads(self.store.get_steps()[0]["tags"])
        self.assertEqual(tags["stage"], "test")

    def test_records_duration(self):
        @track(name="timed", store=self.store)
        def slow():
            time.sleep(0.05)
            return True
        slow()
        self.assertGreaterEqual(self.store.get_steps()[0]["duration_ms"], 40)

    def test_hashes_inputs(self):
        @track(name="hash_test", store=self.store)
        def fn(x, y): return x + y
        fn(1, 2)
        hashes = json.loads(self.store.get_steps()[0]["input_hashes"])
        self.assertIn("arg_0", hashes)
        self.assertIn("arg_1", hashes)

    def test_multiple_calls_logged_separately(self):
        @track(name="multi", store=self.store)
        def fn(x): return x
        fn(1); fn(2); fn(3)
        self.assertEqual(len(self.store.get_steps()), 3)

    def test_hashes_dataframe_inputs(self):
        import pandas as pd
        @track(name="df_test", store=self.store)
        def fn(df): return df.shape[0]
        fn(pd.DataFrame({"a": [1, 2, 3]}))
        hashes = json.loads(self.store.get_steps()[0]["input_hashes"])
        self.assertIn("arg_0", hashes)

    def test_same_data_produces_same_hash(self):
        import pandas as pd
        @track(name="hash_stability", store=self.store)
        def fn(df): return df
        df = pd.DataFrame({"x": [1, 2, 3]})
        fn(df); fn(df)
        steps = self.store.get_steps()
        h1 = json.loads(steps[0]["input_hashes"])["arg_0"]
        h2 = json.loads(steps[1]["input_hashes"])["arg_0"]
        self.assertEqual(h1, h2)

    def test_different_data_produces_different_hash(self):
        import pandas as pd
        @track(name="hash_diff", store=self.store)
        def fn(df): return df
        fn(pd.DataFrame({"x": [1, 2, 3]}))
        fn(pd.DataFrame({"x": [4, 5, 6]}))
        steps = self.store.get_steps()
        h1 = json.loads(steps[0]["input_hashes"])["arg_0"]
        h2 = json.loads(steps[1]["input_hashes"])["arg_0"]
        self.assertNotEqual(h1, h2)


# Snapshot via @track Integration Tests 
class TestSnapshotViaDecorator(unittest.TestCase):

    def setUp(self):
        self.store, self.path = make_store()

    def tearDown(self):
        cleanup(self.store, self.path)

    def test_snapshot_true_logs_before_and_after(self):
        import pandas as pd
        df = pd.DataFrame({"age": [25, None, 38], "gender": ["F", "M", "F"]})
        @track(name="clean", store=self.store, snapshot=True, sensitive_cols=["gender"])
        def clean(df): return df.dropna()
        clean(df)
        snaps = self.store.get_snapshots()
        self.assertEqual(len(snaps), 2)
        positions = {s["position"] for s in snaps}
        self.assertEqual(positions, {"before", "after"})

    def test_snapshot_captures_row_count_change(self):
        import pandas as pd
        df = pd.DataFrame({"age": [25, None, 38, 45, None]})
        @track(name="dropna", store=self.store, snapshot=True)
        def dropna(df): return df.dropna()
        dropna(df)
        before = self.store.get_snapshots(position="before")[0]
        after  = self.store.get_snapshots(position="after")[0]
        self.assertEqual(before["row_count"], 5)
        self.assertEqual(after["row_count"], 3)

    def test_snapshot_gender_shift_detected(self):
        import pandas as pd
        # 4 females (3 without land_title) and 5 males (all with land_title)
        # After dropna: 1 female (25% -> ~17%), F proportion drops significantly
        df = pd.DataFrame({
            "gender":     ["F", "M", "M", "F", "M", "M", "F", "M", "F"],
            "land_title": [None, "yes", "yes", None, "yes", "yes", "yes", "yes", None],
        })
        @track(name="clean", store=self.store, snapshot=True, sensitive_cols=["gender"])
        def clean(df): return df.dropna()
        clean(df)
        before = self.store.get_snapshots(position="before")[0]
        after  = self.store.get_snapshots(position="after")[0]
        # Female proportion should drop because dropna removes null-land_title rows
        female_before = before["sensitive_stats"]["gender"].get("F", 0.0)
        female_after  = after["sensitive_stats"]["gender"].get("F", 0.0)
        self.assertGreater(female_before, female_after)

    def test_snapshot_false_logs_nothing(self):
        import pandas as pd
        @track(name="no_snap", store=self.store, snapshot=False)
        def fn(df): return df
        fn(pd.DataFrame({"x": [1, 2, 3]}))
        self.assertEqual(self.store.get_snapshots(), [])

    def test_snapshot_default_false(self):
        import pandas as pd
        @track(name="default", store=self.store)
        def fn(df): return df
        fn(pd.DataFrame({"x": [1, 2, 3]}))
        self.assertEqual(self.store.get_snapshots(), [])

    def test_snapshot_on_non_dataframe_does_not_raise(self):
        @track(name="scalar", store=self.store, snapshot=True)
        def fn(x, y): return x + y
        result = fn(3, 4)
        self.assertEqual(result, 7)
        self.assertEqual(self.store.get_snapshots(), [])

    def test_snapshot_run_id_matches_step_run_id(self):
        import pandas as pd
        @track(name="linked", store=self.store, snapshot=True)
        def fn(df): return df.dropna()
        fn(pd.DataFrame({"x": [1, None, 3]}))
        step  = self.store.get_steps()[0]
        snaps = self.store.get_snapshots()
        for snap in snaps:
            self.assertEqual(snap["run_id"], step["run_id"])



# Context Tests
class TestLineageContext(unittest.TestCase):

    def setUp(self):
        self.store, self.path = make_store()

    def tearDown(self):
        cleanup(self.store, self.path)

    def test_logs_start_and_end(self):
        with LineageContext(name="test_pipe", store=self.store):
            pass
        pipes = self.store.get_pipelines()
        self.assertEqual(len(pipes), 1)
        self.assertEqual(pipes[0]["status"], "success")
        self.assertIsNotNone(pipes[0]["ended_at"])

    def test_marks_failed_on_exception(self):
        with self.assertRaises(RuntimeError):
            with LineageContext(name="bad_pipe", store=self.store):
                raise RuntimeError("boom")
        self.assertEqual(self.store.get_pipelines()[0]["status"], "failed")

    def test_works_with_tracked_steps(self):
        @track(name="s1", store=self.store)
        def s1(x): return x + 1
        @track(name="s2", store=self.store)
        def s2(x): return x * 2
        with LineageContext(name="full", store=self.store):
            result = s2(s1(5))
        self.assertEqual(result, 12)
        steps = self.store.get_steps()
        self.assertEqual(steps[0]["step_name"], "s1")
        self.assertEqual(steps[1]["step_name"], "s2")


# Global Default Store Tests
class TestGlobalStore(unittest.TestCase):

    def setUp(self):
        dlm.reset()
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp.name

    def tearDown(self):
        dlm.reset()
        try: os.unlink(self.db_path)
        except: pass

    def test_init_returns_store(self):
        store = dlm.init(db_path=self.db_path)
        self.assertIsInstance(store, LineageStore)

    def test_init_sets_default_store(self):
        dlm.init(db_path=self.db_path)
        self.assertIsNotNone(dlm.get_default_store())

    def test_init_creates_db_file(self):
        dlm.init(db_path=self.db_path)
        self.assertTrue(os.path.exists(self.db_path))

    def test_get_default_store_none_before_init(self):
        self.assertIsNone(dlm.get_default_store())

    def test_reset_clears_store(self):
        dlm.init(db_path=self.db_path)
        dlm.reset()
        self.assertIsNone(dlm.get_default_store())

    def test_track_uses_global_store(self):
        dlm.init(db_path=self.db_path)
        @track(name="global_step")
        def fn(x): return x * 2
        fn(5)
        steps = dlm.get_default_store().get_steps()
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["step_name"], "global_step")

    def test_track_explicit_store_overrides_global(self):
        dlm.init(db_path=self.db_path)
        explicit_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        explicit_store = LineageStore(db_path=explicit_tmp.name)
        try:
            @track(name="explicit", store=explicit_store)
            def fn(x): return x
            fn(99)
            self.assertEqual(len(explicit_store.get_steps()), 1)
            self.assertEqual(len(dlm.get_default_store().get_steps()), 0)
        finally:
            explicit_store.close()
            os.unlink(explicit_tmp.name)

    def test_multiple_functions_use_same_global_store(self):
        dlm.init(db_path=self.db_path)
        @track(name="a")
        def a(x): return x
        @track(name="b")
        def b(x): return x
        @track(name="c")
        def c(x): return x
        a(1); b(2); c(3)
        self.assertEqual(len(dlm.get_default_store().get_steps()), 3)

    def test_lineage_context_uses_global_store(self):
        dlm.init(db_path=self.db_path)
        with LineageContext(name="global_pipe"):
            pass
        pipes = dlm.get_default_store().get_pipelines()
        self.assertEqual(len(pipes), 1)
        self.assertEqual(pipes[0]["name"], "global_pipe")

    def test_lineage_context_explicit_overrides_global(self):
        dlm.init(db_path=self.db_path)
        explicit_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        explicit_store = LineageStore(db_path=explicit_tmp.name)
        try:
            with LineageContext(name="explicit_pipe", store=explicit_store):
                pass
            self.assertEqual(len(explicit_store.get_pipelines()), 1)
            self.assertEqual(len(dlm.get_default_store().get_pipelines()), 0)
        finally:
            explicit_store.close()
            os.unlink(explicit_tmp.name)

    def test_end_to_end_global_pipeline(self):
        dlm.init(db_path=self.db_path)
        @track(name="load")
        def load(): return [1, 2, 3]
        @track(name="process")
        def process(data): return [x * 2 for x in data]
        with LineageContext(name="e2e"):
            result = process(load())
        self.assertEqual(result, [2, 4, 6])
        store = dlm.get_default_store()
        self.assertEqual(len(store.get_steps()), 2)
        self.assertEqual(store.get_pipelines()[0]["status"], "success")

    def test_global_store_accumulates_across_calls(self):
        dlm.init(db_path=self.db_path)
        @track(name="repeat")
        def fn(x): return x
        for i in range(5): fn(i)
        self.assertEqual(len(dlm.get_default_store().get_steps()), 5)


# Snapshot integration tests
# (full causal evidence chain — Oyo State scenario)
class TestSnapshotIntegration(unittest.TestCase):
    """
    Tests that validate the full snapshot → demographic shift → metric
    evidence chain. These are the tests closest to the actual research
    contribution: showing that a dropna()-style step produces a measurable,
    attributable demographic shift that can be linked to a bias metric.
    """

    def setUp(self):
        self.store, self.path = make_store()

    def tearDown(self):
        cleanup(self.store, self.path)

    # numeric stats captured correctly

    def test_numeric_stats_mean_std_captured(self):
        """Numeric stats (mean, std, min, max, p25, p75) must be stored and retrieved."""
        import pandas as pd

        @track(name="normalize", snapshot=True, store=self.store)
        def normalize(df):
            return df

        df = pd.DataFrame({"age": [20, 30, 40, 50, 60], "income": [1000, 2000, 3000, 4000, 5000]})
        normalize(df)

        snap = self.store.get_snapshots(position="before")[0]
        age_stats = snap["numeric_stats"]["age"]
        self.assertAlmostEqual(age_stats["mean"], 40.0, places=3)
        self.assertAlmostEqual(age_stats["min"], 20.0, places=3)
        self.assertAlmostEqual(age_stats["max"], 60.0, places=3)
        self.assertIn("std", age_stats)
        self.assertIn("p25", age_stats)
        self.assertIn("p75", age_stats)

    def test_null_rates_captured(self):
        """Null rates per column must be stored as fractions (0–1)."""
        import pandas as pd

        @track(name="clean", snapshot=True, store=self.store)
        def clean(df):
            return df.dropna()

        # 2 nulls out of 5 rows → null_rate = 0.4
        df = pd.DataFrame({
            "age":    [25, None, 35, None, 45],
            "gender": ["F", "M",  "F", "M",  "F"],
        })
        clean(df)

        snap = self.store.get_snapshots(position="before")[0]
        self.assertAlmostEqual(snap["null_rates"]["age"], 0.4, places=4)
        self.assertAlmostEqual(snap["null_rates"]["gender"], 0.0, places=4)

    def test_categorical_stats_top_values_captured(self):
        """Categorical value counts must be stored for non-numeric columns."""
        import pandas as pd

        @track(name="profile", snapshot=True, store=self.store)
        def profile(df):
            return df

        df = pd.DataFrame({
            "zone": ["north"] * 5 + ["south"] * 3 + ["east"] * 2,
        })
        profile(df)

        snap = self.store.get_snapshots(position="before")[0]
        zone_counts = snap["categorical_stats"]["zone"]
        self.assertEqual(zone_counts["north"], 5)
        self.assertEqual(zone_counts["south"], 3)
        self.assertEqual(zone_counts["east"], 2)

    # sensitive column tracking

    def test_multiple_sensitive_columns_tracked(self):
        """All specified sensitive_cols must appear in sensitive_stats."""
        import pandas as pd

        @track(name="step", snapshot=True,
               sensitive_cols=["gender", "zone"], store=self.store)
        def step(df):
            return df

        df = pd.DataFrame({
            "gender": ["F"] * 4 + ["M"] * 6,
            "zone":   ["north"] * 7 + ["south"] * 3,
            "income": list(range(10)),
        })
        step(df)

        snap = self.store.get_snapshots(position="before")[0]
        ss = snap["sensitive_stats"]
        self.assertIn("gender", ss)
        self.assertIn("zone", ss)
        self.assertAlmostEqual(ss["gender"]["F"], 0.4, places=3)
        self.assertAlmostEqual(ss["zone"]["north"], 0.7, places=3)

    def test_sensitive_col_fractions_sum_to_one(self):
        """Sensitive stats fractions must sum to 1.0 (within float tolerance)."""
        import pandas as pd

        @track(name="check", snapshot=True,
               sensitive_cols=["gender"], store=self.store)
        def check(df):
            return df

        df = pd.DataFrame({"gender": ["F"] * 38 + ["M"] * 62})
        check(df)

        snap = self.store.get_snapshots(position="before")[0]
        fractions = snap["sensitive_stats"]["gender"]
        total = sum(fractions.values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_sensitive_col_not_in_df_does_not_raise(self):
        """A sensitive_col that does not exist in the DataFrame is silently skipped."""
        import pandas as pd

        @track(name="step", snapshot=True,
               sensitive_cols=["gender", "nonexistent_col"], store=self.store)
        def step(df):
            return df

        df = pd.DataFrame({"gender": ["F", "M", "F"]})
        step(df)  # must not raise

        snap = self.store.get_snapshots(position="before")[0]
        self.assertIn("gender", snap["sensitive_stats"])
        self.assertNotIn("nonexistent_col", snap["sensitive_stats"])

    # Oyo State scenario: dropna() demographic shift

    def test_oyo_state_dropna_gender_shift(self):
        """
        Core scenario: dropna() on a column correlated with gender removes
        more female records, producing a measurable demographic shift between
        the before and after snapshots.

        Before: 38.4% female (mirrors Oyo State baseline)
        After:  ~21% female  (post-dropna, land_title missing disproportionately)
        """
        import pandas as pd
        import random

        random.seed(42)
        n_total = 1000

        # Female farms: only 11% have land title → most rows will be null
        female_titles = [None if random.random() > 0.11 else "yes"
                         for _ in range(384)]
        # Male farms: 67% have land title → most rows kept
        male_titles   = [None if random.random() > 0.67 else "yes"
                         for _ in range(616)]

        df = pd.DataFrame({
            "gender":     ["F"] * 384 + ["M"] * 616,
            "land_title": female_titles + male_titles,
            "farm_size":  [random.uniform(0.5, 5.0) for _ in range(n_total)],
        })

        @track(name="clean_data", snapshot=True,
               sensitive_cols=["gender"], store=self.store)
        def clean_data(df):
            return df.dropna(subset=["land_title"])

        result = clean_data(df)

        before = self.store.get_snapshots(step_name="clean_data", position="before")[0]
        after  = self.store.get_snapshots(step_name="clean_data", position="after")[0]

        before_female = before["sensitive_stats"]["gender"]["F"]
        after_female  = after["sensitive_stats"]["gender"]["F"]

        # Female fraction must have dropped significantly
        self.assertGreater(before_female, 0.30)  # ~38% before
        self.assertLess(after_female, before_female - 0.05)  # meaningful drop

        # Row count must have fallen
        self.assertGreater(before["row_count"], after["row_count"])

        # Result DataFrame must have no nulls in land_title
        self.assertEqual(result["land_title"].isna().sum(), 0)

    def test_oyo_state_shift_magnitude(self):
        """
        The absolute demographic shift between before and after snapshots
        must exceed 10 percentage points to qualify as HIGH signal.
        This mirrors the real 38.4% → 22.1% shift (16 pp drop).
        """
        import pandas as pd

        # Simplified: 40% female before, ~15% female after (extreme land title gap)
        female_rows = pd.DataFrame({
            "gender": ["F"] * 400,
            "land_title": [None] * 340 + ["yes"] * 60,  # only 15% have title
        })
        male_rows = pd.DataFrame({
            "gender": ["M"] * 600,
            "land_title": [None] * 180 + ["yes"] * 420,  # 70% have title
        })
        df = pd.concat([female_rows, male_rows], ignore_index=True)

        @track(name="clean_oyo", snapshot=True,
               sensitive_cols=["gender"], store=self.store)
        def clean_oyo(df):
            return df.dropna(subset=["land_title"])

        clean_oyo(df)

        before = self.store.get_snapshots(position="before")[0]
        after  = self.store.get_snapshots(position="after")[0]

        shift = (before["sensitive_stats"]["gender"]["F"]
                 - after["sensitive_stats"]["gender"]["F"])

        # Shift must be > 10 percentage points — this is the HIGH signal threshold
        self.assertGreater(shift, 0.10,
            f"Expected shift > 0.10, got {shift:.4f}. "
            "This is the demographic shift that DataLineageML must detect.")

    # metrics linked to snapshot runs 

    def test_metrics_linked_to_snapshot_run_id(self):
        """
        Metrics logged against a run_id must be retrievable alongside the
        snapshots from that same run_id, forming the causal evidence chain:
        snapshot → shift → metric.
        """
        import pandas as pd

        @track(name="train", snapshot=True,
               sensitive_cols=["gender"], store=self.store)
        def train(df):
            return df  # simulate returning a trained model

        df = pd.DataFrame({
            "gender": ["F"] * 22 + ["M"] * 78,
            "score":  list(range(100)),
        })
        train(df)

        step = self.store.get_steps("train")[0]
        run_id = step["run_id"]

        # Log bias metric against the same run_id
        self.store.log_metrics(
            run_id=run_id,
            metrics={"gender_bias_score": 0.34},
            metric_source="equitrace",
            step_name="train",
        )

        # Both snapshot and metric share the same run_id
        snaps   = self.store.get_snapshots(step_name="train")
        metrics = self.store.get_metrics(run_id=run_id)

        self.assertTrue(len(snaps) > 0)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["run_id"], snaps[0]["run_id"])
        self.assertAlmostEqual(metrics[0]["metric_value"], 0.34, places=4)

    def test_before_and_after_metric_comparison(self):
        """
        Log metrics before and after a fix to prove remediation worked.
        This is the counterfactual evidence pattern: original bias vs post-fix bias.
        """
        self.store.log_metrics(
            run_id="run-original",
            metrics={"gender_bias_score": 0.34},
            metric_source="equitrace",
            tags={"version": "before_fix"},
        )
        self.store.log_metrics(
            run_id="run-fixed",
            metrics={"gender_bias_score": 0.09},
            metric_source="equitrace",
            tags={"version": "after_fix"},
        )

        original = self.store.get_metrics(run_id="run-original")[0]
        fixed    = self.store.get_metrics(run_id="run-fixed")[0]

        reduction = (original["metric_value"] - fixed["metric_value"]) / original["metric_value"]
        self.assertGreater(reduction, 0.70,
            f"Expected >70% bias reduction, got {reduction:.1%}")

    # snapshot does not interfere with pipeline execution 

    def test_snapshot_failure_does_not_break_pipeline(self):
        """
        If snapshot logging fails internally (e.g. unexpected data type),
        the tracked function must still return its result correctly.
        """
        # Store will work fine; we simulate a weird input that might confuse the profiler
        @track(name="weird", snapshot=True,
               sensitive_cols=["col"], store=self.store)
        def fn(x):
            return x * 2

        result = fn(21)   # non-DataFrame input with snapshot=True
        self.assertEqual(result, 42)

        # Step must still be logged
        steps = self.store.get_steps("weird")
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["status"], "success")

    def test_snapshot_column_count_matches_dataframe(self):
        """column_count in snapshot must equal the actual number of DataFrame columns."""
        import pandas as pd

        @track(name="count_check", snapshot=True, store=self.store)
        def fn(df):
            return df

        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "e": [5]})
        fn(df)

        snap = self.store.get_snapshots(position="before")[0]
        self.assertEqual(snap["column_count"], 5)
        self.assertEqual(len(snap["column_names"]), 5)

    def test_snapshot_column_names_match_dataframe(self):
        """column_names in snapshot must exactly match df.columns."""
        import pandas as pd

        @track(name="col_names", snapshot=True, store=self.store)
        def fn(df):
            return df

        cols = ["farm_id", "gender", "land_title", "yield_kg", "zone"]
        df = pd.DataFrame({c: [1, 2, 3] for c in cols})
        fn(df)

        snap = self.store.get_snapshots(position="before")[0]
        self.assertEqual(snap["column_names"], cols)

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in [
        TestLineageStore,
        TestSnapshots,
        TestMetrics,
        TestTrackDecorator,
        TestSnapshotViaDecorator,
        TestSnapshotIntegration,
        TestLineageContext,
        TestGlobalStore,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total    = result.testsRun
    failures = len(result.failures) + len(result.errors)
    passed   = total - failures

    print(f"\n{'='*60}")
    print(f"  {passed}/{total} tests passed  {'✓' if failures == 0 else '✗'}")
    print(f"{'='*60}")

    sys.exit(0 if failures == 0 else 1)
