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

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datalineageml.storage.sqlite_store import LineageStore
from datalineageml.trackers.decorator import track
from datalineageml.trackers.context import LineageContext


# ─────────────────────────────────────────────
# Store tests
# ─────────────────────────────────────────────

class TestLineageStore(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.store = LineageStore(db_path=self.tmp.name)

    def tearDown(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def _log(self, step_name="test_step", status="success", error=None):
        self.store.log_step(
            run_id="run-001", step_name=step_name,
            fn_module="m", fn_qualname="f",
            input_hashes={"arg_0": "abc"}, output_hash="def",
            duration_ms=10.0, started_at="2026-03-16T07:00:00",
            status=status, error=error, tags={}
        )

    def test_creates_empty_tables(self):
        self.assertEqual(self.store.get_steps(), [])

    def test_log_and_retrieve_step(self):
        self._log()
        steps = self.store.get_steps()
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["step_name"], "test_step")
        self.assertEqual(steps[0]["status"], "success")

    def test_filter_by_name(self):
        self._log("step_a")
        self._log("step_b")
        self._log("step_a")
        self.assertEqual(len(self.store.get_steps("step_a")), 2)
        self.assertEqual(len(self.store.get_steps("step_b")), 1)

    def test_clear_wipes_all(self):
        self._log()
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
        pipes = self.store.get_pipelines()
        self.assertEqual(pipes[0]["status"], "success")
        self.assertIsNotNone(pipes[0]["ended_at"])


# ─────────────────────────────────────────────
# Decorator tests
# ─────────────────────────────────────────────

class TestTrackDecorator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.store = LineageStore(db_path=self.tmp.name)

    def tearDown(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def test_logs_successful_step(self):
        @track(name="add_numbers", store=self.store)
        def add(a, b):
            return a + b

        result = add(2, 3)
        self.assertEqual(result, 5)
        steps = self.store.get_steps()
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["step_name"], "add_numbers")
        self.assertEqual(steps[0]["status"], "success")

    def test_uses_function_name_by_default(self):
        @track(store=self.store)
        def my_transform(x):
            return x * 2

        my_transform(10)
        self.assertEqual(self.store.get_steps()[0]["step_name"], "my_transform")

    def test_logs_failed_step(self):
        @track(name="failing_step", store=self.store)
        def bad_fn():
            raise ValueError("intentional")

        with self.assertRaises(ValueError):
            bad_fn()

        steps = self.store.get_steps()
        self.assertEqual(steps[0]["status"], "failed")
        self.assertIn("intentional", steps[0]["error"])

    def test_preserves_return_value(self):
        @track(store=self.store)
        def identity(x):
            return x

        self.assertEqual(identity({"k": "v"}), {"k": "v"})
        self.assertEqual(identity([1, 2, 3]), [1, 2, 3])
        self.assertIsNone(identity(None))

    def test_records_tags(self):
        import json

        @track(name="tagged", tags={"stage": "test", "ver": "1"}, store=self.store)
        def fn(x):
            return x

        fn(1)
        tags = json.loads(self.store.get_steps()[0]["tags"])
        self.assertEqual(tags["stage"], "test")
        self.assertEqual(tags["ver"], "1")

    def test_records_duration(self):
        import time

        @track(name="timed", store=self.store)
        def slow():
            time.sleep(0.05)
            return True

        slow()
        self.assertGreaterEqual(self.store.get_steps()[0]["duration_ms"], 40)

    def test_hashes_inputs(self):
        import json

        @track(name="hash_test", store=self.store)
        def fn(x, y):
            return x + y

        fn(1, 2)
        hashes = json.loads(self.store.get_steps()[0]["input_hashes"])
        self.assertIn("arg_0", hashes)
        self.assertIn("arg_1", hashes)

    def test_multiple_calls_logged_separately(self):
        @track(name="multi", store=self.store)
        def fn(x):
            return x

        fn(1); fn(2); fn(3)
        self.assertEqual(len(self.store.get_steps()), 3)

    def test_hashes_dataframe_inputs(self):
        import pandas as pd
        import json

        @track(name="df_test", store=self.store)
        def fn(df):
            return df.shape[0]

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        fn(df)
        hashes = json.loads(self.store.get_steps()[0]["input_hashes"])
        self.assertIn("arg_0", hashes)
        self.assertNotEqual(hashes["arg_0"], "")

    def test_same_data_produces_same_hash(self):
        import pandas as pd
        import json

        @track(name="hash_stability", store=self.store)
        def fn(df):
            return df

        df = pd.DataFrame({"x": [1, 2, 3]})
        fn(df)
        fn(df)
        steps = self.store.get_steps()
        h1 = json.loads(steps[0]["input_hashes"])["arg_0"]
        h2 = json.loads(steps[1]["input_hashes"])["arg_0"]
        self.assertEqual(h1, h2)

    def test_different_data_produces_different_hash(self):
        import pandas as pd
        import json

        @track(name="hash_diff", store=self.store)
        def fn(df):
            return df

        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [4, 5, 6]})
        fn(df1)
        fn(df2)
        steps = self.store.get_steps()
        h1 = json.loads(steps[0]["input_hashes"])["arg_0"]
        h2 = json.loads(steps[1]["input_hashes"])["arg_0"]
        self.assertNotEqual(h1, h2)


# ─────────────────────────────────────────────
# Context tests
# ─────────────────────────────────────────────

class TestLineageContext(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.store = LineageStore(db_path=self.tmp.name)

    def tearDown(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def test_logs_start_and_end(self):
        with LineageContext(name="test_pipe", store=self.store):
            pass

        pipes = self.store.get_pipelines()
        self.assertEqual(len(pipes), 1)
        self.assertEqual(pipes[0]["name"], "test_pipe")
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
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["step_name"], "s1")
        self.assertEqual(steps[1]["step_name"], "s2")


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls in [TestLineageStore, TestTrackDecorator, TestLineageContext]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total = result.testsRun
    failures = len(result.failures) + len(result.errors)
    passed = total - failures

    print(f"\n{'='*55}")
    print(f"  {passed}/{total} tests passed", "✓" if failures == 0 else "✗")
    print(f"{'='*55}")

    sys.exit(0 if failures == 0 else 1)
