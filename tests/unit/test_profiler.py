"""
tests/unit/test_profiler.py

Tests for DataFrameProfiler and print_snapshot_comparison.
Run with: python run_tests.py  OR  pytest tests/unit/test_profiler.py -v
"""

import pytest
import math
import io
import sys
import pandas as pd
import numpy as np

from datalineageml.analysis.profiler import (
    DataFrameProfiler,
    print_snapshot_comparison,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "age":    [25, 32, 45, 28, 38, 52, 29, 41],
        "income": [50000, 72000, 65000, 48000, 91000, 88000, 47000, 76000],
        "score":  [0.72, 0.85, 0.90, 0.55, 0.88, 0.92, 0.58, 0.81],
        "gender": ["F", "M", "M", "F", "M", "M", "F", "M"],
        "target": [0, 1, 1, 0, 1, 1, 0, 1],
    })


@pytest.fixture
def df_with_nulls():
    return pd.DataFrame({
        "age":       [25, None, 45, 28, None, 52],
        "income":    [50000, 72000, None, 48000, 91000, None],
        "gender":    ["F", "M", "M", "F", "M", "F"],
        "land_title":[None, "yes", "yes", None, "yes", None],
    })


@pytest.fixture
def oyo_df():
    """Mirrors the Oyo State structural inequality pattern."""
    np.random.seed(42)
    n = 100
    gender = np.random.choice(["F", "M"], n, p=[0.4, 0.6])
    is_f   = gender == "F"
    land   = np.where(
        is_f,
        np.random.choice(["registered", None], n, p=[0.11, 0.89]), # type: ignore
        np.random.choice(["registered", None], n, p=[0.67, 0.33]), # type: ignore
    ).tolist()
    return pd.DataFrame({
        "gender":     gender,
        "land_title": land,
        "yield_t_ha": np.random.uniform(1.5, 4.0, n).round(2),
    })


# ── basic API ─────────────────────────────────────────────────────────────────

def test_profile_returns_dict(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="test", position="before")
    assert isinstance(result, dict)


def test_profile_required_keys_present(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="test", position="before")
    required = {"run_id", "step_name", "position", "row_count", "column_count",
                "column_names", "null_rates", "numeric_stats",
                "categorical_stats", "sensitive_stats", "recorded_at"}
    assert required.issubset(result.keys())


def test_profile_step_name_and_position_stored(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="my_step", position="after")
    assert result["step_name"] == "my_step"
    assert result["position"]  == "after"


def test_profile_run_id_auto_generated(simple_df):
    profiler = DataFrameProfiler()
    r1 = profiler.profile(simple_df, step_name="s", position="before")
    r2 = profiler.profile(simple_df, step_name="s", position="before")
    assert r1["run_id"] != r2["run_id"]  # auto-generated, unique each call


def test_profile_explicit_run_id(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="s", position="before",
                              run_id="my-run-123")
    assert result["run_id"] == "my-run-123"


def test_profile_invalid_position_raises(simple_df):
    profiler = DataFrameProfiler()
    with pytest.raises(ValueError, match="position must be"):
        profiler.profile(simple_df, step_name="s", position="middle")


def test_profile_non_dataframe_raises():
    profiler = DataFrameProfiler()
    with pytest.raises(TypeError, match="DataFrame"):
        profiler.profile([1, 2, 3], step_name="s", position="before")


# ── shape stats ───────────────────────────────────────────────────────────────

def test_row_count(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="s", position="before")
    assert result["row_count"] == 8


def test_column_count(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="s", position="before")
    assert result["column_count"] == 5


def test_column_names_match(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="s", position="before")
    assert result["column_names"] == list(simple_df.columns)


# ── null rates ────────────────────────────────────────────────────────────────

def test_null_rates_zero_when_no_nulls(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="s", position="before")
    for col, rate in result["null_rates"].items():
        assert rate == 0.0, f"Expected 0 null rate for {col}, got {rate}"


def test_null_rates_nonzero_when_nulls_present(df_with_nulls):
    profiler = DataFrameProfiler()
    result = profiler.profile(df_with_nulls, step_name="s", position="before")
    assert result["null_rates"]["age"]        > 0.0
    assert result["null_rates"]["income"]     > 0.0
    assert result["null_rates"]["land_title"] > 0.0


def test_null_rates_correct_fraction(df_with_nulls):
    profiler = DataFrameProfiler()
    result = profiler.profile(df_with_nulls, step_name="s", position="before")
    # age has 2 nulls out of 6 rows
    assert pytest.approx(result["null_rates"]["age"], abs=1e-4) == 2/6


def test_null_rates_all_columns_present(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="s", position="before")
    assert set(result["null_rates"].keys()) == set(simple_df.columns)


# ── numeric stats ─────────────────────────────────────────────────────────────

def test_numeric_stats_keys(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="s", position="before")
    for col in ["age", "income", "score"]:
        assert col in result["numeric_stats"]
        stats = result["numeric_stats"][col]
        for key in ("mean", "std", "min", "max", "p25", "p75", "n"):
            assert key in stats, f"Missing '{key}' in numeric_stats[{col!r}]"


def test_numeric_stats_mean_correct(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="s", position="before")
    expected_mean = float(simple_df["age"].mean())
    assert pytest.approx(result["numeric_stats"]["age"]["mean"],
                         abs=1e-3) == expected_mean


def test_numeric_stats_n_correct(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="s", position="before")
    assert result["numeric_stats"]["age"]["n"] == 8


def test_numeric_stats_excludes_non_numeric(simple_df):
    profiler = DataFrameProfiler()
    result = profiler.profile(simple_df, step_name="s", position="before")
    assert "gender" not in result["numeric_stats"]
    assert "target" in result["numeric_stats"]  # target is numeric (0/1)


def test_numeric_stats_skips_all_null_column():
    df = pd.DataFrame({"x": [None, None, None], "y": [1, 2, 3]})
    profiler = DataFrameProfiler()
    result = profiler.profile(df, step_name="s", position="before")
    assert "x" not in result["numeric_stats"]
    assert "y" in result["numeric_stats"]


# ── sensitive stats ───────────────────────────────────────────────────────────

def test_sensitive_stats_fractions_sum_to_one(simple_df):
    profiler = DataFrameProfiler(sensitive_cols=["gender"])
    result = profiler.profile(simple_df, step_name="s", position="before")
    total = sum(result["sensitive_stats"]["gender"].values())
    assert pytest.approx(total, abs=1e-4) == 1.0


def test_sensitive_stats_correct_fractions(simple_df):
    # 3 F, 5 M in simple_df
    profiler = DataFrameProfiler(sensitive_cols=["gender"])
    result = profiler.profile(simple_df, step_name="s", position="before")
    dist = result["sensitive_stats"]["gender"]
    assert pytest.approx(dist["F"], abs=1e-4) == 3/8
    assert pytest.approx(dist["M"], abs=1e-4) == 5/8


def test_sensitive_stats_null_tracked_as_null_key(df_with_nulls):
    # land_title has nulls — they should appear as "__null__"
    profiler = DataFrameProfiler(sensitive_cols=["land_title"])
    result = profiler.profile(df_with_nulls, step_name="s", position="before")
    assert "__null__" in result["sensitive_stats"]["land_title"]


def test_sensitive_stats_missing_column_silently_skipped(simple_df):
    profiler = DataFrameProfiler(sensitive_cols=["nonexistent_col", "gender"])
    result = profiler.profile(simple_df, step_name="s", position="before")
    assert "nonexistent_col" not in result["sensitive_stats"]
    assert "gender" in result["sensitive_stats"]


def test_sensitive_stats_not_in_categorical_stats(simple_df):
    """Sensitive cols should not also appear in categorical_stats."""
    profiler = DataFrameProfiler(sensitive_cols=["gender"])
    result = profiler.profile(simple_df, step_name="s", position="before")
    assert "gender" not in result["categorical_stats"]


def test_multiple_sensitive_cols(simple_df):
    df = simple_df.copy()
    df["zone"] = ["A", "B", "A", "B", "A", "B", "A", "B"]
    profiler = DataFrameProfiler(sensitive_cols=["gender", "zone"])
    result = profiler.profile(df, step_name="s", position="before")
    assert "gender" in result["sensitive_stats"]
    assert "zone"   in result["sensitive_stats"]


# ── Oyo State scenario ────────────────────────────────────────────────────────

def test_oyo_dropna_reduces_female_proportion(oyo_df):
    """Core scenario: dropna on land_title column removes more female records."""
    profiler = DataFrameProfiler(sensitive_cols=["gender"])

    before = profiler.profile(oyo_df, step_name="clean", position="before")
    cleaned = oyo_df.dropna()
    after  = profiler.profile(cleaned, step_name="clean", position="after")

    f_before = before["sensitive_stats"]["gender"].get("F", 0.0)
    f_after  = after["sensitive_stats"]["gender"].get("F", 0.0)

    assert f_before > f_after, (
        f"Expected female proportion to drop after dropna, "
        f"got before={f_before:.3f}, after={f_after:.3f}"
    )


def test_oyo_shift_is_significant(oyo_df):
    """The demographic shift from dropna should exceed 0.10 (HIGH threshold)."""
    profiler = DataFrameProfiler(sensitive_cols=["gender"])

    before  = profiler.profile(oyo_df, step_name="clean", position="before")
    after   = profiler.profile(oyo_df.dropna(), step_name="clean", position="after")

    f_before = before["sensitive_stats"]["gender"].get("F", 0.0)
    f_after  = after["sensitive_stats"]["gender"].get("F", 0.0)
    shift    = f_before - f_after

    assert shift > 0.10, (
        f"Expected shift > 0.10 (HIGH), got {shift:.3f}. "
        "Check synthetic data parameters."
    )


# ── sampling ──────────────────────────────────────────────────────────────────

def test_sampling_does_not_affect_sensitive_stats():
    """Sensitive stats always use the full DataFrame even when sampling."""
    np.random.seed(1)
    n = 1000
    df = pd.DataFrame({
        "x":      np.random.randn(n),
        "gender": np.random.choice(["F", "M"], n, p=[0.4, 0.6]),
    })
    # Sample size smaller than DataFrame
    profiler = DataFrameProfiler(sensitive_cols=["gender"], sample_size=100)
    result   = profiler.profile(df, step_name="s", position="before")

    # Row count must reflect full DataFrame
    assert result["row_count"] == n

    # Sensitive stats fractions must still sum to 1.0
    total = sum(result["sensitive_stats"]["gender"].values())
    assert pytest.approx(total, abs=1e-4) == 1.0


def test_no_sampling_when_under_limit(simple_df):
    profiler = DataFrameProfiler(sample_size=1000)  # limit > DataFrame size
    result   = profiler.profile(simple_df, step_name="s", position="before")
    assert result["row_count"] == len(simple_df)


# ── store integration ─────────────────────────────────────────────────────────

def test_profile_compatible_with_log_snapshot(simple_df, tmp_path):
    """Profile output must be directly passable to store.log_snapshot()."""
    from datalineageml.storage.sqlite_store import LineageStore

    store    = LineageStore(db_path=str(tmp_path / "test.db"))
    profiler = DataFrameProfiler(sensitive_cols=["gender"])
    profile  = profiler.profile(simple_df, step_name="clean", position="before")

    # This must not raise
    store.log_snapshot(**profile)

    snaps = store.get_snapshots()
    assert len(snaps) == 1
    assert snaps[0]["row_count"] == 8
    assert snaps[0]["sensitive_stats"]["gender"]["F"] == pytest.approx(3/8, abs=1e-4)
    store.close()


# ── print_snapshot_comparison ─────────────────────────────────────────────────

def test_print_snapshot_comparison_runs_without_error(simple_df):
    """print_snapshot_comparison must not raise for valid before/after profiles."""
    profiler = DataFrameProfiler(sensitive_cols=["gender"])
    before   = profiler.profile(simple_df, step_name="s", position="before")
    after    = profiler.profile(
        simple_df.dropna(), step_name="s", position="after"
    )
    # Capture stdout to avoid polluting test output
    captured = io.StringIO()
    sys.stdout = captured
    try:
        print_snapshot_comparison(before, after, step_name="test_step")
    finally:
        sys.stdout = sys.__stdout__

    output = captured.getvalue()
    assert "test_step" in output
    assert "gender"    in output


def test_print_snapshot_comparison_shows_high_shift(oyo_df):
    """⚠ HIGH SHIFT warning must appear when shift > 0.10."""
    profiler = DataFrameProfiler(sensitive_cols=["gender"])
    before   = profiler.profile(oyo_df,          step_name="s", position="before")
    after    = profiler.profile(oyo_df.dropna(), step_name="s", position="after")

    captured = io.StringIO()
    sys.stdout = captured
    try:
        print_snapshot_comparison(before, after, step_name="oyo_clean")
    finally:
        sys.stdout = sys.__stdout__

    assert "HIGH SHIFT" in captured.getvalue()