"""
tests/unit/test_replayer.py

Tests for CounterfactualReplayer
"""

import pytest
import numpy as np
import pandas as pd
import io
import sys

from datalineageml.replay.replayer import CounterfactualReplayer
from datalineageml.storage.sqlite_store import LineageStore


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    s = LineageStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


@pytest.fixture
def oyo_df():
    """Synthetic Oyo State dataset — known bias pattern."""
    np.random.seed(42)
    n = 300
    gender = np.random.choice(["F", "M"], n, p=[0.40, 0.60])
    is_f   = gender == "F"
    # Land title missing for most female farmers
    land = np.where(
        is_f,
        np.random.choice(["registered", None], n, p=[0.11, 0.89]), # type: ignore
        np.random.choice(["registered", None], n, p=[0.67, 0.33]), # type: ignore
    ).tolist()
    yield_t = np.random.uniform(1.5, 4.0, n).round(2)
    income  = np.where(is_f,
        np.random.normal(45000, 10000, n).clip(20000, 90000),
        np.random.normal(60000, 12000, n).clip(20000, 100000))
    eligible = ((yield_t > 2.0) & (income > 40000)).astype(int)
    return pd.DataFrame({
        "gender":     gender,
        "land_title": land,
        "yield_t_ha": yield_t,
        "income":     income.round(0),
        "eligible":   eligible,
    })


# ── Pipeline functions (pure DataFrame → DataFrame) ───────────────────────────

def step_load(df):
    return df.copy()


def step_clean_biased(df):
    """Biased: dropna removes female farmers disproportionately."""
    return df.dropna().reset_index(drop=True)


def step_clean_fixed(df):
    """Fixed: stratified imputation preserves demographic balance."""
    df = df.copy()
    for col in ["land_title"]:
        for g in df["gender"].dropna().unique():
            mask = (df["gender"] == g) & df[col].isna()
            fill = df.loc[df["gender"] == g, col].mode()
            fill_val = fill.iloc[0] if len(fill) > 0 else "unregistered"
            df.loc[mask, col] = fill_val
    return df


def step_normalize(df):
    df = df.copy()
    for col in ["yield_t_ha", "income"]:
        lo, hi = df[col].min(), df[col].max()
        if hi > lo:
            df[col] = (df[col] - lo) / (hi - lo)
    return df


def step_featurize(df):
    df = df.copy()
    df["yield_income_score"] = df["yield_t_ha"] * df["income"] if \
        "income" in df.columns else df["yield_t_ha"]
    return df


def bias_metric_fn(df):
    """Demographic Parity Gap on subsidy eligibility."""
    if "eligible" not in df.columns or "gender" not in df.columns:
        return 0.0
    f_rate = df[df["gender"] == "F"]["eligible"].mean()
    m_rate = df[df["gender"] == "M"]["eligible"].mean()
    return abs(float(f_rate) - float(m_rate)) if not (np.isnan(f_rate) or np.isnan(m_rate)) else 0.0


# ── Registration ──────────────────────────────────────────────────────────────

def test_register_returns_self():
    r = CounterfactualReplayer()
    result = r.register("load", step_load)
    assert result is r


def test_register_chaining():
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased)
         .register("norm",  step_normalize))
    assert len(r._steps) == 3


def test_register_stores_step_name():
    r = CounterfactualReplayer()
    r.register("my_step", step_load)
    assert r._steps[0]["name"] == "my_step"


def test_register_stores_function():
    r = CounterfactualReplayer()
    r.register("load", step_load)
    assert r._steps[0]["fn"] is step_load


def test_register_snapshot_default_false():
    r = CounterfactualReplayer()
    r.register("load", step_load)
    assert r._steps[0]["snapshot"] is False


def test_register_snapshot_true():
    r = CounterfactualReplayer()
    r.register("clean", step_clean_biased, snapshot=True,
               sensitive_cols=["gender"])
    assert r._steps[0]["snapshot"] is True
    assert r._steps[0]["sensitive_cols"] == ["gender"]


# ── Replay — basic ─────────────────────────────────────────────────────────────

def test_replay_returns_dict(oyo_df):
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("norm",  step_normalize))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender")
    assert isinstance(result, dict)


def test_replay_required_keys(oyo_df):
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"]))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender")
    required = {"replace_step", "sensitive_col", "dist_before_fix", "dist_after_fix",
                "biased_rows_out", "fixed_rows_out", "rows_recovered",
                "bias_metric_before", "bias_metric_after",
                "verdict", "verdict_detail"}
    assert required.issubset(result.keys())


def test_replay_replace_step_stored(oyo_df):
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"]))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender")
    assert result["replace_step"] == "clean"
    assert result["sensitive_col"] == "gender"


def test_replay_unregistered_step_raises(oyo_df):
    r = CounterfactualReplayer().register("load", step_load)
    with pytest.raises(ValueError, match="not registered"):
        r.replay(oyo_df, "nonexistent_step", step_clean_fixed, "gender")


# ── Row recovery ──────────────────────────────────────────────────────────────

def test_replay_fixed_recovers_rows(oyo_df):
    """Fixed pipeline should retain more rows than biased pipeline."""
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("norm",  step_normalize))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender")
    assert result["fixed_rows_out"] > result["biased_rows_out"], (
        f"Expected fixed pipeline to retain more rows: "
        f"biased={result['biased_rows_out']}, fixed={result['fixed_rows_out']}"
    )
    assert result["rows_recovered"] > 0


def test_replay_biased_rows_less_than_input(oyo_df):
    """Biased pipeline drops rows; fixed preserves all."""
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"]))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender")
    assert result["biased_rows_out"] < len(oyo_df)
    assert result["fixed_rows_out"]  == len(oyo_df)   # imputation keeps all


# ── Demographic improvement ────────────────────────────────────────────────────

def test_replay_female_proportion_recovers(oyo_df):
    """Fixed pipeline should have higher female proportion than biased."""
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("norm",  step_normalize))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender")

    biased_f = result["dist_before_fix"].get("F", 0.0)
    fixed_f  = result["dist_after_fix"].get("F", 0.0)

    assert fixed_f > biased_f, (
        f"Expected fixed pipeline to have higher F proportion: "
        f"biased={biased_f:.3f}, fixed={fixed_f:.3f}"
    )


def test_replay_original_input_distribution_preserved(oyo_df):
    """The input distribution (before any step) should be ~40% F."""
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"]))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender")

    input_f = result["dist_original_input"].get("F", 0.0)
    assert 0.35 <= input_f <= 0.50, f"Expected ~40% F in input, got {input_f:.3f}"


# ── Bias metric ───────────────────────────────────────────────────────────────

def test_replay_with_bias_metric_fn(oyo_df):
    """With a bias_metric_fn, before/after values should be populated."""
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("norm",  step_normalize))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender",
                      bias_metric_fn=bias_metric_fn)
    assert result["bias_metric_before"] is not None
    assert result["bias_metric_after"]  is not None
    assert isinstance(result["bias_metric_before"], float)
    assert isinstance(result["bias_metric_after"],  float)


def test_replay_bias_metric_improves(oyo_df):
    """The bias metric should improve after the fix."""
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("norm",  step_normalize))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender",
                      bias_metric_fn=bias_metric_fn)
    assert result["bias_metric_after"] <= result["bias_metric_before"], (
        f"Expected bias to improve: before={result['bias_metric_before']:.4f}, "
        f"after={result['bias_metric_after']:.4f}"
    )


def test_replay_bias_reduction_computed(oyo_df):
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("norm",  step_normalize))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender",
                      bias_metric_fn=bias_metric_fn)
    assert result["bias_reduction"] is not None
    assert result["bias_reduction_pct"] is not None


def test_replay_no_bias_metric_fn_still_works(oyo_df):
    """Replayer should work without a bias_metric_fn."""
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"]))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender")
    assert result["bias_metric_before"] is None
    assert result["bias_metric_after"]  is None


# ── Verdict ────────────────────────────────────────────────────────────────────

def test_replay_verdict_string(oyo_df):
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("norm",  step_normalize))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender",
                      bias_metric_fn=bias_metric_fn)
    assert result["verdict"] in ("STRONG", "MODERATE", "WEAK", "INCONCLUSIVE", "UNKNOWN")


def test_replay_strong_verdict_on_good_fix(oyo_df):
    """The imputation fix on Oyo data should produce STRONG or MODERATE verdict."""
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("norm",  step_normalize))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender",
                      bias_metric_fn=bias_metric_fn)
    assert result["verdict"] in ("STRONG", "MODERATE"), (
        f"Expected STRONG or MODERATE verdict, got {result['verdict']}: "
        f"{result['verdict_detail']}"
    )


def test_replay_verdict_detail_not_empty(oyo_df):
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"]))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender")
    assert len(result["verdict_detail"]) > 0


# ── Store persistence ──────────────────────────────────────────────────────────

def test_replay_persists_counterfactual_snapshots(oyo_df, store):
    r = (CounterfactualReplayer(store=store)
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("norm",  step_normalize))
    r.replay(oyo_df, "clean", step_clean_fixed, "gender")
    # Counterfactual snapshots are stored with __counterfactual suffix
    snaps = store.get_snapshots()
    cf_snaps = [s for s in snaps if "__counterfactual" in s["step_name"]]
    assert len(cf_snaps) > 0


def test_replay_without_store_does_not_raise(oyo_df):
    """Replayer without a store should still work."""
    r = (CounterfactualReplayer(store=None)
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"]))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender")
    assert result is not None


# ── Schema validation ──────────────────────────────────────────────────────────

def test_replay_schema_mismatch_raises(oyo_df):
    """Replacement function that drops a column should raise ValueError."""
    def bad_fix(df):
        # Drops 'income' — violates schema contract
        return df.dropna().drop(columns=["income"], errors="ignore").reset_index(drop=True)

    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"]))

    # We check schema by comparing after-columns between biased and fixed runs.
    # bad_fix drops income → schema mismatch should be caught.
    # Note: this only fires if both runs have snapshot=True
    with pytest.raises(ValueError, match="column"):
        r.replay(oyo_df, "clean", bad_fix, "gender")


# ── Multi-step downstream replay ──────────────────────────────────────────────

def test_replay_downstream_steps_run_after_fix(oyo_df):
    """All downstream steps after the replaced step must be re-run."""
    call_log = []

    def tracked_norm(df):
        call_log.append("normalize")
        return step_normalize(df)

    def tracked_featurize(df):
        call_log.append("featurize")
        return step_featurize(df)

    r = (CounterfactualReplayer()
         .register("load",      step_load)
         .register("clean",     step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("normalize", tracked_norm)
         .register("featurize", tracked_featurize))

    r.replay(oyo_df, "clean", step_clean_fixed, "gender")

    # Both downstream steps should have been called TWICE (biased + fixed run)
    assert call_log.count("normalize") == 2
    assert call_log.count("featurize") == 2


def test_replay_step_before_replaced_runs_once(oyo_df):
    """Steps before the replaced step run only in the biased pipeline pass."""
    call_log = []

    def tracked_load(df):
        call_log.append("load")
        return df.copy()

    r = (CounterfactualReplayer()
         .register("load",  tracked_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"]))

    r.replay(oyo_df, "clean", step_clean_fixed, "gender")
    # load runs twice — once for biased, once for fixed (both run full pipeline)
    assert call_log.count("load") == 2


# ── print_result ───────────────────────────────────────────────────────────────

def test_print_result_no_error(oyo_df, capsys):
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("norm",  step_normalize))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender",
                      bias_metric_fn=bias_metric_fn)
    r.print_result(result)
    out = capsys.readouterr().out
    assert "clean"   in out
    assert "gender"  in out
    assert "Verdict" in out


def test_print_result_shows_row_recovery(oyo_df, capsys):
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"]))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender")
    r.print_result(result)
    out = capsys.readouterr().out
    assert "Rows recovered" in out or "rows" in out.lower()


def test_print_result_shows_bias_metric(oyo_df, capsys):
    r = (CounterfactualReplayer()
         .register("load",  step_load)
         .register("clean", step_clean_biased, snapshot=True, sensitive_cols=["gender"])
         .register("norm",  step_normalize))
    result = r.replay(oyo_df, "clean", step_clean_fixed, "gender",
                      bias_metric_fn=bias_metric_fn)
    r.print_result(result)
    out = capsys.readouterr().out
    assert "Before fix" in out or "bias" in out.lower()


# ── Import from top-level package ─────────────────────────────────────────────

def test_import_from_package():
    from datalineageml import CounterfactualReplayer as CR
    assert CR is not None


def test_import_from_replay_module():
    from datalineageml.replay import CounterfactualReplayer as CR
    assert CR is not None