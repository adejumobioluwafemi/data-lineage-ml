"""
tests/unit/test_sensitive_cols.py

Tests for discover_sensitive_cols, suggest_sensitive_cols,
and print_sensitive_col_report.
"""

import pytest
import io
import sys
import pandas as pd
import numpy as np

from datalineageml.analysis.sensitive_cols import (
    discover_sensitive_cols,
    suggest_sensitive_cols,
    print_sensitive_col_report,
    SensitiveColCandidate,
)

@pytest.fixture
def typical_df():
    """DataFrame with clear sensitive columns mixed with non-sensitive ones."""
    return pd.DataFrame({
        "gender":     ["F", "M", "F", "M", "F"] * 20,
        "age_group":  ["18-24", "25-34", "35-44", "45+", "18-24"] * 20,
        "income":     np.random.normal(50000, 10000, 100),
        "score":      np.random.uniform(0, 1, 100),
        "record_id":  range(100),               # should NOT be detected
        "land_title": ["registered", None, "registered", None, "registered"] * 20,
    })


@pytest.fixture
def oyo_df():
    """Oyo State-style DataFrame with Nigerian demographic terminology."""
    return pd.DataFrame({
        "gender":            ["F", "M"] * 50,
        "lga":               ["Ibadan North", "Ibadan South", "Egbeda", "Akinyele"] * 25,
        "geopolitical_zone": ["SW", "SE", "NC", "SS"] * 25,
        "tribe":             ["Yoruba", "Igbo", "Hausa", "Efik"] * 25,
        "farm_size_ha":      np.random.uniform(0.2, 5.0, 100),
        "yield_t_ha":        np.random.uniform(1.5, 4.0, 100),
        "crop_id":           range(100),         # should NOT be detected
    })


# ── SensitiveColCandidate 

def test_candidate_repr():
    c = SensitiveColCandidate(
        column="gender", confidence=0.97, reasons=["primary keyword match"],
        n_unique=2, dtype="object", top_values=["F", "M"]
    )
    assert "gender" in repr(c)
    assert "0.97" in repr(c)


# ── discover_sensitive_cols 

def test_returns_list(typical_df):
    result = discover_sensitive_cols(typical_df)
    assert isinstance(result, list)


def test_gender_detected(typical_df):
    result = discover_sensitive_cols(typical_df)
    names = [c.column for c in result]
    assert "gender" in names, f"'gender' not found in {names}"


def test_age_group_detected(typical_df):
    result = discover_sensitive_cols(typical_df)
    names = [c.column for c in result]
    assert "age_group" in names, f"'age_group' not found in {names}"


def test_record_id_not_detected(typical_df):
    """High-cardinality numeric IDs should not be detected."""
    result = discover_sensitive_cols(typical_df)
    names = [c.column for c in result]
    assert "record_id" not in names, f"'record_id' incorrectly detected as sensitive"


def test_pure_numeric_income_not_detected(typical_df):
    """Continuous numeric columns should not be detected by default."""
    result = discover_sensitive_cols(typical_df, include_numeric=False)
    names = [c.column for c in result]
    assert "income" not in names
    assert "score" not in names


def test_gender_has_high_confidence(typical_df):
    result = discover_sensitive_cols(typical_df)
    gender = next(c for c in result if c.column == "gender")
    assert gender.confidence >= 0.70, (
        f"Expected gender confidence >= 0.70, got {gender.confidence}"
    )


def test_sorted_by_confidence_descending(typical_df):
    result = discover_sensitive_cols(typical_df)
    confs = [c.confidence for c in result]
    assert confs == sorted(confs, reverse=True), "Results not sorted by confidence"


def test_min_confidence_filter(typical_df):
    high  = discover_sensitive_cols(typical_df, min_confidence=0.8)
    low   = discover_sensitive_cols(typical_df, min_confidence=0.1)
    assert len(high) <= len(low)
    assert all(c.confidence >= 0.8 for c in high)


def test_candidate_has_required_fields(typical_df):
    result = discover_sensitive_cols(typical_df)
    assert len(result) > 0
    c = result[0]
    assert hasattr(c, "column")
    assert hasattr(c, "confidence")
    assert hasattr(c, "reasons")
    assert hasattr(c, "n_unique")
    assert hasattr(c, "dtype")
    assert hasattr(c, "top_values")


def test_reasons_not_empty(typical_df):
    result = discover_sensitive_cols(typical_df)
    for c in result:
        assert len(c.reasons) > 0, f"No reasons for column '{c.column}'"


def test_n_unique_correct(typical_df):
    result = discover_sensitive_cols(typical_df)
    gender = next(c for c in result if c.column == "gender")
    assert gender.n_unique == 2


def test_top_values_populated(typical_df):
    result = discover_sensitive_cols(typical_df)
    gender = next(c for c in result if c.column == "gender")
    assert len(gender.top_values) > 0
    assert "F" in gender.top_values or "M" in gender.top_values


def test_empty_dataframe_returns_empty():
    df = pd.DataFrame()
    result = discover_sensitive_cols(df)
    assert result == []


def test_all_numeric_dataframe_returns_empty():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = discover_sensitive_cols(df, include_numeric=False)
    assert result == []


# ── Nigerian / West African terminology 

def test_lga_detected(oyo_df):
    result = discover_sensitive_cols(oyo_df)
    names = [c.column for c in result]
    assert "lga" in names, f"'lga' not found in {names}"


def test_tribe_detected(oyo_df):
    result = discover_sensitive_cols(oyo_df)
    names = [c.column for c in result]
    assert "tribe" in names, f"'tribe' not found in {names}"


def test_geopolitical_zone_detected(oyo_df):
    result = discover_sensitive_cols(oyo_df)
    names = [c.column for c in result]
    assert "geopolitical_zone" in names, (
        f"'geopolitical_zone' not found in {names}"
    )


def test_crop_id_not_detected(oyo_df):
    result = discover_sensitive_cols(oyo_df)
    names = [c.column for c in result]
    assert "crop_id" not in names


# ── Partial name matching 

def test_partial_match_gender_variant():
    df = pd.DataFrame({"gender_code": ["F", "M", "F", "M"] * 10})
    result = discover_sensitive_cols(df)
    names = [c.column for c in result]
    assert "gender_code" in names


def test_partial_match_age_variant():
    df = pd.DataFrame({"age_grp": ["18-24", "25-34", "35-44", "45+"] * 10})
    result = discover_sensitive_cols(df)
    names = [c.column for c in result]
    assert "age_grp" in names


def test_partial_match_ethnic():
    df = pd.DataFrame({"ethnic_group": ["A", "B", "C", "D"] * 10})
    result = discover_sensitive_cols(df)
    names = [c.column for c in result]
    assert "ethnic_group" in names


# ── suggest_sensitive_cols 

def test_suggest_returns_list(typical_df):
    result = suggest_sensitive_cols(typical_df)
    assert isinstance(result, list)
    assert all(isinstance(c, str) for c in result)


def test_suggest_includes_gender(typical_df):
    result = suggest_sensitive_cols(typical_df, min_confidence=0.6)
    assert "gender" in result


def test_suggest_threshold_filters(typical_df):
    high = suggest_sensitive_cols(typical_df, min_confidence=0.9)
    low  = suggest_sensitive_cols(typical_df, min_confidence=0.1)
    assert len(high) <= len(low)


# ── print_sensitive_col_report 

def test_print_report_no_error(typical_df, capsys):
    print_sensitive_col_report(typical_df)
    out = capsys.readouterr().out
    assert "gender" in out
    assert "Suggested" in out


def test_print_report_empty_df(capsys):
    print_sensitive_col_report(pd.DataFrame())
    out = capsys.readouterr().out
    assert "No candidate" in out


def test_print_report_shows_confidence(typical_df, capsys):
    print_sensitive_col_report(typical_df, min_confidence=0.5)
    out = capsys.readouterr().out
    assert "0." in out   # confidence values should be printed