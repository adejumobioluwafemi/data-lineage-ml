"""
tests/unit/test_report.py

Tests for generate_report — HTML audit report export.
"""

import pytest
import tempfile
import os
import numpy as np
import pandas as pd

from datalineageml.report import generate_report
from datalineageml.storage.sqlite_store import LineageStore


@pytest.fixture
def store(tmp_path):
    s = LineageStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store pre-populated with steps, pipeline, snapshots, and metrics."""
    store.log_pipeline_start(
        pipeline_id="pipe-001", name="test_pipeline",
        started_at="2026-03-26T07:00:00"
    )
    store.log_step(
        run_id="run-001", step_name="load_data",
        fn_module="pipeline", fn_qualname="load_data",
        input_hashes={}, output_hash="abc123",
        duration_ms=5.2, started_at="2026-03-26T07:00:00",
        status="success", error=None, tags={}
    )
    store.log_step(
        run_id="run-002", step_name="clean_data",
        fn_module="pipeline", fn_qualname="clean_data",
        input_hashes={"arg_0": "abc123"}, output_hash="def456",
        duration_ms=18.7, started_at="2026-03-26T07:00:01",
        status="success", error=None, tags={"stage": "preprocessing"}
    )
    store.log_snapshot(
        run_id="run-002", step_name="clean_data", position="before",
        row_count=200, column_count=3, column_names=["gender", "income", "score"],
        null_rates={"gender": 0.0, "income": 0.3},
        numeric_stats={"income": {"mean": 50000, "std": 10000,
                                  "min": 20000, "max": 90000,
                                  "p25": 42000, "p75": 62000, "n": 200}},
        categorical_stats={},
        sensitive_stats={"gender": {"F": 0.50, "M": 0.50}},
        recorded_at="2026-03-26T07:00:01"
    )
    store.log_snapshot(
        run_id="run-002", step_name="clean_data", position="after",
        row_count=140, column_count=3, column_names=["gender", "income", "score"],
        null_rates={"gender": 0.0, "income": 0.0},
        numeric_stats={"income": {"mean": 58000, "std": 9000,
                                  "min": 25000, "max": 90000,
                                  "p25": 51000, "p75": 67000, "n": 140}},
        categorical_stats={},
        sensitive_stats={"gender": {"F": 0.22, "M": 0.78}},
        recorded_at="2026-03-26T07:00:02"
    )
    store.log_metrics(
        run_id="run-002",
        metrics={"gender_jsd": 0.0731, "gender_representation_shift": 0.28},
        metric_source="ShiftDetector",
        step_name="clean_data",
    )
    store.log_pipeline_end(
        pipeline_id="pipe-001", status="success",
        ended_at="2026-03-26T07:05:00"
    )
    return store


def test_generate_report_creates_file(populated_store, tmp_path):
    out = str(tmp_path / "report.html")
    generate_report(store=populated_store, output_path=out)
    assert os.path.exists(out)


def test_generate_report_returns_path(populated_store, tmp_path):
    out = str(tmp_path / "report.html")
    result = generate_report(store=populated_store, output_path=out)
    assert result == out


def test_report_is_valid_html(populated_store, tmp_path):
    out = str(tmp_path / "report.html")
    generate_report(store=populated_store, output_path=out)
    content = open(out).read()
    assert content.startswith("<!DOCTYPE html>")
    assert "</html>" in content


def test_report_has_title(populated_store, tmp_path):
    out = str(tmp_path / "report.html")
    generate_report(
        store=populated_store, output_path=out,
        title="My Custom Audit Report"
    )
    content = open(out).read()
    assert "My Custom Audit Report" in content


def test_report_has_pipeline_name(populated_store, tmp_path):
    out = str(tmp_path / "report.html")
    generate_report(
        store=populated_store, output_path=out,
        pipeline_name="oyo_subsidy_pipeline"
    )
    content = open(out).read()
    assert "oyo_subsidy_pipeline" in content


def test_report_has_step_names(populated_store, tmp_path):
    out = str(tmp_path / "report.html")
    generate_report(store=populated_store, output_path=out)
    content = open(out).read()
    assert "load_data"  in content
    assert "clean_data" in content


def test_report_has_metrics(populated_store, tmp_path):
    out = str(tmp_path / "report.html")
    generate_report(store=populated_store, output_path=out)
    content = open(out).read()
    assert "gender_jsd" in content


def test_report_has_demographic_snapshots(populated_store, tmp_path):
    out = str(tmp_path / "report.html")
    generate_report(store=populated_store, output_path=out,
                    sensitive_col="gender")
    content = open(out).read()
    # Should show the gender distribution snapshots
    assert "gender" in content
    assert "50.0%" in content or "50%" in content   # F proportion before


def test_report_is_self_contained(populated_store, tmp_path):
    """HTML should not reference external resources that could fail offline."""
    out = str(tmp_path / "report.html")
    generate_report(store=populated_store, output_path=out)
    content = open(out).read()
    # No CDN links
    assert "cdn." not in content
    assert "googleapis" not in content
    assert "bootstrap" not in content


def test_report_with_attribution(populated_store, tmp_path):
    """Attribution result should appear in report when provided."""
    out = str(tmp_path / "report.html")
    mock_attribution = {
        "attributed_step":  "clean_data",
        "column":           "gender",
        "test":             "jsd",
        "stat":             0.0731,
        "confidence":       0.95,
        "flag":             "HIGH",
        "rows_removed":     60,
        "removal_rate":     0.30,
        "bias_metric":      {},
        "all_scores":       [],
        "evidence":         "The gender distribution shifted at clean_data.",
        "recommendation":   "Replace dropna() with stratified imputation.",
    }
    generate_report(
        store=populated_store, output_path=out,
        attribution_result=mock_attribution
    )
    content = open(out).read()
    assert "Causal Attribution" in content
    assert "clean_data" in content
    assert "stratified imputation" in content


def test_report_with_counterfactual(populated_store, tmp_path):
    """Counterfactual result should appear in report when provided."""
    out = str(tmp_path / "report.html")
    mock_cf = {
        "replace_step":       "clean_data",
        "sensitive_col":      "gender",
        "dist_before_fix":    {"F": 0.22, "M": 0.78},
        "dist_after_fix":     {"F": 0.50, "M": 0.50},
        "dist_original_input":{"F": 0.50, "M": 0.50},
        "biased_rows_out":    140,
        "fixed_rows_out":     200,
        "rows_recovered":     60,
        "bias_metric_before": 0.34,
        "bias_metric_after":  0.09,
        "bias_reduction":     0.25,
        "bias_reduction_pct": 73.5,
        "jsd_improvement":    0.06,
        "verdict":            "STRONG",
        "verdict_detail":     "Bias reduced by 73.5%.",
        "biased_timing":      {},
        "fixed_timing":       {},
        "_biased_snaps":      {},
        "_fixed_snaps":       {},
    }
    generate_report(
        store=populated_store, output_path=out,
        counterfactual_result=mock_cf
    )
    content = open(out).read()
    assert "Counterfactual" in content
    assert "STRONG" in content
    assert "73" in content


def test_report_empty_store_no_crash(tmp_path, store):
    """Report should generate even on an empty store."""
    out = str(tmp_path / "report.html")
    generate_report(store=store, output_path=out)
    assert os.path.exists(out)
    content = open(out).read()
    assert "<!DOCTYPE html>" in content


def test_report_file_size_reasonable(populated_store, tmp_path):
    """The HTML file should be at least 3KB and at most 500KB."""
    out = str(tmp_path / "report.html")
    generate_report(store=populated_store, output_path=out)
    size_kb = os.path.getsize(out) / 1024
    assert size_kb >= 3,   f"Report too small: {size_kb:.1f} KB"
    assert size_kb <= 500, f"Report too large: {size_kb:.1f} KB"


def test_report_escapes_html_in_pipeline_name(tmp_path, store):
    """Pipeline names with HTML characters should be escaped."""
    out = str(tmp_path / "report.html")
    generate_report(
        store=store, output_path=out,
        pipeline_name="<script>alert('xss')</script>"
    )
    content = open(out).read()
    assert "<script>" not in content
    assert "&lt;script&gt;" in content


def test_report_escapes_html_in_title(tmp_path, store):
    out = str(tmp_path / "report.html")
    generate_report(
        store=store, output_path=out,
        title="Report & Analysis <2026>"
    )
    content = open(out).read()
    # The title should be escaped — no raw < > characters in the title tag
    import re
    title_match = re.search(r"<title>(.*?)</title>", content)
    assert title_match is not None
    title_text = title_match.group(1)
    assert "<" not in title_text
    assert ">" not in title_text