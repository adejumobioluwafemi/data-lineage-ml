"""
src/datalineageml/report.py

Audit report export — generate a self-contained HTML report
from a pipeline's lineage store.

Produces a single .html file containing:
  - Pipeline run summary
  - Shift detection results (JSD + KS table)
  - Attribution finding
  - Counterfactual comparison (if provided)
  - Demographic distribution charts (pure HTML/CSS — no dependencies)

The HTML file is fully self-contained (no external resources) so it can
be emailed, attached to a regulatory submission, or uploaded to a
shared drive without losing any content.

Usage:
    from datalineageml.report import generate_report

    generate_report(
        store=store,
        output_path="oyo_audit_report.html",
        pipeline_name="oyo_subsidy_pipeline_v1",
        sensitive_col="gender",
        attribution_result=attribution_result,   # optional
        counterfactual_result=cf_result,          # optional
        title="Oyo State Agricultural Subsidy Pipeline — Fairness Audit",
    )
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Dict, List, Optional


def generate_report(
    store,
    output_path:           str,
    pipeline_name:         str = "",
    sensitive_col:         str = "gender",
    attribution_result:    Optional[Dict] = None,
    counterfactual_result: Optional[Dict] = None,
    title:                 str = "DataLineageML — Fairness Audit Report",
) -> str:
    """Generate a self-contained HTML audit report.

    Args:
        store:                 ``LineageStore`` instance.
        output_path:           Path to write the HTML file.
        pipeline_name:         Name shown in the report header.
        sensitive_col:         Demographic column that was analysed.
        attribution_result:    Output of ``CausalAttributor.attribute()``.
        counterfactual_result: Output of ``CounterfactualReplayer.replay()``.
        title:                 Report title shown in the browser tab and header.

    Returns:
        The ``output_path`` string (for chaining / logging).
    """
    steps     = store.get_steps()
    pipelines = store.get_pipelines()
    snaps     = store.get_snapshots()
    metrics   = store.get_metrics()

    # Run ShiftDetector on the store
    shift_results = []
    try:
        from datalineageml.analysis.shift_detector import ShiftDetector
        shift_results = ShiftDetector(store=store).detect()
    except Exception:
        pass

    doc = _build_html(
        title=title,
        pipeline_name=pipeline_name,
        sensitive_col=sensitive_col,
        steps=steps,
        pipelines=pipelines,
        snaps=snaps,
        metrics=metrics,
        shift_results=shift_results,
        attribution=attribution_result,
        counterfactual=counterfactual_result,
        generated_at=datetime.utcnow().isoformat(),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc)

    return output_path


# ── HTML builder

def _build_html(title, pipeline_name, sensitive_col, steps, pipelines,
                snaps, metrics, shift_results, attribution,
                counterfactual, generated_at) -> str:

    sections = [
        _section_header(title, pipeline_name, generated_at),
        _section_pipeline_summary(pipelines, steps),
        _section_shift_report(shift_results, sensitive_col),
        _section_demographic_snapshots(snaps, sensitive_col),
    ]

    if attribution:
        sections.append(_section_attribution(attribution))

    if counterfactual:
        sections.append(_section_counterfactual(counterfactual))

    if metrics:
        sections.append(_section_metrics(metrics))

    sections.append(_section_footer())

    body = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
{_CSS}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _section_header(title, pipeline_name, generated_at) -> str:
    sub = f"Pipeline: {html.escape(pipeline_name)}" if pipeline_name else ""
    return f"""
<div class="header">
  <div class="logo">DataLineageML</div>
  <h1>{html.escape(title)}</h1>
  {f'<p class="sub">{sub}</p>' if sub else ''}
  <p class="meta">Generated: {generated_at[:19]} UTC</p>
</div>"""


def _section_pipeline_summary(pipelines, steps) -> str:
    if not pipelines and not steps:
        return ""
    rows = ""
    for p in pipelines:
        status_cls = "ok" if p["status"] == "success" else "fail"
        rows += f"""<tr>
          <td>{html.escape(p.get('name',''))}</td>
          <td class="mono">{p.get('started_at','')[:19]}</td>
          <td class="mono">{p.get('ended_at','')[:19] if p.get('ended_at') else '—'}</td>
          <td><span class="badge {status_cls}">{html.escape(p.get('status',''))}</span></td>
        </tr>"""

    step_rows = ""
    for s in steps:
        icon = "✓" if s["status"] == "success" else "✗"
        cls  = "ok" if s["status"] == "success" else "fail"
        snap = "📸" if True else ""  # placeholder
        step_rows += f"""<tr>
          <td>{html.escape(s.get('step_name',''))}</td>
          <td class="mono">{s.get('started_at','')[:19]}</td>
          <td class="num">{s.get('duration_ms', 0):.1f} ms</td>
          <td><span class="badge {cls}">{icon} {html.escape(s.get('status',''))}</span></td>
          <td class="mono small">{(s.get('output_hash') or '')[:12]}...</td>
        </tr>"""

    return f"""
<div class="section">
  <h2>Pipeline Runs</h2>
  <table>
    <thead><tr><th>Name</th><th>Started</th><th>Ended</th><th>Status</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2 style="margin-top:1.5rem">Steps</h2>
  <table>
    <thead><tr><th>Step</th><th>Started</th><th>Duration</th><th>Status</th><th>Output hash</th></tr></thead>
    <tbody>{step_rows}</tbody>
  </table>
</div>"""


def _section_shift_report(shift_results, sensitive_col) -> str:
    if not shift_results:
        return ""

    rows = ""
    for r in shift_results:
        flag_cls = {"HIGH": "fail", "MEDIUM": "warn", "LOW": "ok"}.get(r["flag"], "ok")
        icon     = "⚠" if r["flag"] == "HIGH" else ("△" if r["flag"] == "MEDIUM" else "")
        rows += f"""<tr>
          <td>{html.escape(r['step_name'])}</td>
          <td>{html.escape(r['column'])}</td>
          <td class="mono">{r['test'].upper()}</td>
          <td class="num">{r['stat']:.4f}</td>
          <td><span class="badge {flag_cls}">{icon} {r['flag']}</span></td>
          <td class="num">{r['rows_removed']:,} ({r['removal_rate']:.1%})</td>
        </tr>"""

    findings = ""
    for r in [x for x in shift_results if x["flag"] == "HIGH"]:
        findings += f"""
        <div class="finding high">
          <strong>[HIGH] {html.escape(r['step_name'])} → {html.escape(r['column'])}</strong>
          <p>{html.escape(r['finding'])}</p>
          {_dist_bars(r.get('before_dist',{}), r.get('after_dist',{}))}
        </div>"""

    return f"""
<div class="section">
  <h2>Distribution Shift Report</h2>
  <table>
    <thead><tr><th>Step</th><th>Column</th><th>Test</th><th>Stat</th><th>Flag</th><th>Rows removed</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {findings}
</div>"""


def _dist_bars(before: Dict, after: Dict) -> str:
    if not before or not after:
        return ""
    all_vals = sorted(set(before) | set(after))
    rows = ""
    for v in all_vals:
        fb = before.get(v, 0.0)
        fa = after.get(v, 0.0)
        delta = fa - fb
        sign_cls = "pos" if delta > 0 else "neg"
        rows += f"""
        <div class="dist-row">
          <span class="dist-label">{html.escape(str(v))}</span>
          <div class="bar-wrap">
            <div class="bar before" style="width:{fb*100:.1f}%"></div>
            <span class="bar-val">{fb:.1%}</span>
          </div>
          <span class="arrow">→</span>
          <div class="bar-wrap">
            <div class="bar after" style="width:{fa*100:.1f}%"></div>
            <span class="bar-val">{fa:.1%}</span>
          </div>
          <span class="delta {sign_cls}">{delta:+.1%}</span>
        </div>"""
    return f'<div class="dist-chart">{rows}</div>'


def _section_demographic_snapshots(snaps, sensitive_col) -> str:
    relevant = [s for s in snaps
                if sensitive_col in s.get("sensitive_stats", {})]
    if not relevant:
        return ""

    cards = ""
    for snap in relevant:
        dist = snap["sensitive_stats"][sensitive_col]
        bars = _dist_single(dist)
        pos_badge = "badge ok" if snap["position"] == "before" else "badge warn"
        cards += f"""
        <div class="snap-card">
          <div class="snap-header">
            <strong>{html.escape(snap.get('step_name',''))}</strong>
            <span class="{pos_badge}">{snap.get('position','')}</span>
            <span class="small mono">{snap.get('recorded_at','')[:19]}</span>
            <span class="small">{snap.get('row_count',0):,} rows</span>
          </div>
          {bars}
        </div>"""

    return f"""
<div class="section">
  <h2>Demographic Snapshots — '{html.escape(sensitive_col)}'</h2>
  <div class="snap-grid">{cards}</div>
</div>"""


def _dist_single(dist: Dict) -> str:
    rows = ""
    for v, frac in sorted(dist.items(), key=lambda x: -x[1]):
        if v == "__null__":
            continue
        rows += f"""
        <div class="dist-row-single">
          <span class="dist-label">{html.escape(str(v))}</span>
          <div class="bar-wrap-single">
            <div class="bar-single" style="width:{frac*100:.1f}%"></div>
          </div>
          <span class="bar-val">{frac:.1%}</span>
        </div>"""
    return f'<div class="dist-single">{rows}</div>'


def _section_attribution(attr: Dict) -> str:
    if not attr or not attr.get("attributed_step"):
        return ""
    conf = attr.get("confidence", 0)
    conf_cls = "ok" if conf >= 0.8 else "warn"
    return f"""
<div class="section">
  <h2>Causal Attribution</h2>
  <div class="attr-box">
    <div class="attr-row">
      <span class="attr-label">Attributed step</span>
      <span class="attr-value mono">{html.escape(attr.get('attributed_step',''))}</span>
    </div>
    <div class="attr-row">
      <span class="attr-label">Sensitive column</span>
      <span class="attr-value">{html.escape(attr.get('column',''))}</span>
    </div>
    <div class="attr-row">
      <span class="attr-label">Shift stat</span>
      <span class="attr-value">{attr.get('stat', 0):.4f} ({(attr.get('test') or '').upper()})
        <span class="badge {'fail' if attr.get('flag')=='HIGH' else 'warn'}">{attr.get('flag','')}</span>
      </span>
    </div>
    <div class="attr-row">
      <span class="attr-label">Confidence</span>
      <span class="attr-value"><span class="badge {conf_cls}">{conf:.0%}</span></span>
    </div>
    <div class="attr-row">
      <span class="attr-label">Rows removed</span>
      <span class="attr-value">{attr.get('rows_removed',0):,}
        ({attr.get('removal_rate',0):.1%} of dataset)</span>
    </div>
    <div class="attr-evidence">
      <strong>Evidence:</strong> {html.escape(attr.get('evidence',''))}
    </div>
    <div class="attr-evidence">
      <strong>Recommendation:</strong>
      <pre class="rec">{html.escape(attr.get('recommendation',''))}</pre>
    </div>
  </div>
</div>"""


def _section_counterfactual(cf: Dict) -> str:
    if not cf:
        return ""
    verdict      = cf.get("verdict", "UNKNOWN")
    verdict_cls  = {"STRONG": "ok", "MODERATE": "warn",
                    "WEAK": "warn", "INCONCLUSIVE": "fail"}.get(verdict, "")
    bias_section = ""
    if cf.get("bias_metric_before") is not None:
        bias_section = f"""
      <div class="cf-metrics">
        <div class="cf-metric">
          <span>Bias before fix</span>
          <strong>{cf['bias_metric_before']:.4f}</strong>
        </div>
        <div class="cf-metric">
          <span>Bias after fix</span>
          <strong>{cf['bias_metric_after']:.4f}</strong>
        </div>
        <div class="cf-metric">
          <span>Reduction</span>
          <strong class="{'ok-text' if cf.get('bias_reduction',0)>0 else 'fail-text'}">{cf.get('bias_reduction_pct',0):+.1f}%</strong>
        </div>
      </div>"""

    dist_section = _dist_bars(cf.get("dist_before_fix",{}), cf.get("dist_after_fix",{}))

    return f"""
<div class="section">
  <h2>Counterfactual Comparison</h2>
  <div class="cf-box">
    <div class="cf-header">
      <span>Replaced step: <strong>{html.escape(cf.get('replace_step',''))}</strong></span>
      <span class="badge {verdict_cls}">{verdict}</span>
    </div>
    <div class="cf-row-counts">
      <span>Biased pipeline: <strong>{cf.get('biased_rows_out',0):,} rows</strong></span>
      <span>Fixed pipeline: <strong>{cf.get('fixed_rows_out',0):,} rows</strong></span>
      <span>Recovered: <strong class="ok-text">{cf.get('rows_recovered',0):+,}</strong></span>
    </div>
    {bias_section}
    <h4>'{html.escape(cf.get('sensitive_col',''))}' distribution</h4>
    <div class="dist-legend"><span class="leg before">■ Before fix</span>
      <span class="leg after">■ After fix</span></div>
    {dist_section}
    <p class="verdict-detail">{html.escape(cf.get('verdict_detail',''))}</p>
  </div>
</div>"""


def _section_metrics(metrics: List[Dict]) -> str:
    if not metrics:
        return ""
    rows = ""
    for m in metrics:
        rows += f"""<tr>
          <td>{html.escape(m.get('metric_name',''))}</td>
          <td class="num">{m.get('metric_value',0):.6f}</td>
          <td>{html.escape(m.get('metric_source','') or '—')}</td>
          <td>{html.escape(m.get('step_name','') or '—')}</td>
          <td class="mono small">{m.get('measured_at','')[:19]}</td>
        </tr>"""
    return f"""
<div class="section">
  <h2>Logged Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th><th>Source</th><th>Step</th><th>Measured</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _section_footer() -> str:
    return """
<div class="footer">
  Generated by <strong>DataLineageML</strong> — Causal data provenance for AI safety.<br>
  <a href="https://github.com/adejumobioluwafemi/data-lineage-ml">github.com/adejumobioluwafemi/data-lineage-ml</a>
</div>"""


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       font-size: 14px; color: #1a1a1a; background: #f5f5f5; line-height: 1.5; }
.header { background: #1a1a2e; color: white; padding: 2rem 2.5rem 1.5rem;
          border-bottom: 3px solid #e94560; }
.logo { font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
        color: #e94560; margin-bottom: .5rem; }
h1 { font-size: 1.6rem; font-weight: 600; margin-bottom: .3rem; }
.sub { color: #aaa; margin-bottom: .2rem; }
.meta { color: #666; font-size: 12px; }
.section { background: white; margin: 1rem 2rem; border-radius: 6px;
           padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
h2 { font-size: 1rem; font-weight: 600; color: #333;
     border-bottom: 1px solid #eee; padding-bottom: .5rem; margin-bottom: 1rem; }
h4 { font-size: .875rem; color: #555; margin: .75rem 0 .4rem; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; font-weight: 600; color: #555; font-size: 11px;
     text-transform: uppercase; letter-spacing: .5px;
     border-bottom: 2px solid #eee; padding: .4rem .5rem; }
td { padding: .4rem .5rem; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 12px;
         font-size: 11px; font-weight: 600; }
.badge.ok   { background: #d1fae5; color: #065f46; }
.badge.warn { background: #fef3c7; color: #92400e; }
.badge.fail { background: #fee2e2; color: #991b1b; }
.mono  { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; }
.small { font-size: 11px; color: #888; }
.num   { text-align: right; font-variant-numeric: tabular-nums; }
pre.rec { background: #f8f8f8; padding: .75rem; border-radius: 4px;
          font-size: 12px; white-space: pre-wrap; margin-top: .3rem; }
.finding.high { background: #fff5f5; border-left: 3px solid #ef4444;
                padding: .75rem 1rem; margin-top: 1rem; border-radius: 0 4px 4px 0; }
.dist-chart { margin-top: .75rem; }
.dist-row { display: flex; align-items: center; gap: .5rem; margin: .3rem 0; }
.dist-label { width: 80px; font-size: 12px; color: #555; flex-shrink: 0; }
.bar-wrap { display: flex; align-items: center; gap: .3rem; width: 180px; }
.bar { height: 14px; border-radius: 2px; min-width: 2px; }
.bar.before { background: #93c5fd; }
.bar.after  { background: #6ee7b7; }
.bar-val { font-size: 11px; color: #666; width: 38px; }
.arrow { color: #aaa; }
.delta { font-size: 11px; font-weight: 600; width: 48px; }
.delta.pos { color: #059669; }
.delta.neg { color: #dc2626; }
.snap-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
             gap: 1rem; margin-top: .75rem; }
.snap-card { background: #fafafa; border: 1px solid #e5e7eb;
             border-radius: 4px; padding: .75rem; }
.snap-header { display: flex; flex-wrap: wrap; gap: .4rem; align-items: center;
               margin-bottom: .5rem; font-size: 12px; }
.dist-single { }
.dist-row-single { display: flex; align-items: center; gap: .4rem; margin: .2rem 0; }
.bar-wrap-single { width: 100px; background: #f0f0f0; border-radius: 2px; height: 12px; }
.bar-single { background: #818cf8; height: 12px; border-radius: 2px; min-width: 2px; }
.attr-box { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 4px;
            padding: 1rem; }
.attr-row { display: flex; gap: 1rem; padding: .3rem 0; border-bottom: 1px solid #f0f0f0; }
.attr-row:last-of-type { border: none; }
.attr-label { width: 140px; color: #666; font-size: 12px; flex-shrink: 0; }
.attr-value { font-size: 13px; }
.attr-evidence { margin-top: .75rem; font-size: 12px; color: #444; }
.cf-box { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 4px;
          padding: 1rem; }
.cf-header { display: flex; justify-content: space-between; align-items: center;
             margin-bottom: .75rem; font-size: 13px; }
.cf-row-counts { display: flex; gap: 1.5rem; font-size: 12px; color: #555;
                 margin-bottom: .75rem; }
.cf-metrics { display: flex; gap: 1.5rem; margin-bottom: .75rem; }
.cf-metric { text-align: center; background: white; padding: .5rem .75rem;
             border-radius: 4px; border: 1px solid #e5e7eb; font-size: 12px; }
.cf-metric strong { display: block; font-size: 1.1rem; margin-top: .1rem; }
.verdict-detail { font-size: 12px; color: #555; margin-top: .75rem; font-style: italic; }
.ok-text   { color: #059669; }
.fail-text { color: #dc2626; }
.dist-legend { display: flex; gap: 1rem; font-size: 11px; color: #666; margin-bottom: .3rem; }
.leg.before::first-letter { color: #93c5fd; }
.leg.after::first-letter  { color: #6ee7b7; }
.footer { text-align: center; padding: 1.5rem; color: #888;
          font-size: 12px; margin-top: 1rem; }
.footer a { color: #6366f1; text-decoration: none; }
"""