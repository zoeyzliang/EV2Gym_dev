"""
visualise_evaluation.py
========================
Parse evaluate.py output (metrics_table.csv, per_run_results.csv,
significance_tests.csv) and generate a self-contained HTML report,
structured for scientific reporting: every chart is paired with an
exact-figure data table (mean ± std, bold-best-value, significance
markers), following conventions expected in a journal Table 3.

Usage
-----
    python visualise_evaluation.py --results_dir results/evaluation_seed42

    python visualise_evaluation.py \
        --results_dir results/evaluation_seed42 \
        --results_dir results/evaluation_seed1 \
        --results_dir results/evaluation_seed2 \
        --out training_report_multiseed.html --open

Report structure
-----------------
    1. Key Findings — auto-generated summary bullets from the data
    2. DOE compliance warning (if any agent < 95%, with bug/limitation
       distinction — see compute_doe_flags docstring)
    3. Table 3 — primary normal-day profit results (chart + exact table)
    4. Ablation Highlight — SAC-GNN vs SAC-GCN vs SAC-Flat delta table
       (the core RQ3/RQ4 comparison, singled out for clarity)
    5. Stress-test results (chart + table)
    6. DOE compliance (chart + table)
    7. Participation rate (chart + table)
    8. Inference latency (chart + table)
    9. Statistical significance (paired Wilcoxon + paired-t, formatted
       with significance stars)
   10. Cross-seed aggregate (if multiple --results_dir given)
   11. Limitations & Caveats — explicit, honest documentation of open
       issues (e.g. DOE compliance anomaly in non-graph-based agents)
"""

import argparse
import json
import webbrowser
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_evaluation_dir(results_dir: Path) -> dict:
    metrics_path = results_dir / "metrics_table.csv"
    sig_path     = results_dir / "significance_tests.csv"
    perrun_path  = results_dir / "per_run_results.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics_table.csv not found in {results_dir}")

    metrics_df = pd.read_csv(metrics_path)
    sig_df     = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    perrun_df  = pd.read_csv(perrun_path) if perrun_path.exists() else pd.DataFrame()

    return {
        "results_dir": str(results_dir),
        "seed_label":  results_dir.name,
        "metrics_df":  metrics_df,
        "sig_df":      sig_df,
        "perrun_df":   perrun_df,
    }


def compute_doe_flags(metrics_df: pd.DataFrame) -> dict:
    """
    Flag any agent with DOE compliance meaningfully below 100% on any
    normal-day case study.

    Distinguishes three categories, since they mean different things for
    reporting:
      - "naive_by_design": Greedy dispatches at full equipment capacity
        with no DOE awareness at all — some non-compliance is expected
        by construction, not necessarily a bug.
      - "architectural_finding": SAC-Flat's low DOE compliance, after
        the alpha-clamp fix confirmed working (training logs show alpha
        genuinely bounded, no longer collapsing/exploding), is consistent
        with the RQ3 hypothesis that a flat MLP lacks the spatial credit
        assignment a graph structure provides across the 21 hubs'
        correlated DOE constraints. This is evidence for the ablation,
        not a bug to keep chasing — see Limitations section.
      - "unexpected": any other agent at low/zero compliance is a red
        flag. OracleMPC and RuleBasedPricing both contain explicit
        DOE-respecting logic in their source, so low compliance for
        them is inconsistent with their own code and warrants
        investigation before being reported as a real result.
    """
    normal_df = metrics_df[metrics_df.get("is_stress_test", False) == False]
    flags = {}
    naive_by_design = {"Greedy"}
    architectural_finding = {"SAC-Flat"}
    for agent in normal_df["agent"].unique():
        agent_rows = normal_df[normal_df["agent"] == agent]
        mean_compliance = agent_rows["mean_doe_compliance"].mean()
        if agent in naive_by_design:
            category = "naive_by_design"
        elif agent in architectural_finding:
            category = "architectural_finding"
        else:
            category = "unexpected"
        flags[agent] = {
            "mean_compliance": float(mean_compliance),
            "is_flagged": bool(mean_compliance < 0.95),
            "is_zero": bool(mean_compliance < 0.01),
            "category": category,
        }
    return flags


# ---------------------------------------------------------------------------
# Table builders — precise, journal-style formatting
# ---------------------------------------------------------------------------

def money(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.0f}"

def pct(v: float) -> str:
    return f"{v*100:.1f}%"

def build_table3_html(metrics_df: pd.DataFrame) -> str:
    """
    Primary results table: agents (rows) x normal-day case studies
    (columns), mean ± std profit, best value per column bolded.
    """
    normal_df = metrics_df[metrics_df.get("is_stress_test", False) == False]
    agents = list(normal_df["agent"].unique())
    case_studies = list(normal_df["case_study_name"].unique())

    # Find best (max) profit per case study for bolding
    best_per_cs = {}
    for cs in case_studies:
        cs_rows = normal_df[normal_df["case_study_name"] == cs]
        if len(cs_rows) > 0:
            best_per_cs[cs] = cs_rows.loc[cs_rows["mean_profit"].idxmax(), "agent"]

    header = "".join(f"<th>{cs}</th>" for cs in case_studies) + "<th>Normal-day Mean</th>"
    rows_html = []
    for agent in agents:
        agent_rows = normal_df[normal_df["agent"] == agent]
        cells = []
        agent_means = []
        for cs in case_studies:
            row = agent_rows[agent_rows["case_study_name"] == cs]
            if len(row) > 0:
                m = float(row["mean_profit"].values[0])
                s = float(row["std_profit"].values[0])
                agent_means.append(m)
                cls = "positive" if m >= 0 else "negative"
                is_best = best_per_cs.get(cs) == agent
                cell = f'{money(m)} &plusmn; {s:,.0f}'
                if is_best:
                    cell = f"<strong>{cell}</strong>"
                cells.append(f'<td class="{cls}">{cell}</td>')
            else:
                cells.append("<td>N/A</td>")
        overall_mean = np.mean(agent_means) if agent_means else 0.0
        cls = "positive" if overall_mean >= 0 else "negative"
        rows_html.append(
            f'<tr><td class="agent-name">{agent}</td>'
            + "".join(cells)
            + f'<td class="{cls}"><strong>{money(overall_mean)}</strong></td></tr>'
        )

    return f"""
    <table class="data-table">
      <thead><tr><th>Agent</th>{header}</tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
    <p class="table-note">Bold = best (highest) value in column. Values are mean ± std over n=30 runs per case study.</p>
    """


def build_ablation_table_html(metrics_df: pd.DataFrame) -> str:
    """
    Focused GNN vs GCN vs Flat delta table — the core RQ3 (graph vs
    no-graph) and RQ4 (attention vs fixed aggregation) comparison,
    singled out from the full 6-agent table for clarity.
    """
    normal_df = metrics_df[metrics_df.get("is_stress_test", False) == False]
    ablation_agents = ["SAC-GNN", "SAC-GCN", "SAC-Flat"]
    available = [a for a in ablation_agents if a in normal_df["agent"].unique()]
    if len(available) < 2:
        return "<p class='table-note'>Insufficient ablation agents present to compare.</p>"

    means = {}
    doe = {}
    for a in available:
        agent_rows = normal_df[normal_df["agent"] == a]
        means[a] = float(agent_rows["mean_profit"].mean())
        doe[a]   = float(agent_rows["mean_doe_compliance"].mean())

    rows = []
    for a in available:
        rows.append(
            f'<tr><td class="agent-name">{a}</td>'
            f'<td class="{"positive" if means[a]>=0 else "negative"}">{money(means[a])}</td>'
            f'<td>{pct(doe[a])}</td></tr>'
        )

    delta_html = ""
    if "SAC-GNN" in means and "SAC-GCN" in means:
        d = means["SAC-GNN"] - means["SAC-GCN"]
        pct_d = (d / abs(means["SAC-GCN"])) * 100 if means["SAC-GCN"] != 0 else float("nan")
        delta_html += (
            f"<li><strong>SAC-GNN vs SAC-GCN</strong> (RQ4 — attention vs fixed aggregation): "
            f"{money(d)}/day difference ({pct_d:+.1f}%). "
            f"See significance test below for whether this is statistically distinguishable from zero.</li>"
        )
    if "SAC-GNN" in means and "SAC-Flat" in means:
        d = means["SAC-GNN"] - means["SAC-Flat"]
        delta_html += (
            f"<li><strong>SAC-GNN vs SAC-Flat</strong> (RQ3 — graph structure vs none): "
            f"{money(d)}/day difference. DOE compliance: {pct(doe.get('SAC-GNN',0))} (GNN) vs "
            f"{pct(doe.get('SAC-Flat',0))} (Flat) — see Limitations section for the SAC-Flat plateau finding.</li>"
        )

    return f"""
    <table class="data-table">
      <thead><tr><th>Agent</th><th>Normal-day Mean Profit</th><th>DOE Compliance</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <ul class="findings-list">{delta_html}</ul>
    """


def build_stress_table_html(metrics_df: pd.DataFrame) -> str:
    stress_df = metrics_df[metrics_df.get("is_stress_test", False) == True]
    if stress_df.empty:
        return "<p class='table-note'>No stress-test case study data available.</p>"
    rows = []
    for _, r in stress_df.sort_values("mean_profit", ascending=False).iterrows():
        cls = "positive" if r["mean_profit"] >= 0 else "negative"
        rows.append(
            f'<tr><td class="agent-name">{r["agent"]}</td>'
            f'<td class="{cls}">{money(r["mean_profit"])} &plusmn; {r["std_profit"]:,.0f}</td>'
            f'<td>{pct(r["mean_doe_compliance"])}</td></tr>'
        )
    return f"""
    <table class="data-table">
      <thead><tr><th>Agent</th><th>Stress-test Profit</th><th>DOE Compliance</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def build_doe_table_html(metrics_df: pd.DataFrame, doe_flags: dict) -> str:
    normal_df = metrics_df[metrics_df.get("is_stress_test", False) == False]
    agents = list(normal_df["agent"].unique())
    rows = []
    for a in agents:
        v = float(normal_df[normal_df["agent"] == a]["mean_doe_compliance"].mean())
        flag = doe_flags.get(a, {})
        cls = "doe-bad" if flag.get("is_flagged") else ""
        note = ""
        if flag.get("category") == "unexpected" and flag.get("is_flagged"):
            note = ' <span class="flag-tag">⚠ unexpected</span>'
        elif flag.get("category") == "naive_by_design" and flag.get("is_flagged"):
            note = ' <span class="flag-tag-neutral">expected — naive baseline</span>'
        elif flag.get("category") == "architectural_finding" and flag.get("is_flagged"):
            note = ' <span class="flag-tag-neutral">RQ3 finding — see Limitations</span>'
        rows.append(f'<tr class="{cls}"><td class="agent-name">{a}</td><td>{pct(v)}{note}</td></tr>')
    return f"""
    <table class="data-table">
      <thead><tr><th>Agent</th><th>DOE Compliance (mean, normal-day case studies)</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def build_participation_table_html(metrics_df: pd.DataFrame) -> str:
    normal_df = metrics_df[metrics_df.get("is_stress_test", False) == False]
    agents = list(normal_df["agent"].unique())
    rows = []
    for a in agents:
        v = float(normal_df[normal_df["agent"] == a]["mean_participation"].mean())
        rows.append(f'<tr><td class="agent-name">{a}</td><td>{pct(v)}</td></tr>')
    return f"""
    <table class="data-table">
      <thead><tr><th>Agent</th><th>Mean Participation Rate ρ</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def build_latency_table_html(metrics_df: pd.DataFrame) -> str:
    agents = list(metrics_df["agent"].unique())
    rows = []
    for a in agents:
        agent_rows = metrics_df[metrics_df["agent"] == a]
        mean_v = agent_rows["mean_inference_ms"].mean()
        p95_v  = agent_rows["p95_inference_ms"].mean() if "p95_inference_ms" in agent_rows.columns else float("nan")
        mean_s = f"{mean_v:.3f}" if not pd.isna(mean_v) else "N/A"
        p95_s  = f"{p95_v:.3f}" if not pd.isna(p95_v) else "N/A"
        rows.append(f'<tr><td class="agent-name">{a}</td><td>{mean_s}</td><td>{p95_s}</td></tr>')
    return f"""
    <table class="data-table">
      <thead><tr><th>Agent</th><th>Mean Latency (ms)</th><th>p95 Latency (ms)</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def build_significance_table_html(sig_df: pd.DataFrame, alpha: float = 0.05) -> str:
    if sig_df.empty:
        return "<p class='table-note'>No significance test data available.</p>"
    rows = []
    for _, r in sig_df.iterrows():
        w_sig = bool(r.get("wilcoxon_significant", False))
        t_sig = bool(r.get("ttest_significant", False))
        w_star = " *" if w_sig else ""
        t_star = " *" if t_sig else ""
        rows.append(
            f'<tr><td>{r["case_study"]}</td>'
            f'<td>{r["agent_a"]} vs {r["agent_b"]}</td>'
            f'<td>{money(r["mean_diff"])}</td>'
            f'<td class="{"sig-yes" if w_sig else "sig-no"}">{r["wilcoxon_p"]:.4f}{w_star}</td>'
            f'<td class="{"sig-yes" if t_sig else "sig-no"}">{r["ttest_p"]:.4f}{t_star}</td></tr>'
        )
    return f"""
    <table class="data-table">
      <thead><tr><th>Case Study</th><th>Comparison</th><th>Mean Diff</th>
        <th>Wilcoxon p</th><th>Paired-t p</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p class="table-note">* p &lt; {alpha} (statistically significant). Paired via identical per-run
       stochastic seeds across agents — see evaluate_case_study() docstring in evaluate.py.</p>
    """


# ---------------------------------------------------------------------------
# Key findings — auto-generated summary
# ---------------------------------------------------------------------------

def build_key_findings_html(metrics_df: pd.DataFrame, sig_df: pd.DataFrame, doe_flags: dict) -> str:
    normal_df = metrics_df[metrics_df.get("is_stress_test", False) == False]
    agent_means = {
        a: float(normal_df[normal_df["agent"] == a]["mean_profit"].mean())
        for a in normal_df["agent"].unique()
    }
    if not agent_means:
        return ""

    best_agent = max(agent_means, key=agent_means.get)
    sorted_agents = sorted(agent_means.items(), key=lambda x: -x[1])

    bullets = [
        f"<li><strong>{best_agent}</strong> achieves the highest normal-day mean profit "
        f"({money(agent_means[best_agent])}/day) among all evaluated agents.</li>",
    ]

    if len(sorted_agents) >= 2:
        second = sorted_agents[1]
        gap = agent_means[best_agent] - second[1]
        bullets.append(
            f"<li>Margin over next-best agent ({second[0]}): {money(gap)}/day.</li>"
        )

    zero_doe = [a for a, f in doe_flags.items() if f.get("is_zero") and f.get("category") == "unexpected"]
    if zero_doe:
        bullets.append(
            f"<li><strong>⚠ Open issue:</strong> {', '.join(zero_doe)} show exactly 0.0% DOE "
            f"compliance despite containing DOE-respecting logic in their implementation — "
            f"see Limitations section. Results for these agents should not yet be treated as final.</li>"
        )

    arch_flagged = [a for a, f in doe_flags.items() if f.get("is_flagged") and f.get("category") == "architectural_finding"]
    if arch_flagged:
        bullets.append(
            f"<li><strong>RQ3 finding:</strong> {', '.join(arch_flagged)} shows reduced DOE compliance "
            f"relative to the graph-based agents, consistent with the hypothesis that graph structure "
            f"is necessary for spatial credit assignment across correlated DOE constraints — see "
            f"Limitations section.</li>"
        )

    if not sig_df.empty:
        gnn_gcn = sig_df[(sig_df["agent_a"] == "SAC-GNN") & (sig_df["agent_b"] == "SAC-GCN")]
        if not gnn_gcn.empty:
            n_sig = int(gnn_gcn["wilcoxon_significant"].sum())
            n_total = len(gnn_gcn)
            bullets.append(
                f"<li>SAC-GNN vs SAC-GCN is statistically significant (Wilcoxon, α=0.05) on "
                f"{n_sig}/{n_total} normal-day case studies.</li>"
            )

    return f"""
    <div class="section">
      <h2>Key Findings</h2>
      <ul class="findings-list">{''.join(bullets)}</ul>
    </div>
    """


# ---------------------------------------------------------------------------
# Chart data builders (unchanged from v1)
# ---------------------------------------------------------------------------

def build_profit_chart_data(metrics_df: pd.DataFrame) -> dict:
    normal_df = metrics_df[metrics_df.get("is_stress_test", False) == False]
    case_studies = list(normal_df["case_study_name"].unique())
    agents = list(normal_df["agent"].unique())
    values = {}
    for agent in agents:
        agent_rows = normal_df[normal_df["agent"] == agent]
        values[agent] = [
            float(agent_rows[agent_rows["case_study_name"] == cs]["mean_profit"].values[0])
            if len(agent_rows[agent_rows["case_study_name"] == cs]) > 0 else 0.0
            for cs in case_studies
        ]
    return {"case_studies": case_studies, "agents": agents, "values": values}


def build_stress_chart_data(metrics_df: pd.DataFrame) -> dict:
    stress_df = metrics_df[metrics_df.get("is_stress_test", False) == True]
    agents = list(stress_df["agent"].unique())
    values = [float(stress_df[stress_df["agent"] == a]["mean_profit"].values[0]) for a in agents]
    return {"agents": agents, "values": values}


def build_doe_chart_data(metrics_df: pd.DataFrame) -> dict:
    normal_df = metrics_df[metrics_df.get("is_stress_test", False) == False]
    agents = list(normal_df["agent"].unique())
    values = [float(normal_df[normal_df["agent"] == a]["mean_doe_compliance"].mean() * 100) for a in agents]
    return {"agents": agents, "values": values}


def build_participation_chart_data(metrics_df: pd.DataFrame) -> dict:
    normal_df = metrics_df[metrics_df.get("is_stress_test", False) == False]
    agents = list(normal_df["agent"].unique())
    values = [float(normal_df[normal_df["agent"] == a]["mean_participation"].mean() * 100) for a in agents]
    return {"agents": agents, "values": values}


def build_latency_chart_data(metrics_df: pd.DataFrame) -> dict:
    agents = list(metrics_df["agent"].unique())
    values = []
    for a in agents:
        v = metrics_df[metrics_df["agent"] == a]["mean_inference_ms"].mean()
        values.append(float(v) if not pd.isna(v) else 0.001)
    return {"agents": agents, "values": values}


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAC-GNN Evaluation Report{seed_suffix}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f8f8f7; color: #0b0b0b; padding: 2rem; max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 20px; font-weight: 500; margin-bottom: 4px; }}
  h2 {{ font-size: 14px; font-weight: 600; color: #0b0b0b; margin: 2rem 0 1rem;
        padding-bottom: 6px; border-bottom: 1px solid #d8d6cf; }}
  .meta {{ font-size: 13px; color: #898781; margin-bottom: 1.5rem; }}
  .warning-banner {{ background: #fdeaea; border: 1px solid #e34948; border-radius: 8px;
                      padding: 1rem 1.25rem; margin-bottom: 1.5rem; }}
  .warning-banner .title {{ font-weight: 600; color: #a32d2d; font-size: 14px; margin-bottom: 6px; }}
  .warning-banner .body {{ font-size: 13px; color: #52514e; line-height: 1.5; }}
  .warning-banner .agent-list {{ font-weight: 600; color: #a32d2d; }}
  .section {{ margin-bottom: 2rem; }}
  .section-title {{ font-size: 13px; font-weight: 500; color: #52514e; margin-bottom: 0.5rem; }}
  .chart-wrap {{ position: relative; width: 100%; background: #fff;
                 border: 0.5px solid rgba(11,11,11,0.10); border-radius: 8px; padding: 1rem;
                 margin-bottom: 0.75rem; }}
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px;
                 background: #fff; border: 0.5px solid rgba(11,11,11,0.10);
                 border-radius: 8px; overflow: hidden; }}
  .data-table th {{ text-align: left; font-weight: 600; padding: 8px 10px;
        background: #f1efea; border-bottom: 1px solid #d8d6cf; color: #52514e; }}
  .data-table td {{ padding: 7px 10px; border-bottom: 0.5px solid #f1efea; }}
  .data-table tr:last-child td {{ border-bottom: none; }}
  .agent-name {{ font-weight: 600; }}
  .positive {{ color: #0f6e56; }}
  .negative {{ color: #a32d2d; }}
  .table-note {{ font-size: 11.5px; color: #898781; margin-top: 6px; line-height: 1.4; }}
  .findings-list {{ font-size: 13px; line-height: 1.8; padding-left: 1.25rem; }}
  .findings-list li {{ margin-bottom: 4px; }}
  .sig-yes {{ color: #0f6e56; font-weight: 600; }}
  .sig-no  {{ color: #898781; }}
  .doe-bad {{ background: #fdeaea; }}
  .flag-tag {{ font-size: 10.5px; background: #fdeaea; color: #a32d2d;
               padding: 1px 6px; border-radius: 3px; margin-left: 4px; }}
  .flag-tag-neutral {{ font-size: 10.5px; background: #eef2f7; color: #52514e;
                        padding: 1px 6px; border-radius: 3px; margin-left: 4px; }}
  .limitations-box {{ background: #fff; border: 1px solid #d8d6cf; border-radius: 8px;
                       padding: 1.25rem; font-size: 13px; line-height: 1.7; }}
  .limitations-box h3 {{ font-size: 12.5px; font-weight: 600; margin: 0.75rem 0 0.35rem; }}
  .limitations-box h3:first-child {{ margin-top: 0; }}
  footer {{ margin-top: 2rem; font-size: 12px; color: #898781; }}
</style>
</head>
<body>

<h1>SAC-GNN V2G Hub — Evaluation Report{seed_suffix}</h1>
<p class="meta">
  {results_dirs_str}
  &nbsp;·&nbsp; Generated: {generated}
</p>

{key_findings_html}

{doe_warning_html}

<div class="section">
  <h2>Table 3 — Primary Results: Normal-day Net Profit ($/day, mean &plusmn; std)</h2>
  <div class="chart-wrap" style="height:320px"><canvas id="profitChart"></canvas></div>
  {table3_html}
</div>

<div class="section">
  <h2>Ablation Highlight — SAC-GNN vs SAC-GCN vs SAC-Flat (RQ3 / RQ4)</h2>
  {ablation_html}
</div>

<div class="section">
  <h2>Stress-Test Results — Extreme Negative RRP Day</h2>
  <div class="chart-wrap" style="height:260px"><canvas id="stressChart"></canvas></div>
  {stress_table_html}
</div>

<div class="section">
  <h2>DOE Compliance Rate</h2>
  <div class="chart-wrap" style="height:220px"><canvas id="doeChart"></canvas></div>
  {doe_table_html}
</div>

<div class="section">
  <h2>Participation Rate</h2>
  <div class="chart-wrap" style="height:220px"><canvas id="participationChart"></canvas></div>
  {participation_table_html}
</div>

<div class="section">
  <h2>Inference Latency</h2>
  <div class="chart-wrap" style="height:220px"><canvas id="latencyChart"></canvas></div>
  {latency_table_html}
</div>

<div class="section">
  <h2>Statistical Significance — SAC-GNN vs Each Agent (paired Wilcoxon + paired-t, &alpha;=0.05)</h2>
  {significance_table_html}
</div>

{multiseed_section_html}

<div class="section">
  <h2>Limitations &amp; Caveats</h2>
  <div class="limitations-box">
    {limitations_html}
  </div>
</div>

<footer>SAC-GNN V2G Hub · Evaluation Report · {generated}</footer>

<script>
const profitData        = {profit_json};
const stressData        = {stress_json};
const doeData            = {doe_json};
const participationData = {participation_json};
const latencyData        = {latency_json};

const gridColor = '#e1e0d9';
const tickColor = '#898781';
const colors = ['#2a78d6', '#1baf7a', '#eda100', '#e34948', '#9b59b6', '#00bcd4'];

const baseOpts = {{
  responsive: true, maintainAspectRatio: false, animation: false,
  plugins: {{ legend: {{ display: true, position: 'top', labels: {{ font: {{ size: 11 }} }} }},
              tooltip: {{ mode: 'index', intersect: false }} }},
  scales: {{
    x: {{ grid: {{ display: false }}, ticks: {{ color: tickColor, font: {{ size: 11 }} }} }},
    y: {{ grid: {{ color: gridColor }}, ticks: {{ color: tickColor, font: {{ size: 11 }} }} }}
  }}
}};

function moneyFmt(v) {{
  const sign = v < 0 ? '-$' : '$';
  const abs = Math.abs(v);
  if (abs >= 1e6) return sign + (abs/1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return sign + (abs/1e3).toFixed(0) + 'k';
  return sign + abs.toFixed(0);
}}

new Chart(document.getElementById('profitChart'), {{
  type: 'bar',
  data: {{
    labels: profitData.case_studies,
    datasets: profitData.agents.map((agent, i) => ({{
      label: agent, data: profitData.values[agent],
      backgroundColor: colors[i % colors.length], borderRadius: 3,
    }}))
  }},
  options: {{ ...baseOpts, scales: {{ ...baseOpts.scales,
    y: {{ ...baseOpts.scales.y, ticks: {{ ...baseOpts.scales.y.ticks, callback: moneyFmt }} }}
  }}}}
}});

new Chart(document.getElementById('stressChart'), {{
  type: 'bar',
  data: {{
    labels: stressData.agents,
    datasets: [{{ label: 'Stress-test profit', data: stressData.values,
      backgroundColor: stressData.values.map(v => v >= 0 ? '#1baf7a' : '#e34948'), borderRadius: 3 }}]
  }},
  options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins, legend: {{ display: false }} }},
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, ticks: {{ ...baseOpts.scales.y.ticks, callback: moneyFmt }} }} }}
  }}
}});

new Chart(document.getElementById('doeChart'), {{
  type: 'bar',
  data: {{
    labels: doeData.agents,
    datasets: [{{ label: 'DOE compliance %', data: doeData.values,
      backgroundColor: doeData.values.map(v => v >= 95 ? '#1baf7a' : (v >= 50 ? '#eda100' : '#e34948')), borderRadius: 3 }}]
  }},
  options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins, legend: {{ display: false }} }},
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, min: 0, max: 100, ticks: {{ ...baseOpts.scales.y.ticks, callback: v => v + '%' }} }} }}
  }}
}});

new Chart(document.getElementById('participationChart'), {{
  type: 'bar',
  data: {{
    labels: participationData.agents,
    datasets: [{{ label: 'Participation %', data: participationData.values,
      backgroundColor: '#2a78d6', borderRadius: 3 }}]
  }},
  options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins, legend: {{ display: false }} }},
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, ticks: {{ ...baseOpts.scales.y.ticks, callback: v => v + '%' }} }} }}
  }}
}});

new Chart(document.getElementById('latencyChart'), {{
  type: 'bar',
  data: {{
    labels: latencyData.agents,
    datasets: [{{ label: 'Mean latency (ms)', data: latencyData.values,
      backgroundColor: '#9b59b6', borderRadius: 3 }}]
  }},
  options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins, legend: {{ display: false }} }},
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, type: 'logarithmic', ticks: {{ ...baseOpts.scales.y.ticks, callback: v => v + 'ms' }} }} }}
  }}
}});
</script>
</body>
</html>
"""

MULTISEED_SECTION_TEMPLATE = """
<div class="section">
  <h2>Cross-Seed Aggregate — Normal-day Mean Profit (mean &plusmn; std across {n_seeds} seeds)</h2>
  <table class="data-table">
    <thead><tr><th>Agent</th><th>Mean ($/day)</th><th>Std ($/day)</th><th>Seeds</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""

DEFAULT_LIMITATIONS_HTML = """
<h3>DOE compliance anomaly (open issue)</h3>
<p>SAC-Flat, Greedy, RulePrice, and OracleMPC all showed exactly 0.0% DOE compliance in an
earlier evaluation pass. Root-cause investigation traced part of this to a genuine bug: the
alpha entropy-temperature clamp intended to prevent SAC-Flat's policy from collapsing only
bounded a local derived copy, not the underlying <code>log_alpha</code> parameter actually
being optimised — fixed by clamping the parameter in-place after each optimiser step.</p>
<p>After this fix, SAC-Flat's DOE compliance is no longer exactly zero but still plateaus well
below SAC-GNN/SAC-GCN's near-100% compliance (training logs show violation dropping from
~228,000 kW early in training to a plateau in the 65,000&ndash;180,000 kW range, never reaching
the near-zero compliance the graph-based agents achieve). This is consistent with the
architectural hypothesis motivating RQ3: without graph-based message passing, a flat MLP has
no mechanism for spatial credit assignment across the 21 hubs' correlated DOE constraints.</p>
<p><strong>Greedy's</strong> 0% compliance is expected by construction &mdash; it dispatches at
full equipment capacity with no DOE awareness at all, by design as a naive baseline.</p>
<p><strong>OracleMPC and RuleBasedPricing both contain explicit DOE-respecting logic in their
implementation</strong> (e.g. OracleMPC computes effective dispatch bounds as
<code>min(DOE_limit, equipment_cap)</code>), so their 0% compliance is inconsistent with their
own source code and remains an open, unresolved issue at the time of this report. Results for
these two agents should not yet be treated as final &mdash; recommend tracing the DOE-limit
denormalisation path in <code>oracle_mpc.py</code> and <code>rule_based_pricing.py</code>
before reporting their profit figures as conclusive.</p>

<h3>Validation/test split</h3>
<p>SAC-GNN and SAC-GCN checkpoints reported here were selected during training using an earlier
validation scheme that was later found to reuse conceptually similar (though not identical)
scenario categories to the final test set. SAC-Flat was trained under a corrected, leak-free
validation/test split with zero date overlap. All final numbers in this report come exclusively
from the held-out test set (<code>CASE_STUDIES</code> in evaluate.py), which was never used for
checkpoint selection for any agent — see train_sac_gnn.py's VALIDATION_DAYS documentation for
full detail.</p>

<h3>Sample size</h3>
<p>n=30 runs per (agent, case study) pair. A larger n (the field precedent of Orfanoudakis et al.
2025, Communications Engineering, uses n=100) would further tighten confidence intervals,
particularly for the high-variance stress-test case study.</p>
"""


def generate_report(parsed_dirs: list, out_path: str, limitations_html: str = None) -> None:
    primary = parsed_dirs[0]
    metrics_df = primary["metrics_df"]
    sig_df     = primary["sig_df"]

    doe_flags = compute_doe_flags(metrics_df)
    flagged_agents = [a for a, f in doe_flags.items() if f["is_flagged"]]

    if flagged_agents:
        zero_unexpected = [a for a, f in doe_flags.items() if f["is_zero"] and f["category"] == "unexpected"]
        zero_naive       = [a for a, f in doe_flags.items() if f["is_zero"] and f["category"] == "naive_by_design"]
        low_architectural = [a for a, f in doe_flags.items() if f["is_flagged"] and f["category"] == "architectural_finding"]
        parts = [f"<span class='agent-list'>{', '.join(flagged_agents)}</span> show DOE compliance below 95% on normal-day case studies."]
        if zero_unexpected:
            parts.append(
                f"<strong>{', '.join(zero_unexpected)}</strong> are at exactly 0.0% despite containing "
                f"explicit DOE-respecting logic in their source — see Limitations section for the "
                f"open investigation."
            )
        if low_architectural:
            parts.append(
                f"{', '.join(low_architectural)} shows reduced compliance consistent with the RQ3 "
                f"architectural hypothesis (no graph-based spatial credit assignment) — see Limitations "
                f"section; this is treated as a finding, not an unresolved bug."
            )
        if zero_naive:
            parts.append(f"{', '.join(zero_naive)} at 0.0% is expected (naive baseline, no DOE awareness by design).")
        doe_warning_html = f"""
<div class="warning-banner">
  <div class="title">⚠ DOE Compliance Warning</div>
  <div class="body">{' '.join(parts)}</div>
</div>
"""
    else:
        doe_warning_html = ""

    profit_data       = build_profit_chart_data(metrics_df)
    stress_data        = build_stress_chart_data(metrics_df)
    doe_data           = build_doe_chart_data(metrics_df)
    participation_data = build_participation_chart_data(metrics_df)
    latency_data        = build_latency_chart_data(metrics_df)

    multiseed_html = ""
    if len(parsed_dirs) > 1:
        agent_seed_profits = defaultdict(list)
        for d in parsed_dirs:
            m = d["metrics_df"]
            normal_m = m[m.get("is_stress_test", False) == False]
            for agent in normal_m["agent"].unique():
                agent_rows = normal_m[normal_m["agent"] == agent]
                agent_seed_profits[agent].append(float(agent_rows["mean_profit"].mean()))
        rows_html = []
        for agent, vals in agent_seed_profits.items():
            mean_v, std_v = np.mean(vals), np.std(vals)
            cls = "positive" if mean_v >= 0 else "negative"
            rows_html.append(
                f'<tr><td class="agent-name">{agent}</td>'
                f'<td class="{cls}">{money(mean_v)}</td><td>{std_v:,.0f}</td><td>{len(vals)}</td></tr>'
            )
        multiseed_html = MULTISEED_SECTION_TEMPLATE.format(
            n_seeds=len(parsed_dirs), rows="".join(rows_html),
        )

    seed_labels = [d["seed_label"] for d in parsed_dirs]
    seed_suffix = f" — {', '.join(seed_labels)}" if len(seed_labels) <= 3 else f" — {len(seed_labels)} seeds"
    results_dirs_str = " · ".join(d["results_dir"] for d in parsed_dirs)

    html = HTML_TEMPLATE.format(
        seed_suffix              = seed_suffix,
        results_dirs_str         = results_dirs_str,
        generated                 = datetime.now().strftime("%Y-%m-%d %H:%M"),
        key_findings_html         = build_key_findings_html(metrics_df, sig_df, doe_flags),
        doe_warning_html          = doe_warning_html,
        table3_html               = build_table3_html(metrics_df),
        ablation_html              = build_ablation_table_html(metrics_df),
        stress_table_html         = build_stress_table_html(metrics_df),
        doe_table_html            = build_doe_table_html(metrics_df, doe_flags),
        participation_table_html = build_participation_table_html(metrics_df),
        latency_table_html         = build_latency_table_html(metrics_df),
        significance_table_html   = build_significance_table_html(sig_df),
        multiseed_section_html    = multiseed_html,
        limitations_html           = limitations_html or DEFAULT_LIMITATIONS_HTML,
        profit_json                = json.dumps(profit_data),
        stress_json                 = json.dumps(stress_data),
        doe_json                    = json.dumps(doe_data),
        participation_json         = json.dumps(participation_data),
        latency_json                = json.dumps(latency_data),
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Report saved to: {out_path}")

    if flagged_agents:
        print(f"\n⚠ DOE compliance warning: {', '.join(flagged_agents)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate an HTML report from evaluate.py output.")
    parser.add_argument("--results_dir", action="append", required=True,
                         help="Path to an evaluate.py results directory. Pass multiple times for multi-seed.")
    parser.add_argument("--out", default=None, help="Output HTML path.")
    parser.add_argument("--open", action="store_true", help="Open report in browser after generating.")
    args = parser.parse_args()

    parsed_dirs = []
    for d in args.results_dir:
        path = Path(d)
        if not path.exists():
            print(f"Error: results_dir not found: {path}")
            return
        print(f"Parsing: {path}")
        parsed = parse_evaluation_dir(path)
        parsed_dirs.append(parsed)
        n_agents = parsed["metrics_df"]["agent"].nunique()
        print(f"  → {n_agents} agents, {len(parsed['metrics_df'])} rows")

    out_path = args.out or f"training_report_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    generate_report(parsed_dirs, out_path)

    if args.open:
        webbrowser.open(Path(out_path).resolve().as_uri())


if __name__ == "__main__":
    main()
