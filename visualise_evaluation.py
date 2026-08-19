"""
visualise_evaluation.py
========================
Parse evaluate.py output (metrics_table.csv, per_run_results.csv,
significance_tests.csv) and generate a self-contained HTML report.

Companion to visualise_training.py — that script visualises training
curves from SLURM logs, this one visualises the FINAL evaluation
results (Table 3) from evaluate.py's output directory.

Usage
-----
    # Single seed
    python visualise_evaluation.py --results_dir results/evaluation_seed42

    # Multiple seeds — aggregates mean±std across seeds per agent
    python visualise_evaluation.py \
        --results_dir results/evaluation_seed42 \
        --results_dir results/evaluation_seed1 \
        --results_dir results/evaluation_seed2 \
        --out training_report_multiseed.html --open

Output
------
A single self-contained HTML file with:
    - Summary cards (best agent, DOE compliance red flags)
    - Normal-day profit bar chart (grouped by case study)
    - Stress-test profit bar chart (separate scale)
    - DOE compliance rate bar chart (flags any agent below 100%)
    - Participation rate bar chart
    - Inference latency bar chart
    - Significance test table (Wilcoxon + paired-t, highlighted)
    - If multiple --results_dir given: cross-seed aggregate section
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
    """
    Load metrics_table.csv, significance_tests.csv, per_run_results.csv
    from a single evaluate.py output directory.
    """
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
    normal-day case study — used to render a prominent warning banner,
    since 0% compliance is a strong signal of a bug (see conversation
    context: SAC-Flat/Greedy/RulePrice/OracleMPC observed at exactly
    0.0% compliance, consistent with hitting the per-step DOE penalty
    cap on every single step of every episode).
    """
    normal_df = metrics_df[metrics_df.get("is_stress_test", False) == False]
    flags = {}
    for agent in normal_df["agent"].unique():
        agent_rows = normal_df[normal_df["agent"] == agent]
        mean_compliance = agent_rows["mean_doe_compliance"].mean()
        flags[agent] = {
            "mean_compliance": float(mean_compliance),
            "is_flagged": bool(mean_compliance < 0.95),
            "is_zero": bool(mean_compliance < 0.01),
        }
    return flags


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
  h2 {{ font-size: 14px; font-weight: 500; color: #52514e; margin: 2rem 0 1rem;
        padding-bottom: 6px; border-bottom: 0.5px solid #e1e0d9; }}
  .meta {{ font-size: 13px; color: #898781; margin-bottom: 1.5rem; }}
  .warning-banner {{ background: #fdeaea; border: 1px solid #e34948; border-radius: 8px;
                      padding: 1rem 1.25rem; margin-bottom: 1.5rem; }}
  .warning-banner .title {{ font-weight: 600; color: #a32d2d; font-size: 14px; margin-bottom: 6px; }}
  .warning-banner .body {{ font-size: 13px; color: #52514e; line-height: 1.5; }}
  .warning-banner .agent-list {{ font-weight: 600; color: #a32d2d; }}
  .metric-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                 gap: 12px; margin-bottom: 1.5rem; }}
  .card {{ background: #fff; border: 0.5px solid rgba(11,11,11,0.10);
           border-radius: 8px; padding: 1rem; }}
  .card-label {{ font-size: 12px; color: #898781; margin-bottom: 4px; }}
  .card-value {{ font-size: 20px; font-weight: 500; }}
  .card-sub   {{ font-size: 12px; color: #898781; margin-top: 4px; }}
  .positive {{ color: #0f6e56; }}
  .negative {{ color: #a32d2d; }}
  .neutral  {{ color: #185fa5; }}
  .section {{ margin-bottom: 2rem; }}
  .section-title {{ font-size: 13px; font-weight: 500; color: #52514e; margin-bottom: 0.5rem; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px;
             color: #898781; margin-bottom: 8px; }}
  .chart-wrap {{ position: relative; width: 100%; background: #fff;
                 border: 0.5px solid rgba(11,11,11,0.10); border-radius: 8px; padding: 1rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ text-align: left; font-weight: 500; padding: 6px 8px;
        border-bottom: 0.5px solid #e1e0d9; color: #52514e; }}
  td {{ padding: 6px 8px; border-bottom: 0.5px solid #f1efea; }}
  tr:last-child td {{ border-bottom: none; }}
  .sig-yes {{ color: #0f6e56; font-weight: 600; }}
  .sig-no  {{ color: #898781; }}
  .doe-bad {{ background: #fdeaea; }}
  footer {{ margin-top: 2rem; font-size: 12px; color: #898781; }}
</style>
</head>
<body>

<h1>SAC-GNN V2G Hub — Evaluation Report{seed_suffix}</h1>
<p class="meta">
  {results_dirs_str}
  &nbsp;·&nbsp; Generated: {generated}
</p>

{doe_warning_html}

<div class="section">
  <p class="section-title">Normal-day Net Profit ($/day) — by agent, by case study</p>
  <div class="chart-wrap" style="height:320px">
    <canvas id="profitChart"></canvas>
  </div>
</div>

<div class="section">
  <p class="section-title">Stress-Test Net Profit ($/day) — extreme negative RRP day (separate scale)</p>
  <div class="chart-wrap" style="height:280px">
    <canvas id="stressChart"></canvas>
  </div>
</div>

<div class="section">
  <p class="section-title">DOE Compliance Rate (%) — steps with zero violation, averaged over normal-day case studies</p>
  <div class="chart-wrap" style="height:220px">
    <canvas id="doeChart"></canvas>
  </div>
</div>

<div class="section">
  <p class="section-title">Participation Rate ρ (%) — averaged over normal-day case studies</p>
  <div class="chart-wrap" style="height:220px">
    <canvas id="participationChart"></canvas>
  </div>
</div>

<div class="section">
  <p class="section-title">Inference Latency (ms/step) — mean, log scale</p>
  <div class="chart-wrap" style="height:220px">
    <canvas id="latencyChart"></canvas>
  </div>
</div>

<div class="section">
  <p class="section-title">Statistical Significance — SAC-GNN vs each agent, paired Wilcoxon signed-rank (α=0.05)</p>
  <div class="card" style="padding:0;overflow:hidden">
    <table>
      <thead>
        <tr>
          <th>Case Study</th><th>Comparison</th><th>Mean Diff ($)</th>
          <th>Wilcoxon p</th><th>Sig.</th><th>Paired-t p</th><th>Sig.</th>
        </tr>
      </thead>
      <tbody id="sigTable"></tbody>
    </table>
  </div>
</div>

{multiseed_section_html}

<footer>SAC-GNN V2G Hub · Evaluation Report · {generated}</footer>

<script>
const profitData      = {profit_json};
const stressData      = {stress_json};
const doeData         = {doe_json};
const participationData = {participation_json};
const latencyData     = {latency_json};
const sigData          = {sig_json};

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

// --- Normal-day profit chart (grouped bars: case study x agent) ---
new Chart(document.getElementById('profitChart'), {{
  type: 'bar',
  data: {{
    labels: profitData.case_studies,
    datasets: profitData.agents.map((agent, i) => ({{
      label: agent,
      data: profitData.values[agent],
      backgroundColor: colors[i % colors.length],
      borderRadius: 3,
    }}))
  }},
  options: {{ ...baseOpts, scales: {{ ...baseOpts.scales,
    y: {{ ...baseOpts.scales.y, ticks: {{ ...baseOpts.scales.y.ticks, callback: moneyFmt }} }}
  }}}}
}});

// --- Stress test chart ---
new Chart(document.getElementById('stressChart'), {{
  type: 'bar',
  data: {{
    labels: stressData.agents,
    datasets: [{{
      label: 'Stress-test profit',
      data: stressData.values,
      backgroundColor: stressData.values.map(v => v >= 0 ? '#1baf7a' : '#e34948'),
      borderRadius: 3,
    }}]
  }},
  options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins, legend: {{ display: false }} }},
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, ticks: {{ ...baseOpts.scales.y.ticks, callback: moneyFmt }} }}
    }}
  }}
}});

// --- DOE compliance chart ---
new Chart(document.getElementById('doeChart'), {{
  type: 'bar',
  data: {{
    labels: doeData.agents,
    datasets: [{{
      label: 'DOE compliance %',
      data: doeData.values,
      backgroundColor: doeData.values.map(v => v >= 95 ? '#1baf7a' : (v >= 50 ? '#eda100' : '#e34948')),
      borderRadius: 3,
    }}]
  }},
  options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins, legend: {{ display: false }} }},
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, min: 0, max: 100, ticks: {{ ...baseOpts.scales.y.ticks, callback: v => v + '%' }} }}
    }}
  }}
}});

// --- Participation chart ---
new Chart(document.getElementById('participationChart'), {{
  type: 'bar',
  data: {{
    labels: participationData.agents,
    datasets: [{{
      label: 'Participation %',
      data: participationData.values,
      backgroundColor: '#2a78d6',
      borderRadius: 3,
    }}]
  }},
  options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins, legend: {{ display: false }} }},
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, ticks: {{ ...baseOpts.scales.y.ticks, callback: v => v + '%' }} }}
    }}
  }}
}});

// --- Latency chart (log scale) ---
new Chart(document.getElementById('latencyChart'), {{
  type: 'bar',
  data: {{
    labels: latencyData.agents,
    datasets: [{{
      label: 'Mean latency (ms)',
      data: latencyData.values,
      backgroundColor: '#9b59b6',
      borderRadius: 3,
    }}]
  }},
  options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins, legend: {{ display: false }} }},
    scales: {{ ...baseOpts.scales,
      y: {{ ...baseOpts.scales.y, type: 'logarithmic',
        ticks: {{ ...baseOpts.scales.y.ticks, callback: v => v + 'ms' }} }}
    }}
  }}
}});

// --- Significance table ---
const sigTbody = document.getElementById('sigTable');
sigData.forEach(row => {{
  const wSig = row.wilcoxon_significant;
  const tSig = row.ttest_significant;
  sigTbody.innerHTML += `<tr>
    <td>${{row.case_study}}</td>
    <td>${{row.agent_a}} vs ${{row.agent_b}}</td>
    <td>${{row.mean_diff >= 0 ? '+' : ''}}${{row.mean_diff.toLocaleString('en-AU', {{maximumFractionDigits:0}})}}</td>
    <td>${{row.wilcoxon_p.toFixed(4)}}</td>
    <td class="${{wSig ? 'sig-yes' : 'sig-no'}}">${{wSig ? '✓ Sig.' : '—'}}</td>
    <td>${{row.ttest_p.toFixed(4)}}</td>
    <td class="${{tSig ? 'sig-yes' : 'sig-no'}}">${{tSig ? '✓ Sig.' : '—'}}</td>
  </tr>`;
}});
</script>
</body>
</html>
"""

MULTISEED_SECTION_TEMPLATE = """
<div class="section">
  <p class="section-title">Cross-Seed Aggregate — Normal-day Mean Profit (mean ± std across {n_seeds} seeds)</p>
  <div class="card" style="padding:0;overflow:hidden">
    <table>
      <thead><tr><th>Agent</th><th>Mean ($/day)</th><th>Std ($/day)</th><th>Seeds</th></tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>
</div>
"""


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


def generate_report(
    parsed_dirs: list,
    out_path: str,
) -> None:
    """
    parsed_dirs : list of dicts from parse_evaluation_dir(), one per seed.
    Uses the FIRST dir's metrics for the main charts; if more than one
    dir is given, adds a cross-seed aggregate table at the bottom.
    """
    primary = parsed_dirs[0]
    metrics_df = primary["metrics_df"]
    sig_df     = primary["sig_df"]

    doe_flags = compute_doe_flags(metrics_df)
    flagged_agents = [a for a, f in doe_flags.items() if f["is_flagged"]]

    if flagged_agents:
        zero_agents = [a for a, f in doe_flags.items() if f["is_zero"]]
        doe_warning_html = f"""
<div class="warning-banner">
  <div class="title">⚠ DOE Compliance Warning</div>
  <div class="body">
    <span class="agent-list">{', '.join(flagged_agents)}</span> show DOE compliance
    below 95% on normal-day case studies.
    {"Agents at exactly 0.0% compliance (" + ', '.join(zero_agents) + ") are consistent with hitting the per-step DOE penalty cap on every single step — this is a strong signal of a bug in equipment-cap or DOE-limit reading, not a genuine policy failure. Recommend checking the training log's DOE violation trajectory for these agents before trusting these evaluation numbers." if zero_agents else ""}
  </div>
</div>
"""
    else:
        doe_warning_html = ""

    profit_data        = build_profit_chart_data(metrics_df)
    stress_data         = build_stress_chart_data(metrics_df)
    doe_data            = build_doe_chart_data(metrics_df)
    participation_data  = build_participation_chart_data(metrics_df)
    latency_data        = build_latency_chart_data(metrics_df)
    sig_records          = sig_df.to_dict("records") if not sig_df.empty else []

    # Multi-seed aggregate section
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
            mean_v = np.mean(vals)
            std_v  = np.std(vals)
            cls = "positive" if mean_v >= 0 else "negative"
            rows_html.append(
                f'<tr><td>{agent}</td>'
                f'<td class="{cls}">{"+" if mean_v >= 0 else ""}{mean_v:,.0f}</td>'
                f'<td>{std_v:,.0f}</td>'
                f'<td>{len(vals)}</td></tr>'
            )
        multiseed_html = MULTISEED_SECTION_TEMPLATE.format(
            n_seeds=len(parsed_dirs),
            rows="\n        ".join(rows_html),
        )

    seed_labels = [d["seed_label"] for d in parsed_dirs]
    seed_suffix = f" — {', '.join(seed_labels)}" if len(seed_labels) <= 3 else f" — {len(seed_labels)} seeds"
    results_dirs_str = " · ".join(d["results_dir"] for d in parsed_dirs)

    html = HTML_TEMPLATE.format(
        seed_suffix         = seed_suffix,
        results_dirs_str    = results_dirs_str,
        generated           = datetime.now().strftime("%Y-%m-%d %H:%M"),
        doe_warning_html    = doe_warning_html,
        multiseed_section_html = multiseed_html,
        profit_json         = json.dumps(profit_data),
        stress_json          = json.dumps(stress_data),
        doe_json             = json.dumps(doe_data),
        participation_json  = json.dumps(participation_data),
        latency_json         = json.dumps(latency_data),
        sig_json             = json.dumps(sig_records),
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Report saved to: {out_path}")

    if flagged_agents:
        print(f"\n⚠ DOE compliance warning: {', '.join(flagged_agents)}")
        if any(doe_flags[a]["is_zero"] for a in flagged_agents):
            print("  Some agents are at EXACTLY 0.0% compliance — likely a bug, not a real result.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML report from evaluate.py output."
    )
    parser.add_argument(
        "--results_dir", action="append", required=True,
        help="Path to an evaluate.py results directory (e.g. results/evaluation_seed42). "
             "Pass multiple times to aggregate across seeds.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output HTML path. Default: training_report_evaluation_<timestamp>.html",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open the report in your default browser after generating.",
    )
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
