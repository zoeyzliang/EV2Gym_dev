"""
visualise_training.py
=====================
Parse one or more SLURM .err logs from M3 and generate a self-contained
HTML training report for the SAC-GNN V2G Hub training run.

Usage:
    # Single log
    python visualise_training.py --log logs/slurm_57942364.err --open

    # Merge two logs into one 2000-episode view (original run + resume run)
    python visualise_training.py \
        --log logs/slurm_57942364.err logs/slurm_58308859.err \
        --open

    # Custom output name
    python visualise_training.py --log logs/slurm_57942364.err --out reports/my_report.html

Default output name: training_report_<jobid>_<YYYYMMDD>.html
When merging logs:   training_report_<jobid1>+<jobid2>_<YYYYMMDD>.html

New in v3:
    - Accept multiple --log files and merge them (deduplication by episode)
    - Auto-naming: training_report_{jobid}_{date}.html
    - Reward chart y-axis clipped to remove catastrophic early outliers
      (clip at 5th/95th percentile so the interesting range is visible)
    - Clip toggle button in the report to switch between clipped/full view
"""

import re
import numpy as np
import argparse
import json
import webbrowser
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_time(t: str) -> int:
    h, m, s = map(int, t.split(':'))
    return h * 3600 + m * 60 + s


def parse_log(log_path: str) -> dict:
    train_re   = re.compile(
        r'(\d{2}:\d{2}:\d{2})\s+\[INFO\]\s+Ep\s+(\d+)/(\d+)'
        r'.*?reward=\s*([-\d.]+)'
        r'(?:.*?avg10=\s*([-\d.]+))?'
        r'(?:.*?[rρ]=([\d.]+))?'
        r'(?:.*?doe_viol=([\d.]+)kW)?'
        r'(?:.*?buf=\s*(\d+))?'
        r'(?:.*?[aα]=([\d.e+-]+))?'
    )
    # Handles both the old log format (profit=X ±Y) and the new one that
    # separates pooled vs normal-day-only profit (profit(all 5)=X ...
    # profit(normal 4)=Y ...) after the stress-test-day fix. Group 2 is
    # always the pooled/primary profit value for backward compatibility;
    # group 3 (optional) captures the normal-day value when present.
    eval_re    = re.compile(
        r'(\d{2}:\d{2}:\d{2})\s+\[INFO\].*?Eval:\s+profit(?:\(all 5\))?=([-\d.]+)'
        r'(?:.*?profit\(normal 4\)=([-\d.]+))?'
        r'.*?participation=([\d.]+)'
        r'.*?doe_viol=([\d.]+)kW'
    )
    # Matches both "New best net profit:" (pre-fix logs) and
    # "New best normal-day profit:" (post-fix logs, using the
    # stress-test-excluded metric for checkpoint selection)
    best_re    = re.compile(r'\[INFO\].*?New best (?:net|normal-day) profit:\s*([-\d.]+)')
    cancel_re  = re.compile(r'CANCELLED.*DUE TO TIME LIMIT')
    job_re     = re.compile(r'JOB\s+(\d+)\s+ON\s+(\S+)')
    started_re = re.compile(r'(\d{2}:\d{2}:\d{2})\s+\[INFO\]\s+Starting training:\s+(\d+) episodes')
    resume_re  = re.compile(r'(\d{2}:\d{2}:\d{2})\s+\[INFO\]\s+Resuming from episode\s+(\d+)')
    jobstart_re= re.compile(r'Job started:.*?(\d{2}:\d{2}:\d{2})')
    jobend_re  = re.compile(r'Job finished:.*?(\d{2}:\d{2}:\d{2})')
    complete_re= re.compile(r'Training complete in ([\d.]+) minutes')

    train_rows, eval_rows, best_profits = [], [], []
    cancelled = False
    completed = False
    job_id = node = total_episodes = None
    train_start_time = job_wall_start = job_wall_end = None
    last_train_ep = None
    start_episode = 1
    training_minutes = None

    with open(log_path, 'r', errors='replace') as f:
        lines = f.readlines()

    for line in lines:
        m = job_re.search(line)
        if m:
            job_id = m.group(1)
            node   = m.group(2)

        m = jobstart_re.search(line)
        if m and job_wall_start is None:
            job_wall_start = m.group(1)

        m = jobend_re.search(line)
        if m:
            job_wall_end = m.group(1)

        m = complete_re.search(line)
        if m:
            training_minutes = float(m.group(1))
            completed = True

        m = started_re.search(line)
        if m and train_start_time is None:
            train_start_time = m.group(1)
            total_episodes   = int(m.group(2))

        m = resume_re.search(line)
        if m and train_start_time is None:
            train_start_time = m.group(1)
            start_episode    = int(m.group(2))

        m = train_re.search(line)
        if m:
            ep = int(m.group(2))
            last_train_ep = ep
            train_rows.append({
                'time':     m.group(1),
                'episode':  ep,
                'total':    int(m.group(3)),
                'reward':   float(m.group(4)),
                'avg10':    float(m.group(5)) if m.group(5) else None,
                'rho':      float(m.group(6)) if m.group(6) else None,
                'doe_viol': float(m.group(7)) if m.group(7) else None,
                'buf':      int(m.group(8))   if m.group(8) else None,
                'alpha':    float(m.group(9)) if m.group(9) else None,
            })
            continue

        m = eval_re.search(line)
        if m:
            eval_rows.append({
                'time':          m.group(1),
                'episode':       last_train_ep,
                'profit':        float(m.group(2)),
                # profit_normal (excludes stress-test day) is only present
                # in post-fix logs; falls back to the pooled value for
                # older logs so charts/tables still render sensibly
                'profit_normal': float(m.group(3)) if m.group(3) else float(m.group(2)),
                'participation': float(m.group(4)),
                'doe_viol':      float(m.group(5)),
            })
            continue

        m = best_re.search(line)
        if m:
            best_profits.append(float(m.group(1)))

        if cancel_re.search(line):
            cancelled = True

    return {
        'train':            train_rows,
        'eval':             eval_rows,
        'best_profits':     best_profits,
        'cancelled':        cancelled,
        'completed':        completed,
        'job_id':           job_id,
        'node':             node,
        'total_episodes':   total_episodes or (train_rows[-1]['total'] if train_rows else None),
        'train_start_time': train_start_time,
        'start_episode':    start_episode,
        'job_wall_start':   job_wall_start,
        'job_wall_end':     job_wall_end,
        'training_minutes': training_minutes,
        'log_path':         str(log_path),
    }


def merge_logs(datas: list) -> dict:
    """
    Merge multiple parsed log dicts into one, deduplicating by episode number.
    Later logs take precedence for duplicate episodes (resume overwrites original).
    Combines job IDs and nodes for display.
    """
    # Collect all rows, keyed by episode
    train_by_ep = {}
    eval_by_ep  = {}

    for d in datas:
        for r in d['train']:
            train_by_ep[r['episode']] = r
        for r in d['eval']:
            if r['episode'] not in eval_by_ep:
                eval_by_ep[r['episode']] = r
            else:
                # Keep the one with higher profit (prefer original good run)
                if r['profit'] > eval_by_ep[r['episode']]['profit']:
                    eval_by_ep[r['episode']] = r

    train_rows = sorted(train_by_ep.values(), key=lambda r: r['episode'])
    eval_rows  = sorted(eval_by_ep.values(),  key=lambda r: r['episode'])

    job_ids = [d['job_id'] for d in datas if d['job_id']]
    nodes   = list({d['node'] for d in datas if d['node']})

    # Total wall time = sum of all job wall times
    total_wall_min = None
    wall_mins = []
    for d in datas:
        if d.get('training_minutes'):
            wall_mins.append(d['training_minutes'])
        elif d.get('job_wall_start') and d.get('job_wall_end'):
            j0 = parse_time(d['job_wall_start'])
            j1 = parse_time(d['job_wall_end'])
            if j1 < j0: j1 += 86400
            wall_mins.append((j1 - j0) / 60)
    if wall_mins:
        total_wall_min = sum(wall_mins)

    cancelled = any(d['cancelled'] for d in datas)
    completed = any(d['completed'] for d in datas)

    return {
        'train':            train_rows,
        'eval':             eval_rows,
        'best_profits':     [p for d in datas for p in d['best_profits']],
        'cancelled':        cancelled,
        'completed':        completed,
        'job_id':           '+'.join(job_ids),
        'node':             ', '.join(nodes),
        'total_episodes':   max((d['total_episodes'] or 0) for d in datas),
        'train_start_time': datas[0]['train_start_time'],
        'job_wall_start':   datas[0]['job_wall_start'],
        'job_wall_end':     datas[-1]['job_wall_end'],
        'training_minutes': total_wall_min,
        'log_path':         ' + '.join(d['log_path'] for d in datas),
        'n_logs':           len(datas),
    }


# ---------------------------------------------------------------------------
# Timing & summary
# ---------------------------------------------------------------------------

STEPS_PER_EPISODE = 288

def compute_timing(data: dict) -> dict:
    train = data['train']
    if len(train) < 2:
        return {}

    secs = [parse_time(r['time']) for r in train]
    # Handle midnight rollover within a segment
    for i in range(1, len(secs)):
        if secs[i] < secs[i-1]:
            secs[i] += 86400

    durations_sec = []
    for i in range(1, len(secs)):
        ep_gap   = train[i]['episode'] - train[i-1]['episode']
        time_gap = secs[i] - secs[i-1]
        # Skip: episode went backward (dedup artefact), time gap negative
        # (cross-job boundary where second log starts earlier in the day),
        # or time gap > 2h (implausible for a single batch of episodes)
        if ep_gap <= 0 or time_gap <= 0 or time_gap > 7200:
            continue
        durations_sec.append({
            'episode':      train[i]['episode'],
            'duration_sec': time_gap,
            'ep_gap':       ep_gap,
            'min_per_ep':   time_gap / 60.0 / ep_gap,
        })

    if not durations_sec:
        return {}

    last_ep   = train[-1]['episode']
    total_eps = data['total_episodes'] or last_ep
    remaining = total_eps - last_ep

    warmup_idx = max(0, len(durations_sec) - max(1, len(durations_sec) // 5))
    recent = durations_sec[warmup_idx:]
    avg_min = sum(d['min_per_ep'] for d in recent) / len(recent)

    all_min = [d['min_per_ep'] for d in durations_sec]
    eps_hr  = 60 / avg_min if avg_min > 0 else 0

    # Compute total elapsed from episode count x avg time per episode.
    # Raw HH:MM:SS timestamps only roll over once per midnight (+86400s),
    # so a 48h job spanning 2 midnights shows ~24h from raw timestamps.
    # Using avg_min x n_episodes is accurate regardless of job duration.
    n_eps_completed = last_ep - (train[0]['episode'] - 1)
    total_elapsed_min_from_eps = avg_min * n_eps_completed

    # Cross-check against reported wall time if available
    reported = data.get('training_minutes')
    if reported and reported > total_elapsed_min_from_eps * 0.5:
        total_elapsed_min = reported
    else:
        total_elapsed_min = total_elapsed_min_from_eps

    return {
        'durations':               durations_sec,
        'total_elapsed_min':       total_elapsed_min,
        'avg_min_per_ep':          avg_min,
        'min_min_per_ep':          min(all_min),
        'max_min_per_ep':          max(all_min),
        'eps_per_hour':            eps_hr,
        'steps_per_hour':          eps_hr * STEPS_PER_EPISODE,
        'projected_remaining_min': avg_min * remaining,
        'remaining_eps':           remaining,
    }


def compute_summary(data: dict) -> dict:
    train = data['train']
    evals = data['eval']
    if not train:
        return {}

    best_eval   = max((e.get('profit_normal', e['profit']) for e in evals), default=None)
    best_ep     = next((e['episode'] for e in evals if e.get('profit_normal', e['profit']) == best_eval), None) if best_eval is not None else None
    last_ep     = train[-1]['episode']
    total_eps   = data['total_episodes'] or last_ep
    last_doe    = train[-1]['doe_viol']
    rhos        = [r['rho'] for r in train if r['rho'] is not None]
    mean_rho    = sum(rhos) / len(rhos) if rhos else 0
    final_alpha = train[-1]['alpha']
    doe_zero_ep = next((t['episode'] for t in train if t['doe_viol'] == 0.0), None)

    return {
        'best_eval_profit': best_eval,
        'best_eval_ep':     best_ep,
        'last_episode':     last_ep,
        'total_episodes':   total_eps,
        'last_doe_viol':    last_doe,
        'mean_rho':         mean_rho,
        'final_alpha':      final_alpha,
        'doe_zero_ep':      doe_zero_ep,
    }


def fmt_profit(v):
    if v is None: return 'N/A'
    return ('+' if v >= 0 else '-') + f'${abs(v):,.0f}'

def fmt_duration(m):
    if m is None: return 'N/A'
    h, mn = int(m // 60), int(m % 60)
    return f'{h}h {mn}m' if h > 0 else f'{mn}m'

def percentile(lst, p):
    s = sorted(lst)
    idx = int(len(s) * p / 100)
    return s[max(0, min(idx, len(s)-1))]


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAC-GNN Training Report — {job_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f8f8f7; color: #0b0b0b; padding: 2rem;
          max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 20px; font-weight: 500; margin-bottom: 4px; }}
  h2 {{ font-size: 14px; font-weight: 500; color: #52514e; margin: 2rem 0 1rem;
        padding-bottom: 6px; border-bottom: 0.5px solid #e1e0d9; }}
  .meta {{ font-size: 13px; color: #898781; margin-bottom: 1.5rem; line-height: 1.6; }}
  .badge {{ display: inline-block; font-size: 11px; padding: 2px 8px;
            border-radius: 4px; margin-left: 6px; vertical-align: middle; }}
  .badge-warn {{ background:#faeeda; color:#854f0b; }}
  .badge-ok   {{ background:#eaf3de; color:#3b6d11; }}
  .badge-info {{ background:#e8eef8; color:#185fa5; }}
  .metric-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(148px,1fr));
                 gap: 12px; margin-bottom: 1.5rem; }}
  .card {{ background:#fff; border:0.5px solid rgba(11,11,11,.10);
           border-radius:8px; padding:1rem; }}
  .card-label {{ font-size:12px; color:#898781; margin-bottom:4px; }}
  .card-value {{ font-size:22px; font-weight:500; }}
  .card-sub   {{ font-size:12px; color:#898781; margin-top:4px; }}
  .positive {{ color:#0f6e56; }} .negative {{ color:#a32d2d; }} .neutral {{ color:#185fa5; }}
  .section {{ margin-bottom:2rem; }}
  .section-header {{ display:flex; align-items:center; justify-content:space-between;
                     margin-bottom:0.5rem; }}
  .section-title {{ font-size:13px; font-weight:500; color:#52514e; }}
  .btn {{ font-size:11px; padding:3px 10px; border-radius:4px; border:0.5px solid #c8c7c0;
          background:#fff; cursor:pointer; color:#52514e; }}
  .btn:hover {{ background:#f1efea; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:16px; font-size:12px;
             color:#898781; margin-bottom:8px; }}
  .legend span {{ display:flex; align-items:center; gap:4px; }}
  .ldot {{ width:10px; height:10px; border-radius:2px; display:inline-block; }}
  .chart-wrap {{ position:relative; width:100%; background:#fff;
                 border:0.5px solid rgba(11,11,11,.10); border-radius:8px; padding:1rem; }}
  .progress-bar {{ background:#f1efea; border-radius:4px; height:8px; margin-top:6px; }}
  .progress-fill {{ height:8px; border-radius:4px; background:#2a78d6; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; font-weight:500; padding:6px 10px;
        border-bottom:0.5px solid #e1e0d9; color:#52514e; }}
  td {{ padding:6px 10px; border-bottom:0.5px solid #f1efea; }}
  tr:last-child td {{ border-bottom:none; }}
  .pos {{ color:#0f6e56; font-weight:500; }} .neg {{ color:#a32d2d; }}
  footer {{ margin-top:2rem; font-size:12px; color:#898781; }}
</style>
</head>
<body>

<h1>SAC-GNN V2G Hub — Training Report</h1>
<p class="meta">
  Job: <strong>{job_id}</strong>
  <span class="badge badge-info">{n_logs} log(s) merged</span>
  {status_badge}
  <br>
  Node: {node} &nbsp;·&nbsp; Generated: {generated}
  <br>
  <span style="font-size:11px">{log_path}</span>
</p>

<h2>Training outcomes</h2>
<div class="metric-row">
  <div class="card">
    <p class="card-label">Best eval profit</p>
    <p class="card-value {profit_class}">{best_profit}</p>
    <p class="card-sub">Episode {best_ep}</p>
  </div>
  <div class="card">
    <p class="card-label">Episodes completed</p>
    <p class="card-value">{last_ep:,} / {total_eps:,}</p>
    <div class="progress-bar">
      <div class="progress-fill" style="width:{progress_pct:.1f}%"></div>
    </div>
    <p class="card-sub">{progress_pct:.1f}% complete</p>
  </div>
  <div class="card">
    <p class="card-label">Final DOE violation</p>
    <p class="card-value {doe_class}">{last_doe} kW</p>
    <p class="card-sub">{doe_sub}</p>
  </div>
  <div class="card">
    <p class="card-label">Mean participation ρ</p>
    <p class="card-value">{mean_rho}</p>
    <p class="card-sub">Avg across training</p>
  </div>
  <div class="card">
    <p class="card-label">Final alpha</p>
    <p class="card-value">{final_alpha}</p>
    <p class="card-sub">SAC entropy temp</p>
  </div>
</div>

<h2>Performance &amp; timing</h2>
<div class="metric-row">
  <div class="card">
    <p class="card-label">Total training time</p>
    <p class="card-value neutral">{total_elapsed}</p>
    <p class="card-sub">All jobs combined</p>
  </div>
  <div class="card">
    <p class="card-label">Avg time / episode</p>
    <p class="card-value neutral">{avg_min_per_ep}</p>
    <p class="card-sub">Recent avg (last 20%)</p>
  </div>
  <div class="card">
    <p class="card-label">Throughput</p>
    <p class="card-value neutral">{eps_per_hour}</p>
    <p class="card-sub">{steps_per_hour} steps/hr</p>
  </div>
  <div class="card">
    <p class="card-label">Remaining</p>
    <p class="card-value {remaining_class}">{projected_remaining}</p>
    <p class="card-sub">{remaining_eps:,} episodes left</p>
  </div>
</div>

<div class="section">
  <div class="section-header">
    <p class="section-title">Episode duration (min/episode)</p>
  </div>
  <div class="legend">
    <span><span class="ldot" style="background:#4a3aa7"></span>Min/episode</span>
    <span><span class="ldot" style="border:1.5px dashed #eda100; display:inline-block; width:10px; height:10px; border-radius:2px;"></span>Rolling avg</span>
  </div>
  <div class="chart-wrap" style="height:180px">
    <canvas id="timingChart"></canvas>
  </div>
</div>

<div class="section">
  <div class="section-header">
    <p class="section-title">Episode reward (training)</p>
    <button class="btn" id="clipBtn" onclick="toggleClip()">Show full range</button>
  </div>
  <div class="legend">
    <span><span class="ldot" style="background:#2a78d6"></span>Episode reward</span>
    <span><span class="ldot" style="background:#1baf7a"></span>10-ep moving avg</span>
    <span style="font-size:11px;color:#eda100">⚡ Y-axis clipped by default (median ± 4×MAD)</span>
  </div>
  <div class="chart-wrap" style="height:260px">
    <canvas id="rewardChart"></canvas>
  </div>
</div>

<div class="section">
  <div class="section-header">
    <p class="section-title">Eval net profit at checkpoints (normal-day mean, excl. stress test)</p>
    <button class="btn" id="evalClipBtn" onclick="toggleEvalClip()">Show full range</button>
  </div>
  <div class="legend">
    <span><span class="ldot" style="background:#1baf7a"></span>Positive</span>
    <span><span class="ldot" style="background:#e34948"></span>Negative</span>
    <span style="font-size:11px;color:#eda100">⚡ Y-axis clipped by default (median ± 4×MAD)</span>
    <span style="font-size:11px;color:#898781">Excludes the extreme negative-RRP stress-test day — matches best.pt selection metric</span>
  </div>
  <div class="chart-wrap" style="height:200px">
    <canvas id="evalChart"></canvas>
  </div>
</div>

<div class="section">
  <div class="section-header">
    <p class="section-title">Eval net profit — all 5 days pooled (for reference)</p>
  </div>
  <div class="legend">
    <span><span class="ldot" style="background:#9b59b6"></span>Pooled (incl. stress test)</span>
    <span style="font-size:11px;color:#898781">Dominated by the extreme negative-RRP day — shown for reference only, not used for checkpoint selection</span>
  </div>
  <div class="chart-wrap" style="height:200px">
    <canvas id="evalPooledChart"></canvas>
  </div>
</div>

<div class="section">
  <p class="section-title">DOE violation (kW) per episode</p>
  <div class="legend">
    <span><span class="ldot" style="background:#e34948"></span>DOE violation kW</span>
  </div>
  <div class="chart-wrap" style="height:180px">
    <canvas id="doeChart"></canvas>
  </div>
</div>

<div class="section">
  <p class="section-title">Alpha (entropy temperature) over training</p>
  <div class="legend">
    <span><span class="ldot" style="background:#eda100"></span>Alpha</span>
  </div>
  <div class="chart-wrap" style="height:160px">
    <canvas id="alphaChart"></canvas>
  </div>
</div>

<div class="section">
  <p class="section-title">Evaluation checkpoints</p>
  <div class="card" style="padding:0;overflow:hidden">
    <table>
      <thead>
        <tr>
          <th>Episode</th><th>Eval profit — Normal-day ($)</th>
          <th>Eval profit — Pooled all 5 ($)</th>
          <th>Participation ρ</th><th>DOE violation (kW)</th>
        </tr>
      </thead>
      <tbody id="evalTable"></tbody>
    </table>
  </div>
</div>

<footer>SAC-GNN V2G Hub · Monash University FIT5126 · {generated}</footer>

<script>
const trainData  = {train_json};
const evalData   = {eval_json};
const timingData = {timing_json};
const rewardClipLow  = {clip_low};
const rewardClipHigh = {clip_high};
const evalClipLow    = {eval_clip_low};
const evalClipHigh   = {eval_clip_high};

const eps     = trainData.map(r => r.episode);
const rewards = trainData.map(r => r.reward);
const avg10   = trainData.map((_, i) => {{
  const sl = trainData.slice(Math.max(0,i-9), i+1);
  return sl.reduce((a,b)=>a+b.reward,0)/sl.length;
}});
const doeViol = trainData.map(r => r.doe_viol ?? 0);
const alphas  = trainData.map(r => r.alpha ?? null);
const timingEps = timingData.map(d => d.episode);
const minPerEp  = timingData.map(d => +d.min_per_ep.toFixed(2));
const rollingAvg = (arr, n) => arr.map((_,i) => {{
  const sl = arr.slice(Math.max(0,i-n+1),i+1);
  return sl.reduce((a,b)=>a+b,0)/sl.length;
}});
const timingAvg = rollingAvg(minPerEp, 10);

const grid = '#e1e0d9', tick = '#898781';
const base = {{
  responsive:true, maintainAspectRatio:false, animation:false,
  plugins:{{ legend:{{display:false}}, tooltip:{{mode:'index',intersect:false}} }},
  scales:{{
    x:{{grid:{{color:grid}}, ticks:{{color:tick,maxTicksLimit:12,font:{{size:11}}}}}},
    y:{{grid:{{color:grid}}, ticks:{{color:tick,font:{{size:11}}}}}}
  }}
}};
const profitFmt = v => (v<0?'-$':'$')+Math.abs(v/1000).toFixed(0)+'k';

new Chart(document.getElementById('timingChart'), {{
  type:'line',
  data:{{
    labels:timingEps,
    datasets:[
      {{label:'Min/ep', data:minPerEp, borderColor:'#4a3aa7', borderWidth:1.5, pointRadius:0, tension:0.2, fill:false}},
      {{label:'Avg',    data:timingAvg, borderColor:'#eda100', borderWidth:2, borderDash:[5,3], pointRadius:0, tension:0.3, fill:false}}
    ]
  }},
  options:{{...base, scales:{{...base.scales,
    y:{{...base.scales.y, ticks:{{...base.scales.y.ticks, callback:v=>v.toFixed(1)+' min'}}}}
  }}}}
}});

let clipped = true;
const rewardChart = new Chart(document.getElementById('rewardChart'), {{
  type:'line',
  data:{{
    labels:eps,
    datasets:[
      {{label:'Reward',   data:rewards, borderColor:'#2a78d6', borderWidth:1.5, pointRadius:0, tension:0.2, fill:false}},
      {{label:'10-ep avg',data:avg10,   borderColor:'#1baf7a', borderWidth:2, borderDash:[5,3], pointRadius:0, tension:0.3, fill:false}}
    ]
  }},
  options:{{...base, scales:{{...base.scales,
    y:{{...base.scales.y,
      min:rewardClipLow, max:rewardClipHigh,
      ticks:{{...base.scales.y.ticks, callback:profitFmt}}
    }}
  }}}}
}});

function toggleClip() {{
  clipped = !clipped;
  const sc = rewardChart.options.scales.y;
  if (clipped) {{
    sc.min = rewardClipLow; sc.max = rewardClipHigh;
    document.getElementById('clipBtn').textContent = 'Show full range';
  }} else {{
    delete sc.min; delete sc.max;
    document.getElementById('clipBtn').textContent = 'Clip outliers';
  }}
  rewardChart.update();
}}

let evalClipped = true;
let evalChart;
// Primary chart uses profit_normal (excludes the stress-test day) — this
// is the metric that actually drives best.pt selection and convergence
// detection in train_sac_gnn.py, so this chart should match what the
// training loop was actually optimising against.
evalChart = new Chart(document.getElementById('evalChart'), {{
  type:'bar',
  data:{{
    labels:evalData.map(e=>e.episode),
    datasets:[{{
      label:'Eval profit (normal-day, excl. stress test)',
      data:evalData.map(e=>e.profit_normal),
      backgroundColor:evalData.map(e=>e.profit_normal>=0?'#1baf7a':'#e34948'),
      borderRadius:3, borderSkipped:false
    }}]
  }},
  options:{{...base, scales:{{...base.scales,
    y:{{...base.scales.y,
      min:evalClipLow, max:evalClipHigh,
      ticks:{{...base.scales.y.ticks, callback:profitFmt}}
    }}
  }}}}
}});

function toggleEvalClip() {{
  evalClipped = !evalClipped;
  const sc = evalChart.options.scales.y;
  if (evalClipped) {{
    sc.min = evalClipLow; sc.max = evalClipHigh;
    document.getElementById('evalClipBtn').textContent = 'Show full range';
  }} else {{
    delete sc.min; delete sc.max;
    document.getElementById('evalClipBtn').textContent = 'Clip outliers';
  }}
  evalChart.update();
}}

// Secondary chart: all 5 days pooled (includes the extreme stress-test
// day) — shown for reference/transparency only, un-clipped since its
// whole purpose here is to show the reader how dominant that one day's
// magnitude is relative to the normal-day chart above.
new Chart(document.getElementById('evalPooledChart'), {{
  type:'bar',
  data:{{
    labels:evalData.map(e=>e.episode),
    datasets:[{{
      label:'Eval profit (pooled, all 5 days)',
      data:evalData.map(e=>e.profit),
      backgroundColor:'#9b59b6',
      borderRadius:3, borderSkipped:false
    }}]
  }},
  options:{{...base, scales:{{...base.scales,
    y:{{...base.scales.y, ticks:{{...base.scales.y.ticks, callback:profitFmt}}}}
  }}}}
}});

new Chart(document.getElementById('doeChart'), {{
  type:'line',
  data:{{
    labels:eps,
    datasets:[{{
      label:'DOE kW', data:doeViol,
      borderColor:'#e34948', backgroundColor:'rgba(227,73,72,0.08)',
      borderWidth:1.5, pointRadius:0, tension:0.2, fill:true
    }}]
  }},
  options:{{...base, scales:{{...base.scales,
    y:{{...base.scales.y, ticks:{{...base.scales.y.ticks, callback:v=>v+' kW'}}}}
  }}}}
}});

new Chart(document.getElementById('alphaChart'), {{
  type:'line',
  data:{{
    labels:eps,
    datasets:[{{
      label:'Alpha', data:alphas,
      borderColor:'#eda100', borderWidth:1.5, pointRadius:0, tension:0.2, fill:false
    }}]
  }},
  options:{{...base, scales:{{...base.scales,
    y:{{...base.scales.y, ticks:{{...base.scales.y.ticks, callback:v=>v.toExponential(1)}}}}
  }}}}
}});

const tbody = document.getElementById('evalTable');
evalData.forEach(e => {{
  const clsN = e.profit_normal>=0?'pos':'neg', signN=e.profit_normal>=0?'+':'';
  const clsP = e.profit>=0?'pos':'neg', signP=e.profit>=0?'+':'';
  tbody.innerHTML += `<tr>
    <td>${{e.episode}}</td>
    <td class="${{clsN}}">${{signN}}${{e.profit_normal.toLocaleString('en-AU',{{maximumFractionDigits:0}})}}</td>
    <td class="${{clsP}}">${{signP}}${{e.profit.toLocaleString('en-AU',{{maximumFractionDigits:0}})}}</td>
    <td>${{(e.participation*100).toFixed(1)}}%</td>
    <td>${{e.doe_viol.toFixed(1)}}</td>
  </tr>`;
}});
</script>
</body>
</html>
"""


def generate_report(data: dict, out_path: str) -> None:
    summary = compute_summary(data)
    timing  = compute_timing(data)
    train   = data['train']
    evals   = data['eval']

    # Compute reward clip bounds using median + MAD (median absolute deviation).
    #
    # Why not simple percentiles (the old approach):
    # The 3rd/97th percentile clip only excluded the first 20 episodes from
    # the calculation. When catastrophic losses extend well past episode 20
    # (e.g. still recovering through episode ~100-150, as seen in several
    # real-price runs), 3% of a ~750-episode run is only ~22 episodes —
    # not enough margin, so the percentile itself still lands on a
    # catastrophic value and the "clipped" view still spans millions.
    #
    # Median + MAD is far more robust: MAD requires >50% of the data to be
    # extreme before it gets pulled off course, versus percentiles needing
    # only >3%. This means even a training run with a long catastrophic
    # recovery phase (dozens or hundreds of early episodes) still produces
    # a sensible clip range showing the settled late-training behaviour.
    rewards = [r['reward'] for r in train]
    stable_rewards = [r['reward'] for r in train if r['episode'] > 20]
    if stable_rewards:
        arr = np.array(stable_rewards)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        # 1.4826 scales MAD to be comparable to std for normally-distributed
        # data; k=4 gives a generous but not excessive window
        robust_std = mad * 1.4826
        if robust_std < 1.0:
            # Degenerate case: nearly all values identical (e.g. flat-zero
            # late-training plateau) — fall back to a small fixed margin
            # around the median so the chart isn't a zero-height line
            clip_low  = median - 5000
            clip_high = median + 5000
        else:
            clip_low  = round(median - 4 * robust_std, -3)
            clip_high = round(median + 4 * robust_std, -3)
        clip_high = max(clip_high, 10000)
        clip_low  = min(clip_low, -30000)
    elif rewards:
        clip_low, clip_high = -100000, 100000
    else:
        clip_low, clip_high = -100000, 100000

    # Eval profit clip — same median+MAD approach, applied independently
    # since eval profit is on a different scale (5-episode averages, fewer
    # points, and typically smaller in magnitude than raw training reward).
    # Previously the eval chart had NO clipping at all — a single
    # catastrophic early eval (e.g. -$3.2M at episode 50) would set the
    # y-axis scale for the entire chart, making all later checkpoints
    # (which might be in the tens of thousands) look like a flat zero line.
    eval_profits = [e.get('profit_normal', e['profit']) for e in evals]
    if eval_profits:
        arr = np.array(eval_profits)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        robust_std = mad * 1.4826
        if robust_std < 1.0:
            eval_clip_low  = median - 5000
            eval_clip_high = median + 5000
        else:
            eval_clip_low  = round(median - 4 * robust_std, -3)
            eval_clip_high = round(median + 4 * robust_std, -3)
        eval_clip_high = max(eval_clip_high, 10000)
        eval_clip_low  = min(eval_clip_low, -10000)
    else:
        eval_clip_low, eval_clip_high = -50000, 50000

    best_profit   = fmt_profit(summary.get('best_eval_profit'))
    profit_class  = 'positive' if (summary.get('best_eval_profit') or 0) >= 0 else 'negative'
    last_doe      = summary.get('last_doe_viol', 0) or 0
    doe_class     = 'positive' if last_doe == 0 else 'negative'
    doe_zero_ep   = summary.get('doe_zero_ep')
    doe_sub       = f'Zero from ep {doe_zero_ep}' if doe_zero_ep else 'Still present'
    mean_rho      = f"{summary.get('mean_rho', 0)*100:.1f}%"
    final_alpha   = f"{summary.get('final_alpha', 0):.4f}" if summary.get('final_alpha') else 'N/A'
    last_ep       = summary.get('last_episode', 0)
    total_eps     = summary.get('total_episodes', 0) or last_ep
    progress_pct  = 100 * last_ep / total_eps if total_eps else 0

    completed = data.get('completed', False)
    cancelled = data.get('cancelled', False)
    if completed:
        status_badge = '<span class="badge badge-ok">✓ Completed</span>'
    elif cancelled:
        status_badge = '<span class="badge badge-warn">⚠ Time limit cancelled</span>'
    else:
        status_badge = '<span class="badge badge-info">▶ In progress</span>'

    avg_min  = timing.get('avg_min_per_ep')
    eps_hr   = timing.get('eps_per_hour', 0)
    remaining_eps = timing.get('remaining_eps', 0)
    remaining_class = 'positive' if remaining_eps == 0 else 'neutral'

    html = HTML_TEMPLATE.format(
        job_id           = data['job_id'] or 'unknown',
        n_logs           = data.get('n_logs', 1),
        log_path         = data['log_path'],
        node             = data['node'] or 'unknown',
        generated        = datetime.now().strftime('%Y-%m-%d %H:%M'),
        status_badge     = status_badge,
        best_profit      = best_profit,
        profit_class     = profit_class,
        best_ep          = summary.get('best_eval_ep') or 'N/A',
        last_ep          = last_ep,
        total_eps        = total_eps,
        progress_pct     = progress_pct,
        last_doe         = f'{last_doe:.1f}',
        doe_class        = doe_class,
        doe_sub          = doe_sub,
        mean_rho         = mean_rho,
        final_alpha      = final_alpha,
        total_elapsed    = fmt_duration(timing.get('total_elapsed_min')),
        avg_min_per_ep   = f"{avg_min:.1f} min" if avg_min else 'N/A',
        eps_per_hour     = f"{eps_hr:.1f} ep/hr" if eps_hr else 'N/A',
        steps_per_hour   = f"{timing.get('steps_per_hour',0):,.0f}",
        projected_remaining = fmt_duration(timing.get('projected_remaining_min')),
        remaining_eps    = remaining_eps,
        remaining_class  = remaining_class,
        train_json       = json.dumps(train),
        eval_json        = json.dumps(evals),
        timing_json      = json.dumps(timing.get('durations', [])),
        clip_low         = clip_low,
        clip_high        = clip_high,
        eval_clip_low    = eval_clip_low,
        eval_clip_high   = eval_clip_high,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding='utf-8')
    print(f'Report saved: {out_path}')
    if timing:
        print(f'  Episodes     : {last_ep}/{total_eps} ({progress_pct:.0f}%)')
        print(f'  Best profit  : {best_profit} @ ep {summary.get("best_eval_ep")}')
        print(f'  Training time: {fmt_duration(timing.get("total_elapsed_min"))}')
        print(f'  Throughput   : {eps_hr:.1f} ep/hr')
        print(f'  Reward clip  : [{clip_low:,.0f}, {clip_high:,.0f}]')
        print(f'  Eval clip    : [{eval_clip_low:,.0f}, {eval_clip_high:,.0f}]')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate SAC-GNN training report from one or more SLURM logs.'
    )
    parser.add_argument('--log', required=True, nargs='+',
        help='Path(s) to SLURM .err log file(s). Pass multiple to merge.')
    parser.add_argument('--out', default=None,
        help='Output HTML path. Default: training_report_<jobid>_<date>.html')
    parser.add_argument('--open', action='store_true',
        help='Open in browser after generating.')
    args = parser.parse_args()

    datas = []
    for log_path in args.log:
        p = Path(log_path)
        if not p.exists():
            print(f'Error: not found: {p}')
            return
        print(f'Parsing: {p}')
        d = parse_log(str(p))
        print(f'  → {len(d["train"])} train rows, {len(d["eval"])} eval rows'
              + (' [CANCELLED]' if d['cancelled'] else '')
              + (' [COMPLETED]' if d['completed'] else ''))
        datas.append(d)

    data = merge_logs(datas) if len(datas) > 1 else datas[0]
    if len(datas) == 1:
        data['n_logs'] = 1

    # Auto-name: training_report_<jobid>_<YYYYMMDD>.html
    if args.out:
        out_path = args.out
    else:
        date_str = datetime.now().strftime('%Y%m%d')
        job_id   = (data['job_id'] or 'unknown').replace('+', '_')
        out_path = f'training_report_{job_id}_{date_str}.html'

    generate_report(data, out_path)

    if args.open:
        webbrowser.open(Path(out_path).resolve().as_uri())


if __name__ == '__main__':
    main()
