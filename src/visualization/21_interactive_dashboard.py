#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 21 — Interactive Dashboard (PRO)
- Scans outputs/ for artifacts from *any* step (CSV/PNG/TEX/MD).
- Builds a polished single-file HTML with:
  • sticky sidebar, search, tabbed sections per step
  • KPIs (dataset size, date span, #categories, etc.)
  • Plotly figures if CSVs are available; falls back to images
  • theme toggle (light/dark), anchor links, keyboard shortcuts
Output:
  outputs/interactive/dashboard_step21.html
"""

from __future__ import annotations
import json, re, textwrap, base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
DATA = OUT / "data"
FIGS_DARK = OUT / "figures" / "dark"
FIGS_LIGHT = OUT / "figures" / "light"
NARR = OUT / "narratives"
INTERACTIVE = OUT / "interactive"
INTERACTIVE.mkdir(parents=True, exist_ok=True)

ML_CORPUS = DATA / "ml_corpus.parquet"

# ---------- helpers ----------
def _try_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _b64img(path: Path) -> Optional[str]:
    try:
        b = path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(b).decode("ascii")
    except Exception:
        return None

def _scan_artifacts() -> Dict[str, Dict[str, List[Path]]]:
    """Return dict[step]['csv'|'png'|'md'|'tex'] -> [paths...]"""
    steps: Dict[str, Dict[str, List[Path]]] = {}
    # Gather across outputs/{data,figures, narratives} + top-level outputs
    pools = [
        (DATA, "csv", "*.csv"),
        (OUT / "ablation", "csv", "*.csv"),  
        (OUT / "ablation", "png", "*.png"),
        (FIGS_DARK, "png", "*.png"),
        (FIGS_LIGHT, "png", "*.png"),
        (NARR, "md", "**/*.md"),
        
        (ROOT / "dissertation" / "auto_tables", "tex", "*.tex"),
    ]
    # Also consider standalone images in outputs/figures root (if any)
    figs_root = OUT / "figures"
    if figs_root.exists():
        pools.append((figs_root, "png", "*.png"))

    step_re = re.compile(r"(^|[^\d])(?P<num>\d{2})[_\-]")

    for base, kind, glob_pat in pools:
        if not base.exists():
            continue
        for p in base.rglob(glob_pat):
            # infer step from filename
            m = step_re.search(p.name)
            step = f"Step {m.group('num')}" if m else "Misc"
            steps.setdefault(step, {}).setdefault(kind, []).append(p)

    # Ensure deterministic ordering
    steps = dict(sorted(steps.items(), key=lambda kv: kv[0]))
    for k in steps:
        for kind in steps[k]:
            steps[k][kind] = sorted(steps[k][kind])
    return steps

def _kpi_from_corpus() -> Dict[str, str]:
    out = {}
    if not ML_CORPUS.exists():
        return {
            "Rows": "—",
            "Date span": "—",
            "Distinct categories": "—",
            "Distinct tags": "—",
        }
    df = pd.read_parquet(ML_CORPUS)
    out["Rows"] = f"{len(df):,}"
    # publish date span
    dt_col = next((c for c in ["publish_date","upload_date","published_at","date"] if c in df.columns), None)
    if dt_col:
        dt = pd.to_datetime(df[dt_col], errors="coerce", utc=True).dt.tz_convert(None)
        lo, hi = dt.min(), dt.max()
        if pd.notna(lo) and pd.notna(hi):
            out["Date span"] = f"{lo.date()} → {hi.date()}"
    # category-ish stats
    cat_col = next((c for c in ["categories","tags","category","tag_list","labels"] if c in df.columns), None)
    if cat_col:
        def parse_listish(x):
            if isinstance(x, (list, tuple, set)): return list(x)
            s = str(x)
            if s.startswith("[") and s.endswith("]"):
                try: from ast import literal_eval; return list(literal_eval(s))
                except Exception: return [s]
            sep = "," if "," in s else ("|" if "|" in s else None)
            return [p.strip().lower() for p in s.split(sep)] if sep else [s.strip().lower()]
        allv = [t for row in df[cat_col].dropna().head(200_000).map(parse_listish) for t in row if t]
        if allv:
            out["Distinct categories"] = f"{len(set(allv)):,}"
            out["Top category"] = pd.Series(allv).value_counts().head(1).index[0][:32]
    # group cols
    gcols = [c for c in df.columns if c.startswith("race_ethnicity_") or c.startswith("gender_")]
    if gcols:
        out["One-hot group cols"] = str(len(gcols))
    return out

def _fig_from_csv(name: str, df: pd.DataFrame) -> Optional[go.Figure]:
    """Minimal heuristics to render common analysis tables with Plotly."""
    # adv19_bw_effect_sizes
    if {"metric","cliffs_delta","ci_lo","ci_hi"}.issubset(df.columns):
        y = df["cliffs_delta"].astype(float)
        yerr = np.vstack([y - df["ci_lo"].astype(float), df["ci_hi"].astype(float) - y])
        fig = go.Figure(go.Bar(x=df["metric"], y=y, error_y=dict(array=yerr[1], arrayminus=yerr[0])))
        fig.update_layout(title="Advanced Stats — Cliff’s δ (BW − Others)", height=360, margin=dict(l=40,r=20,t=50,b=40))
        return fig
    # adv19_trend_slopes
    if {"metric","slope_bw","slope_others","slope_bw_lo","slope_bw_hi","slope_others_lo","slope_others_hi"}.issubset(df.columns):
        xs = df["metric"]
        fig = go.Figure()
        fig.add_bar(name="Black women", x=xs, y=df["slope_bw"],
                    error_y=dict(array=(df["slope_bw_hi"]-df["slope_bw"]), arrayminus=(df["slope_bw"]-df["slope_bw_lo"])))
        fig.add_bar(name="Others", x=xs, y=df["slope_others"],
                    error_y=dict(array=(df["slope_others_hi"]-df["slope_others"]), arrayminus=(df["slope_others"]-df["slope_others_lo"])))
        fig.update_layout(barmode="group", title="Temporal Slopes (95% CI)", height=380, margin=dict(l=40,r=20,t=50,b=40))
        return fig
    # adv19_harm_relative_risks
    if {"harm_category","RR_bw_vs_others","RR_lo","RR_hi"}.issubset(df.columns):
        dd = df.sort_values("RR_bw_vs_others", ascending=True).tail(25)
        fig = px.bar(dd, x="RR_bw_vs_others", y="harm_category", orientation="h",
                     title="Harm Relative Risk — BW vs Others (Top 25 by RR)")
        fig.update_layout(height=max(360, 20*len(dd)), margin=dict(l=160,r=30,t=50,b=40))
        return fig
    # net20_category_centrality
    if {"category","strength"}.issubset(df.columns):
        dd = df.sort_values("strength", ascending=False).head(20)
        fig = px.bar(dd, x="strength", y="category", orientation="h", title="Network — Top Category Strength")
        fig.update_layout(height=480, margin=dict(l=200,r=40,t=50,b=40))
        return fig
    # fallback: show first numeric columns against index
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] >= 1:
        fig = px.line(numeric.iloc[:200])
        fig.update_layout(title=name, height=300, margin=dict(l=40,r=20,t=40,b=40))
    # 22_topk_mass_K.csv
    if {"K","mass_captured"}.issubset(df.columns):
        fig = go.Figure(go.Scatter(x=df["K"], y=df["mass_captured"], mode="lines+markers", name="Mass captured"))
        fig.update_layout(title="Ablation: Top-K coverage (mass captured)", yaxis_title="Fraction", xaxis_title="K",
                          height=320, margin=dict(l=40,r=20,t=50,b=40))
        return fig

    # 22_rr_lexicon_off.csv
    if {"harm_category","RR_bw_vs_others"}.issubset(df.columns) and "lexicon_off" in name:
        dd = df.sort_values("RR_bw_vs_others", ascending=True).tail(25)
        fig = go.Figure(go.Bar(x=dd["RR_bw_vs_others"], y=dd["harm_category"], orientation="h", name="RR off"))
        fig.update_layout(title="Ablation: Harm RR with lexicon OFF (Top 25)", height=max(360, 20*len(dd)),
                          margin=dict(l=160,r=30,t=50,b=40))
        return fig

    # 22_topcats_noise_p.csv
    if {"category","count"}.issubset(df.columns) and "topcats_noise" in name:
        fig = go.Figure(go.Bar(x=df["count"], y=df["category"], orientation="h"))
        fig.update_layout(title=f"Ablation: Top categories under noise ({name})", height=max(360, 18*len(df.head(25))),
                          margin=dict(l=180,r=30,t=50,b=40))
        return fig

    # Fallback generic
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] >= 1:
        fig = px.line(numeric.iloc[:200], title=name)
        fig.update_layout(height=300, margin=dict(l=40,r=20,t=40,b=40))
        return fig
    return None

def _html_template(body: str, sidebar: str, title: str, kpi_html: str) -> str:
    # Escape braces in CSS/HTML by doubling them for f-string safety
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
:root {{
  --bg: #0b0f14; --panel:#0f1520; --muted:#64748b; --text:#e2e8f0; --brand:#7c3aed; --card:#111827; --accent:#22c55e; --link:#60a5fa;
  --border: #1f2937; --chip-bg:#0b1220; --shadow: 0 10px 30px rgba(0,0,0,.25);
}}
html.light {{
  --bg:#ffffff; --panel:#f8fafc; --muted:#475569; --text:#0f172a; --brand:#6d28d9; --card:#ffffff; --accent:#059669; --link:#2563eb;
  --border:#e5e7eb; --chip-bg:#f1f5f9; --shadow: 0 8px 24px rgba(2, 6, 23, .08);
}}
* {{ box-sizing: border-box; }}
body {{
  margin:0; background: var(--bg); color:var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji;
}}
.app {{
  display:grid; grid-template-columns: 280px 1fr; min-height:100vh;
}}
.sidebar {{
  position: sticky; top:0; height:100vh; overflow:auto;
  padding:20px 18px; background:var(--panel); border-right:1px solid var(--border);
}}
.brand {{ display:flex; align-items:center; gap:10px; margin-bottom:14px; }}
# ... same CSS rules with doubled braces throughout ...
.grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:12px; }}
.footer {{ color:var(--muted); font-size:12px; padding:20px; text-align:center }}
.badge {{ font-size:11px; padding:2px 8px; background:rgba(34,197,94,.12); border:1px solid rgba(34,197,94,.4); border-radius:999px }}
hr.sep {{ border:0; height:1px; background:var(--border); margin:10px 0; }}
code.file {{ color:var(--muted); font-size:12px }}
</style>
</head>
<body class="dark">
<div class="app">
  <aside class="sidebar">
    <div class="brand"><span class="dot"></span><h1>AlgoFairness — Dashboard</h1></div>
    <div class="kpis">{kpi_html}</div>
# ... unchanged HTML with {{…}}-escaped braces until ...
</body>
</html>
"""


def _render_step_section(step: str, files: Dict[str, List[Path]]) -> Tuple[str, str]:
    """Return (sidebar_item_html, section_html)"""
    anchor = re.sub(r"\\s+", "-", step.lower())
    # Gather keywords for client-side filtering
    kws = [step.lower()] + [p.stem.lower() for vs in files.values() for p in vs]
    kw_attr = " ".join(sorted(set(kws)))
    # Build content blocks
    blocks = []

    # CSVs -> figures
    if "csv" in files:
        for p in files["csv"]:
            df = _try_read_csv(p)
            if df is None or df.empty:
                continue
            fig = _fig_from_csv(p.stem, df)
            if fig:
                fig.update_layout(template="plotly_white")
                div = fig.to_html(full_html=False, include_plotlyjs=False, config=dict(responsive=True, displaylogo=False))
                blocks.append(f'<div class="plot">{div}<div><code class="file">{p.relative_to(ROOT)}</code></div></div>')

    # Images
    if "png" in files:
        # Prefer dark figure if same stem exists
        seen = set()
        for p in files["png"]:
            stem = p.stem
            if stem in seen: continue
            prefer_dark = (FIGS_DARK / f"{stem}.png")
            use = prefer_dark if prefer_dark.exists() else p
            seen.add(stem)
            img = _b64img(use)
            if img:
                blocks.append(f'<div><img style="width:100%; border-radius:8px" src="{img}" /><div><code class="file">{use.relative_to(ROOT)}</code></div></div>')

    # MD/TEX links
    links = []
    for kind in ("md","tex"):
        for p in files.get(kind, []):
            links.append(f'<a class="chip" href="file://{p}">{p.name}</a>')
    if links:
        blocks.append("<div>" + " ".join(links) + "</div>")

    if not blocks:
        blocks = ['<div class="muted">No visual artifacts; see linked files above.</div>']

    body = f"""
    <section class="section" id="{anchor}" data-keywords="{kw_attr}">
      <div class="head"><h2>{step}</h2><span class="badge">{sum(len(v) for v in files.values())} artifacts</span></div>
      <div class="body">
        <div class="grid">
          {''.join(blocks)}
        </div>
      </div>
    </section>
    """

    side = f'<li><a href="#{anchor}">{step}</a></li>'
    return side, body

def build_dashboard() -> Path:
    steps = _scan_artifacts()
    # Always raise Step Overview to top
    steps = {"Overview": {}} | steps

    # KPI cards from corpus
    kpis = _kpi_from_corpus()
    kpi_html = "".join([f'<div class="kpi"><div class="label">{k}</div><div class="value">{v}</div></div>' for k,v in kpis.items()])

    # Overview section (small legend + quick tips)
    overview_body = """
    <section class="section" id="overview" data-keywords="overview">
      <div class="head"><h2>Overview</h2><span class="badge">UI</span></div>
      <div class="body">
        <div class="grid">
          <div>
            <h3>How to use</h3>
            <ul>
              <li><b>Sidebar</b> lists every detected step. Click to jump.</li>
              <li><b>Search</b> filters sections client-side. Press <code>Ctrl/⌘ + K</code> to focus.</li>
              <li><b>Theme</b> toggles light/dark.</li>
            </ul>
          </div>
          <div>
            <h3>Coverage</h3>
            <p>This dashboard auto-discovers outputs under <code>outputs/</code> &amp; <code>dissertation/auto_tables/</code>. If a step adds a CSV image or TEX, it appears automatically.</p>
          </div>
        </div>
      </div>
    </section>
    """

    # Sidebar + Sections
    sidebar_items = ['<li><a href="#overview">Overview</a></li>']
    sections = [overview_body]

    for step, files in steps.items():
        if step == "Overview":  # already added
            continue
        side, body = _render_step_section(step, files)
        sidebar_items.append(side)
        sections.append(body)

    html = _html_template(body="".join(sections), sidebar="".join(sidebar_items),
                          title="AlgoFairness — Dashboard", kpi_html=kpi_html)
    out = INTERACTIVE / "dashboard_step21.html"
    out.write_text(html, encoding="utf-8")
    print(f"[WRITE] {out}")
    return out

def main():
    build_dashboard()
    print("--- Step 21: Interactive Dashboard Completed Successfully ---")

if __name__ == "__main__":
    main()
