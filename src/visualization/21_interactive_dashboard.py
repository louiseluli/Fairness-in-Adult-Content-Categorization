#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
21_interactive_dashboard.py

Purpose:
  Build a single, self-contained HTML dashboard (Plotly) aggregating key
  analyses from Steps 16–20 & 22 (ablations) with a Black-women–centric focus.

Inputs (optional; robust to missing):
  outputs/data/temporal_black_women_share.csv
  outputs/data/temporal_group_representation.csv
  outputs/data/eng17_yearly_bw_gaps.csv
  outputs/data/eng17_quantiles_race_ethnicity.csv
  outputs/data/cgd_black_women_under_over.csv
  outputs/data/cgd_category_group_matrix.csv
  outputs/data/net20_category_centrality.csv
  outputs/data/adv19_bw_effect_sizes.csv
  outputs/data/adv19_trend_slopes.csv
  outputs/data/adv19_category_divergence.csv
  outputs/data/22_rr_lexicon_off.csv
  outputs/data/22_topcats_noise_10.csv
  outputs/data/22_topcats_noise_25.csv
  outputs/data/22_topcats_noise_50.csv
  outputs/data/22_topk_mass_10.csv
  outputs/data/22_topk_mass_20.csv
  outputs/data/22_topk_mass_30.csv
  outputs/data/22_topk_mass_50.csv

Output:
  outputs/interactive/dashboard_step21.html

Prints:
  timing + success banner
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.offline import plot as _offline  # not used; keep for completeness

# ---------------------- Config ----------------------
SEED = 75
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs" / "data"
INTERACTIVE = ROOT / "outputs" / "interactive"
INTERACTIVE.mkdir(parents=True, exist_ok=True)
DASHBOARD_HTML = INTERACTIVE / "dashboard_step21.html"

def _read_csv(p: Path, **kwargs) -> pd.DataFrame | None:
    try:
        if p.exists():
            return pd.read_csv(p, **kwargs)
        return None
    except Exception:
        return None

# ---------------------- Builders ----------------------
def section_header(text: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(text=f"<b>{text}</b>", x=0, y=1, xref="paper", yref="paper",
                          xanchor="left", yanchor="top", showarrow=False, font=dict(size=22))],
        height=80, margin=dict(l=20, r=20, t=20, b=10)
    )
    return fig

def build_temporal_representation() -> list[go.Figure]:
    figs = []
    bw_share = _read_csv(DATA / "temporal_black_women_share.csv")
    if bw_share is not None and {"year","share_bw"}.issubset(bw_share.columns):
        figs.append(section_header("Temporal Representation — Black Women"))
        tr = go.Figure()
        tr.add_trace(go.Scatter(x=bw_share["year"], y=bw_share["share_bw"],
                                mode="lines+markers", name="Black women share"))
        tr.update_layout(yaxis_title="Share of videos (%)", xaxis_title="Year",
                         height=350, margin=dict(l=40, r=20, t=40, b=40))
        figs.append(tr)

    rep = _read_csv(DATA / "temporal_group_representation.csv")
    if rep is not None and {"year","group","share_pct"}.issubset(rep.columns):
        figs.append(section_header("Temporal Representation — All Groups (Top 6)"))
        top_groups = (rep.groupby("group")["share_pct"].mean()
                        .sort_values(ascending=False).head(6).index.tolist())
        rep6 = rep[rep["group"].isin(top_groups)]
        tr_all = go.Figure()
        for g in top_groups:
            sub = rep6[rep6["group"] == g]
            tr_all.add_trace(go.Scatter(x=sub["year"], y=sub["share_pct"],
                                        mode="lines+markers", name=str(g)))
        tr_all.update_layout(yaxis_title="Share (%)", xaxis_title="Year",
                             height=380, legend=dict(orientation="h", y=1.15),
                             margin=dict(l=40, r=20, t=40, b=40))
        figs.append(tr_all)
    return figs

def build_engagement_gaps() -> list[go.Figure]:
    figs = []
    gaps = _read_csv(DATA / "eng17_yearly_bw_gaps.csv")
    if gaps is not None and {"year","gap_views_per_day","gap_views_ci_lo","gap_views_ci_hi",
                             "gap_rating","gap_rating_ci_lo","gap_rating_ci_hi"}.issubset(gaps.columns):
        figs.append(section_header("Engagement Gaps — Black Women vs Others"))
        # Views/day
        g1 = go.Figure()
        g1.add_trace(go.Scatter(x=gaps["year"], y=gaps["gap_views_per_day"],
                                mode="lines+markers", name="Δ views/day"))
        g1.add_trace(go.Scatter(
            x=pd.concat([gaps["year"], gaps["year"][::-1]]),
            y=pd.concat([gaps["gap_views_ci_hi"], gaps["gap_views_ci_lo"][::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.2)", line=dict(width=0),
            name="95% CI", showlegend=True
        ))
        g1.add_hline(y=0, line_dash="dash", line_width=1)
        g1.update_layout(yaxis_title="Δ views/day", xaxis_title="Year",
                         height=330, margin=dict(l=40, r=20, t=40, b=40))
        figs.append(g1)
        # Rating
        g2 = go.Figure()
        g2.add_trace(go.Scatter(x=gaps["year"], y=gaps["gap_rating"],
                                mode="lines+markers", name="Δ rating"))
        g2.add_trace(go.Scatter(
            x=pd.concat([gaps["year"], gaps["year"][::-1]]),
            y=pd.concat([gaps["gap_rating_ci_hi"], gaps["gap_rating_ci_lo"][::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.2)", line=dict(width=0),
            name="95% CI", showlegend=True
        ))
        g2.add_hline(y=0, line_dash="dash", line_width=1)
        g2.update_layout(yaxis_title="Δ rating", xaxis_title="Year",
                         height=330, margin=dict(l=40, r=20, t=40, b=40))
        figs.append(g2)

    q = _read_csv(DATA / "eng17_quantiles_race_ethnicity.csv")
    if q is not None and {"group","q10_views_per_day","q50_views_per_day","q90_views_per_day"}.issubset(q.columns):
        figs.append(section_header("Engagement Quantiles — by Race/Ethnicity"))
        qf = go.Figure()
        topg = (q.groupby("group")["q50_views_per_day"].mean()
                  .sort_values(ascending=False).head(6).index.tolist())
        q6 = q[q["group"].isin(topg)]
        for g in topg:
            sub = q6[q6["group"] == g]
            qf.add_trace(go.Bar(
                x=["q10","q50","q90"],
                y=[sub["q10_views_per_day"].mean(), sub["q50_views_per_day"].mean(), sub["q90_views_per_day"].mean()],
                name=str(g)
            ))
        qf.update_layout(barmode="group", yaxis_title="Views/day", height=380,
                         legend=dict(orientation="h", y=1.12),
                         margin=dict(l=40, r=20, t=40, b=40))
        figs.append(qf)
    return figs

def build_category_group_dynamics() -> list[go.Figure]:
    figs = []
    bw = _read_csv(DATA / "cgd_black_women_under_over.csv")
    if bw is not None and {"category","log2_rr_bw"}.issubset(bw.columns):
        figs.append(section_header("Category Over/Under-Representation — Black Women"))
        top_under = bw.nsmallest(15, "log2_rr_bw")
        top_over  = bw.nlargest(15, "log2_rr_bw")

        fig_under = go.Figure()
        fig_under.add_trace(go.Bar(x=top_under["log2_rr_bw"], y=top_under["category"],
                                   orientation="h", name="Under-represented (log2 RR)"))
        fig_under.update_layout(height=420, margin=dict(l=160, r=30, t=40, b=40),
                                xaxis_title="log2(BW RR)", yaxis_title=None)
        figs.append(fig_under)

        fig_over = go.Figure()
        fig_over.add_trace(go.Bar(x=top_over["log2_rr_bw"], y=top_over["category"],
                                  orientation="h", name="Over-represented (log2 RR)"))
        fig_over.update_layout(height=420, margin=dict(l=160, r=30, t=40, b=40),
                               xaxis_title="log2(BW RR)", yaxis_title=None)
        figs.append(fig_over)

    mat = _read_csv(DATA / "cgd_category_group_matrix.csv", index_col=0)
    if mat is not None:
        try:
            m30 = mat.head(30) if len(mat) > 30 else mat
            figs.append(section_header("Category × Group Matrix (Top 30 categories)"))
            hm = go.Figure(data=go.Heatmap(z=m30.values, x=list(m30.columns), y=list(m30.index),
                                           coloraxis="coloraxis"))
            hm.update_layout(coloraxis=dict(colorscale="RdBu"),
                             height=700, margin=dict(l=160, r=40, t=40, b=60))
            figs.append(hm)
        except Exception:
            pass
    return figs

def build_network_and_corr() -> list[go.Figure]:
    figs = []
    net = _read_csv(DATA / "net20_category_centrality.csv")
    if net is not None and {"category","strength"}.issubset(net.columns):
        figs.append(section_header("Category Co-occurrence Network Metrics"))
        top = net.nlargest(20, "strength")
        bars = go.Figure()
        bars.add_trace(go.Bar(x=top["strength"], y=top["category"], orientation="h", name="Strength"))
        bars.update_layout(height=480, margin=dict(l=200, r=40, t=40, b=40),
                           xaxis_title="Strength", yaxis_title=None)
        figs.append(bars)

        if "log2_rr_bw" in net.columns and net["log2_rr_bw"].notna().any():
            sc = go.Figure()
            sc.add_trace(go.Scatter(x=net["strength"], y=net["log2_rr_bw"], mode="markers",
                                    text=net["category"], name="Category"))
            sc.add_hline(y=0, line_dash="dash", line_width=1)
            sc.update_layout(xaxis_title="Strength", yaxis_title="log2(BW RR)",
                             height=420, margin=dict(l=60, r=30, t=40, b=40))
            figs.append(sc)
    return figs

def build_advanced_stats() -> list[go.Figure]:
    figs = []
    eff = _read_csv(DATA / "adv19_bw_effect_sizes.csv")
    if eff is not None and {"metric","cliffs_delta","ci_lo","ci_hi"}.issubset(eff.columns):
        figs.append(section_header("Advanced Stats — Cliff’s δ (BW − Others)"))
        x = eff["metric"]
        y = eff["cliffs_delta"].astype(float)
        lo = eff["ci_lo"].astype(float); hi = eff["ci_hi"].astype(float)
        err_lo = y - lo; err_hi = hi - y
        ef = go.Figure()
        ef.add_trace(go.Bar(x=x, y=y, name="Cliff’s δ",
                            error_y=dict(type="data", array=err_hi, arrayminus=err_lo, visible=True)))
        ef.add_hline(y=0, line_dash="dash", line_width=1)
        ef.update_layout(yaxis_title="Cliff’s δ", height=360, margin=dict(l=60, r=30, t=40, b=40))
        figs.append(ef)

    sl = _read_csv(DATA / "adv19_trend_slopes.csv")
    needed = {"metric","slope_bw","slope_others","slope_bw_lo","slope_bw_hi","slope_others_lo","slope_others_hi"}
    if sl is not None and needed.issubset(sl.columns):
        figs.append(section_header("Advanced Stats — Temporal Slopes (BW vs Others)"))
        slf = go.Figure()
        slf.add_trace(go.Bar(x=sl["metric"], y=sl["slope_bw"], name="Black women",
                             error_y=dict(type="data",
                                          array=(sl["slope_bw_hi"]-sl["slope_bw"]),
                                          arrayminus=(sl["slope_bw"]-sl["slope_bw_lo"]))))
        slf.add_trace(go.Bar(x=sl["metric"], y=sl["slope_others"], name="Others",
                             error_y=dict(type="data",
                                          array=(sl["slope_others_hi"]-sl["slope_others"]),
                                          arrayminus=(sl["slope_others"]-sl["slope_others_lo"]))))
        slf.add_hline(y=0, line_dash="dash", line_width=1)
        slf.update_layout(barmode="group", yaxis_title="Yearly slope",
                          height=380, margin=dict(l=60, r=30, t=40, b=40))
        figs.append(slf)

    kl = _read_csv(DATA / "adv19_category_divergence.csv")
    if kl is not None and {"group","kl_vs_global"}.issubset(kl.columns):
        figs.append(section_header("Advanced Stats — Category KL Divergence vs Global"))
        kf = go.Figure()
        kf.add_trace(go.Bar(x=kl["group"], y=kl["kl_vs_global"]))
        kf.update_layout(yaxis_title="KL divergence", height=360, margin=dict(l=60, r=30, t=40, b=40))
        figs.append(kf)
    return figs

def build_ablation_panel() -> list[go.Figure]:
    """Aggregates Step 22 ablation CSVs into KPIs + a bar chart."""
    figs = []
    # Collect scenarios if present
    items = [
        ("Lexicon OFF", DATA / "22_rr_lexicon_off.csv"),
        ("Noise 10%",  DATA / "22_topcats_noise_10.csv"),
        ("Noise 25%",  DATA / "22_topcats_noise_25.csv"),
        ("Noise 50%",  DATA / "22_topcats_noise_50.csv"),
        ("TopK 10",    DATA / "22_topk_mass_10.csv"),
        ("TopK 20",    DATA / "22_topk_mass_20.csv"),
        ("TopK 30",    DATA / "22_topk_mass_30.csv"),
        ("TopK 50",    DATA / "22_topk_mass_50.csv"),
    ]
    rows = []
    for name, path in items:
        df = _read_csv(path)
        if df is None or df.empty:
            continue
        # Expect a single-row delta column; otherwise try any numeric
        if "delta" in df.columns:
            val = float(df["delta"].iloc[0])
        else:
            # take first numeric column mean as fallback
            nums = df.select_dtypes(include=[np.number])
            if nums.empty:
                continue
            val = float(nums.mean(axis=1).iloc[0])
        rows.append({"scenario": name, "delta": val})

    if not rows:
        return figs

    import plotly.graph_objs as go
    import pandas as pd
    data = pd.DataFrame(rows)

    figs.append(section_header("Ablations — Sensitivity of Key Metric (Step 22)"))
    # KPI cards (as bars with labels)
    figk = go.Figure()
    figk.add_trace(go.Bar(x=data["scenario"], y=data["delta"]))
    figk.update_layout(yaxis_title="Δ (scenario vs baseline)", height=360,
                       margin=dict(l=60, r=30, t=40, b=120))
    figs.append(figk)
    return figs

# ---------------------- Main ----------------------
def main() -> None:
    t0 = time.time()
    print("--- Starting Step 21: Interactive Dashboard ---")

    figs: list[go.Figure] = []
    intro = go.Figure()
    intro.update_layout(
        annotations=[dict(
            text="<b>Fairness Dashboard</b><br>Focus: Representation & Engagement bias with a Black-women lens.<br>"
                 "Use the index in your browser to jump between sections.",
            x=0, y=1, xref="paper", yref="paper", xanchor="left", yanchor="top",
            showarrow=False, font=dict(size=18)
        )],
        height=120, margin=dict(l=20, r=20, t=20, b=20)
    )
    figs.append(intro)

    # Sections
    figs += build_temporal_representation()
    figs += build_engagement_gaps()
    figs += build_category_group_dynamics()
    figs += build_network_and_corr()
    figs += build_advanced_stats()
    figs += build_ablation_panel()

    if not figs or len(figs) == 1:
        empty = go.Figure()
        empty.update_layout(
            annotations=[dict(text="No artefacts found yet. Please run steps 16–20 and 22.", x=0.5, y=0.5, showarrow=False)],
            height=200
        )
        figs.append(empty)

    # Join figures
    html_parts = []
    for i, f in enumerate(figs, start=1):
        f.update_layout(title_x=0.0)
        html_parts.append(f.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig{i}"))

    nav_links = "".join([f'<li><a href="#fig{i}">Section {i}</a></li>' for i in range(1, len(figs)+1)])

    # Use .format() to avoid f-string brace issues with CSS
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AlgoFairness — Dashboard (Step 21)</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0 16px; background:#0b0f17; color:#e8eef9; }}
    .nav {{ position: sticky; top: 0; background: #0d1320; border-bottom: 1px solid #1b2333; padding: 8px 0; margin-bottom: 8px; }}
    .nav ul {{ list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 12px; }}
    .nav a {{ text-decoration: none; color: #7aa2ff; padding:6px 10px; border-radius:8px; border:1px solid transparent; }}
    .nav a:hover {{ border-color:#25304a; background:#121a2a; }}
    .section {{ margin: 8px 0 28px 0; background:#121826; border:1px solid #1b2333; border-radius:12px; padding:10px; box-shadow:0 8px 24px rgba(0,0,0,.25); }}
  </style>
</head>
<body>
  <div class="nav">
    <strong>Sections:</strong>
    <ul>{nav_links}</ul>
  </div>
  {sections}
</body>
</html>
    """.format(
        nav_links=nav_links,
        sections="".join(f'<div class="section" id="fig{i}">{part}</div>' for i, part in enumerate(html_parts, start=1))
    )

    with open(DASHBOARD_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[WRITE] {DASHBOARD_HTML}")

    elapsed = time.time() - t0
    print(f"[TIME] Step 21 runtime: {elapsed:.2f}s")
    print("--- Step 21: Interactive Dashboard Completed Successfully ---")

if __name__ == "__main__":
    main()
