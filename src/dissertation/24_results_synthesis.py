#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
24_results_synthesis.py

Purpose:
  Robust, schema-tolerant synthesis of key findings across steps 16–23 with a
  Black-women–centric lens. No hard-coded column names; gracefully degrades if
  some artefacts are missing.

Inputs (auto-detected if present):
  outputs/data/eng17_yearly_bw_gaps.csv
  outputs/data/adv19_bw_effect_sizes.csv
  outputs/data/adv19_trend_slopes.csv
  outputs/data/adv19_category_divergence.csv
  outputs/data/cgd_black_women_under_over.csv
  outputs/data/lim23_summary_checklist.csv

Outputs:
  outputs/data/res24_keyfindings.csv
  dissertation/auto_tables/res24_summary.tex
  outputs/narratives/automated/24_results_synthesis.md

Prints:
  timing + success banner
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd

# ---------------------- Paths ----------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs" / "data"
DATA.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / "dissertation" / "auto_tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
NARR_DIR = ROOT / "outputs" / "narratives" / "automated"
NARR_DIR.mkdir(parents=True, exist_ok=True)

KEY_CSV = DATA / "res24_keyfindings.csv"
KEY_TEX = TABLE_DIR / "res24_summary.tex"
KEY_MD  = NARR_DIR / "24_results_synthesis.md"

# ---------------------- Helpers ----------------------
def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _fmt_ci(lo: float, hi: float, k: int = 3) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"[{lo:.{k}f}, {hi:.{k}f}]"

def _safe_latest(df: pd.DataFrame, year_candidates: List[str]) -> pd.DataFrame:
    """Return last row sorted by the first present year-like column; empty DF if none."""
    ycol = _first_col(df, year_candidates)
    if not ycol:
        return pd.DataFrame()
    tmp = df.copy()
    tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
    tmp = tmp.dropna(subset=[ycol])
    if tmp.empty:
        return pd.DataFrame()
    return tmp.sort_values(ycol, kind="stable").tail(1)

# ---------------------- Main ----------------------
def main() -> None:
    t0 = time.time()
    print("--- Starting Step 24: Results Synthesis ---")

    rows = []  # list of dicts -> final KPI CSV

    # ---------- Step 17: Engagement gaps (robust column detection) ----------
    gaps = _read_csv(DATA / "eng17_yearly_bw_gaps.csv")
    if not gaps.empty:
        year_col     = _first_col(gaps, ["year", "publish_year", "_publish_year", "yr", "year_int"])
        gap_vpd_col  = _first_col(gaps, ["gap_views_per_day", "views_per_day_gap", "gap_vpd"])
        lo_vpd_col   = _first_col(gaps, ["ci_lo_views_per_day", "vpd_ci_lo", "ci_lo_vpd"])
        hi_vpd_col   = _first_col(gaps, ["ci_hi_views_per_day", "vpd_ci_hi", "ci_hi_vpd"])
        gap_rat_col  = _first_col(gaps, ["gap_rating", "rating_gap"])
        lo_rat_col   = _first_col(gaps, ["ci_lo_rating", "rating_ci_lo"])
        hi_rat_col   = _first_col(gaps, ["ci_hi_rating", "rating_ci_hi"])

        latest = _safe_latest(gaps, ["year", "publish_year", "_publish_year", "yr", "year_int"])

        # Views/day gap KPI
        if not latest.empty and gap_vpd_col and lo_vpd_col and hi_vpd_col:
            v = float(latest[gap_vpd_col].iloc[0])
            lo = float(latest[lo_vpd_col].iloc[0])
            hi = float(latest[hi_vpd_col].iloc[0])
            rows.append({
                "section": "Engagement",
                "metric": "BW − Others views/day (latest year)",
                "value": v, "ci": _fmt_ci(lo, hi),
                "note": ""
            })
        else:
            print("[INFO] Step 24: gaps present but missing VPD columns; skipping that KPI.")

        # Rating gap KPI (optional)
        if not latest.empty and gap_rat_col and lo_rat_col and hi_rat_col:
            v = float(latest[gap_rat_col].iloc[0])
            lo = float(latest[lo_rat_col].iloc[0])
            hi = float(latest[hi_rat_col].iloc[0])
            rows.append({
                "section": "Engagement",
                "metric": "BW − Others rating (latest year)",
                "value": v, "ci": _fmt_ci(lo, hi),
                "note": ""
            })

    # ---------- Step 19: Effect sizes (Cliff's δ) ----------
    eff = _read_csv(DATA / "adv19_bw_effect_sizes.csv")
    if not eff.empty:
        # Prefer rating; otherwise first available
        target_row = None
        if "metric" in eff.columns:
            have_rating = eff[eff["metric"].astype(str).str.lower() == "rating"]
            target_row = have_rating.iloc[0] if not have_rating.empty else eff.iloc[0]
        else:
            target_row = eff.iloc[0]

        met = str(target_row.get("metric", "metric"))
        d   = float(target_row.get("cliffs_delta", np.nan))
        lo  = float(target_row.get("ci_lo", np.nan))
        hi  = float(target_row.get("ci_hi", np.nan))
        rows.append({
            "section": "Advanced Stats",
            "metric": f"Cliff’s δ (BW − Others) — {met}",
            "value": d, "ci": _fmt_ci(lo, hi),
            "note": "Bootstrap 95% CI"
        })

    # ---------- Step 19: Temporal slopes ----------
    sl = _read_csv(DATA / "adv19_trend_slopes.csv")
    if not sl.empty and {"metric","slope_bw","slope_others"}.issubset(sl.columns):
        # compute slope gap for rating if present; else first row
        target = None
        if "metric" in sl.columns:
            rat = sl[sl["metric"].astype(str).str.lower() == "rating"]
            target = rat.iloc[0] if not rat.empty else sl.iloc[0]
        else:
            target = sl.iloc[0]
        mname = str(target.get("metric", "metric"))
        bw = float(target.get("slope_bw", np.nan))
        ot = float(target.get("slope_others", np.nan))
        gap = bw - ot
        lo_g = float(target.get("slope_gap_lo", np.nan))
        hi_g = float(target.get("slope_gap_hi", np.nan))
        rows.append({
            "section": "Advanced Stats",
            "metric": f"Temporal slope gap (BW − Others) — {mname}",
            "value": gap, "ci": _fmt_ci(lo_g, hi_g),
            "note": "Yearly change; bootstrap CI for gap if available"
        })

    # ---------- Step 19: Category divergence ----------
    kl = _read_csv(DATA / "adv19_category_divergence.csv")
    if not kl.empty and {"group","kl_vs_global"}.issubset(kl.columns):
        top3 = kl.sort_values("kl_vs_global", ascending=False).head(3)
        for _, r in top3.iterrows():
            rows.append({
                "section": "Category Divergence",
                "metric": f"KL vs global — {str(r['group'])}",
                "value": float(r["kl_vs_global"]),
                "ci": "",
                "note": "Top divergence"
            })

    # ---------- Step 18: Categories over/under for BW ----------
    bw = _read_csv(DATA / "cgd_black_women_under_over.csv")
    if not bw.empty and {"category","log2_rr_bw"}.issubset(bw.columns):
        # 3 strongest under + 3 strongest over
        under = bw.nsmallest(3, "log2_rr_bw")
        over  = bw.nlargest(3, "log2_rr_bw")
        if not under.empty:
            catlist = ", ".join([str(x) for x in under["category"].tolist()])
            rows.append({
                "section": "Category Representation",
                "metric": "Most under-represented (BW) — log2 RR",
                "value": float(under["log2_rr_bw"].min()),
                "ci": "",
                "note": catlist
            })
        if not over.empty:
            catlist = ", ".join([str(x) for x in over["category"].tolist()])
            rows.append({
                "section": "Category Representation",
                "metric": "Most over-represented (BW) — log2 RR",
                "value": float(over["log2_rr_bw"].max()),
                "ci": "",
                "note": catlist
            })

    # ---------- Step 23: Limitations checklist (optional) ----------
    lim = _read_csv(DATA / "lim23_summary_checklist.csv")
    if not lim.empty and {"dimension","status"}.issubset(lim.columns):
        # simple count of critical "attention" statuses
        att = lim[lim["status"].astype(str).str.lower().isin({"attention","critical","warning"})]
        rows.append({
            "section": "Limitations",
            "metric": "Data/Method caveats flagged",
            "value": float(len(att)),
            "ci": "",
            "note": "From Step 23 checklist"
        })

    # ---------- Write artefacts ----------
    out = pd.DataFrame(rows, columns=["section","metric","value","ci","note"])
    out.to_csv(KEY_CSV, index=False)
    print(f"[WRITE] {KEY_CSV}")

    # LaTeX
    try:
        KEY_TEX.write_text(out.to_latex(index=False, float_format="%.4f"))
        print(f"[TEX]   {KEY_TEX}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # Narrative
    lines = ["# Step 24 — Results Synthesis (Robust)","",
             "This summary consolidates the latest engagement gaps, effect sizes, temporal slopes,",
             "category divergence, and representation extremes for Black women vs others.",
             "",
             out.to_markdown(index=False)]
    KEY_MD.write_text("\n".join(lines))
    print(f"[WRITE] {KEY_MD}")

    dt = time.time() - t0
    print(f"[TIME] Step 24 runtime: {dt:.2f}s")
    print("--- Step 24: Results Synthesis Completed Successfully ---")

if __name__ == "__main__":
    main()
