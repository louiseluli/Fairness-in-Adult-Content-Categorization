#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16_deep_data_analysis.py

Temporal evolution + engagement-bias diagnostics with robust fallbacks:
- If protected-attribute columns are missing, a synthetic single group "__all__" is used so the analysis still runs.
- Uses only Matplotlib (no seaborn).

Deliverables:
  CSV:
    outputs/data/temporal_group_representation.csv
    outputs/data/temporal_rating_view_trends.csv
    outputs/data/engagement_bias_by_group.csv
  TEX:
    dissertation/auto_tables/temporal_group_representation.tex
    dissertation/auto_tables/temporal_rating_view_trends.tex
    dissertation/auto_tables/engagement_bias_by_group.tex
  FIG:
    outputs/figures/temporal_representation_line_dark.png
    outputs/figures/temporal_representation_line_light.png
    outputs/figures/temporal_rating_trends_dark.png
    outputs/figures/temporal_rating_trends_light.png
    outputs/figures/temporal_views_trends_dark.png
    outputs/figures/temporal_views_trends_light.png
  Narrative:
    outputs/narratives/automated/16_temporal_engagement_summary.md
"""

from __future__ import annotations

# ---------- stdlib ----------
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- third-party ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- local (soft) ----------
THEME = None
try:
    from src.utils.theme_manager import ThemeManager
    THEME = ThemeManager()
except Exception:
    THEME = None

TABLES = None
try:
    from src.utils.academic_tables import write_latex_table
    TABLES = "v1"
except Exception:
    TABLES = None


# ----------------------- Configuration -----------------------
SEED = 75
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "outputs" / "data"
FIG_DIR_DARK = ROOT / "outputs" / "figures" / "dark"
FIG_DIR_LIGHT = ROOT / "outputs" / "figures" / "light"
for d in (DATA_DIR, FIG_DIR_DARK, FIG_DIR_LIGHT):
    d.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / "dissertation" / "auto_tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
NARR_DIR = ROOT / "outputs" / "narratives" / "automated"
NARR_DIR.mkdir(parents=True, exist_ok=True)

ML_CORPUS = DATA_DIR / "ml_corpus.parquet"


# ----------------------- Utilities -----------------------
def set_mpl_theme(dark: bool) -> None:
    if THEME is not None:
        THEME.apply(dark=dark)
        return
    plt.rcParams.update({
        "figure.facecolor": "black" if dark else "white",
        "axes.facecolor": "black" if dark else "white",
        "axes.edgecolor": "white" if dark else "black",
        "axes.labelcolor": "white" if dark else "black",
        "xtick.color": "white" if dark else "black",
        "ytick.color": "white" if dark else "black",
        "text.color": "white" if dark else "black",
        "savefig.facecolor": "black" if dark else "white",
        "savefig.edgecolor": "black" if dark else "white",
        "grid.color": "gray",
        "grid.alpha": 0.25,
    })


def _round(x: float, k: int = 3) -> float:
    try:
        return round(float(x), k)
    except Exception:
        return x


def _ensure_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return s.dt.tz_convert(None)
    except Exception:
        return s


def _load_corpus() -> pd.DataFrame:
    if ML_CORPUS.exists():
        df = pd.read_parquet(ML_CORPUS)
        print(f"[LOAD] {ML_CORPUS} (rows={len(df):,})")
        return df
    raise FileNotFoundError("Expected outputs/data/ml_corpus.parquet")


def _standardise_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {c.lower(): c for c in df.columns}

    # publish date
    publish_col = None
    for c in ("publish_date", "upload_date", "published_at", "date"):
        if c in cols:
            publish_col = cols[c]
            break

    # views
    views_col = None
    for c in ("views", "view_count", "views_count", "play_count"):
        if c in cols:
            views_col = cols[c]
            break

    # rating mean
    rating_col = None
    for c in ("rating", "rating_mean", "average_rating", "avg_rating", "score"):
        if c in cols:
            rating_col = cols[c]
            break

    # rating count
    rating_n_col = None
    for c in ("ratings", "rating_count", "num_ratings", "n_ratings", "votes"):
        if c in cols:
            rating_n_col = cols[c]
            break

    # time features
    if publish_col is not None:
        dt = _ensure_datetime(df[publish_col])
        df["_publish_dt"] = dt
        df["_publish_year"] = dt.dt.year
        today = pd.Timestamp.utcnow().tz_convert(None)
        df["_age_days"] = (today - dt).dt.days
    else:
        df["_publish_dt"] = pd.NaT
        df["_publish_year"] = np.nan
        df["_age_days"] = np.nan

    # engagement
    df["_views"] = pd.to_numeric(df[views_col], errors="coerce") if views_col else np.nan
    df["_rating"] = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else np.nan
    df["_rating_n"] = pd.to_numeric(df[rating_n_col], errors="coerce") if rating_n_col else np.nan

    return {
        "publish_date": publish_col,
        "views": views_col,
        "rating_mean": rating_col,
        "rating_count": rating_n_col,
    }


def _detect_protected_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    # Preferred names
    for c in ["race_ethnicity", "gender", "sexual_orientation"]:
        if c in df.columns:
            candidates.append(c)
    # Heuristics if none
    if not candidates:
        for c in df.columns:
            lc = c.lower()
            if ("race" in lc) or ("ethnic" in lc) or (lc == "gender") or ("sex_orient" in lc) or ("sexual_orientation" in lc):
                candidates.append(c)
        if candidates:
            print(f"[WARN] Using heuristic protected columns: {candidates}")
    # Final fallback: synthetic single group
    if not candidates:
        df["__all__"] = "__all__"
        candidates = ["__all__"]
        print("[WARN] No protected-attribute columns found. Using synthetic group '__all__' so analysis can proceed.")
    return candidates


def gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = arr[arr >= 0]
    if arr.size == 0:
        return float("nan")
    arr.sort()
    n = arr.size
    cum = np.cumsum(arr, dtype=float)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Safe O(na*nb) with subsample for very large arrays.
    MAX = 20000
    if a.size * b.size > MAX * MAX:
        rng = np.random.default_rng(SEED)
        a = rng.choice(a, size=min(a.size, MAX), replace=False)
        b = rng.choice(b, size=min(b.size, MAX), replace=False)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return float((gt - lt) / (a.size * b.size))


# ----------------------- Core Analyses -----------------------
def build_temporal_group_representation(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = (
        df.dropna(subset=["_publish_year"])
          .groupby(["_publish_year", group_col])
          .size()
          .reset_index(name="count")
    )
    total_per_year = out.groupby("_publish_year")["count"].transform("sum")
    out["share"] = out["count"] / total_per_year
    out["_publish_year"] = out["_publish_year"].astype(int)

    csv_path = DATA_DIR / "temporal_group_representation.csv"
    out.sort_values(["_publish_year", "share"], ascending=[True, False]).to_csv(csv_path, index=False)
    print(f"[WRITE] {csv_path}")

    tex_path = TABLE_DIR / "temporal_group_representation.tex"
    try:
        if TABLES == "v1":
            write_latex_table(out, tex_path)
        else:
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(out.to_latex(index=False))
        print(f"[TEX]   {tex_path}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    for dark in (True, False):
        set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(10, 5))
        for g, sub in out.groupby(group_col):
            ax.plot(sub["_publish_year"], sub["share"], label=str(g), marker="o", linewidth=2)
        ax.set_title(f"Temporal Representation by Year — {group_col}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Share")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncols=2, frameon=False)
        f = "temporal_representation_line_dark.png" if dark else "temporal_representation_line_light.png"
        fpath = (FIG_DIR_DARK if dark else FIG_DIR_LIGHT) / f
        fig.tight_layout()
        fig.savefig(fpath, dpi=200)
        plt.close(fig)
        print(f"[PLOT] {fpath}")

    return out


def build_temporal_rating_view_trends(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    df = df.copy()
    df["_age_days_clamped"] = df["_age_days"].clip(lower=1)
    df["_views_per_day"] = df["_views"] / df["_age_days_clamped"]
    df["_ratings_per_day"] = df["_rating_n"] / df["_age_days_clamped"]

    agg = (
        df.dropna(subset=["_publish_year"])
          .groupby(["_publish_year", group_col])
          .agg(
              mean_rating=("_rating", "mean"),
              mean_views=("_views", "mean"),
              mean_views_per_day=("_views_per_day", "mean"),
              mean_ratings_per_day=("_ratings_per_day", "mean"),
              n=(group_col, "count"),
          )
          .reset_index()
    )
    agg["_publish_year"] = agg["_publish_year"].astype(int)

    csv_path = DATA_DIR / "temporal_rating_view_trends.csv"
    r = agg.copy()
    for c in ("mean_rating", "mean_views", "mean_views_per_day", "mean_ratings_per_day"):
        r[c] = r[c].map(lambda x: _round(x, 3))
    r["n"] = r["n"].astype(int)
    r.to_csv(csv_path, index=False)
    print(f"[WRITE] {csv_path}")

    tex_path = TABLE_DIR / "temporal_rating_view_trends.tex"
    try:
        if TABLES == "v1":
            write_latex_table(r, tex_path)
        else:
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(r.to_latex(index=False))
        print(f"[TEX]   {tex_path}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    for metric, title, fname in [
        ("mean_rating", "Temporal Rating Trends", "temporal_rating_trends"),
        ("mean_views", "Temporal Views Trends", "temporal_views_trends"),
    ]:
        for dark in (True, False):
            set_mpl_theme(dark=dark)
            fig, ax = plt.subplots(figsize=(10, 5))
            for g, sub in agg.groupby(group_col):
                ax.plot(sub["_publish_year"], sub[metric], label=str(g), marker="o", linewidth=2)
            ax.set_title(f"{title} — {group_col}")
            ax.set_xlabel("Year")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", ncols=2, frameon=False)
            f = f"{fname}_{'dark' if dark else 'light'}.png"
            fpath = (FIG_DIR_DARK if dark else FIG_DIR_LIGHT) / f
            fig.tight_layout()
            fig.savefig(fpath, dpi=200)
            plt.close(fig)
            print(f"[PLOT] {fpath}")

    return agg


def build_engagement_bias_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    def _safe_int(x):
        try:
            return int(x)
        except Exception:
            return x

    summary = (
        df.groupby(group_col)
          .agg(
              n=(group_col, "count"),
              views_mean=("_views", "mean"),
              views_median=("_views", "median"),
              rating_mean=("_rating", "mean"),
              rating_median=("_rating", "median"),
              ratings_count_mean=("_rating_n", "mean"),
              ratings_count_median=("_rating_n", "median"),
          )
          .reset_index()
    )

    gv = []
    gr = []
    for g, sub in df.groupby(group_col):
        gv.append(gini(sub["_views"].values))
        gr.append(gini(sub["_rating_n"].values))
    summary["gini_views"] = [ _round(x, 4) for x in gv ]
    summary["gini_rating_count"] = [ _round(x, 4) for x in gr ]

    # simple max |Cliff's delta| on views
    groups = summary[group_col].tolist()
    max_abs_delta = 0.0
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            a = df.loc[df[group_col]==groups[i], "_views"].values
            b = df.loc[df[group_col]==groups[j], "_views"].values
            max_abs_delta = max(max_abs_delta, abs(cliffs_delta(a, b)))
    summary["max_abs_cliffs_delta_views"] = _round(max_abs_delta, 4)

    for c in ("views_mean","views_median","rating_mean","rating_median","ratings_count_mean","ratings_count_median"):
        summary[c] = summary[c].map(lambda x: _round(x, 3))
    summary["n"] = summary["n"].map(_safe_int)

    csv_path = DATA_DIR / "engagement_bias_by_group.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[WRITE] {csv_path}")

    tex_path = TABLE_DIR / "engagement_bias_by_group.tex"
    try:
        if TABLES == "v1":
            write_latex_table(summary, tex_path)
        else:
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(summary.to_latex(index=False))
        print(f"[TEX]   {tex_path}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    return summary


# ----------------------- Orchestrator -----------------------
def main() -> None:
    print("--- Starting Step 16: Deep Data Analysis ---")
    df = _load_corpus()
    _ = _standardise_columns(df)

    prot_cols = _detect_protected_columns(df)
    primary = prot_cols[0]
    print(f"[INFO] Protected attribute columns used: {prot_cols}. Primary for plots: {primary}")

    print("Analyzing temporal representation trends...")
    rep = build_temporal_group_representation(df, primary)

    print("Analyzing engagement disparities...")
    summ = build_engagement_bias_summary(df, primary)

    print("Analyzing rating/views temporal trends...")
    trends = build_temporal_rating_view_trends(df, primary)

    # Narrative
    narrative = [
        "# 16 — Temporal & Engagement Bias Summary",
        f"- Primary attribute analysed: **{primary}**.",
        "- Outputs: temporal representation, rating/views trends (age-normalised), and engagement inequality (Gini) with effect size.",
    ]
    md_path = NARR_DIR / "16_temporal_engagement_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(narrative) + "\n")
    print(f"[WRITE] {md_path}")

    print("--- Step 16: Completed ---")


if __name__ == "__main__":
    main()
