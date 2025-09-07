#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
17_engagement_bias_analysis.py

Deep engagement-bias analysis with explicit Black-women focus:
- Age-normalised engagement: views/day, ratings/day
- Yearly engagement gaps: Black women vs Others (with bootstrap CIs)
- Quantile profiles (p25, p50, p75, p90, p99) by group
- Head vs Long-tail composition (top 1% by views/day)
- Point-biserial correlations (Black-women flag vs log-views/log-ratings) with bootstrap CIs

Outputs (CSV):
  outputs/data/eng17_yearly_bw_gaps.csv
  outputs/data/eng17_quantiles_race_ethnicity.csv
  outputs/data/eng17_head_tail_composition.csv
  outputs/data/eng17_bw_correlations.csv

Outputs (LaTeX):
  dissertation/auto_tables/eng17_yearly_bw_gaps.tex
  dissertation/auto_tables/eng17_quantiles_race_ethnicity.tex
  dissertation/auto_tables/eng17_head_tail_composition.tex
  dissertation/auto_tables/eng17_bw_correlations.tex

Outputs (Figures):
  outputs/figures/dark/eng17_gap_views_per_day_dark.png
  outputs/figures/light/eng17_gap_views_per_day_light.png
  outputs/figures/dark/eng17_gap_rating_dark.png
  outputs/figures/light/eng17_gap_rating_light.png
  outputs/figures/dark/eng17_head_tail_stack_dark.png
  outputs/figures/light/eng17_head_tail_stack_light.png
  outputs/figures/dark/eng17_quantiles_bw_vs_others_dark.png
  outputs/figures/light/eng17_quantiles_bw_vs_others_light.png

Narrative:
  outputs/narratives/automated/17_engagement_bias_summary.md
"""

from __future__ import annotations

# ---------- stdlib ----------
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

WRITE_TEX = None
try:
    from src.utils.academic_tables import write_latex_table
    WRITE_TEX = write_latex_table
except Exception:
    WRITE_TEX = None


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

N_BOOT = 1000
ALPHA = 0.95


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

    publish_col = None
    for c in ("publish_date", "upload_date", "published_at", "date"):
        if c in cols:
            publish_col = cols[c]; break

    views_col = None
    for c in ("views", "view_count", "views_count", "play_count"):
        if c in cols:
            views_col = cols[c]; break

    rating_col = None
    for c in ("rating", "rating_mean", "average_rating", "avg_rating", "score"):
        if c in cols:
            rating_col = cols[c]; break

    rating_n_col = None
    for c in ("ratings", "rating_count", "num_ratings", "n_ratings", "votes"):
        if c in cols:
            rating_n_col = cols[c]; break

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

    df["_views"] = pd.to_numeric(df[views_col], errors="coerce") if views_col else np.nan
    df["_rating"] = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else np.nan
    df["_rating_n"] = pd.to_numeric(df[rating_n_col], errors="coerce") if rating_n_col else np.nan

    return {
        "publish_date": publish_col,
        "views": views_col,
        "rating_mean": rating_col,
        "rating_count": rating_n_col,
    }


def _derive_categorical_from_onehot(df: pd.DataFrame, prefix: str, out_col: str, fallback: str = "unknown") -> Optional[str]:
    pref = f"{prefix}_"
    onehot_cols = [c for c in df.columns if c.lower().startswith(pref)]
    if not onehot_cols:
        return None

    labels = [c[len(pref):].lower() for c in onehot_cols]
    mixed_token = "mixed_or_other" if any(l == "mixed_or_other" for l in labels) else "mixed"

    oh = df[onehot_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    oh = (oh > 0.5).astype(int)

    vals = []
    for i in range(len(df)):
        row = oh.iloc[i].values
        hits = np.where(row == 1)[0]
        if len(hits) == 1:
            vals.append(labels[hits[0]])
        elif len(hits) > 1:
            vals.append(mixed_token)
        else:
            vals.append(fallback)

    df[out_col] = pd.Series(vals, index=df.index, dtype="object")
    return out_col


def _norm_gender_label(x: str) -> str:
    x = (x or "").lower()
    if x in {"female", "woman", "women", "cis_female", "cis_woman"}:
        return "female"
    if x in {"male", "man", "men", "cis_male", "cis_man"}:
        return "male"
    return x or "unknown"


def _prepare_protected_columns(df: pd.DataFrame) -> List[str]:
    created = []
    re_col = _derive_categorical_from_onehot(df, "race_ethnicity", "race_ethnicity")
    if re_col: created.append(re_col)
    g_col = _derive_categorical_from_onehot(df, "gender", "gender")
    if g_col: created.append(g_col)
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(_norm_gender_label)
    candidates = []
    for c in ["race_ethnicity", "gender", "sexual_orientation"]:
        if c in df.columns:
            candidates.append(c)
    if not candidates:
        df["__all__"] = "__all__"
        candidates = ["__all__"]
        print("[WARN] No protected columns found. Using synthetic '__all__'.")
    else:
        print(f"[INFO] Protected columns: {candidates}")
    return candidates


# ----------------------- Metrics / Stats -----------------------
def _bootstrap_ci(values: np.ndarray, func, n=N_BOOT, alpha=ALPHA) -> Tuple[float, float, float]:
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.arange(vals.size)
    res = []
    for _ in range(n):
        samp = np.random.choice(idx, size=idx.size, replace=True)
        res.append(func(vals[samp]))
    res = np.array(res, dtype=float)
    point = func(vals)
    lo, hi = np.quantile(res, [(1 - alpha) / 2.0, 1 - (1 - alpha) / 2.0])
    return float(point), float(lo), float(hi)


def _mean_diff_ci(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    na, nb = a.size, b.size
    ia = np.arange(na); ib = np.arange(nb)
    diffs = []
    for _ in range(N_BOOT):
        sa = np.random.choice(ia, size=na, replace=True)
        sb = np.random.choice(ib, size=nb, replace=True)
        diffs.append(a[sa].mean() - b[sb].mean())
    diffs = np.array(diffs, dtype=float)
    point = a.mean() - b.mean()
    lo, hi = np.quantile(diffs, [(1 - ALPHA) / 2.0, 1 - (1 - ALPHA) / 2.0])
    return float(point), float(lo), float(hi)


def _corr_ci(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    m = (~np.isnan(x)) & (~np.isnan(y))
    x = x[m]; y = y[m]
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")
    idx = np.arange(x.size)
    vals = []
    for _ in range(N_BOOT):
        s = np.random.choice(idx, size=idx.size, replace=True)
        xv = x[s]; yv = y[s]
        vx = xv.std(ddof=1); vy = yv.std(ddof=1)
        if vx == 0 or vy == 0:
            vals.append(0.0)
        else:
            vals.append(float(np.corrcoef(xv, yv)[0,1]))
    vals = np.array(vals, dtype=float)
    point = float(np.corrcoef(x, y)[0,1]) if x.std(ddof=1) > 0 and y.std(ddof=1) > 0 else 0.0
    lo, hi = np.quantile(vals, [(1 - ALPHA) / 2.0, 1 - (1 - ALPHA) / 2.0])
    return point, float(lo), float(hi)


# ----------------------- Analyses -----------------------
def _compute_engagement_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_age_days_clamped"] = df["_age_days"].clip(lower=1)
    df["_views_per_day"] = df["_views"] / df["_age_days_clamped"]
    df["_ratings_per_day"] = df["_rating_n"] / df["_age_days_clamped"]
    df["_log_views"] = np.log1p(df["_views"].astype(float))
    df["_log_vpd"] = np.log1p(df["_views_per_day"].astype(float))
    df["_log_rpd"] = np.log1p(df["_ratings_per_day"].astype(float))
    return df


def yearly_black_women_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Yearly mean gaps (BW minus Others) for views/day and rating."""
    if "race_ethnicity" not in df.columns or "gender" not in df.columns:
        print("[INFO] Missing race_ethnicity or gender; skipping yearly BW gaps.")
        return pd.DataFrame()

    is_bw = (df["race_ethnicity"].str.lower() == "black") & (df["gender"].str.lower() == "female")
    tmp = df.dropna(subset=["_publish_year"]).copy()
    tmp["is_bw"] = is_bw

    rows = []
    for y, sub in tmp.groupby("_publish_year"):
        a = sub.loc[sub["is_bw"], "_views_per_day"].to_numpy(dtype=float)
        b = sub.loc[~sub["is_bw"], "_views_per_day"].to_numpy(dtype=float)
        d_vpd, lo_vpd, hi_vpd = _mean_diff_ci(a, b)

        a2 = sub.loc[sub["is_bw"], "_rating"].to_numpy(dtype=float)
        b2 = sub.loc[~sub["is_bw"], "_rating"].to_numpy(dtype=float)
        d_r, lo_r, hi_r = _mean_diff_ci(a2, b2)

        rows.append({
            "_publish_year": int(y),
            "gap_views_per_day": _round(d_vpd, 4),
            "gap_views_per_day_lo": _round(lo_vpd, 4),
            "gap_views_per_day_hi": _round(hi_vpd, 4),
            "gap_rating": _round(d_r, 4),
            "gap_rating_lo": _round(lo_r, 4),
            "gap_rating_hi": _round(hi_r, 4),
            "n_bw": int(np.isfinite(a).sum()),
            "n_others": int(np.isfinite(b).sum()),
        })
    out = pd.DataFrame(rows).sort_values("_publish_year")
    csv = DATA_DIR / "eng17_yearly_bw_gaps.csv"
    out.to_csv(csv, index=False)
    print(f"[WRITE] {csv}")

    tex = TABLE_DIR / "eng17_yearly_bw_gaps.tex"
    try:
        if WRITE_TEX:
            WRITE_TEX(out, tex)
        else:
            with open(tex, "w", encoding="utf-8") as f:
                f.write(out.to_latex(index=False))
        print(f"[TEX]   {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # Plots
    for metric, title, fname in [
        ("gap_views_per_day", "Yearly Gap — Views per day (BW - Others)", "eng17_gap_views_per_day"),
        ("gap_rating", "Yearly Gap — Rating (BW - Others)", "eng17_gap_rating"),
    ]:
        for dark in (True, False):
            set_mpl_theme(dark=dark)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(out["_publish_year"], out[metric], marker="o", linewidth=2)
            # CI ribbons
            ax.fill_between(out["_publish_year"], out[f"{metric}_lo"], out[f"{metric}_hi"], alpha=0.2)
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("Year"); ax.set_ylabel("Difference in means")
            ax.grid(True, alpha=0.3)
            f = f"{fname}_{'dark' if dark else 'light'}.png"
            fpath = (FIG_DIR_DARK if dark else FIG_DIR_LIGHT) / f
            fig.tight_layout(); fig.savefig(fpath, dpi=200); plt.close(fig)
            print(f"[PLOT] {fpath}")

    return out


def quantile_profiles_by_race(df: pd.DataFrame) -> pd.DataFrame:
    """Race-ethnicity quantiles for views/day and rating."""
    if "race_ethnicity" not in df.columns:
        print("[INFO] No race_ethnicity; skipping quantile profiles.")
        return pd.DataFrame()

    qs = [0.25, 0.5, 0.75, 0.9, 0.99]
    rows = []
    for g, sub in df.groupby("race_ethnicity"):
        vpd = sub["_views_per_day"].to_numpy(dtype=float)
        rt = sub["_rating"].to_numpy(dtype=float)
        qv = np.quantile(vpd[~np.isnan(vpd)], qs) if np.isfinite(vpd).any() else [np.nan]*len(qs)
        qr = np.quantile(rt[~np.isnan(rt)], qs) if np.isfinite(rt).any() else [np.nan]*len(qs)
        row = {"race_ethnicity": g}
        for q, val in zip(qs, qv):
            row[f"vpd_q{int(q*100)}"] = _round(val, 3)
        for q, val in zip(qs, qr):
            row[f"rating_q{int(q*100)}"] = _round(val, 3)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("race_ethnicity")
    csv = DATA_DIR / "eng17_quantiles_race_ethnicity.csv"
    out.to_csv(csv, index=False)
    print(f"[WRITE] {csv}")

    tex = TABLE_DIR / "eng17_quantiles_race_ethnicity.tex"
    try:
        if WRITE_TEX:
            WRITE_TEX(out, tex)
        else:
            with open(tex, "w", encoding="utf-8") as f:
                f.write(out.to_latex(index=False))
        print(f"[TEX]   {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # BW vs Others quantile lines for views/day
    if "gender" in df.columns:
        is_bw = (df["race_ethnicity"].str.lower() == "black") & (df["gender"].str.lower() == "female")
        a = df.loc[is_bw, "_views_per_day"].to_numpy(dtype=float)
        b = df.loc[~is_bw, "_views_per_day"].to_numpy(dtype=float)
        qa = np.quantile(a[~np.isnan(a)], qs) if np.isfinite(a).any() else [np.nan]*len(qs)
        qb = np.quantile(b[~np.isnan(b)], qs) if np.isfinite(b).any() else [np.nan]*len(qs)

        for dark in (True, False):
            set_mpl_theme(dark=dark)
            fig, ax = plt.subplots(figsize=(8.5, 5))
            x = np.array([25, 50, 75, 90, 99], dtype=int)
            ax.plot(x, qa, marker="o", linewidth=2, label="Black women")
            ax.plot(x, qb, marker="o", linewidth=2, label="Others")
            ax.set_title("Quantile Profile — Views per day")
            ax.set_xlabel("Quantile (%)"); ax.set_ylabel("Views per day")
            ax.grid(True, alpha=0.3); ax.legend(frameon=False)
            f = f"eng17_quantiles_bw_vs_others_{'dark' if dark else 'light'}.png"
            fpath = (FIG_DIR_DARK if dark else FIG_DIR_LIGHT) / f
            fig.tight_layout(); fig.savefig(fpath, dpi=200); plt.close(fig)
            print(f"[PLOT] {fpath}")

    return out


def head_tail_composition(df: pd.DataFrame, top_pct: float = 0.01) -> pd.DataFrame:
    """Composition of top 'head' (top_pct by views/day) vs tail, by race_ethnicity and BW flag."""
    df = df.copy()
    if "_views_per_day" not in df.columns:
        return pd.DataFrame()
    thr = df["_views_per_day"].quantile(1 - top_pct)
    df["_is_head"] = df["_views_per_day"] >= thr

    # Race-ethnicity composition
    rows = []
    for head_flag, sub in df.groupby("_is_head"):
        total = len(sub)
        if total == 0:
            continue
        for g, sub2 in sub.groupby("race_ethnicity") if "race_ethnicity" in df.columns else [("all", sub)]:
            rows.append({
                "segment": "head" if head_flag else "tail",
                "race_ethnicity": g,
                "count": int(len(sub2)),
                "share": _round(len(sub2) / total, 4),
            })
    out = pd.DataFrame(rows).sort_values(["segment", "share"], ascending=[True, False])
    csv = DATA_DIR / "eng17_head_tail_composition.csv"
    out.to_csv(csv, index=False)
    print(f"[WRITE] {csv}")

    tex = TABLE_DIR / "eng17_head_tail_composition.tex"
    try:
        if WRITE_TEX:
            WRITE_TEX(out, tex)
        else:
            with open(tex, "w", encoding="utf-8") as f:
                f.write(out.to_latex(index=False))
        print(f"[TEX]   {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # Stacked bar plot (BW vs Others if possible, else race_ethnicity)
    if "race_ethnicity" in df.columns and "gender" in df.columns:
        is_bw = (df["race_ethnicity"].str.lower() == "black") & (df["gender"].str.lower() == "female")
        df["_is_bw"] = is_bw
        shares = (
            df.groupby(["_is_head", "_is_bw"]).size().reset_index(name="count")
        )
        # to shares per segment
        shares["total_seg"] = shares.groupby("_is_head")["count"].transform("sum")
        shares["share"] = shares["count"] / shares["total_seg"]
        seg_map = {True: "Head (top 1%)", False: "Tail (bottom 99%)"}

        for dark in (True, False):
            set_mpl_theme(dark=dark)
            fig, ax = plt.subplots(figsize=(7.5, 5))
            for seg in [False, True]:
                sub = shares.loc[shares["_is_head"] == seg]
                bw_share = float(sub.loc[sub["_is_bw"], "share"]) if any(sub["_is_bw"]) else 0.0
                oth_share = 1.0 - bw_share
                left = 0 if not seg else 1
                ax.bar(left, bw_share, width=0.8, label="Black women" if left==0 else None)
                ax.bar(left, oth_share, width=0.8, bottom=bw_share, label="Others" if left==0 else None)
            ax.set_xticks([0,1]); ax.set_xticklabels(["Tail (99%)", "Head (1%)"])
            ax.set_ylim(0,1); ax.set_ylabel("Share")
            ax.set_title("Head vs Tail Composition — Black women share")
            ax.legend(frameon=False)
            ax.grid(True, axis="y", alpha=0.3)
            f = f"eng17_head_tail_stack_{'dark' if dark else 'light'}.png"
            fpath = (FIG_DIR_DARK if dark else FIG_DIR_LIGHT) / f
            fig.tight_layout(); fig.savefig(fpath, dpi=200); plt.close(fig)
            print(f"[PLOT] {fpath}")

    return out


def bw_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Point-biserial correlations (equivalently Pearson with binary) between is_black_woman
    and engagement proxies (log-views, log-views/day, rating).
    """
    if "race_ethnicity" not in df.columns or "gender" not in df.columns:
        print("[INFO] Missing race_ethnicity or gender; skipping BW correlations.")
        return pd.DataFrame()

    is_bw = ((df["race_ethnicity"].str.lower() == "black") &
             (df["gender"].str.lower() == "female")).astype(float)

    metrics = {
        "log_views": df["_log_views"].to_numpy(dtype=float),
        "log_vpd": df["_log_vpd"].to_numpy(dtype=float),
        "rating": df["_rating"].to_numpy(dtype=float),
    }
    rows = []
    for name, arr in metrics.items():
        r, lo, hi = _corr_ci(is_bw.to_numpy(dtype=float), arr)
        rows.append({"metric": name, "corr": _round(r, 4),
                     "corr_lo": _round(lo, 4), "corr_hi": _round(hi, 4),
                     "n": int(np.isfinite(arr).sum())})
    out = pd.DataFrame(rows)

    csv = DATA_DIR / "eng17_bw_correlations.csv"
    out.to_csv(csv, index=False)
    print(f"[WRITE] {csv}")

    tex = TABLE_DIR / "eng17_bw_correlations.tex"
    try:
        if WRITE_TEX:
            WRITE_TEX(out, tex)
        else:
            with open(tex, "w", encoding="utf-8") as f:
                f.write(out.to_latex(index=False))
        print(f"[TEX]   {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    return out


# ----------------------- Orchestrator -----------------------
def main() -> None:
    print("--- Starting Step 17: Engagement Bias Analysis ---")
    df = _load_corpus()
    _ = _standardise_columns(df)
    _ = _prepare_protected_columns(df)  # creates tidy 'race_ethnicity'/'gender' when one-hot is present

    df = _compute_engagement_fields(df)

    # 1) Yearly BW gaps
    bw_yearly = yearly_black_women_gaps(df)

    # 2) Quantiles by race_ethnicity (+ BW vs Others quantile figure)
    quantiles = quantile_profiles_by_race(df)

    # 3) Head vs Tail composition (top 1% by views/day); BW stacked plot
    composition = head_tail_composition(df, top_pct=0.01)

    # 4) BW correlations with log-engagement
    corr = bw_correlations(df)

    # Narrative
    lines = [
        "# 17 — Engagement Bias Deep Dive",
        "- Age-normalised metrics (views/day, ratings/day) computed to mitigate recency/popularity effects.",
        "- Yearly differences (BW − Others) reported with bootstrap 95% CIs.",
        "- Quantile profiles reveal tail vs head disparities; composition charts show BW presence at the very top (1%).",
        "- Point-biserial correlations quantify association between the BW flag and engagement proxies.",
    ]
    md_path = NARR_DIR / "17_engagement_bias_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[WRITE] {md_path}")

    print("--- Step 17: Completed ---")


if __name__ == "__main__":
    main()
