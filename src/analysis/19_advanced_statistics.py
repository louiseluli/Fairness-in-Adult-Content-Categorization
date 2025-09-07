#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
19_advanced_statistics.py  —  FAST version

Black-women–centric, uncertainty-aware analytics:
A) Cliff’s delta (fast, rank-based) + bootstrap CIs for:
   - views/day, ratings/day, rating
B) Temporal slopes (views/day, rating) for BW vs Others + bootstrap CIs and slope-gap CI
C) Category divergence by group via KL (top-K categories)
D) Optional: Harm relative risk if outputs/data/harm_category_by_group.csv exists

Optimisations:
- Try reading only needed columns from parquet; fall back to full read if needed.
- Cliff’s δ via Mann–Whitney U / ranks (O(n log n)).
- Vectorised bootstrap for slopes (no nested bootstraps).

Outputs (CSV):
  outputs/data/adv19_bw_effect_sizes.csv
  outputs/data/adv19_trend_slopes.csv
  outputs/data/adv19_category_divergence.csv
  outputs/data/adv19_harm_relative_risks.csv (if harm file available)

LaTeX:
  dissertation/auto_tables/adv19_*.tex

Figures:
  outputs/figures/dark|light/adv19_*.png
"""

from __future__ import annotations

# ---------- stdlib ----------
import time
from ast import literal_eval
from pathlib import Path
from typing import List, Optional, Tuple

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
HARM_BY_GROUP = DATA_DIR / "harm_category_by_group.csv"

TOP_K_CATEGORIES = 30
N_BOOT = 1000
ALPHA = 0.95
EPS = 1e-12


# ----------------------- Utils -----------------------
def set_mpl_theme(dark: bool) -> None:
    if THEME is not None:
        THEME.apply(dark=dark); return
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
    try: return round(float(x), k)
    except Exception: return x


def _ensure_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    try: return s.dt.tz_convert(None)
    except Exception: return s


def _load_corpus_fast() -> pd.DataFrame:
    """Try minimal columns; fall back to full read if needed."""
    if not ML_CORPUS.exists():
        raise FileNotFoundError("Expected outputs/data/ml_corpus.parquet")

    # Minimal set that covers most schemas seen in this project
    minimal_cols = [
        "publish_date","upload_date","published_at","date",
        "views","view_count","views_count","play_count",
        "rating","rating_mean","average_rating","avg_rating","score",
        "ratings","rating_count","num_ratings","n_ratings","votes",
        "race_ethnicity","gender","categories","tags","category","tag_list","labels"
    ]
    try:
        df = pd.read_parquet(ML_CORPUS, columns=minimal_cols)
        print(f"[LOAD] {ML_CORPUS} (filtered cols, rows={len(df):,})")
        return df
    except Exception:
        df = pd.read_parquet(ML_CORPUS)
        print(f"[LOAD] {ML_CORPUS} (rows={len(df):,})")
        return df


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
        if len(hits) == 1: vals.append(labels[hits[0]])
        elif len(hits) > 1: vals.append(mixed_token)
        else: vals.append(fallback)
    df[out_col] = pd.Series(vals, index=df.index, dtype="object")
    return out_col


def _norm_gender_label(x: str) -> str:
    x = (x or "").lower()
    if x in {"female","woman","women","cis_female","cis_woman"}: return "female"
    if x in {"male","man","men","cis_male","cis_man"}: return "male"
    return x or "unknown"


def _prepare_columns(df: pd.DataFrame) -> None:
    # publish date
    publish_col = next((c for c in ["publish_date","upload_date","published_at","date"] if c in df.columns), None)
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

    # numeric fields (robust to schema variants)
    def _num(colnames, default=np.nan):
        c = next((c for c in colnames if c in df.columns), None)
        return pd.to_numeric(df[c], errors="coerce") if c else default
    df["_views"] = _num(["views","view_count","views_count","play_count"])
    df["_rating"] = _num(["rating","rating_mean","average_rating","avg_rating","score"])
    df["_rating_n"] = _num(["ratings","rating_count","num_ratings","n_ratings","votes"])

    # protected
    if "race_ethnicity" not in df.columns:
        _derive_categorical_from_onehot(df, "race_ethnicity", "race_ethnicity")
    if "gender" not in df.columns:
        _derive_categorical_from_onehot(df, "gender", "gender")
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(_norm_gender_label)

    # engagement per day
    df["_age_days_clamped"] = df["_age_days"].clip(lower=1)
    df["_views_per_day"] = df["_views"] / df["_age_days_clamped"]
    df["_ratings_per_day"] = df["_rating_n"] / df["_age_days_clamped"]


# ----------------------- Fast Stats -----------------------
def cliffs_delta_fast(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-based Cliff's δ using Mann–Whitney U equivalence."""
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    m, n = a.size, b.size
    if m == 0 or n == 0:
        return float("nan")
    x = np.concatenate([a, b])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    Ra = ranks[:m].sum()  # because we concatenated as [a, b]
    U = Ra - m*(m+1)/2.0
    delta = 2.0*U/(m*n) - 1.0
    return float(delta)


def bootstrap_ci_delta(a: np.ndarray, b: np.ndarray, n=N_BOOT, alpha=ALPHA) -> Tuple[float, float, float]:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    ia = np.arange(a.size); ib = np.arange(b.size)
    vals = []
    for _ in range(n):
        sa = a[np.random.choice(ia, size=ia.size, replace=True)]
        sb = b[np.random.choice(ib, size=ib.size, replace=True)]
        vals.append(cliffs_delta_fast(sa, sb))
    vals = np.array(vals, dtype=float)
    point = cliffs_delta_fast(a, b)
    lo, hi = np.quantile(vals, [(1-alpha)/2.0, 1-(1-alpha)/2.0])
    return point, float(lo), float(hi)


def slopes_bootstrap(x: np.ndarray, y: np.ndarray, n=N_BOOT) -> np.ndarray:
    """Return n bootstrap slope samples for simple linear regression (degree=1)."""
    m = (~np.isnan(x)) & (~np.isnan(y))
    x = x[m].astype(float); y = y[m].astype(float)
    if x.size < 3:
        return np.array([], dtype=float)
    idx = np.arange(x.size)
    out = np.empty(n, dtype=float)
    for i in range(n):
        s = np.random.choice(idx, size=idx.size, replace=True)
        coef = np.polyfit(x[s], y[s], 1)[0]
        out[i] = coef
    return out


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, None); q = np.clip(q, EPS, None)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ----------------------- Category parsing -----------------------
def _parse_listish(x) -> List[str]:
    if x is None: return []
    if isinstance(x, (list, tuple, set)): vals = list(x)
    else:
        s = str(x).strip()
        if not s: return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                obj = literal_eval(s)
                vals = list(obj) if isinstance(obj, (list, tuple, set)) else [s]
            except Exception:
                vals = [s]
        else:
            sep = "," if ("," in s and "|" not in s) else ("|" if "|" in s else None)
            vals = [p for p in s.split(sep)] if sep else s.split()
    out = []
    for v in vals:
        t = str(v).strip().strip("'\"`").lower()
        if t and t not in {"nan","none","null"}:
            out.append(t)
    return out


def _extract_categories(df: pd.DataFrame) -> Optional[pd.Series]:
    candidates = [c for c in ["categories","tags","category","tag_list","labels"] if c in df.columns]
    if not candidates: return None
    col = next((c for c in ["categories","tags","category","tag_list","labels"] if c in df.columns), candidates[0])
    return df[col].apply(_parse_listish)


# ----------------------- Analyses -----------------------
def effect_sizes_bw_vs_others(df: pd.DataFrame) -> pd.DataFrame:
    if "race_ethnicity" not in df.columns or "gender" not in df.columns:
        print("[INFO] Missing race_ethnicity or gender; skipping BW effect sizes.")
        return pd.DataFrame()
    is_bw = (df["race_ethnicity"].str.lower()=="black") & (df["gender"].str.lower()=="female")
    metrics = {
        "views_per_day": df.loc[is_bw, "_views_per_day"].to_numpy(float), 
        "ratings_per_day": df.loc[is_bw, "_ratings_per_day"].to_numpy(float),
        "rating": df.loc[is_bw, "_rating"].to_numpy(float),
    }
    others = {
        "views_per_day": df.loc[~is_bw, "_views_per_day"].to_numpy(float),
        "ratings_per_day": df.loc[~is_bw, "_ratings_per_day"].to_numpy(float),
        "rating": df.loc[~is_bw, "_rating"].to_numpy(float),
    }
    rows = []
    for name in ["views_per_day","ratings_per_day","rating"]:
        point, lo, hi = bootstrap_ci_delta(metrics[name], others[name], n=N_BOOT, alpha=ALPHA)
        rows.append({
            "metric": name, "cliffs_delta": _round(point, 4), 
            "ci_lo": _round(lo, 4), "ci_hi": _round(hi, 4),
            "n_bw": int(np.isfinite(metrics[name]).sum()),
            "n_others": int(np.isfinite(others[name]).sum()),
        })
    out = pd.DataFrame(rows)
    p = DATA_DIR / "adv19_bw_effect_sizes.csv"; out.to_csv(p, index=False); print(f"[WRITE] {p}")
    t = TABLE_DIR / "adv19_bw_effect_sizes.tex"
    try:
        if WRITE_TEX: WRITE_TEX(out, t)
        else: open(t, "w", encoding="utf-8").write(out.to_latex(index=False))
        print(f"[TEX]   {t}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # Small CI bar figure
    for dark in (True, False):
        set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = np.arange(len(out))
        y = out["cliffs_delta"].astype(float).to_numpy()
        ylo = y - out["ci_lo"].astype(float).to_numpy()
        yhi = out["ci_hi"].astype(float).to_numpy() - y
        ax.bar(xs, y, yerr=[ylo, yhi], capsize=5)
        ax.axhline(0.0, ls="--", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(out["metric"].tolist())
        ax.set_ylabel("Cliff's δ (BW − Others)")
        ax.set_title("Effect sizes with 95% bootstrap CI")
        ax.grid(True, axis="y", alpha=0.3)
        fname = f"adv19_bw_effectsizes_{'dark' if dark else 'light'}.png"
        fig.tight_layout(); fig.savefig((FIG_DIR_DARK if dark else FIG_DIR_LIGHT)/fname, dpi=200); plt.close(fig)
        print(f"[PLOT] {(FIG_DIR_DARK if dark else FIG_DIR_LIGHT)/fname}")
    return out


def temporal_slopes_bw_vs_others(df: pd.DataFrame) -> pd.DataFrame:
    if "race_ethnicity" not in df.columns or "gender" not in df.columns:
        print("[INFO] Missing race_ethnicity or gender; skipping BW temporal slopes."); return pd.DataFrame()
    m = ~df["_publish_year"].isna()
    if not m.any():
        print("[INFO] No publish years; skipping temporal slopes."); return pd.DataFrame()

    is_bw = (df["race_ethnicity"].str.lower()=="black") & (df["gender"].str.lower()=="female")
    rows = []
    for metric, label in [("_views_per_day","views_per_day"), ("_rating","rating")]:
        a_y = df.loc[is_bw & m, "_publish_year"].to_numpy(float)
        a_v = df.loc[is_bw & m, metric].to_numpy(float)
        b_y = df.loc[~is_bw & m, "_publish_year"].to_numpy(float)
        b_v = df.loc[~is_bw & m, metric].to_numpy(float)

        # bootstrap slopes
        boot_a = slopes_bootstrap(a_y, a_v, n=N_BOOT)
        boot_b = slopes_bootstrap(b_y, b_v, n=N_BOOT)
        s_a = float(np.polyfit(a_y[~np.isnan(a_v)], a_v[~np.isnan(a_v)], 1)[0]) if a_y.size>2 else np.nan
        s_b = float(np.polyfit(b_y[~np.isnan(b_v)], b_v[~np.isnan(b_v)], 1)[0]) if b_y.size>2 else np.nan

        lo_a, hi_a = np.quantile(boot_a, [(1-ALPHA)/2.0, 1-(1-ALPHA)/2.0]) if boot_a.size else (np.nan, np.nan)
        lo_b, hi_b = np.quantile(boot_b, [(1-ALPHA)/2.0, 1-(1-ALPHA)/2.0]) if boot_b.size else (np.nan, np.nan)
        # gap distribution by random pairing (same length via min)
        k = int(min(len(boot_a), len(boot_b)))
        gap = s_a - s_b
        gap_samples = (np.random.choice(boot_a, size=k, replace=True) - np.random.choice(boot_b, size=k, replace=True)) if k>0 else np.array([])
        lo_g, hi_g = (np.quantile(gap_samples, [(1-ALPHA)/2.0, 1-(1-ALPHA)/2.0]) if gap_samples.size else (np.nan, np.nan))

        rows.append({
            "metric": label,
            "slope_bw": _round(s_a, 4), "slope_bw_lo": _round(lo_a, 4), "slope_bw_hi": _round(hi_a, 4),
            "slope_others": _round(s_b, 4), "slope_others_lo": _round(lo_b, 4), "slope_others_hi": _round(hi_b, 4),
            "slope_gap_bw_minus_others": _round(gap, 4),
            "slope_gap_lo": _round(float(lo_g), 4), "slope_gap_hi": _round(float(hi_g), 4),
            "n_bw": int(np.isfinite(a_v).sum()), "n_others": int(np.isfinite(b_v).sum()),
        })

    out = pd.DataFrame(rows)
    p = DATA_DIR / "adv19_trend_slopes.csv"; out.to_csv(p, index=False); print(f"[WRITE] {p}")
    t = TABLE_DIR / "adv19_trend_slopes.tex"
    try:
        if WRITE_TEX: WRITE_TEX(out, t)
        else: open(t, "w", encoding="utf-8").write(out.to_latex(index=False))
        print(f"[TEX]   {t}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # BW vs Others slope bars
    for dark in (True, False):
        set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(8.6, 5))
        xs = np.arange(len(out))
        width = 0.35
        s_bw = out["slope_bw"].astype(float).to_numpy()
        s_ot = out["slope_others"].astype(float).to_numpy()
        eb_bw = [s_bw - out["slope_bw_lo"].astype(float), out["slope_bw_hi"].astype(float) - s_bw]
        eb_ot = [s_ot - out["slope_others_lo"].astype(float), out["slope_others_hi"].astype(float) - s_ot]
        ax.bar(xs - width/2, s_bw, width=width, yerr=eb_bw, capsize=5, label="Black women")
        ax.bar(xs + width/2, s_ot, width=width, yerr=eb_ot, capsize=5, label="Others")
        ax.axhline(0.0, ls="--", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(out["metric"].tolist())
        ax.set_ylabel("Yearly slope")
        ax.set_title("Temporal slopes (95% bootstrap CI)")
        ax.legend(frameon=False); ax.grid(True, axis="y", alpha=0.3)
        fname = f"adv19_slope_bw_vs_others_{'dark' if dark else 'light'}.png"
        fig.tight_layout(); fig.savefig((FIG_DIR_DARK if dark else FIG_DIR_LIGHT)/fname, dpi=200); plt.close(fig)
        print(f"[PLOT] {(FIG_DIR_DARK if dark else FIG_DIR_LIGHT)/fname}")
    return out


def category_divergence_by_group(df: pd.DataFrame) -> pd.DataFrame:
    cat_series = _extract_categories(df)
    if cat_series is None:
        print("[INFO] No categories; skipping category divergence."); return pd.DataFrame()
    # vocab (top-K)
    counts = {}
    for lst in cat_series:
        for t in lst: counts[t] = counts.get(t, 0) + 1
    vocab = [w for w,_ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K_CATEGORIES]]
    if not vocab: return pd.DataFrame()
    # global p
    g = np.zeros(len(vocab), float)
    for lst in cat_series:
        for t in lst:
            if t in vocab: g[vocab.index(t)] += 1
    gp = (g + EPS) / (g.sum() + EPS*len(vocab))
    # per group
    group_col = "race_ethnicity" if "race_ethnicity" in df.columns else ("gender" if "gender" in df.columns else None)
    if group_col is None:
        print("[INFO] No group column; skipping category divergence."); return pd.DataFrame()
    rows = []
    for gname, sub in df.groupby(group_col):
        s = np.zeros(len(vocab), float)
        subs = cat_series.loc[sub.index]
        for lst in subs:
            for t in lst:
                if t in vocab: s[vocab.index(t)] += 1
        sp = (s + EPS) / (s.sum() + EPS*len(vocab))
        D = kl_divergence(sp, gp)
        rows.append({"group": str(gname), "kl_vs_global": _round(D, 4), "tokens": int(s.sum())})
    out = pd.DataFrame(rows).sort_values("kl_vs_global", ascending=False)
    p = DATA_DIR / "adv19_category_divergence.csv"; out.to_csv(p, index=False); print(f"[WRITE] {p}")
    t = TABLE_DIR / "adv19_category_divergence.tex"
    try:
        if WRITE_TEX: WRITE_TEX(out, t)
        else: open(t, "w", encoding="utf-8").write(out.to_latex(index=False))
        print(f"[TEX]   {t}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")
    # bar
    for dark in (True, False):
        set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(out["group"], out["kl_vs_global"].astype(float))
        ax.set_ylabel("KL divergence vs global"); ax.set_title("Category divergence by group")
        ax.grid(True, axis="y", alpha=0.3)
        fname = f"adv19_category_kl_bar_{'dark' if dark else 'light'}.png"
        fig.tight_layout(); fig.savefig((FIG_DIR_DARK if dark else FIG_DIR_LIGHT)/fname, dpi=200); plt.close(fig)
        print(f"[PLOT] {(FIG_DIR_DARK if dark else FIG_DIR_LIGHT)/fname}")
    return out


def harm_relative_risk() -> pd.DataFrame:
    if not HARM_BY_GROUP.exists():
        print("[INFO] harm_category_by_group.csv not found; skipping harm RR."); return pd.DataFrame()
    df = pd.read_csv(HARM_BY_GROUP)
    cols = {c.lower(): c for c in df.columns}
    if not {"group","harm_category"}.issubset(set(k.lower() for k in df.columns)):
        print("[INFO] harm file missing required columns; skipping harm RR."); return pd.DataFrame()
    group_col = cols.get("group"); harm_col = cols.get("harm_category")
    count_col = cols.get("count") or cols.get("n") or cols.get("freq")
    if count_col is None:
        print("[INFO] harm file missing count column; skipping harm RR."); return pd.DataFrame()

    # Determine BW group names present
    bw_mask = df[group_col].astype(str).str.contains("black", case=False) & df[group_col].astype(str).str.contains("female|woman|women", case=False)

    totals = df.groupby(group_col)[count_col].sum().rename("total").reset_index()
    merged = df.merge(totals, on=group_col)
    merged["p_harm"] = merged[count_col] / merged["total"]
    merged["_is_bw"] = merged[group_col].astype(str).isin(df.loc[bw_mask, group_col].astype(str).unique())

    rows = []
    for h, sub in merged.groupby(harm_col):
        a = sub.loc[sub["_is_bw"], [count_col, "total"]].sum()
        b = sub.loc[~sub["_is_bw"], [count_col, "total"]].sum()
        a1, a0 = float(a[count_col]), float(a["total"])
        b1, b0 = float(b[count_col]), float(b["total"])
        p1 = (a1 + EPS) / (a0 + EPS); p0 = (b1 + EPS) / (b0 + EPS)
        rr = p1 / p0
        log_rr = np.log(rr)
        se = np.sqrt((1/(a1+EPS)) - (1/(a0+EPS)) + (1/(b1+EPS)) - (1/(b0+EPS)))
        lo = np.exp(log_rr - 1.96*se); hi = np.exp(log_rr + 1.96*se)
        rows.append({
            "harm_category": h, "RR_bw_vs_others": _round(rr, 4),
            "RR_lo": _round(lo, 4), "RR_hi": _round(hi, 4),
            "bw_count": int(a1), "bw_total": int(a0), "others_count": int(b1), "others_total": int(b0),
        })
    out = pd.DataFrame(rows).sort_values("RR_bw_vs_others")
    p = DATA_DIR / "adv19_harm_relative_risks.csv"; out.to_csv(p, index=False); print(f"[WRITE] {p}")
    t = TABLE_DIR / "adv19_harm_relative_risks.tex"
    try:
        if WRITE_TEX: WRITE_TEX(out, t)
        else: open(t, "w", encoding="utf-8").write(out.to_latex(index=False))
        print(f"[TEX]   {t}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")
    return out


# ----------------------- Orchestrator -----------------------
def main() -> None:
    t0 = time.time()
    print("--- Starting Step 19: Advanced Statistics ---")

    df = _load_corpus_fast()
    _prepare_columns(df)

    eff = effect_sizes_bw_vs_others(df)
    slopes = temporal_slopes_bw_vs_others(df)
    cat_div = category_divergence_by_group(df)
    harr = harm_relative_risk()

    # Narrative
    lines = [
        "# 19 — Advanced Statistics (Fast, Uncertainty-aware)",
        "- Rank-based Cliff’s δ with bootstrap CIs for BW vs Others (views/day, ratings/day, rating).",
        "- Temporal slopes with bootstrap CIs and BW–Others slope gap.",
        "- KL divergence of category distribution by group (top-K).",
        "- Harm relative risks (if available). Seed=75.",
    ]
    md = NARR_DIR / "19_advanced_statistics_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[WRITE] {md}")

    elapsed = time.time() - t0
    print(f"[TIME] Step 19 runtime: {elapsed:.2f}s")
    print("--- Step 19: Advanced Statistics Completed Successfully ---")


if __name__ == "__main__":
    main()
