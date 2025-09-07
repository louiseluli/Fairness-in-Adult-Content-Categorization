#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
18_category_group_dynamics.py

Category × group dynamics with intersectional focus on Black women:
- Robust category parsing from common fields (categories/tags/etc.)
- Tidy protected attributes derived from one-hot if needed
- Category × Group prevalence, shares, representation ratios and log2 over/under-representation
- Black-women under/over-represented categories vs global baseline (dissertation-ready tables)
- Category co-occurrence matrix among top-K categories (for later network plots)

Outputs (CSV):
  outputs/data/cgd_category_group_matrix.csv
  outputs/data/cgd_black_women_under_over.csv
  outputs/data/cgd_category_cooccurrence.csv

Outputs (LaTeX):
  dissertation/auto_tables/cgd_category_group_matrix.tex
  dissertation/auto_tables/cgd_black_women_under_over.tex

Figures (Matplotlib only):
  outputs/figures/dark/cgd_heatmap_log2rr_dark.png
  outputs/figures/light/cgd_heatmap_log2rr_light.png
  outputs/figures/dark/cgd_bw_underrep_bar_dark.png
  outputs/figures/light/cgd_bw_underrep_bar_light.png
  outputs/figures/dark/cgd_bw_overrep_bar_dark.png
  outputs/figures/light/cgd_bw_overrep_bar_light.png

Notes:
- Smoothing: Laplace (+1) to avoid zeros in ratios.
- Log2 ratio centered at 0 (over >0, under <0) for visual symmetry.
- Seed fixed at 75. No seaborn used.

References (for documentation/commentary):
- Monroe, Colaresi, & Quinn (2008). "Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict." Political Analysis.
- Caliskan et al. (2017) association framing (for representation/log-odds intuition).
"""

from __future__ import annotations

# ---------- stdlib ----------
from ast import literal_eval
from collections import Counter
from itertools import combinations
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

TOP_K_CATEGORIES = 30   # limit matrix & co-occurrence to top-K for speed/readability
LAPLACE = 1.0           # Laplace smoothing for ratios


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


def _load_corpus() -> pd.DataFrame:
    if ML_CORPUS.exists():
        df = pd.read_parquet(ML_CORPUS)
        print(f"[LOAD] {ML_CORPUS} (rows={len(df):,})")
        return df
    raise FileNotFoundError("Expected outputs/data/ml_corpus.parquet")


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
    if "race_ethnicity" not in df.columns:
        re_col = _derive_categorical_from_onehot(df, "race_ethnicity", "race_ethnicity")
        if re_col: created.append(re_col)
    if "gender" not in df.columns:
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


# ----------------------- Category parsing -----------------------
def _parse_listish(x) -> List[str]:
    """
    Parse list-like strings or sequences into a list of cleaned category tokens.
    Handles: python-list-in-string, comma/pipe-separated, actual list/tuple.
    Returns lowercase tokens; filtering of empties applied.
    """
    if x is None:
        return []
    # already list-like
    if isinstance(x, (list, tuple, set)):
        vals = list(x)
    else:
        s = str(x).strip()
        if not s:
            return []
        # list-in-string?
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                obj = literal_eval(s)
                if isinstance(obj, (list, tuple, set)):
                    vals = list(obj)
                else:
                    vals = [s]
            except Exception:
                vals = [s]
        else:
            # split by common separators
            sep = "," if ("," in s and "|" not in s) else ("|" if "|" in s else None)
            if sep:
                vals = [p for p in s.split(sep)]
            else:
                # fallback: whitespace split (rare)
                vals = s.split()

    # Clean
    out = []
    for v in vals:
        t = str(v).strip().strip("'\"`").lower()
        if t and t not in {"nan", "none", "null"}:
            out.append(t)
    return out


def _extract_categories(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Try several likely fields. Returns a Series of list[str] categories per row or None if not found.
    """
    candidates = [c for c in ["categories", "category", "tags", "tag_list", "labels"] if c in df.columns]
    if not candidates:
        print("[INFO] No category/tags fields found; skipping category-group dynamics.")
        return None
    # Prefer 'categories' then 'tags'
    for c in ["categories", "tags", "category", "tag_list", "labels"]:
        if c in df.columns:
            series = df[c]
            break
    return series.apply(_parse_listish)


# ----------------------- Core computations -----------------------
def _top_k_categories(cat_lists: pd.Series, k: int = TOP_K_CATEGORIES) -> List[str]:
    cnt = Counter()
    for lst in cat_lists:
        cnt.update(lst)
    top = [w for w, _ in cnt.most_common(k)]
    return top


def _explode_categories(df: pd.DataFrame, cat_lists: pd.Series, keep_only: Optional[List[str]] = None) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_categories"] = cat_lists
    tmp = tmp.loc[tmp["_categories"].map(bool)]  # keep rows with non-empty lists
    tmp = tmp.explode("_categories")
    if keep_only:
        tmp = tmp.loc[tmp["_categories"].isin(keep_only)]
    return tmp


def category_group_matrix(df: pd.DataFrame, cat_lists: pd.Series, group_col: str, top_k: int = TOP_K_CATEGORIES) -> pd.DataFrame:
    """Return Category × Group matrix with counts, shares, representation ratios and log2 rr."""
    top = _top_k_categories(cat_lists, k=top_k)
    ex = _explode_categories(df, cat_lists, keep_only=top)

    # counts
    grp = ex.groupby(["_categories", group_col]).size().reset_index(name="count")
    # totals
    total_cat = grp.groupby("_categories")["count"].transform("sum")
    total_group = grp.groupby(group_col)["count"].transform("sum")
    grand_total = float(grp["count"].sum())

    # Probabilities with smoothing
    # P(group | category) ~ (count + L) / (total_cat + L*G)
    G = grp[group_col].nunique()
    p_g_given_c = (grp["count"] + LAPLACE) / (total_cat + LAPLACE * G)
    # P(group) ~ (sum over categories for that group + L) / (grand_total + L*G)
    # but since grp is per category×group, we need group marginal counts:
    group_totals = grp.groupby(group_col)["count"].transform("sum")
    # equal length series aligned with grp rows:
    p_g = (group_totals + LAPLACE) / (grand_total + LAPLACE * G)

    rr = p_g_given_c / p_g
    log2_rr = np.log2(rr)

    out = grp.copy()
    out["share_in_category"] = (grp["count"] / total_cat).astype(float)
    out["global_group_share"] = (group_totals / grand_total).astype(float)
    out["repr_ratio"] = rr.astype(float)
    out["log2_rr"] = log2_rr.astype(float)

    # nice sorting for each category by most over-represented
    out = out.sort_values(["_categories", "log2_rr"], ascending=[True, False])

    # tidy rounding for export
    exp = out.copy()
    for c in ["share_in_category", "global_group_share", "repr_ratio", "log2_rr"]:
        exp[c] = exp[c].map(lambda x: _round(x, 4))
    exp.rename(columns={"_categories": "category"}, inplace=True)

    # write
    csv = DATA_DIR / "cgd_category_group_matrix.csv"
    exp.to_csv(csv, index=False)
    print(f"[WRITE] {csv}")

    tex = TABLE_DIR / "cgd_category_group_matrix.tex"
    try:
        if WRITE_TEX:
            WRITE_TEX(exp, tex)
        else:
            with open(tex, "w", encoding="utf-8") as f:
                f.write(exp.to_latex(index=False))
        print(f"[TEX]   {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # heatmap of log2_rr (categories × groups)
    # pivot (rows=groups, cols=categories) for legible axis on x
    pv = out.pivot(index=group_col, columns="_categories", values="log2_rr").reindex(columns=top)
    for dark in (True, False):
        set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(min(18, 1.0 + 0.5 * len(top)), 6))
        im = ax.imshow(pv.to_numpy(dtype=float), aspect="auto", origin="upper", cmap="coolwarm", vmin=-2, vmax=2)
        ax.set_xticks(range(len(top))); ax.set_xticklabels([t[:22] for t in top], rotation=45, ha="right")
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        ax.set_title(f"Category × {group_col} — log2 over/under-representation (smoothed)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log2 ratio")
        fig.tight_layout()
        f = f"cgd_heatmap_log2rr_{'dark' if dark else 'light'}.png"
        fig.savefig((FIG_DIR_DARK if dark else FIG_DIR_LIGHT) / f, dpi=200)
        plt.close(fig)
        print(f"[PLOT] {(FIG_DIR_DARK if dark else FIG_DIR_LIGHT) / f}")

    return out, top


def black_women_under_over(df: pd.DataFrame, cat_lists: pd.Series, top: List[str]) -> pd.DataFrame:
    """Under/over-representation of Black women by category vs baseline global share."""
    if "race_ethnicity" not in df.columns or "gender" not in df.columns:
        print("[INFO] Missing race_ethnicity or gender; skipping BW under/over analysis.")
        return pd.DataFrame()

    ex = _explode_categories(df, cat_lists, keep_only=top)
    is_bw = (ex["race_ethnicity"].str.lower() == "black") & (ex["gender"].str.lower() == "female")

    # totals per category
    cat_totals = ex.groupby("_categories").size().rename("n_cat")
    # BW counts per category
    bw_counts = ex.loc[is_bw].groupby("_categories").size().reindex(cat_totals.index, fill_value=0).rename("n_bw_cat")
    # global totals
    n_total = len(ex)
    n_bw_global = int(is_bw.sum())
    p_bw_global = (n_bw_global + LAPLACE) / (n_total + 2 * LAPLACE)  # binary smoothing

    # per-category BW share (smoothed)
    p_bw_cat = (bw_counts + LAPLACE) / (cat_totals + 2 * LAPLACE)
    rr = p_bw_cat / p_bw_global
    log2_rr = np.log2(rr)

    out = pd.DataFrame({
        "category": cat_totals.index,
        "n_cat": cat_totals.values.astype(int),
        "n_bw_cat": bw_counts.values.astype(int),
        "share_bw_in_cat": p_bw_cat.values.astype(float),
        "global_share_bw": float(p_bw_global),
        "repr_ratio_bw": rr.values.astype(float),
        "log2_rr_bw": log2_rr.values.astype(float),
    }).sort_values("log2_rr_bw", ascending=True)  # ascending so underrep first

    # round for export
    for c in ["share_bw_in_cat", "global_share_bw", "repr_ratio_bw", "log2_rr_bw"]:
        out[c] = out[c].map(lambda x: _round(x, 4))

    csv = DATA_DIR / "cgd_black_women_under_over.csv"
    out.to_csv(csv, index=False)
    print(f"[WRITE] {csv}")

    tex = TABLE_DIR / "cgd_black_women_under_over.tex"
    try:
        if WRITE_TEX:
            WRITE_TEX(out, tex)
        else:
            with open(tex, "w", encoding="utf-8") as f:
                f.write(out.to_latex(index=False))
        print(f"[TEX]   {tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # Plots: top-15 underrep & overrep bars by log2_rr_bw
    under = out.nsmallest(15, "log2_rr_bw")
    over = out.nlargest(15, "log2_rr_bw")

    def _bar(df_sub: pd.DataFrame, title: str, fname: str):
        xs = df_sub["category"].tolist()
        ys = df_sub["log2_rr_bw"].astype(float).tolist()
        for dark in (True, False):
            set_mpl_theme(dark=dark)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(xs)), ys)
            ax.set_yticks(range(len(xs))); ax.set_yticklabels([x[:28] for x in xs])
            ax.set_xlabel("log2 over/under-representation")
            ax.set_title(title)
            ax.axvline(0.0, linestyle="--", linewidth=1)
            ax.grid(True, axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig((FIG_DIR_DARK if dark else FIG_DIR_LIGHT) / f"{fname}_{'dark' if dark else 'light'}.png", dpi=200)
            plt.close(fig)
            print(f"[PLOT] {(FIG_DIR_DARK if dark else FIG_DIR_LIGHT) / f'{fname}_{'dark' if dark else 'light'}.png'}")

    _bar(under, "Most Under-represented Categories — Black Women", "cgd_bw_underrep_bar")
    _bar(over, "Most Over-represented Categories — Black Women", "cgd_bw_overrep_bar")

    return out


def category_cooccurrence(cat_lists: pd.Series, top: List[str]) -> pd.DataFrame:
    """Simple symmetric co-occurrence matrix among top categories."""
    # build counts
    idx = {c: i for i, c in enumerate(top)}
    mat = np.zeros((len(top), len(top)), dtype=int)
    for lst in cat_lists:
        row = [c for c in lst if c in idx]
        if len(row) < 2:
            continue
        for a, b in combinations(sorted(set(row)), 2):
            ia, ib = idx[a], idx[b]
            mat[ia, ib] += 1
            mat[ib, ia] += 1
    # to dataframe
    out = pd.DataFrame(mat, index=top, columns=top)
    csv = DATA_DIR / "cgd_category_cooccurrence.csv"
    out.to_csv(csv)
    print(f"[WRITE] {csv}")
    return out


# ----------------------- Orchestrator -----------------------
def main() -> None:
    print("--- Starting Step 18: Category–Group Dynamics ---")
    df = _load_corpus()
    _prepare_protected_columns(df)

    # Categories/tags
    cat_series = _extract_categories(df)
    if cat_series is None:
        print("--- Step 18: Skipped (no categories/tags found) ---")
        return

    # Build matrices
    print("Building Category × Group matrix...")
    group_col = "race_ethnicity" if "race_ethnicity" in df.columns else ("gender" if "gender" in df.columns else "__all__")
    cgm, top = category_group_matrix(df, cat_series, group_col=group_col, top_k=TOP_K_CATEGORIES)

    print("Analyzing Black-women under/over-representation by category...")
    bw = black_women_under_over(df, cat_series, top=top)

    print("Computing category co-occurrence among top categories...")
    _ = category_cooccurrence(cat_series, top=top)

    # Narrative (short)
    lines = [
        "# 18 — Category × Group Dynamics",
        f"- Grouping attribute: **{group_col}**.",
        "- We compute smoothed representation ratios log2(P(group|category)/P(group)). 0 means parity; <0 under-representation; >0 over-representation.",
        "- Black-women focus: top under/over-represented categories vs their global baseline.",
        "- Co-occurrence matrix saved for downstream network visualisation.",
    ]
    md_path = NARR_DIR / "18_category_group_dynamics_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[WRITE] {md_path}")

    print("--- Step 18: Completed ---")


if __name__ == "__main__":
    main()
