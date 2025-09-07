#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20_network_analysis.py

Category co-occurrence network analytics (top-K categories):
- Degree (unweighted), strength (sum of weights)
- Eigenvector centrality (power iteration)
- PageRank (weighted)
- Local clustering coefficient (unweighted)
- Correlate centrality with Black-women log2 over/under-representation

Inputs:
  outputs/data/cgd_category_cooccurrence.csv        (from Step 18)
  outputs/data/cgd_black_women_under_over.csv       (from Step 18; optional)

Outputs:
  outputs/data/net20_category_centrality.csv
  dissertation/auto_tables/net20_category_centrality.tex
  outputs/figures/dark|light/net20_top_strength.png
  outputs/figures/dark|light/net20_strength_vs_bw_log2rr.png
  outputs/narratives/automated/20_network_analysis_summary.md
"""

from __future__ import annotations

# ---------- stdlib ----------
import time
from ast import literal_eval
from pathlib import Path
from typing import List, Optional

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

COOC_PATH = DATA_DIR / "cgd_category_cooccurrence.csv"
BW_PATH = DATA_DIR / "cgd_black_women_under_over.csv"
CORPUS = DATA_DIR / "ml_corpus.parquet"
TOP_K_FALLBACK = 30
EPS = 1e-12


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


def _parse_listish(x) -> List[str]:
    if x is None: return []
    if isinstance(x, (list, tuple, set)): vals = list(x)
    else:
        s = str(x).strip()
        if not s: return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            from ast import literal_eval
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


def _rebuild_cooccurrence_from_corpus(top_k: int = TOP_K_FALLBACK) -> pd.DataFrame:
    print("[INFO] Rebuilding co-occurrence from corpus (fallback).")
    df = pd.read_parquet(CORPUS)
    # categories/tags column
    col = next((c for c in ["categories","tags","category","tag_list","labels"] if c in df.columns), None)
    if col is None:
        raise RuntimeError("No categories/tags columns found to rebuild co-occurrence.")
    lists = df[col].apply(_parse_listish)
    # build vocab
    counts = {}
    for lst in lists:
        for t in lst: counts[t] = counts.get(t, 0) + 1
    vocab = [w for w,_ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    idx = {c:i for i,c in enumerate(vocab)}
    W = np.zeros((len(vocab), len(vocab)), dtype=int)
    from itertools import combinations
    for lst in lists:
        row = [c for c in set(lst) if c in idx]
        for a,b in combinations(sorted(row), 2):
            ia, ib = idx[a], idx[b]
            W[ia, ib] += 1; W[ib, ia] += 1
    mat = pd.DataFrame(W, index=vocab, columns=vocab)
    mat.to_csv(COOC_PATH)
    print(f"[WRITE] {COOC_PATH}")
    return mat


def _load_cooccurrence() -> pd.DataFrame:
    if COOC_PATH.exists():
        try:
            return pd.read_csv(COOC_PATH, index_col=0)
        except Exception:
            # try without index_col (rare CSV mishap)
            df = pd.read_csv(COOC_PATH)
            df = df.set_index(df.columns[0])
            return df
    else:
        return _rebuild_cooccurrence_from_corpus()


def eigenvector_centrality(W: np.ndarray, tol=1e-8, max_iter=500) -> np.ndarray:
    n = W.shape[0]
    v = np.ones(n, dtype=float) / np.sqrt(n)
    for _ in range(max_iter):
        v_new = W @ v
        norm = np.linalg.norm(v_new)
        if norm == 0: return v
        v_new /= norm
        if np.linalg.norm(v_new - v) < tol:
            return v_new
        v = v_new
    return v


def pagerank(W: np.ndarray, d: float = 0.85, tol=1e-8, max_iter=100):
    n = W.shape[0]
    # row-normalized weights
    S = W.sum(axis=1, keepdims=True)
    P = np.divide(W, np.where(S==0, 1.0, S), where=~(S==0))
    r = np.ones(n, dtype=float) / n
    teleport = (1.0 - d) / n
    for _ in range(max_iter):
        r_new = teleport + d * (P.T @ r)
        if np.linalg.norm(r_new - r, 1) < tol:
            return r_new
        r = r_new
    return r


def clustering_coefficients(A: np.ndarray) -> np.ndarray:
    """Unweighted local clustering coefficient for each node."""
    n = A.shape[0]
    C = np.zeros(n, dtype=float)
    for i in range(n):
        nbrs = np.where(A[i] > 0)[0]
        k = nbrs.size
        if k < 2: 
            C[i] = 0.0
            continue
        sub = A[np.ix_(nbrs, nbrs)]
        # each triangle counted twice in upper+lower; divide by 2
        tri = (np.triu(sub, 1) > 0).sum()
        C[i] = (2.0 * tri) / (k*(k-1))
    return C


def main() -> None:
    t0 = time.time()
    print("--- Starting Step 20: Network Analysis ---")

    # 1) Load co-occurrence matrix
    M = _load_cooccurrence()
    categories = list(M.index.astype(str))
    W = M.to_numpy(dtype=float)
    # ensure non-negative symmetry
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)

    # 2) Metrics
    A = (W > 0).astype(int)
    degree = A.sum(axis=1).astype(int)
    strength = W.sum(axis=1)
    eig = eigenvector_centrality(W)
    pr = pagerank(W, d=0.85)
    cc = clustering_coefficients(A)

    out = pd.DataFrame({
        "category": categories,
        "degree": degree,
        "strength": np.round(strength, 3),
        "eigencentrality": np.round(eig / (eig.max() + EPS), 4),  # normalised to [0,1]
        "pagerank": np.round(pr / (pr.max() + EPS), 4),           # normalised to [0,1]
        "clustering": np.round(cc, 4),
    }).sort_values("strength", ascending=False)

    # 3) Merge BW under/over representation (optional)
    corr_note = "n/a"
    if BW_PATH.exists():
        bw = pd.read_csv(BW_PATH)
        if {"category","log2_rr_bw"}.issubset(bw.columns):
            merged = out.merge(bw[["category","log2_rr_bw"]], on="category", how="left")
            x = merged["strength"].to_numpy(float)
            y = merged["log2_rr_bw"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 3:
                r = np.corrcoef(x[m], y[m])[0,1]
                corr_note = f"{r:.3f}"
                out = merged
        else:
            out["log2_rr_bw"] = np.nan
    else:
        out["log2_rr_bw"] = np.nan

    # 4) Save
    p = DATA_DIR / "net20_category_centrality.csv"
    out.to_csv(p, index=False); print(f"[WRITE] {p}")
    t = TABLE_DIR / "net20_category_centrality.tex"
    try:
        if WRITE_TEX: WRITE_TEX(out, t)
        else: open(t, "w", encoding="utf-8").write(out.to_latex(index=False))
        print(f"[TEX]   {t}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # 5) Figures
    # Top-20 by strength
    top = out.nlargest(20, "strength").copy()
    for dark in (True, False):
        set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top)), top["strength"].astype(float).to_numpy())
        ax.set_yticks(range(len(top))); ax.set_yticklabels([c[:28] for c in top["category"]][::-1])
        ax.invert_yaxis()
        ax.set_xlabel("Strength (sum of co-occurrence weights)")
        ax.set_title("Top-20 categories by network strength")
        ax.grid(True, axis="x", alpha=0.3)
        fname = f"net20_top_strength_{'dark' if dark else 'light'}.png"
        fig.tight_layout(); fig.savefig((FIG_DIR_DARK if dark else FIG_DIR_LIGHT)/fname, dpi=200); plt.close(fig)
        print(f"[PLOT] {(FIG_DIR_DARK if dark else FIG_DIR_LIGHT)/fname}")

    # Scatter: strength vs BW log2_rr (if available)
    if "log2_rr_bw" in out.columns and out["log2_rr_bw"].notna().any():
        for dark in (True, False):
            set_mpl_theme(dark=dark)
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            ax.scatter(out["strength"].astype(float), out["log2_rr_bw"].astype(float), s=20)
            ax.axhline(0.0, ls="--", lw=1)
            ax.set_xlabel("Strength"); ax.set_ylabel("Black-women log2 RR")
            ax.set_title("Centrality vs Black-women over/under-representation")
            ax.grid(True, alpha=0.3)
            fname = f"net20_strength_vs_bw_log2rr_{'dark' if dark else 'light'}.png"
            fig.tight_layout(); fig.savefig((FIG_DIR_DARK if dark else FIG_DIR_LIGHT)/fname, dpi=200); plt.close(fig)
            print(f"[PLOT] {(FIG_DIR_DARK if dark else FIG_DIR_LIGHT)/fname}")

    # 6) Narrative
    lines = [
        "# 20 — Network Analysis of Category Co-occurrence",
        "- Degree/strength/eigenvector/PageRank/clustering computed on the top-K category graph.",
        f"- Correlation(strength, BW log2 RR) = {corr_note}. Higher negative values indicate central categories under-represent Black women.",
        "Seed=75; Matplotlib only.",
    ]
    md = NARR_DIR / "20_network_analysis_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[WRITE] {md}")

    elapsed = time.time() - t0
    print(f"[TIME] Step 20 runtime: {elapsed:.2f}s")
    print("--- Step 20: Network Analysis Completed Successfully ---")


if __name__ == "__main__":
    main()
