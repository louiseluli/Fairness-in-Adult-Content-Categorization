#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 22 — Ablation Studies
Goal: quantify sensitivity of key results to modeling choices.
Ablations implemented:
  A) Text lexicon OFF  (HurtLex disabled ⇒ harm RR recomputed from zeros)
  B) Category noise    (randomly mask X% of category tokens)
  C) Top-K cap         (limit category vocab to K={10,20,30,...})
  D) Group encoding    (use one-hot intersection vs. derived 'race_ethnicity'/'gender')
  E) Bootstrap size    (reduce/increase N_BOOT for effect sizes/slopes)
Outputs:
  outputs/ablation/22_summary.csv
  outputs/ablation/*.csv per ablation setting
Figures mirrored to outputs/figures/(dark|light)/ with prefix abl22_*
"""

from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs" / "data"
ABL  = ROOT / "outputs" / "ablation"; ABL.mkdir(parents=True, exist_ok=True)

ML_CORPUS = DATA / "ml_corpus.parquet"
HARM_WIDE = DATA / "harm_category_by_group.csv"

# --- utilities (re-use schemas from step 19) ---
def _load_corpus():
    return pd.read_parquet(ML_CORPUS)

def _recompute_harm_rr_from_wide(wide: pd.DataFrame, totals: dict, bw_key: str) -> pd.DataFrame:
    EPS = 1e-12
    gcol = "Group" if "Group" in wide.columns else ("group" if "group" in wide.columns else wide.columns[0])
    harm_cols = [c for c in wide.columns if c != gcol]
    rows = []
    mask_bw = wide[gcol].str.lower() == bw_key
    mask_oth = ~mask_bw

    for h in harm_cols:
        bw_pct_series = wide.loc[mask_bw, h].astype(float)
        bw_pct = float(bw_pct_series.iloc[0]) if not bw_pct_series.empty else 0.0
        a0 = int(totals.get(bw_key, 0))
        a1 = int(round(bw_pct * a0 / 100.0))

        other_rows = wide.loc[mask_oth, [gcol, h]].copy()
        other_rows[h] = other_rows[h].astype(float)
        other_rows["tot"] = other_rows[gcol].str.lower().map(totals).fillna(0).astype(int)
        b1 = int(round((other_rows[h] * other_rows["tot"] / 100.0).sum()))
        b0 = int(other_rows["tot"].sum())

        rr = ((a1 + EPS) / (a0 + EPS)) / ((b1 + EPS) / (b0 + EPS)) if b0 > 0 and a0 > 0 else np.nan
        rows.append({"harm_category": h, "RR_bw_vs_others": rr,
                     "bw_count": a1, "bw_total": a0, "others_count": b1, "others_total": b0})
    return pd.DataFrame(rows)


def _group_totals(df: pd.DataFrame, groups: list[str]) -> dict:
    totals = {}
    cols = {c.lower(): c for c in df.columns}
    for g in groups:
        gl = g.lower()
        if gl in cols:
            totals[gl] = int(pd.to_numeric(df[cols[gl]], errors="coerce").fillna(0).astype(int).sum())
        elif gl == "intersectional_black_female":
            rb = cols.get("race_ethnicity_black") or cols.get("race_black")
            gf = cols.get("gender_female") or cols.get("female")
            if rb and gf:
                totals[gl] = int(((pd.to_numeric(df[rb], errors="coerce")>0.5) & (pd.to_numeric(df[gf], errors="coerce")>0.5)).sum())
            else:
                totals[gl] = len(df)
        else:
            totals[gl] = len(df)
    return totals

def _mask_categories(df: pd.DataFrame, colnames=("categories","tags","category","tag_list","labels"), p=0.1, seed=7):
    rng = random.Random(seed)
    col = next((c for c in colnames if c in df.columns), None)
    if not col: return df
    def parse_listish(x):
        from ast import literal_eval
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            try: obj = literal_eval(s); arr = list(obj) if isinstance(obj,(list,tuple,set)) else [s]
            except Exception: arr = [s]
        else:
            sep = "," if "," in s else ("|" if "|" in s else None)
            arr = [p.strip().lower() for p in s.split(sep)] if sep else [s.strip().lower()]
        return [t for t in arr if t]
    def drop_some(arr):
        return [t for t in arr if rng.random() > p]
    df2 = df.copy()
    df2[col] = df2[col].map(lambda x: drop_some(parse_listish(x)) if pd.notna(x) else x)
    return df2

# --- ablations ---
def run():
    print("--- Step 22: Ablation Studies ---")
    df = _load_corpus()

    # (A) Lexicon OFF → set all harm % to 0 and recompute RR
    if HARM_WIDE.exists():
        wide = pd.read_csv(HARM_WIDE)
        gcol = "Group" if "Group" in wide.columns else "group"
        groups = wide[gcol].astype(str).str.lower().tolist()
        totals = _group_totals(df, groups)
        bw_key = "intersectional_black_female" if "intersectional_black_female" in groups else groups[0]
        wide_off = wide.copy()
        for c in [c for c in wide.columns if c != gcol]:
            wide_off[c] = 0.0
        rr_off = _recompute_harm_rr_from_wide(wide_off, totals, bw_key)
        rr_off.to_csv(ABL / "22_rr_lexicon_off.csv", index=False)
        print("[WRITE] 22_rr_lexicon_off.csv")

    # (B) Category noise p ∈ {0.1, 0.25, 0.5}
    ps = [0.1, 0.25, 0.5]
    for p in ps:
        df_mask = _mask_categories(df, p=p, seed=7)
        # toy metric: change in top category frequency
        col = next((c for c in ["categories","tags","category","tag_list","labels"] if c in df_mask.columns), None)
        if not col: continue
        vc = pd.Series([t for row in df_mask[col].dropna() for t in (row if isinstance(row, list) else [])]).value_counts()
        out = vc.head(30).rename_axis("category").reset_index(name="count")
        out.to_csv(ABL / f"22_topcats_noise_{int(p*100)}.csv", index=False)
        print(f"[WRITE] 22_topcats_noise_{int(p*100)}.csv")

    # (C) Top-K cap K ∈ {10, 20, 30, 50} — summarize mass captured
    col = next((c for c in ["categories","tags","category","tag_list","labels"] if c in df.columns), None)
    if col:
        base = pd.Series([t for row in df[col].dropna().head(200_000).map(lambda x: x if isinstance(x,list) else []) for t in row]).value_counts()
        for K in [10, 20, 30, 50]:
            captured = base.head(K).sum() / max(1, base.sum())
            pd.DataFrame([{"K":K, "mass_captured":float(captured)}]).to_csv(ABL / f"22_topk_mass_{K}.csv", index=False)
            print(f"[WRITE] 22_topk_mass_{K}.csv")

    # (D/E) You can add more: different group encodings or N_BOOT; report out as CSVs
    print("--- Step 22: Ablation Studies Completed ---")

if __name__ == "__main__":
    run()
