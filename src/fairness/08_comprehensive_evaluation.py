#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_comprehensive_evaluation.py

Now robust to predictions without 'id':
- If 'id' missing, try to attach from outputs/data/val_ids.csv (length must match).
- If still no 'id', compute overall metrics & calibration; skip group-wise metrics.

Outputs:
  CSV:
    outputs/data/fairness_overall_metrics_with_ci.csv
    outputs/data/fairness_group_metrics_with_ci.csv   (only if group info available)
  TEX:
    dissertation/auto_tables/fairness_overall_metrics_with_ci.tex
    dissertation/auto_tables/fairness_group_metrics_with_ci.tex  (only if available)
  FIG:
    outputs/figures/fairness/reliability_<model>_overall_(dark|light).png
    outputs/figures/fairness/reliability_<model>_<group>=<value>_(dark|light).png
    outputs/figures/fairness/confmat_<group>=<value>.png
"""

from __future__ import annotations

# ---------- stdlib ----------
import json
import re
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

try:
    from src.fairness.fairness_evaluation_utils import compute_metrics_by_group
except Exception:
    compute_metrics_by_group = None


# ---------------------- Configuration ----------------------
SEED = 75
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "outputs" / "data"
FIG_ROOT = ROOT / "outputs" / "figures" / "fairness"
FIG_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / "dissertation" / "auto_tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

ML_CORPUS = DATA_DIR / "ml_corpus.parquet"
VAL_IDS = DATA_DIR / "val_ids.csv"  # optional, used to backfill 'id'

PRED_FILES = [
    DATA_DIR / "rf_baseline_val_predictions.csv",
    DATA_DIR / "rf_inprocessing_val_predictions.csv",
    DATA_DIR / "rf_postprocessing_val_predictions.csv",
    DATA_DIR / "rf_reweighed_val_predictions.csv",
    DATA_DIR / "bert_baseline_val_predictions.csv",
]

N_BINS = 10
ALPHA_CI = 0.95
N_BOOT = 1000


# ---------------------- Helpers ----------------------
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


def _load_val_ids() -> Optional[pd.Series]:
    if not VAL_IDS.exists():
        return None
    df = pd.read_csv(VAL_IDS)
    # try common id column names
    for cand in ["id", "video_id", "Id", "ID"]:
        if cand in df.columns:
            s = df[cand].astype(str)
            if len(s) > 0:
                return s
    # if single unnamed column, take it
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str)
    return None


def _load_predictions(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)

    cols = {c.lower(): c for c in df.columns}
    # Must have y_true/y_pred
    for need in ("y_true", "y_pred"):
        if need not in cols and need.upper() not in df.columns:
            print(f"[WARN] {path.name} missing column: {need} — skipping.")
            return None
    y_true_col = cols.get("y_true", "y_true")
    y_pred_col = cols.get("y_pred", "y_pred")

    out = pd.DataFrame({
        "y_true": df[y_true_col],
        "y_pred": df[y_pred_col],
    })

    # Attach 'id' when present; otherwise try val_ids.csv
    id_col = cols.get("id")
    if id_col:
        out["id"] = df[id_col].astype(str)
    else:
        val_ids = _load_val_ids()
        if val_ids is not None and len(val_ids) == len(out):
            out["id"] = val_ids.values
            print(f"[INFO] {path.name}: 'id' backfilled from val_ids.csv")
        else:
            print(f"[INFO] {path.name}: proceeding without 'id' (overall metrics only).")

    # Probability information (optional)
    if "y_proba" in cols:
        out["y_proba"] = pd.to_numeric(df[cols["y_proba"]], errors="coerce")
    elif "proba" in cols:
        out["proba"] = df[cols["proba"]]
    else:
        proba_cols = [c for c in df.columns if str(c).startswith("proba_")]
        if proba_cols:
            out = out.join(df[proba_cols])

    return out


def _join_groups(df_pred: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if "id" not in df_pred.columns or not ML_CORPUS.exists():
        return df_pred, []
    base = pd.read_parquet(ML_CORPUS, columns=["id", "race_ethnicity", "gender", "sexual_orientation"])
    base["id"] = base["id"].astype(str)
    out = df_pred.merge(base, on="id", how="left")
    gcols = [c for c in ["race_ethnicity", "gender", "sexual_orientation"] if c in out.columns]
    return out, gcols


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for lab in labels:
        tp = np.sum((y_true == lab) & (y_pred == lab))
        fp = np.sum((y_true != lab) & (y_pred == lab))
        fn = np.sum((y_true == lab) & (y_pred != lab))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")


def _bootstrap_ci(metric_fn, y_true, y_pred, n=N_BOOT, alpha=ALPHA_CI) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_obs = len(y_true)
    idx = np.arange(n_obs)
    vals = []
    for _ in range(n):
        samp = np.random.choice(idx, size=n_obs, replace=True)
        vals.append(metric_fn(y_true[samp], y_pred[samp]))
    vals = np.array(vals, dtype=float)
    point = metric_fn(y_true, y_pred)
    lo, hi = np.quantile(vals, [(1 - alpha) / 2.0, 1 - (1 - alpha) / 2.0])
    return float(point), float(lo), float(hi)


def _parse_probabilities(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    proba_cols = [c for c in df.columns if str(c).startswith("proba_")]
    if proba_cols:
        labs = [c.replace("proba_", "", 1) for c in proba_cols]
        proba = df[proba_cols].to_numpy(dtype=float)
        return proba, labs

    if "proba" in df.columns:
        try:
            parsed = df["proba"].apply(lambda s: np.array(json.loads(s) if isinstance(s, str) else s, dtype=float))
            k = int(parsed.iloc[0].shape[0])
            proba = np.vstack(parsed.values)
            labs = sorted(pd.unique(np.concatenate([df["y_true"].to_numpy(), df["y_pred"].to_numpy()])))
            if len(labs) != k:
                labs = [f"class{i}"] * k
            return proba, labs
        except Exception:
            pass

    if "y_proba" in df.columns:
        p = pd.to_numeric(df["y_proba"], errors="coerce").to_numpy()
        proba = np.vstack([1.0 - p, p]).T
        labs = sorted(pd.unique(np.concatenate([df["y_true"].to_numpy(), df["y_pred"].to_numpy()])))[:2]
        if len(labs) < 2:
            labs = ["neg","pos"]
        return proba, labs

    return None, None


def _ece(probs: np.ndarray, y_true_idx: np.ndarray, n_bins: int = N_BINS) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _brier_score(probs: np.ndarray, y_true_idx: np.ndarray, k: int) -> float:
    y = np.zeros_like(probs)
    y[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    return float(np.mean((y - probs) ** 2))


def _reliability_plot(probs: np.ndarray, y_true_idx: np.ndarray, title: str, fpath_dark: Path, fpath_light: Path) -> None:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true_idx).astype(float)
    bins = np.linspace(0, 1, N_BINS + 1)

    xs, ys = [], []
    for i in range(N_BINS):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())

    for dark, path in [(True, fpath_dark), (False, fpath_light)]:
        set_mpl_theme(dark=dark)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
        ax.scatter(xs, ys, s=60)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Observed accuracy")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[PLOT] {path}")


def _safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "_", str(x))


def _confusion_by_group(df: pd.DataFrame, group_col: str, group_val) -> None:
    sub = df.loc[df[group_col] == group_val]
    if sub.empty:
        return
    y_true = sub["y_true"].to_numpy()
    y_pred = sub["y_pred"].to_numpy()
    labels = sorted(pd.unique(np.concatenate([y_true, y_pred])))
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    mat = np.zeros((k, k), dtype=float)
    for yt, yp in zip(y_true, y_pred):
        mat[idx[yt], idx[yp]] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    mat = mat / row_sums

    set_mpl_theme(dark=True)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(mat, aspect="auto", origin="upper")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(f"Confusion — {group_col}={group_val}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fpath = FIG_ROOT / f"confmat_{_safe_name(group_col)}={_safe_name(group_val)}.png"
    fig.savefig(fpath, dpi=200)
    plt.close(fig)
    print(f"[PLOT] {fpath}")


# ---------------------- Evaluation ----------------------
def evaluate_one_model(df: pd.DataFrame, model_name: str, group_cols: List[str]) -> Tuple[Dict, Optional[pd.DataFrame]]:
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    acc_p, acc_lo, acc_hi = _bootstrap_ci(_accuracy, y_true, y_pred)
    f1_p, f1_lo, f1_hi = _bootstrap_ci(_macro_f1, y_true, y_pred)

    overall = {
        "model": model_name,
        "accuracy": _round(acc_p, 4), "accuracy_lo": _round(acc_lo, 4), "accuracy_hi": _round(acc_hi, 4),
        "macro_f1": _round(f1_p, 4), "macro_f1_lo": _round(f1_lo, 4), "macro_f1_hi": _round(f1_hi, 4),
    }

    gm_df = None
    if compute_metrics_by_group is not None and group_cols:
        # Single pass (no-bootstrap inside to keep runtime reasonable) — already improved overall via bootstrap above
        frames = []
        for gcol in group_cols:
            gm = compute_metrics_by_group(df, group_col=gcol)
            gm.insert(0, "model", model_name)
            gm.insert(1, "group_by", gcol)
            frames.append(gm)
        gm_df = pd.concat(frames, ignore_index=True) if frames else None

    return overall, gm_df


def main() -> None:
    models = []
    group_metrics_all = []

    for p in PRED_FILES:
        dfp = _load_predictions(p)
        if dfp is None:
            continue

        # Attach groups (if id available)
        dfp, gcols = _join_groups(dfp)
        model_name = p.stem.replace("_val_predictions", "")

        # Evaluate + (optionally) group metrics
        overall, gm = evaluate_one_model(dfp, model_name, gcols)
        models.append(overall)
        if gm is not None and not gm.empty:
            group_metrics_all.append(gm)

        # Calibration plots if probabilities exist
        probs, labels = _parse_probabilities(dfp)
        if probs is not None and labels is not None:
            # map y_true to indices in labels
            lab_to_idx = {lab: i for i, lab in enumerate(labels)}
            y_true_idx = np.array([lab_to_idx.get(y, -1) for y in dfp["y_true"].to_numpy()])
            mask = y_true_idx >= 0
            probs_v = probs[mask]
            y_true_v = y_true_idx[mask]

            # Overall reliability
            f_dark = FIG_ROOT / f"reliability_{_safe_name(model_name)}_overall_dark.png"
            f_light = FIG_ROOT / f"reliability_{_safe_name(model_name)}_overall_light.png"
            _reliability_plot(probs_v, y_true_v,
                              title=f"Reliability — {model_name} (overall)",
                              fpath_dark=f_dark, fpath_light=f_light)
            try:
                ece = _ece(probs_v, y_true_v, n_bins=N_BINS)
                models[-1]["ece"] = _round(ece, 4)
            except Exception:
                models[-1]["ece"] = np.nan
            try:
                models[-1]["brier"] = _round(_brier_score(probs_v, y_true_v, probs_v.shape[1]), 4)
            except Exception:
                models[-1]["brier"] = np.nan

            # Per-group reliability (first available group column only)
            if gcols:
                gcol = gcols[0]
                for gv, sub in dfp.loc[mask].groupby(gcol):
                    gmask = (dfp.index.isin(sub.index)) & mask
                    probs_g = probs[gmask]
                    y_true_g = y_true_idx[gmask]
                    if len(probs_g) == 0:
                        continue
                    f_dark = FIG_ROOT / f"reliability_{_safe_name(model_name)}_{_safe_name(gcol)}={_safe_name(gv)}_dark.png"
                    f_light = FIG_ROOT / f"reliability_{_safe_name(model_name)}_{_safe_name(gcol)}={_safe_name(gv)}_light.png"
                    _reliability_plot(probs_g, y_true_g,
                                      title=f"Reliability — {model_name} ({gcol}={gv})",
                                      fpath_dark=f_dark, fpath_light=f_light)
                    _confusion_by_group(dfp, gcol, gv)
        else:
            print(f"[INFO] {p.name}: no probability distributions found; skipping reliability/ECE/Brier.")

    # Save overall metrics
    overall_df = pd.DataFrame(models)
    overall_csv = DATA_DIR / "fairness_overall_metrics_with_ci.csv"
    overall_df.to_csv(overall_csv, index=False)
    print(f"[WRITE] {overall_csv}")

    overall_tex = TABLE_DIR / "fairness_overall_metrics_with_ci.tex"
    try:
        if TABLES == "v1":
            write_latex_table(overall_df, overall_tex)
        else:
            with open(overall_tex, "w", encoding="utf-8") as f:
                f.write(overall_df.to_latex(index=False))
        print(f"[TEX]   {overall_tex}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")

    # Save group metrics if available
    if group_metrics_all:
        gm_df = pd.concat(group_metrics_all, ignore_index=True)
        gm_csv = DATA_DIR / "fairness_group_metrics_with_ci.csv"
        gm_df.to_csv(gm_csv, index=False)
        print(f"[WRITE] {gm_csv}")

        gm_tex = TABLE_DIR / "fairness_group_metrics_with_ci.tex"
        try:
            if TABLES == "v1":
                write_latex_table(gm_df, gm_tex)
            else:
                with open(gm_tex, "w", encoding="utf-8") as f:
                    f.write(gm_df.to_latex(index=False))
            print(f"[TEX]   {gm_tex}")
        except Exception as e:
            print(f"[WARN] LaTeX export failed: {e}")
    else:
        print("[INFO] No group metrics written (missing id or group columns).")

    print("[DONE] 08_comprehensive_evaluation — CIs + calibration complete.")


if __name__ == "__main__":
    main()
