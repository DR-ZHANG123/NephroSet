"""Participant-level evaluation of spot predictions against measured programme scores."""
from __future__ import annotations

from pathlib import Path as _Path


def ext_file(ext_dir: _Path, model: str) -> _Path:
    """External predictions: the ten-seed ensemble when it exists, otherwise the three-seed one."""
    f = ext_dir / f"{model}_s10.parquet"
    return f if f.exists() else ext_dir / f"{model}.parquet"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

PROGS = ["pt_injury", "inflammation", "pt_transport"]
COMP = ["PT", "LOH", "DCT_CNT", "CD", "Glomerular", "Endothelial", "Fibroblast", "Mural", "Immune"]


def _rho(a, b):
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return spearmanr(a, b)[0]


def _resid(v: np.ndarray, C: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(C)), C[:, :-1]])  # fractions sum to one: drop one column
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


def section_metrics(merged: pd.DataFrame, pt_min: float) -> pd.DataFrame:
    """merged: one row per spot with y_<prog>, pred_<prog>, comp_<class>, section."""
    rows = []
    for s, g in merged.groupby("section"):
        C = g[[f"comp_{c}" for c in COMP]].values
        pt = g.comp_PT.values >= pt_min
        for k in PROGS:
            y, yp = g[f"y_{k}"].values, g[f"pred_{k}"].values
            rows.append(dict(section=s, programme=k, n_spots=len(g),
                             rho=_rho(y, yp),
                             rho_pt_rich=_rho(y[pt], yp[pt]) if pt.sum() >= 30 else np.nan,
                             rho_beyond_comp=_rho(_resid(y, C), _resid(yp, C)),
                             nmae=np.mean(np.abs(y - yp)) / (np.std(y) + 1e-9),
                             mean_y=y.mean(), mean_pred=yp.mean()))
    return pd.DataFrame(rows)


def participant_table(sec: pd.DataFrame, part_of: dict) -> pd.DataFrame:
    sec = sec.assign(participant=sec.section.map(part_of))
    return sec.groupby(["participant", "programme"]).mean(numeric_only=True).reset_index()


def boot_ci(v: np.ndarray, n: int, rng) -> tuple[float, float]:
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    bs = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(a: pd.Series, b: pd.Series, n: int, rng) -> dict:
    d = (a - b).dropna()
    lo, hi = boot_ci(d.values, n, rng)
    try:
        pw = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
    except ValueError:
        pw = np.nan
    return dict(diff=float(d.mean()), lo=lo, hi=hi, p_wilcoxon=float(pw), n=int(len(d)),
                frac_positive=float((d > 0).mean()))
