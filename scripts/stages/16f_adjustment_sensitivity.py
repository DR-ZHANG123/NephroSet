"""Is the information beyond composition robust to how composition is modelled and measured?

Usage: python 16f_adjustment_sensitivity.py

For held-out KPMP participants (cross-validated predictions) and the external donors, agreement
between predicted and measured programme scores is recomputed after removing composition in
several ways, within each section:
  linear       linear regression on the nine measured class fractions (the main analysis)
  poly2        second-degree polynomial with all pairwise interactions (ridge, alpha 1)
  gbm          gradient-boosted trees on the fractions, cross-fitted over five random spot
               folds within the section, so residuals are out-of-sample
  m20 / m80    linear, composition from 20 or 80 markers per class
  fine         linear, composition over 16 reference subclasses
  score        linear, marker-expression scores instead of deconvolved fractions
  fine_gbm     cross-fitted trees on the 16-subclass composition
and, without adjustment, within spots whose measured PT fraction is at least 0.5, 0.7 or 0.8.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import COMP, PROGS, boot_ci, ext_file, paired  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
E = cfg["evaluate"]
S = stage_dir("16f_adjustment_sensitivity")
CACHE = p(cfg["paths"]["cache"])
INT = p(cfg["paths"]["interim"])
MODELS = ["tok_abmil_ctx", "tok_abmil", "mlp", "ridge_image", "gbm_comp_morph", "gbm_measured_comp"]
PT_MIN = [0.5, 0.7, 0.8]


def rho(a, b):
    if len(a) < 30 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return spearmanr(a, b)[0]


def r_lin(v, C):
    X = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


def r_poly(v, C):
    X = PolynomialFeatures(2, include_bias=False).fit_transform(C)
    return v - Ridge(alpha=1.0).fit(X, v).predict(X)


def r_gbm(v, C, seed):
    out = np.zeros_like(v)
    for a, b in KFold(5, shuffle=True, random_state=seed).split(C):
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, random_state=seed, verbose=-1, n_jobs=1)
        out[b] = v[b] - m.fit(C[a], v[a]).predict(C[b])
    return out


def section(cohort: str, s: str, preds: dict) -> list[dict]:
    t = pd.read_parquet(INT / f"{cohort}_spots" / f"{s}.parquet")
    alt = pd.read_parquet(CACHE / "comp_alt" / cohort / f"{s}.parquet").loc[t.index]
    C = t[[f"comp_{c}" for c in COMP]].values
    covs = {"linear": C, "poly2": C, "gbm": C,
            "m20": alt.filter(like="m20_").values, "m80": alt.filter(like="m80_").values,
            "fine": alt.filter(like="fine_").values, "score": alt.filter(like="score_").values,
            "fine_gbm": alt.filter(like="fine_").values}
    seed = cfg["seed"]

    def resid(v, how):
        X = covs[how]
        if how in ("gbm", "fine_gbm"):
            return r_gbm(v, X, seed)
        if how == "poly2":
            return r_poly(v, X)
        return r_lin(v, X)
    rows = []
    for k in PROGS:
        y = t[f"y_{k}"].values
        ry = {h: resid(y, h) for h in covs}
        for m, pr in preds.items():
            if s not in pr:
                continue
            yp = pr[s].loc[t.index, f"pred_{k}"].values
            row = dict(cohort=cohort, section=s, programme=k, model=m, unadjusted=rho(y, yp))
            for h in covs:
                row[h] = rho(ry[h], resid(yp, h))
            for q in PT_MIN:
                w = t.comp_PT.values >= q
                row[f"pt_rich_{q}"] = rho(y[w], yp[w])
                row[f"n_pt_rich_{q}"] = int(w.sum())
            rows.append(row)
    return rows


def load_preds(cohort: str) -> dict:
    out = {}
    for m in MODELS:
        if cohort == "kpmp":
            fs = sorted((CACHE / "cv_predictions").glob(f"{m}_fold*.parquet"))
            if not fs:
                continue
            pr = pd.concat([pd.read_parquet(f) for f in fs])
        else:
            f = ext_file(CACHE / "external_predictions", m)
            if not f.exists():
                continue
            pr = pd.read_parquet(f)
        pr = pr.reset_index().groupby(["section", "barcode"]).mean(numeric_only=True)
        out[m] = {s: g.droplevel(0) for s, g in pr.groupby(level=0)}
    return out


def main() -> None:
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    part = dict(zip(idx.section, idx.participant))
    rows = []
    for cohort in ("kpmp", "gse211785"):
        preds = load_preds(cohort)
        secs = sorted(set().union(*[set(v) for v in preds.values()]))
        res = Parallel(n_jobs=48)(delayed(section)(cohort, s, {m: {s: v[s]} for m, v in preds.items() if s in v})
                                       for s in secs)
        rows += [r for rr in res for r in rr]
    df = pd.DataFrame(rows)
    df["participant"] = [part.get(s, s) for s in df.section]
    df.to_csv(S / "section_metrics.tsv", sep="\t", index=False)
    metrics = ["unadjusted", "linear", "poly2", "gbm", "m20", "m80", "fine", "score", "fine_gbm"] + \
        [f"pt_rich_{q}" for q in PT_MIN]
    pp = df.groupby(["cohort", "model", "programme", "participant"])[metrics].mean().reset_index()
    rng = np.random.default_rng(cfg["seed"])
    summ, cons = [], []
    for (c, m, k), q in pp.groupby(["cohort", "model", "programme"]):
        for col in metrics:
            lo, hi = boot_ci(q[col].values, E["n_boot"], rng)
            summ.append(dict(cohort=c, model=m, programme=k, metric=col, mean=q[col].mean(), lo=lo, hi=hi,
                             n=int(q[col].notna().sum())))
    for c in ("kpmp", "gse211785"):
        for b in ["gbm_measured_comp", "gbm_comp_morph", "mlp"]:
            for k in PROGS:
                for col in metrics:
                    a_ = pp[(pp.cohort == c) & (pp.model == "tok_abmil_ctx") & (pp.programme == k)].set_index("participant")[col]
                    b_ = pp[(pp.cohort == c) & (pp.model == b) & (pp.programme == k)].set_index("participant")[col]
                    if len(a_) and len(b_):
                        cons.append(dict(cohort=c, a="tok_abmil_ctx", b=b, programme=k, metric=col,
                                         **paired(a_, b_, E["n_boot"], rng)))
    Sm = pd.DataFrame(summ)
    Sm.to_csv(S / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(cons).to_csv(S / "contrasts.tsv", sep="\t", index=False)
    print(Sm[(Sm.model == "tok_abmil_ctx")].pivot_table(index=["cohort", "metric"], columns="programme",
                                                        values="mean").round(3).to_string())
    write_provenance("16f_adjustment_sensitivity", cfg, [], [S / "summary.tsv", S / "contrasts.tsv"])


if __name__ == "__main__":
    main()
