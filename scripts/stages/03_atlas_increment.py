"""How much information about PT programme state does each level of spatial organisation add?

Gradient-boosted models are trained on nested descriptor sets (A; A+B; A+B+C; A+B+D;
A+B+C+D) with folds that hold out whole donors. Performance is the within-section Spearman
correlation between predicted and measured programme score, averaged with equal weight per
donor. Increments are summarised with donor-level bootstrap intervals and compared with the
same increments computed after interstitial labels were permuted within 100 um tiles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
set_global_seed(cfg["seed"])
S = stage_dir("03_atlas_increment")
M = cfg["atlas_model"]
LEVELS = {"A": ["A"], "AB": ["A", "B"], "ABC": ["A", "B", "C"], "ABD": ["A", "B", "D"], "ABCD": ["A", "B", "C", "D"],
          "ABDu": ["A", "B", "D_unit"], "ABDh": ["A", "B", "D_halo"]}


def cols(df: pd.DataFrame, levels: list[str], null: bool) -> list[str]:
    pre = "null_" if null else ""
    c = [k for k in df.columns if any(k.startswith(pre + lv + ("" if lv.startswith("D_") else "_")) for lv in levels)]
    if not null:
        c = [k for k in c if not k.startswith("null_")]
    return c + ["log_counts", "is_xenium"]


def cv_predict(df: pd.DataFrame, X: list[str], y: str, seed: int) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=M["n_folds"])
    binary = y == "is_iPT"
    for tr, te in gkf.split(df, groups=df.donor):
        if binary:
            mdl = lgb.LGBMClassifier(n_estimators=M["n_estimators"], learning_rate=M["learning_rate"],
                                     num_leaves=M["num_leaves"], min_child_samples=M["min_child_samples"],
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                     random_state=seed, verbose=-1, n_jobs=M["n_jobs"])
            mdl.fit(df[X].values[tr], df[y].values[tr])
            pred[te] = mdl.predict_proba(df[X].values[te])[:, 1]
            continue
        # within-section centring of the target removes section offsets (platform, depth)
        ytr = df[y].values[tr] - df.groupby("sample")[y].transform("mean").values[tr]
        mdl = lgb.LGBMRegressor(n_estimators=M["n_estimators"], learning_rate=M["learning_rate"],
                                num_leaves=M["num_leaves"], min_child_samples=M["min_child_samples"],
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                random_state=seed, verbose=-1, n_jobs=M["n_jobs"])
        mdl.fit(df[X].values[tr], ytr)
        pred[te] = mdl.predict(df[X].values[te])
    return pred


def per_section(df: pd.DataFrame, y: str, pred: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    out = []
    for s, g in df.assign(_p=pred).groupby("sample"):
        if g[y].std() > 0:
            v = roc_auc_score(g[y], g._p) if y == "is_iPT" else spearmanr(g[y], g._p)[0]
            if y == "is_iPT" and min(g[y].sum(), (1 - g[y]).sum()) < 20:
                continue
            out.append(dict(sample=s, donor=g.donor.iloc[0], platform=g.platform.iloc[0],
                            condition=g.condition.iloc[0], rho=v, n=len(g)))
    return pd.DataFrame(out)


def donor_mean(sec: pd.DataFrame) -> pd.Series:
    return sec.groupby("donor").rho.mean()


def boot_diff(a: pd.Series, b: pd.Series, n: int, rng) -> tuple[float, float, float, float]:
    d = (a - b).dropna()
    bs = np.array([rng.choice(d.values, len(d)).mean() for _ in range(n)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), float((d > 0).mean())


def main() -> None:
    df = pd.read_parquet(p(cfg["paths"]["interim"]) / "atlas_features" / "all_targets.parquet")
    df["is_xenium"] = (df.platform == "Xenium").astype(int)
    rng = np.random.default_rng(cfg["seed"])
    rows, secs, incs = [], [], []
    targets = [f"yres_{k}" for k in cfg["programmes"]] + [f"yunit_{k}" for k in cfg["programmes"]] + ["is_iPT"]
    for y in targets:
        d = df[df[y].notna()].reset_index(drop=True)
        perf = {}
        for null in (False, True):
            for name, lv in LEVELS.items():
                X = cols(d, lv, null)
                preds = cv_predict(d, X, y, cfg["seed"])
                sec = per_section(d, y, preds).assign(programme=y, model=name, null=null)
                secs.append(sec)
                perf[(name, null)] = donor_mean(sec)
                rows.append(dict(programme=y, model=name, null=null, n_features=len(X),
                                 donor_mean_rho=perf[(name, null)].mean(),
                                 n_donors=perf[(name, null)].notna().sum()))
        for a, b in [("AB", "A"), ("ABC", "AB"), ("ABD", "AB"), ("ABCD", "ABC"), ("ABD", "ABC"),
                     ("ABDu", "AB"), ("ABDh", "AB")]:
            for null in (False, True):
                m, lo, hi, fpos = boot_diff(perf[(a, null)], perf[(b, null)], M["n_boot"], rng)
                incs.append(dict(programme=y, contrast=f"{a}-{b}", null=null, mean=m, lo=lo, hi=hi,
                                 frac_donors_positive=fpos))
        # real increment of anatomical descriptors versus the same increment under the null
        for lv in ["ABD", "ABDu", "ABDh"]:
            real = perf[(lv, False)] - perf[("AB", False)]
            nul = perf[(lv, True)] - perf[("AB", True)]
            m, lo, hi, fpos = boot_diff(real, nul, M["n_boot"], rng)
            incs.append(dict(programme=y, contrast=f"{lv}-AB real minus null", null=False, mean=m, lo=lo,
                             hi=hi, frac_donors_positive=fpos))
    pd.DataFrame(rows).to_csv(S / "model_performance.tsv", sep="\t", index=False)
    pd.concat(secs).to_csv(S / "section_performance.tsv", sep="\t", index=False)
    pd.DataFrame(incs).to_csv(S / "increments.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).to_string())
    print(pd.DataFrame(incs).round(4).to_string())
    write_provenance("03_atlas_increment", cfg, [], [S / "model_performance.tsv", S / "increments.tsv"])


if __name__ == "__main__":
    main()
