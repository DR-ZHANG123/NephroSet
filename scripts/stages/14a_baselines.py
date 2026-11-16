"""Non-neural reference predictors in the same participant folds as stage 14.

* ridge_image      : ridge regression on frozen encoder features (two scales) + morphology
* gbm_comp_morph   : gradient boosting on image-predicted class fractions + morphology
                     (can programme scores be read off tissue composition seen in the image?)
* gbm_measured_comp: gradient boosting on the measured (sequencing-derived) class fractions,
                     an upper reference for what composition alone explains; not an image model
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
SN = "--secnorm" in sys.argv
if SN:
    sys.argv.remove("--secnorm")
SUF = "_sn" if SN else ""
set_global_seed(cfg["seed"])
T = cfg["train"]
OUTC = p(cfg["paths"]["cache"]) / "cv_predictions"
OUTC.mkdir(parents=True, exist_ok=True)


def gbm(seed):
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                             verbose=-1, n_jobs=32)


def main() -> None:
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    enc = set(pd.read_csv(p("results/12_image_features_kpmp/encoded_sections.tsv"), sep="\t").section)
    sm = pd.read_csv(p("results/11b_kpmp_targets/spot_summary.tsv"), sep="\t")
    sections = idx[idx.section.isin(set(sm.section[sm.status == "ok"]) & enc)].reset_index(drop=True)
    data = D.load_cohort(p(cfg["paths"]["interim"]) / "kpmp_spots", p(cfg["paths"]["cache"]) / "img_kpmp",
                         sections.section, section_norm=SN)
    preds = {k: [] for k in ["ridge_image", "gbm_comp_morph", "gbm_measured_comp"]}
    for fi, (tr_all, te) in enumerate(D.participant_folds(sections, T["n_folds"], cfg["seed"])):
        rng = np.random.default_rng(cfg["seed"] + fi)
        tr, va = D.split_validation(sections, tr_all, T["val_frac"], rng)
        tr = tr + va  # no early stopping needed
        X = lambda ss: np.concatenate([np.concatenate([data[s]["x"], data[s]["morph"]], 1) for s in ss])
        Y = lambda ss: np.concatenate([data[s]["table"][D.PROGS].values for s in ss])
        idx_te = np.concatenate([np.full(len(data[s]["table"]), s) for s in te])
        bc_te = np.concatenate([data[s]["table"].index.values for s in te])
        rid = Ridge(alpha=T["ridge_alpha"]).fit(X(tr), Y(tr))
        out = {"ridge_image": rid.predict(X(te))}
        roles_tr = D.oof_roles(data, tr, T["role_alpha"], 3, rng)
        rm = D.role_model(data, tr, T["role_alpha"])
        Z = lambda ss, rr: np.concatenate([np.concatenate([rr[s], data[s]["morph"]], 1) for s in ss])
        roles_te = {s: D.predict_roles(rm, data[s]) for s in te}
        Mc = lambda ss: np.concatenate([data[s]["table"][D.COMP].values for s in ss])
        out["gbm_comp_morph"] = np.stack([np.mean([gbm(sd).fit(Z(tr, roles_tr), Y(tr)[:, j]).predict(Z(te, roles_te))
                                                   for sd in cfg["seeds"]], 0) for j in range(3)], 1)
        out["gbm_measured_comp"] = np.stack([np.mean([gbm(sd).fit(Mc(tr), Y(tr)[:, j]).predict(Mc(te))
                                                      for sd in cfg["seeds"]], 0) for j in range(3)], 1)
        for k, v in out.items():
            df = pd.DataFrame(v, columns=[f"pred_{c[2:]}" for c in D.PROGS], index=bc_te)
            df.index.name = "barcode"
            df["section"], df["seed"], df["fold"] = idx_te, 0, fi
            if k == "gbm_comp_morph":
                rc = np.concatenate([roles_te[s] for s in te])
                for j, c in enumerate(D.COMP):
                    df[f"predc_{c[5:]}"] = rc[:, j]
            preds[k].append(df)
        print("fold", fi, flush=True)
    for k, v in preds.items():
        for fi in range(T["n_folds"]):
            pd.concat(v)[lambda d: d.fold == fi].to_parquet(OUTC / f"{k}{SUF}_fold{fi}.parquet")
        S = stage_dir(f"14_cv/{k}{SUF}")
        (S / "done.json").write_text('{"done": true}')
        write_provenance(f"14_cv/{k}{SUF}", cfg, [], [S / "done.json"], extra={"model": k + SUF})


if __name__ == "__main__":
    main()
