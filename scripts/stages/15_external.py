"""Train each model once on all KPMP participants and predict the external cohort (GSE211785).

The structure size of each model is the value chosen most often across the cross-validation
folds. Nothing about the external cohort is used before its predictions are written.
Usage: python 15_external.py <model> [<model> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
cfg = load_config()
# --seeds10: ten training seeds instead of three (external estimates are seed-sensitive)
S10 = "--seeds10" in sys.argv
if S10:
    sys.argv.remove("--seeds10")
    cfg["seeds"] = [cfg["seed"] + i for i in range(10)]
SN = "--secnorm" in sys.argv
if SN:
    sys.argv.remove("--secnorm")
SUF = "_sn" if SN else ""
T = cfg["train"]
DEV = T["device"]
OUT = p(cfg["paths"]["cache"]) / "external_predictions"
OUT.mkdir(parents=True, exist_ok=True)


def chosen(name):
    f = p(f"results/14_cv/{name}{SUF}/folds.tsv")
    if not f.exists():
        return None
    c = pd.read_csv(f, sep="\t").chosen.dropna()
    return None if c.empty else int(c.mode().iloc[0])


def main() -> None:
    from importlib import import_module
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    tcv = import_module("14_train_cv")
    names = sys.argv[1:]
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    enc = set(pd.read_csv(p("results/12_image_features_kpmp/encoded_sections.tsv"), sep="\t").section)
    sm = pd.read_csv(p("results/11b_kpmp_targets/spot_summary.tsv"), sep="\t")
    sections = idx[idx.section.isin(set(sm.section[sm.status == "ok"]) & enc)].reset_index(drop=True)
    kd = D.load_cohort(p(cfg["paths"]["interim"]) / "kpmp_spots", p(cfg["paths"]["cache"]) / "img_kpmp",
                       sections.section, section_norm=SN)
    ext = pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t").section.tolist()
    xd = D.load_cohort(p(cfg["paths"]["interim"]) / "gse211785_spots", p(cfg["paths"]["cache"]) / "img_gse211785",
                       ext, section_norm=SN)
    rng = np.random.default_rng(cfg["seed"])
    tr, va = D.split_validation(sections, list(sections.section), T["val_frac"], rng)
    roles_tr = D.oof_roles(kd, tr, T["role_alpha"], 3, rng)
    rm = D.role_model(kd, tr, T["role_alpha"])
    roles = {**roles_tr, **{s: D.predict_roles(rm, kd[s]) for s in va}}
    roles_x = {s: D.predict_roles(rm, xd[s]) for s in ext}
    mm = np.concatenate([kd[s]["morph"] for s in tr])
    mmu, msd = mm.mean(0), mm.std(0) + 1e-6
    yy = np.concatenate([kd[s]["table"][D.PROGS].values for s in tr])
    ymu, ysd = yy.mean(0), yy.std(0) + 1e-6
    for name in names:
        preds = []
        if name in ("ridge_image", "gbm_comp_morph", "gbm_measured_comp"):
            X = lambda dd, ss: np.concatenate([np.concatenate([dd[s]["x"], dd[s]["morph"]], 1) for s in ss])
            Y = np.concatenate([kd[s]["table"][D.PROGS].values for s in tr + va])
            if name == "ridge_image":
                yp = Ridge(alpha=T["ridge_alpha"]).fit(X(kd, tr + va), Y).predict(X(xd, ext))
            else:
                if name == "gbm_comp_morph":
                    Ztr = np.concatenate([np.concatenate([roles[s], kd[s]["morph"]], 1) for s in tr + va])
                    Zte = np.concatenate([np.concatenate([roles_x[s], xd[s]["morph"]], 1) for s in ext])
                else:
                    Ztr = np.concatenate([kd[s]["table"][D.COMP].values for s in tr + va])
                    Zte = np.concatenate([xd[s]["table"][D.COMP].values for s in ext])
                yp = np.stack([np.mean([lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                                          min_child_samples=50, subsample=0.8, subsample_freq=1,
                                                          colsample_bytree=0.8, random_state=sd, verbose=-1,
                                                          n_jobs=32).fit(Ztr, Y[:, j]).predict(Zte)
                                        for sd in cfg["seeds"]], 0) for j in range(3)], 1)
            bc = np.concatenate([xd[s]["table"].index.values for s in ext])
            sec = np.concatenate([np.full(len(xd[s]["table"]), s) for s in ext])
            df = pd.DataFrame(yp, columns=[f"pred_{c[2:]}" for c in D.PROGS], index=bc)
            df.index.name = "barcode"
            df["section"], df["seed"] = sec, 0
            preds.append(df)
        else:
            c = chosen(name)
            mu = c if c is not None and ("anat" in name or name == "clique_gcn") else T["unit_grid"][0]
            rk = c if c is not None and name in ("deepsets", "hgnn_ring", "set_ring") else 1
            for seed in cfg["seeds"]:
                gr = np.random.default_rng(seed)
                G = {s: D.build_graph(kd[s], roles[s], mmu, msd, ymu, ysd, mu, rk, gr, DEV) for s in tr + va}
                GX = {s: D.build_graph(xd[s], roles_x[s], mmu, msd, ymu, ysd, mu, rk, gr, DEV) for s in ext}
                model, val, ep = tcv.train_one(name, [G[s] for s in tr], [G[s] for s in va], seed,
                                               G[tr[0]]["x"].shape[1])
                model.eval()
                with torch.no_grad():
                    for s in ext:
                        yp, lc = model(GX[s])
                        df = pd.DataFrame(yp.cpu().numpy() * ysd + ymu, columns=[f"pred_{c[2:]}" for c in D.PROGS],
                                          index=xd[s]["table"].index)
                        df["section"], df["seed"] = s, seed
                        preds.append(df)
                print(name, seed, round(val, 3), ep, flush=True)
        pd.concat(preds).to_parquet(OUT / f"{name}{SUF}{'_s10' if S10 else ''}.parquet")
        S = stage_dir(f"15_external/{name}{SUF}{'_s10' if S10 else ''}")
        (S / "done.json").write_text('{"done": true}')
        write_provenance(f"15_external/{name}{SUF}", cfg, [], [S / "done.json"], extra={"model": name + SUF})


if __name__ == "__main__":
    main()
