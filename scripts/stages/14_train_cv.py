"""Participant-held-out cross-validation of all spot-level predictors on KPMP.

Usage: python 14_train_cv.py <model> [<model> ...]
For each outer fold (participants never shared between folds), roles are estimated by ridge
regression trained on the fold's training participants (out-of-fold for the training sections
themselves), a model is trained with early stopping on held-aside training participants, and
the held-out participants are predicted. Structure size (unit size or ring radius) is chosen on
the held-aside participants of each fold with the first seed and reused for the other seeds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib.models import REGISTRY  # noqa: E402
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
SN = "--secnorm" in sys.argv
if SN:
    sys.argv.remove("--secnorm")
SUF = "_sn" if SN else ""
T = cfg["train"]
DEV = T["device"]
OUTC = p(cfg["paths"]["cache"]) / "cv_predictions"
OUTC.mkdir(parents=True, exist_ok=True)


def within_rho(pred: np.ndarray, y: np.ndarray) -> float:
    return spearmanr(pred, y)[0] if np.std(y) > 0 and np.std(pred) > 0 else np.nan


def train_one(name, graphs_tr, graphs_va, seed, d_in):
    set_global_seed(seed)
    cls, kw = REGISTRY[name]
    model = cls(d_in, d=T["hidden"], p=T["dropout"], **kw).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    w = torch.tensor(T["programme_weights"], device=DEV)
    best, best_state, bad = -np.inf, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(T["max_epochs"]):
        model.train()
        for i in rng.permutation(len(graphs_tr)):
            g = graphs_tr[i]
            yp, lc = model(g)
            loss = ((yp - g["y"]) ** 2 * w).mean() + T["comp_weight"] * F.kl_div(lc, g["comp"], reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            r = [within_rho(model(g)[0][:, 0].cpu().numpy(), g["y"][:, 0].cpu().numpy()) for g in graphs_va]
        score = np.nanmean(r)
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= T["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def main() -> None:
    names = sys.argv[1:]
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    enc = set(pd.read_csv(p("results/12_image_features_kpmp/encoded_sections.tsv"), sep="\t").section)
    sm = pd.read_csv(p("results/11b_kpmp_targets/spot_summary.tsv"), sep="\t")
    keep = set(sm.section[sm.status == "ok"]) & enc
    sections = idx[idx.section.isin(keep)].reset_index(drop=True)
    data = D.load_cohort(p(cfg["paths"]["interim"]) / "kpmp_spots",
                         p(cfg["paths"]["cache"]) / "img_kpmp", sections.section, section_norm=SN)
    log = []
    for name in names:
        S = stage_dir(f"14_cv/{name}{SUF}")
        rows = []
        for fi, (tr_all, te) in enumerate(D.participant_folds(sections, T["n_folds"], cfg["seed"])):
            rng = np.random.default_rng(cfg["seed"] + fi)
            tr, va = D.split_validation(sections, tr_all, T["val_frac"], rng)
            roles_tr = D.oof_roles(data, tr, T["role_alpha"], 3, rng)
            rm = D.role_model(data, tr, T["role_alpha"])
            roles = {**roles_tr, **{s: D.predict_roles(rm, data[s]) for s in va + te}}
            mm = np.concatenate([data[s]["morph"] for s in tr])
            mmu, msd = mm.mean(0), mm.std(0) + 1e-6
            yy = np.concatenate([data[s]["table"][D.PROGS].values for s in tr])
            ymu, ysd = yy.mean(0), yy.std(0) + 1e-6
            grid = T["unit_grid"] if "anat" in name or name == "clique_gcn" else \
                (T["ring_grid"] if name in ("deepsets", "hgnn_ring", "set_ring") else [None])
            chosen, best_val = None, -np.inf
            preds = []
            for si, seed in enumerate(cfg["seeds"]):
                cand = grid if si == 0 else [chosen]
                for c in cand:
                    mu = c if c is not None and ("anat" in name or name == "clique_gcn") else T["unit_grid"][0]
                    rk = c if c is not None and name in ("deepsets", "hgnn_ring", "set_ring") else 1
                    gr = np.random.default_rng(seed)
                    G = {s: D.build_graph(data[s], roles[s], mmu, msd, ymu, ysd, mu, rk, gr, DEV)
                         for s in tr + va + te}
                    d_in = G[tr[0]]["x"].shape[1]
                    model, val, ep = train_one(name, [G[s] for s in tr], [G[s] for s in va], seed, d_in)
                    if si == 0 and val > best_val:
                        best_val, chosen = val, c
                        keep_model, keep_G = model, G
                    elif si > 0:
                        keep_model, keep_G = model, G
                    log.append(dict(model=name, fold=fi, seed=seed, choice=c, val_rho=val, epochs=ep))
                model, G = keep_model, keep_G
                model.eval()
                with torch.no_grad():
                    for s in te:
                        yp, lc = model(G[s])
                        yp = yp.cpu().numpy() * ysd + ymu
                        t = data[s]["table"]
                        df = pd.DataFrame(yp, columns=[f"pred_{c[2:]}" for c in D.PROGS], index=t.index)
                        cp = np.exp(lc.cpu().numpy())
                        for j, c in enumerate(D.COMP):
                            df[f"predc_{c[5:]}"] = cp[:, j]
                        df["section"], df["seed"], df["fold"] = s, seed, fi
                        preds.append(df)
            pr = pd.concat(preds)
            pr.to_parquet(OUTC / f"{name}{SUF}_fold{fi}.parquet")
            rows.append(dict(model=name, fold=fi, chosen=chosen, n_train=len(tr), n_val=len(va), n_test=len(te)))
            print(name, fi, chosen, round(best_val, 3), time.strftime("%H:%M"), flush=True)
        pd.DataFrame(rows).to_csv(S / "folds.tsv", sep="\t", index=False)
        pd.DataFrame([r for r in log if r["model"] == name]).to_csv(S / "training_log.tsv", sep="\t", index=False)
        write_provenance(f"14_cv/{name}{SUF}", cfg, [], [S / "folds.tsv"], extra={"model": name + SUF})


if __name__ == "__main__":
    main()
