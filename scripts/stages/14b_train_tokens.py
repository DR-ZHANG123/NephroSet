"""Participant-held-out cross-validation of models that read structure inside the spot.

Usage: python 14b_train_tokens.py <model> [<model> ...]   (models from lib.models.TOKEN_REGISTRY)

Folds, validation participants, spot roles, seeds, training and prediction format are those of
14_train_cv.py. In addition, a linear token-role head is trained per fold on the training
participants (out-of-fold for the training sections) and the token hyperedges of every section
are built from its predicted token roles.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib import tokens as TK  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

_spec = importlib.util.spec_from_file_location("cv14", Path(__file__).with_name("14_train_cv.py"))
cv14 = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(cv14)
sys.argv = _argv

cfg = load_config()
T = cfg["train"]
DEV = T["device"]
OUTC = p(cfg["paths"]["cache"]) / "cv_predictions"


def load_tokens(sections, cohort="kpmp"):
    out = {}
    for s in sections:
        z = np.load(p(cfg["paths"]["cache"]) / f"tok_{cohort}" / f"{s}.npz", allow_pickle=True)
        out[s] = (torch.as_tensor(z["tok"]), torch.as_tensor(z["col"]), z["barcode"])
    return out


def role_heads(tok, data, tr, others, seed):
    """Token roles: out-of-fold for training sections, full-training head for the rest."""
    comp = {s: torch.as_tensor(data[s]["table"][D.COMP].values, dtype=torch.float32, device=DEV)
            for s in tr}
    rng = np.random.default_rng(seed)
    grp = rng.permutation(len(tr)) % 3
    roles = {}
    trn = np.array(tr)
    for k in range(3):
        fit = list(trn[grp != k])
        h = TK.train_role_head([tok[s][0] for s in fit], [tok[s][1] for s in fit], [comp[s] for s in fit], DEV,
                               seed=seed + k)
        for s in trn[grp == k]:
            roles[s] = TK.token_roles(h, tok[s][0], tok[s][1])
    h = TK.train_role_head([tok[s][0] for s in tr], [tok[s][1] for s in tr], [comp[s] for s in tr], DEV, seed=seed)
    for s in others:
        roles[s] = TK.token_roles(h, tok[s][0], tok[s][1])
    return roles


def add_tokens(g, tk, troles, rng, fams=None):
    """fams: hyperedge families to keep on the device (all if None)."""
    tok, col, _ = tk
    n = tok.shape[0]
    # token tensors stay in pinned host memory and are moved to the device inside the model
    g["tok"] = torch.cat([tok, col], -1).reshape(n * TK.NT, -1).cpu().pin_memory()
    tissue = col[..., 0].float().cpu().numpy()
    g["tmask"] = torch.as_tensor((tissue >= TK.TISSUE_MIN).astype(np.float32).ravel()).pin_memory()
    g["tpos"] = torch.as_tensor(TK.token_pos(), dtype=torch.float32, device=DEV)
    H = TK.hyperedges(troles, tissue, rng)
    pn, ph, pt = H["profile"]
    hn, hh, ht = H["halo"]
    H["anat"] = (np.concatenate([pn, hn]), np.concatenate([ph, hh + len(pt)]), np.concatenate([pt, ht]))
    for f, (node, he, typ) in H.items():
        if fams is not None and f not in fams:
            continue
        g[f"t_{f}_node"] = torch.as_tensor(node).pin_memory()
        g[f"t_{f}_he"] = torch.as_tensor(he).pin_memory()
        g[f"t_{f}_n"] = len(typ)
        g[f"t_{f}_type"] = torch.as_tensor(typ).pin_memory()
    return g


def sections_kpmp():
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    enc = set(pd.read_csv(p("results/12_image_features_kpmp/encoded_sections.tsv"), sep="\t").section)
    sm = pd.read_csv(p("results/11b_kpmp_targets/spot_summary.tsv"), sep="\t")
    keep = set(sm.section[sm.status == "ok"]) & enc
    return idx[idx.section.isin(keep)].reset_index(drop=True)


def main() -> None:
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    from lib.models import REGISTRY
    fams = {REGISTRY[n][1].get("fam") for n in names} - {None}
    sections = sections_kpmp()
    data = D.load_cohort(p(cfg["paths"]["interim"]) / "kpmp_spots",
                         p(cfg["paths"]["cache"]) / "img_kpmp", sections.section)
    tok = load_tokens(sections.section)
    for s in sections.section:
        assert (tok[s][2] == data[s]["table"].index.values).all(), s
    log, rows = [], {n: [] for n in names}
    for fi, (tr_all, te) in enumerate(D.participant_folds(sections, T["n_folds"], cfg["seed"])):
        done = all((OUTC / f"{n}_fold{fi}.parquet").exists() for n in names)
        for name in names:
            rows[name].append(dict(model=name, fold=fi, n_test=len(te)))
        if done and "--resume" in _argv:  # every fold is seeded independently, so finished folds are reused
            continue
        rng = np.random.default_rng(cfg["seed"] + fi)
        tr, va = D.split_validation(sections, tr_all, T["val_frac"], rng)
        roles_tr = D.oof_roles(data, tr, T["role_alpha"], 3, rng)
        rm = D.role_model(data, tr, T["role_alpha"])
        roles = {**roles_tr, **{s: D.predict_roles(rm, data[s]) for s in va + te}}
        mm = np.concatenate([data[s]["morph"] for s in tr])
        mmu, msd = mm.mean(0), mm.std(0) + 1e-6
        yy = np.concatenate([data[s]["table"][D.PROGS].values for s in tr])
        ymu, ysd = yy.mean(0), yy.std(0) + 1e-6
        troles = role_heads(tok, data, tr, va + te, cfg["seed"] + fi)
        preds = {n: [] for n in names}
        for seed in cfg["seeds"]:
            gr = np.random.default_rng(seed)
            G = {}
            for s in tr + va + te:
                g = D.build_graph(data[s], roles[s], mmu, msd, ymu, ysd, T["unit_grid"][0], 1, gr, DEV)
                G[s] = add_tokens(g, tok[s], troles[s], gr, fams)
            d_in = G[tr[0]]["x"].shape[1]
            for name in names:
                model, val, ep = cv14.train_one(name, [G[s] for s in tr], [G[s] for s in va], seed, d_in)
                log.append(dict(model=name, fold=fi, seed=seed, val_rho=val, epochs=ep))
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
                        preds[name].append(df)
                print(name, fi, seed, round(val, 3), ep, time.strftime("%H:%M"), flush=True)
                del model
                torch.cuda.empty_cache()
            del G
        for name in names:
            pd.concat(preds[name]).to_parquet(OUTC / f"{name}_fold{fi}.parquet")
            rows[name][-1].update(n_train=len(tr), n_val=len(va))
    for name in names:
        S = stage_dir(f"14_cv/{name}")
        pd.DataFrame(rows[name]).to_csv(S / "folds.tsv", sep="\t", index=False)
        pd.DataFrame([r for r in log if r["model"] == name]).to_csv(S / "training_log.tsv", sep="\t", index=False)
        write_provenance(f"14_cv/{name}", cfg, [], [S / "folds.tsv"], extra={"model": name})


if __name__ == "__main__":
    main()
