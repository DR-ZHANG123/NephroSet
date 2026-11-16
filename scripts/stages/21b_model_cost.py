"""Size and computational cost of the image models.

Usage: python 21b_model_cost.py

For each model: number of trainable parameters; inference time per section (median over 20
KPMP sections, GPU, after warm-up; image encoding excluded because it is shared by all models);
time per training epoch over 40 sections; and the mean number of epochs to early stopping in
cross-validation (from the training logs). Cost of one model = epoch time scaled to the
training set x mean epochs.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib.models import REGISTRY  # noqa: E402
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

_spec = importlib.util.spec_from_file_location("m14b", Path(__file__).with_name("14b_train_tokens.py"))
m14b = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(m14b)
sys.argv = _argv

cfg = load_config()
set_global_seed(cfg["seed"])
T = cfg["train"]
DEV = T["device"]
S = stage_dir("21b_model_cost")
MODELS = ["mlp", "deepsets", "gcn", "gat", "hgnn_anat", "set_anat", "tok_mean", "tok_abmil", "tok_gat",
          "tok_hyper_anat", "tok_pred_mean", "tok_abmil_ctx", "tok_hyper_anat_ctx"]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    sections = m14b.sections_kpmp()
    tr_all, te = D.participant_folds(sections, T["n_folds"], cfg["seed"])[0]
    rng = np.random.default_rng(cfg["seed"])
    tr = list(rng.choice(tr_all, 40, replace=False))
    te = list(rng.choice(te, 20, replace=False))
    data = D.load_cohort(p(cfg["paths"]["interim"]) / "kpmp_spots", p(cfg["paths"]["cache"]) / "img_kpmp", tr + te)
    tok = m14b.load_tokens(tr + te, fold=0)
    roles = {**D.oof_roles(data, tr, T["role_alpha"], 3, rng)}
    rm = D.role_model(data, tr, T["role_alpha"])
    roles.update({s: D.predict_roles(rm, data[s]) for s in te})
    mm = np.concatenate([data[s]["morph"] for s in tr])
    yy = np.concatenate([data[s]["table"][D.PROGS].values for s in tr])
    troles = m14b.role_heads(tok, data, tr, te, cfg["seed"])
    G = {}
    for s in tr + te:
        g = D.build_graph(data[s], roles[s], mm.mean(0), mm.std(0) + 1e-6, yy.mean(0), yy.std(0) + 1e-6,
                          T["unit_grid"][0], 1, rng, DEV)
        G[s] = m14b.add_tokens(g, tok[s], troles[s], rng)
    d_in = G[tr[0]]["x"].shape[1]
    logs = []
    for f in Path(p("results/14_cv")).glob("*/training_log.tsv"):
        logs.append(pd.read_csv(f, sep="\t"))
    logs = pd.concat(logs)
    n_train = int(np.mean([len(D.split_validation(sections, a, T["val_frac"], np.random.default_rng(0))[0])
                           for a, _ in D.participant_folds(sections, T["n_folds"], cfg["seed"])]))
    rows = []
    w = torch.tensor(T["programme_weights"], device=DEV)
    for name in MODELS:
        cls, kw = REGISTRY[name]
        model = cls(d_in, d=T["hidden"], p=T["dropout"], **kw).to(DEV)
        npar = sum(x.numel() for x in model.parameters() if x.requires_grad)
        model.eval()
        with torch.no_grad():
            for s in te[:3]:
                model(G[s])
            ts = []
            for s in te:
                sync()
                t0 = time.perf_counter()
                model(G[s])
                sync()
                ts.append(time.perf_counter() - t0)
        opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
        model.train()
        sync()
        t0 = time.perf_counter()
        for s in tr:
            yp, lc = model(G[s])
            loss = ((yp - G[s]["y"]) ** 2 * w).mean() + T["comp_weight"] * F.kl_div(lc, G[s]["comp"], reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
        sync()
        ep_time = (time.perf_counter() - t0) / len(tr) * n_train
        ep = logs[logs.model == name].epochs.mean() if name in set(logs.model) else np.nan
        rows.append(dict(model=name, parameters=npar, inference_s_per_section=float(np.median(ts)),
                         epoch_s=ep_time, mean_epochs=ep, train_min_per_model=ep_time * ep / 60,
                         median_spots=int(np.median([len(data[s]["table"]) for s in te]))))
        print(rows[-1], flush=True)
        del model, opt
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(S / "model_cost.tsv", sep="\t", index=False)
    write_provenance("21b_model_cost", cfg, [], [S / "model_cost.tsv"], extra={"gpu": torch.cuda.get_device_name(0)})


if __name__ == "__main__":
    main()
