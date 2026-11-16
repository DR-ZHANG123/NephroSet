"""Fold-specific token projections for participant-held-out cross-validation.

Usage: python 12c_token_fold_projection.py

The token projection of 12b is fitted on KPMP sections and serves the external cohort, for
which all KPMP participants are training data. Within cross-validation, the projection must
see only the training participants of each fold. For every fold, principal components are
fitted on tokens of 24 sections drawn from that fold's training participants (same sampling
as 12b); every KPMP section is then encoded once and projected with each fold's components.
Output: <cache>/tok_kpmp_fold{f}/<section>.npz with the same fields as 12b.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib.imagefeat import Encoder  # noqa: E402
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

_spec = importlib.util.spec_from_file_location("tok12b", Path(__file__).with_name("12b_token_features.py"))
tok12b = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = [sys.argv[0], "kpmp"]
_spec.loader.exec_module(tok12b)
sys.argv = _argv

cfg = load_config()
set_global_seed(cfg["seed"])
I, T, TR = cfg["image"], cfg["tokens"], cfg["train"]
S = stage_dir("12c_token_fold_projection")
CACHE = p(cfg["paths"]["cache"])


def sections_kpmp() -> pd.DataFrame:
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    enc = set(pd.read_csv(p("results/12_image_features_kpmp/encoded_sections.tsv"), sep="\t").section)
    sm = pd.read_csv(p("results/11b_kpmp_targets/spot_summary.tsv"), sep="\t")
    keep = set(sm.section[sm.status == "ok"]) & enc
    return idx[idx.section.isin(keep)].reset_index(drop=True)


def fit_projection(enc, fit_sections, rng) -> dict:
    bs = I["batch"]
    chunk = Parallel(n_jobs=I["prefetch"], max_nbytes=None)(delayed(tok12b.crops)(s) for s in fit_sections)
    acc = []
    for _, c1, _ in chunk:
        idx = rng.choice(len(c1), min(len(c1), T["n_fit_spots"]), replace=False)
        for j in range(0, len(idx), bs):
            h = tok12b.tokens(enc, c1[idx[j:j + bs]]).reshape(-1, 1024)
            acc.append(h[torch.randperm(h.shape[0])[:4000]].cpu())
    X = torch.cat(acc).double()
    mu = X.mean(0)
    _, Sv, V = torch.linalg.svd(X - mu, full_matrices=False)
    k = T["n_pcs"]
    return dict(mu=mu.float().to(enc.device), W=V[:k].T.float().to(enc.device),
                var_kept=float((Sv[:k] ** 2).sum() / (Sv ** 2).sum()), n_tokens=int(X.shape[0]))


def main() -> None:
    sections = sections_kpmp()
    folds = D.participant_folds(sections, TR["n_folds"], cfg["seed"])
    enc = Encoder(I["encoder"], I["device"])
    proj, rows = {}, []
    for fi, (tr_all, te) in enumerate(folds):
        rng = np.random.default_rng(cfg["seed"] + 100 * (fi + 1))
        fit = list(rng.choice(sorted(tr_all), T["n_fit_sections"], replace=False))
        assert not set(fit) & set(te)
        proj[fi] = fit_projection(enc, fit, rng)
        rows.append(dict(fold=fi, n_fit_sections=len(fit), n_tokens=proj[fi]["n_tokens"],
                         variance_kept=proj[fi]["var_kept"], fit_sections=",".join(fit)))
        print("fold", fi, "variance kept", round(proj[fi]["var_kept"], 4), flush=True)
    outs = {fi: CACHE / f"tok_kpmp_fold{fi}" for fi in proj}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)
    todo = [s for s in sections.section if not all((outs[f] / f"{s}.npz").exists() for f in outs)]
    bs = I["batch"]
    for i in range(0, len(todo), I["prefetch"]):
        chunk = Parallel(n_jobs=I["prefetch"], max_nbytes=None)(
            delayed(tok12b.crops)(s) for s in todo[i:i + I["prefetch"]])
        for sid, c1, bc in chunk:
            hs = [tok12b.tokens(enc, c1[j:j + bs]) for j in range(0, len(c1), bs)]
            col = tok12b.colour(c1).astype(np.float16)
            for fi, pr in proj.items():
                z = np.concatenate([((h - pr["mu"]) @ pr["W"]).cpu().numpy() for h in hs])
                np.savez(outs[fi] / f"{sid}.npz", tok=z.astype(np.float16), col=col, barcode=bc)
            print(sid, len(bc), flush=True)
    pd.DataFrame(rows).to_csv(S / "fold_projections.tsv", sep="\t", index=False)
    write_provenance("12c_token_fold_projection", cfg, [], [S / "fold_projections.tsv"],
                     extra={"encoder": I["encoder"], "n_sections": len(sections), "n_pcs": T["n_pcs"]})


if __name__ == "__main__":
    main()
