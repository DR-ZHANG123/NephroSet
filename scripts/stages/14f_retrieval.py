"""Same-encoder retrieval reference: nearest training spots in the Phikon-v2 feature space.

Usage: python 14f_retrieval.py            (cross-validation on KPMP and the external cohort)

This is the prediction rule of BLEEP (average of the measured programme scores of the k
nearest training spots, k = 50) applied to the frozen encoder used by all models of this study,
so that the comparison with the published method separates the encoder from the rule.
Features (both scales and morphology) are standardized with training statistics and projected
onto 256 principal components fitted on training spots; similarity is cosine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
set_global_seed(cfg["seed"])
T = cfg["train"]
K, NPC = 50, 256
S = stage_dir("14f_retrieval")
OUTC = p(cfg["paths"]["cache"]) / "cv_predictions"
EXT = p(cfg["paths"]["cache"]) / "external_predictions"
DEV = T["device"]
NAME = "knn_encoder"


def feats(data, ss):
    return np.concatenate([np.concatenate([data[s]["x"], data[s]["morph"]], 1) for s in ss])


def fit_space(X: np.ndarray, rng):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    sub = X[rng.choice(len(X), min(len(X), 60000), replace=False)]
    Z = torch.as_tensor((sub - mu) / sd, device=DEV)
    _, _, V = torch.linalg.svd(Z - Z.mean(0), full_matrices=False)
    W = V[:NPC].T.contiguous()

    def proj(A):
        z = torch.as_tensor((A - mu) / sd, device=DEV) @ W
        return torch.nn.functional.normalize(z, dim=1)
    return proj


def knn_predict(qz: torch.Tensor, bank: torch.Tensor, ybank: torch.Tensor) -> np.ndarray:
    out = []
    for i in range(0, len(qz), 4096):
        sim = qz[i:i + 4096] @ bank.T
        idx = sim.topk(K, dim=1).indices
        out.append(ybank[idx].mean(1).cpu().numpy())
    return np.concatenate(out)


def frame(v, bc, sec, fold) -> pd.DataFrame:
    df = pd.DataFrame(v, columns=[f"pred_{c[2:]}" for c in D.PROGS], index=bc)
    df.index.name = "barcode"
    df["section"], df["seed"], df["fold"] = sec, 0, fold
    return df


def main() -> None:
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    enc = set(pd.read_csv(p("results/12_image_features_kpmp/encoded_sections.tsv"), sep="\t").section)
    sm = pd.read_csv(p("results/11b_kpmp_targets/spot_summary.tsv"), sep="\t")
    sections = idx[idx.section.isin(set(sm.section[sm.status == "ok"]) & enc)].reset_index(drop=True)
    data = D.load_cohort(p(cfg["paths"]["interim"]) / "kpmp_spots", p(cfg["paths"]["cache"]) / "img_kpmp",
                         sections.section)
    Y = lambda ss: np.concatenate([data[s]["table"][D.PROGS].values for s in ss])  # noqa: E731
    rows = []
    for fi, (tr_all, te) in enumerate(D.participant_folds(sections, T["n_folds"], cfg["seed"])):
        rng = np.random.default_rng(cfg["seed"] + fi)
        proj = fit_space(feats(data, tr_all), rng)
        bank, yb = proj(feats(data, tr_all)), torch.as_tensor(Y(tr_all), device=DEV)
        v = knn_predict(proj(feats(data, te)), bank, yb)
        bc = np.concatenate([data[s]["table"].index.values for s in te])
        sec = np.concatenate([np.full(len(data[s]["table"]), s) for s in te])
        frame(v, bc, sec, fi).to_parquet(OUTC / f"{NAME}_fold{fi}.parquet")
        rows.append(dict(fold=fi, n_train_spots=len(yb), n_test_spots=len(v)))
        print("fold", fi, flush=True)
    # external: bank of all KPMP spots
    es = pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t").section
    ext = D.load_cohort(p(cfg["paths"]["interim"]) / "gse211785_spots", p(cfg["paths"]["cache"]) / "img_gse211785", es)
    allk = list(sections.section)
    proj = fit_space(feats(data, allk), np.random.default_rng(cfg["seed"]))
    bank, yb = proj(feats(data, allk)), torch.as_tensor(Y(allk), device=DEV)
    out = []
    for s in es:
        v = knn_predict(proj(feats(ext, [s])), bank, yb)
        out.append(frame(v, ext[s]["table"].index.values, s, -1))
    pd.concat(out).to_parquet(EXT / f"{NAME}.parquet")
    pd.DataFrame(rows).to_csv(S / "folds.tsv", sep="\t", index=False)
    write_provenance("14f_retrieval", cfg, [], [S / "folds.tsv"], extra={"k": K, "n_pcs": NPC})


if __name__ == "__main__":
    main()
