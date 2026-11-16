"""How much does the split-half reliability reference depend on how it is computed?

Usage: python 16g_reliability_sensitivity.py

For every section of both cohorts and every programme, spot-level reliability is estimated in
five ways (20 splits each where splitting applies):
  random      random halves of the z-scored genes, Spearman-Brown (the reference used in the text)
  balanced    halves matched for mean expression: genes are ranked by mean log expression and
              one gene of each consecutive pair goes to each half
  scored      random halves, each half scored exactly as the programme (mean minus
              expression-matched control genes)
  alpha       Cronbach's alpha of the z-scored genes (all genes, no split)
  half_depth  random halves after binomial thinning of counts to 50%
The square root of each reliability is the reference against which agreement is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy import sparse
from scipy.io import mmread
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import PROGS  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402
from lib.visium import normalise  # noqa: E402

cfg = load_config()
S = stage_dir("16g_reliability_sensitivity")
INT = p(cfg["paths"]["interim"])
NS = cfg["region"]["n_splits"]


def sb(r):
    return 2 * r / (1 + r) if r > -1 else np.nan


def dense(X):
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)


def zgenes(ad, g):
    X = dense(ad[:, g].X)
    return (X - X.mean(0)) / (X.std(0) + 1e-9), X.mean(0)


def split_rel(Z, halves):
    return sb(np.nanmean([spearmanr(Z[:, a].mean(1), Z[:, b].mean(1))[0] for a, b in halves]))


def random_halves(n, rng):
    out = []
    for _ in range(NS):
        q = rng.permutation(n)
        out.append((q[: n // 2], q[n // 2:]))
    return out


def balanced_halves(mean_expr, rng):
    order = np.argsort(mean_expr)
    out = []
    for _ in range(NS):
        a, b = [], []
        for j in range(0, len(order) - 1, 2):
            x, y = order[j], order[j + 1]
            if rng.random() < 0.5:
                x, y = y, x
            a.append(x)
            b.append(y)
        if len(order) % 2:
            (a if rng.random() < 0.5 else b).append(order[-1])
        out.append((np.array(a), np.array(b)))
    return out


def cronbach(Z):
    k = Z.shape[1]
    return k / (k - 1) * (1 - Z.var(0).sum() / Z.sum(1).var())


def scored_rel(ad, g, rng):
    rs = []
    for _ in range(NS):
        q = rng.permutation(len(g))
        a, b = [g[i] for i in q[: len(g) // 2]], [g[i] for i in q[len(g) // 2:]]
        sc.tl.score_genes(ad, a, ctrl_size=cfg["visium"]["ctrl_size"], score_name="_a", random_state=int(rng.integers(1e9)))
        sc.tl.score_genes(ad, b, ctrl_size=cfg["visium"]["ctrl_size"], score_name="_b", random_state=int(rng.integers(1e9)))
        rs.append(spearmanr(ad.obs["_a"], ad.obs["_b"])[0])
    return sb(np.nanmean(rs))


def one(cohort: str, sid: str, counts: sparse.csr_matrix, genes) -> list[dict]:
    rng = np.random.default_rng(cfg["seed"])
    ad = sc.AnnData(counts.astype(np.float32))
    ad.obs_names, ad.var_names = [f"s{i}" for i in range(counts.shape[0])], genes
    ad.var_names_make_unique()
    adn = normalise(ad)
    thin = counts.copy()
    thin.data = rng.binomial(thin.data.astype(np.int64), 0.5).astype(np.float32)
    ad2 = sc.AnnData(thin)
    ad2.obs_names, ad2.var_names = ad.obs_names, ad.var_names
    adt = normalise(ad2)
    out = []
    for k in PROGS:
        g = [x for x in cfg["programmes"][k] if x in adn.var_names]
        Z, mu = zgenes(adn, g)
        Zt, _ = zgenes(adt, g)
        out.append(dict(cohort=cohort, section=sid, programme=k, n_genes=len(g),
                        random=split_rel(Z, random_halves(len(g), rng)),
                        balanced=split_rel(Z, balanced_halves(mu, rng)),
                        scored=scored_rel(adn, g, rng), alpha=cronbach(Z),
                        half_depth=split_rel(Zt, random_halves(len(g), rng))))
    return out


def kpmp_jobs():
    for f in sorted((INT / "kpmp_spots").glob("*.parquet")):
        t = pd.read_parquet(f)
        ad = sc.read_10x_h5(INT / "kpmp_sections" / f.stem / "filtered_feature_bc_matrix.h5")
        ad.var_names_make_unique()
        ad = ad[t.index]
        yield "kpmp", f.stem, sparse.csr_matrix(ad.X), np.array(ad.var_names)


def ext_jobs():
    mm = mmread(INT / "gse211785_counts" / "counts.mtx").tocsr().T.tocsr()
    genes = pd.read_csv(INT / "gse211785_counts" / "genes.tsv", header=None)[0].values
    bcs = pd.Index(pd.read_csv(INT / "gse211785_counts" / "barcodes.tsv", header=None)[0].values)
    for sid in pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t").section:
        t = pd.read_parquet(INT / "gse211785_spots" / f"{sid}.parquet")
        yield "gse211785", sid, mm[bcs.get_indexer(t.index)], genes


def main() -> None:
    res = Parallel(n_jobs=32)(delayed(one)(*j) for j in list(kpmp_jobs()) + list(ext_jobs()))
    df = pd.DataFrame([r for rr in res for r in rr])
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    part = dict(zip(idx.section, idx.participant))
    df["participant"] = [part.get(s, s) for s in df.section]
    df.to_csv(S / "reliability_sections.tsv", sep="\t", index=False)
    meth = ["random", "balanced", "scored", "alpha", "half_depth"]
    pp = df.groupby(["cohort", "participant", "programme"])[meth].mean().reset_index()
    summ = pp.groupby(["cohort", "programme"])[meth].agg(lambda v: np.sqrt(v.clip(lower=0)).mean()).reset_index()
    summ.to_csv(S / "summary.tsv", sep="\t", index=False)
    print(summ.round(3).to_string())
    write_provenance("16g_reliability_sensitivity", cfg, [], [S / "summary.tsv"])


if __name__ == "__main__":
    main()
