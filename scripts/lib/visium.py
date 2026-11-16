"""Spot-level measured programme scores and reference-based coarse composition for Visium."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import nnls

CLASSES = ["PT", "LOH", "DCT_CNT", "CD", "Glomerular", "Endothelial", "Fibroblast", "Mural", "Immune"]


def normalise(ad: sc.AnnData) -> sc.AnnData:
    ad = ad.copy()
    ad.var_names_make_unique()
    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    return ad


def programme_scores(ad: sc.AnnData, programmes: dict, ctrl_size: int, seed: int) -> pd.DataFrame:
    """Programme score per spot: mean of programme genes minus expression-matched control genes."""
    out = {}
    for k, genes in programmes.items():
        g = [x for x in genes if x in ad.var_names]
        sc.tl.score_genes(ad, g, ctrl_size=ctrl_size, score_name=f"y_{k}", random_state=seed)
        out[f"y_{k}"] = ad.obs[f"y_{k}"].values
        out[f"n_genes_{k}"] = len(g)
    return pd.DataFrame(out, index=ad.obs_names)


def composition(ad: sc.AnnData, profiles: pd.DataFrame, markers: pd.DataFrame, n_iter: int = 3) -> pd.DataFrame:
    """Non-negative least squares on marker genes with a per-gene platform factor.

    Reference profiles come from single-nucleus data; the capture efficiency of each gene on
    the spatial platform differs. As in RCTD, a gene-wise factor is estimated within the
    section by comparing observed with fitted marker totals, and the fit is repeated.
    Each marker gene is scaled to equal weight. Returns class fractions per spot.
    """
    mk = markers.drop_duplicates("gene")
    mk = mk[mk.gene.isin(ad.var_names) & mk.gene.isin(profiles.index)]
    genes, cls_of = list(mk.gene), mk.cls.reset_index(drop=True)
    R0 = profiles.loc[genes, CLASSES].values.astype(float)
    X = ad[:, genes].layers["counts"]
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X = X / np.maximum(np.asarray(ad.layers["counts"].sum(1)).ravel()[:, None], 1) * 1e4
    gamma = np.ones(len(genes))
    for _ in range(n_iter):
        R = R0 * gamma[:, None]
        scale = 1.0 / np.maximum(R.max(1), 1e-6)
        Rs, Xs = R * scale[:, None], X * scale[None, :]
        W = np.stack([nnls(Rs, x)[0] for x in Xs])
        fit = W @ R.T
        gamma = gamma * np.clip(X.sum(0) / np.maximum(fit.sum(0), 1e-9), 0.05, 20)
        # keep each class's overall scale from the reference (only relative gene capture is
        # re-estimated), otherwise class abundance and marker scaling are not identifiable
        for c in cls_of.unique():
            m = (cls_of == c).values
            gamma[m] = gamma[m] / np.median(gamma[m])
    frac = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    return pd.DataFrame(frac, index=ad.obs_names, columns=[f"comp_{c}" for c in CLASSES])
