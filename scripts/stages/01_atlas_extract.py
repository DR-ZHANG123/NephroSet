"""Extract per-sample cell tables from the CosMx/Xenium kidney atlas (Zenodo 19868428).

For every cell: sample, donor, platform, condition, coordinates in micrometres, coarse class and
log-normalised expression of the programme genes. Panel coverage of each programme gene is
recorded per platform so that only directly measured genes enter the programme scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
H5 = p(cfg["paths"]["raw"]) / "zenodo" / "spatial_adata_xenium_cosmx_zenodo.h5ad"
OUT = p(cfg["paths"]["interim"]) / "atlas_cells"
OUT.mkdir(parents=True, exist_ok=True)
S = stage_dir("01_atlas_extract")
COARSE = cfg["atlas"]["coarse_map"]


def cat(f, k):
    g = f["obs"][k]
    return pd.Categorical.from_codes(g["codes"][:], g["categories"][:].astype(str))


def main() -> None:
    f = h5py.File(H5, "r")
    obs = pd.DataFrame({k: cat(f, k) for k in ["orig_ident", "tech", "Condition", "annotation_updated"]})
    obs["n_counts"] = f["obs"]["nCount_RNA"][:]
    genes = pd.Index(f["var"]["_index"][:].astype(str))
    xy = f["obsm"]["spatial"][:]
    xy_fov = f["obsm"]["spatial_fov"][:]
    g = f["layers"]["counts"]
    X = sparse.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=(len(obs), len(genes)))
    obs["coarse"] = obs.annotation_updated.astype(str).map(COARSE)
    assert obs.coarse.notna().all(), set(obs.annotation_updated[obs.coarse.isna()])

    progs = cfg["programmes"]
    want = sorted({x for v in progs.values() for x in v})
    present = [x for x in want if x in genes]
    # panel coverage: fraction of cells with >=1 count, per platform
    cov = []
    Xp = X[:, genes.get_indexer(present)].tocsc()
    for t in ["CosMx", "Xenium"]:
        m = (obs.tech == t).values
        det = np.asarray((Xp[m] > 0).mean(axis=0)).ravel()
        tot = np.asarray(Xp[m].sum(axis=0)).ravel() / m.sum()
        cov += [dict(platform=t, gene=x, frac_cells_detected=d, mean_counts=c) for x, d, c in zip(present, det, tot)]
    cov = pd.DataFrame(cov)
    for x in sorted(set(want) - set(present)):
        for t in ["CosMx", "Xenium"]:
            cov.loc[len(cov)] = dict(platform=t, gene=x, frac_cells_detected=0.0, mean_counts=0.0)
    cov["programme"] = cov.gene.map({x: k for k, v in progs.items() for x in v})
    cov.to_csv(S / "panel_coverage.tsv", sep="\t", index=False)

    # log-normalised expression (counts per 1,000 transcripts, log1p) of programme genes
    lib = np.maximum(obs.n_counts.values.astype(float), 1.0)
    E = Xp.tocsr().multiply(1e3 / lib[:, None]).tocsr()
    E.data = np.log1p(E.data)
    E = pd.DataFrame(E.toarray().astype(np.float32), columns=present)

    rows = []
    for (s, t), idx in obs.groupby(["orig_ident", "tech"], observed=True).groups.items():
        idx = np.asarray(idx)
        if t == "Xenium":
            c = xy[idx]
        else:
            c = xy_fov[idx] * cfg["atlas"]["cosmx_um_per_px"]
        d, _ = cKDTree(c).query(c, k=2)
        nn = float(np.median(d[:, 1]))
        df = obs.iloc[idx][["orig_ident", "tech", "Condition", "annotation_updated", "coarse", "n_counts"]].copy()
        df["x_um"], df["y_um"] = c[:, 0], c[:, 1]
        df = pd.concat([df.reset_index(drop=True), E.iloc[idx].reset_index(drop=True)], axis=1)
        name = f"{s}_{t}"
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        rows.append(dict(sample=name, donor=str(s).split("_")[0], platform=t,
                         condition=df.Condition.iloc[0], n_cells=len(df),
                         n_pt=int((df.coarse == "PT").sum()), median_nn_um=nn))
    man = pd.DataFrame(rows)
    man.to_csv(S / "atlas_samples.tsv", sep="\t", index=False)
    print(man.to_string())
    print(cov.pivot_table(index="gene", columns="platform", values="frac_cells_detected").round(3).to_string())
    write_provenance("01_atlas_extract", cfg, [H5], [S / "atlas_samples.tsv", S / "panel_coverage.tsv"])


if __name__ == "__main__":
    main()
