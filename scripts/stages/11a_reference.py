"""Coarse-class expression profiles and marker genes from the KPMP single-nucleus atlas.

The reference only fixes what each coarse class looks like; it provides no expression of the
participants whose images are modelled. Programme genes are excluded from the markers so that
composition estimates cannot carry programme information.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
set_global_seed(cfg["seed"])
R = cfg["reference"]
S = stage_dir("11a_reference")
H5 = p(cfg["paths"]["raw"]) / "reference" / R["file"]


def main() -> None:
    f = h5py.File(H5, "r")
    g = f["obs"][R["label"]]
    lab = pd.Categorical.from_codes(g["codes"][:], g["categories"][:].astype(str)).astype(str)
    coarse = pd.Series(lab).map(R["coarse_map"])
    genes = f["raw/var"]["feature_name"]
    genes = pd.Categorical.from_codes(genes["codes"][:], genes["categories"][:].astype(str)).astype(str) \
        if isinstance(genes, h5py.Group) else genes[:].astype(str)
    genes = pd.Index(genes)
    rng = np.random.default_rng(cfg["seed"])
    ip = f["raw/X/indptr"]
    prof, n_used = {}, {}
    for c in sorted(coarse.dropna().unique()):
        idx = np.sort(rng.choice(np.where(coarse == c)[0], size=min(R["n_per_class"], int((coarse == c).sum())),
                                 replace=False))
        acc = np.zeros(len(genes))
        for i in idx:
            a, b = ip[i], ip[i + 1]
            x = f["raw/X/data"][a:b]
            j = f["raw/X/indices"][a:b]
            acc[j] += x / max(x.sum(), 1) * 1e4
        prof[c] = acc / len(idx)
        n_used[c] = len(idx)
    P = pd.DataFrame(prof, index=genes)
    P = P.groupby(level=0).sum()
    excl = {x for v in cfg["programmes"].values() for x in v}
    lp = np.log1p(P)
    markers = []
    for c in P.columns:
        lfc = lp[c] - lp.drop(columns=c).max(axis=1)
        noisy = P.index.str.match(r"^(ENSG|LINC|MT-|RPL|RPS|IG[HKL][VCJDGAM]|HLA-)") | P.index.str.contains(r"-AS\d|\.")
        ok = (P[c] >= R["min_cp10k"]) & ~P.index.isin(excl) & ~noisy
        top = lfc[ok].sort_values(ascending=False).head(R["n_markers"])
        markers += [dict(cls=c, gene=gname, log_fc=v) for gname, v in top.items()]
    P.to_csv(S / "reference_profiles_cp10k.tsv.gz", sep="\t")
    pd.DataFrame(markers).to_csv(S / "markers.tsv", sep="\t", index=False)
    pd.Series(n_used).rename("n_nuclei").to_csv(S / "nuclei_per_class.tsv", sep="\t")
    print(pd.Series(n_used))
    write_provenance("11a_reference", cfg, [H5], [S / "markers.tsv", S / "reference_profiles_cp10k.tsv.gz"])


if __name__ == "__main__":
    main()
