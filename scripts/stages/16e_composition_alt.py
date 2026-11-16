"""Alternative composition estimates for every Visium spot, for composition-adjusted agreement.

Usage: python 16e_composition_alt.py

The main composition is NNLS on 40 markers per coarse class (11b). Alternatives differ in the
amount of marker information, in the granularity of the reference and in the estimation
principle:
  m20, m80   coarse classes, 20 or 80 markers per class (same NNLS with capture factors)
  fine       16 KPMP subclasses (immune split into lymphoid and myeloid, tubule segments
             separated), 40 markers per subclass
  score      no deconvolution: mean z-scored log expression of the 40 coarse markers per class
Programme genes are excluded from all markers. Output: <cache>/comp_alt/<cohort>/<section>.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy.io import mmread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402
from lib.visium import CLASSES, composition, normalise  # noqa: E402

cfg = load_config()
set_global_seed(cfg["seed"])
R = cfg["reference"]
S = stage_dir("16e_composition_alt")
OUT = p(cfg["paths"]["cache"]) / "comp_alt"
INT = p(cfg["paths"]["interim"])
NOISY = r"^(ENSG|LINC|MT-|RPL|RPS|IG[HKL][VCJDGAM]|HLA-)"


def fine_profiles() -> pd.DataFrame:
    """Mean CP10k profile of each KPMP subclass (SubclassLevel1), as in 11a for coarse classes."""
    f = h5py.File(p(cfg["paths"]["raw"]) / "reference" / R["file"], "r")
    g = f["obs"][R["label"]]
    lab = pd.Series(pd.Categorical.from_codes(g["codes"][:], g["categories"][:].astype(str)).astype(str))
    lab = lab.where(lab.isin(R["coarse_map"].keys()))
    genes = f["raw/var"]["feature_name"]
    genes = pd.Index(pd.Categorical.from_codes(genes["codes"][:], genes["categories"][:].astype(str)).astype(str)
                     if isinstance(genes, h5py.Group) else genes[:].astype(str))
    rng = np.random.default_rng(cfg["seed"])
    ip = f["raw/X/indptr"]
    prof = {}
    for c in sorted(lab.dropna().unique()):
        pool = np.where(lab == c)[0]
        idx = np.sort(rng.choice(pool, size=min(R["n_per_class"], len(pool)), replace=False))
        acc = np.zeros(len(genes))
        for i in idx:
            a, b = ip[i], ip[i + 1]
            x = f["raw/X/data"][a:b]
            acc[f["raw/X/indices"][a:b]] += x / max(x.sum(), 1) * 1e4
        prof[c.replace("/", "_")] = acc / len(idx)
    return pd.DataFrame(prof, index=genes).groupby(level=0).sum()


def markers(P: pd.DataFrame, n: int) -> pd.DataFrame:
    excl = {x for v in cfg["programmes"].values() for x in v}
    lp = np.log1p(P)
    noisy = P.index.str.match(NOISY) | P.index.str.contains(r"-AS\d|\.")
    out = []
    for c in P.columns:
        lfc = lp[c] - lp.drop(columns=c).max(axis=1)
        ok = (P[c] >= R["min_cp10k"]) & ~P.index.isin(excl) & ~noisy
        out += [dict(cls=c, gene=g, log_fc=v) for g, v in lfc[ok].sort_values(ascending=False).head(n).items()]
    return pd.DataFrame(out)


def marker_scores(ad: sc.AnnData, mk: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in CLASSES:
        g = [x for x in mk.gene[mk.cls == c] if x in ad.var_names]
        X = ad[:, g].X
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        out[f"score_{c}"] = ((X - X.mean(0)) / (X.std(0) + 1e-9)).mean(1)
    return pd.DataFrame(out, index=ad.obs_names)


def variants(ad, refs) -> pd.DataFrame:
    (Pc, Pf), (m20, m40, m80, mf) = refs
    fine = list(Pf.columns)
    parts = [composition(ad, Pc, m20).add_prefix("m20_"), composition(ad, Pc, m80).add_prefix("m80_"),
             composition(ad, Pf, mf, classes=fine).add_prefix("fine_"), marker_scores(ad, m40)]
    return pd.concat(parts, axis=1)


def kpmp_one(sid, refs):
    t = pd.read_parquet(INT / "kpmp_spots" / f"{sid}.parquet")
    ad = sc.read_10x_h5(INT / "kpmp_sections" / sid / "filtered_feature_bc_matrix.h5")
    ad.var_names_make_unique()
    ad = normalise(ad[t.index].copy())
    v = variants(ad, refs)
    v.index.name = "barcode"
    v.to_parquet(OUT / "kpmp" / f"{sid}.parquet")
    return sid


def main() -> None:
    for c in ("kpmp", "gse211785"):
        (OUT / c).mkdir(parents=True, exist_ok=True)
    Pc = pd.read_csv(p("results/11a_reference/reference_profiles_cp10k.tsv.gz"), sep="\t", index_col=0)
    Pf = fine_profiles()
    Pf.to_csv(S / "fine_profiles_cp10k.tsv.gz", sep="\t")
    mk = {n: markers(Pc, n) for n in (20, 40, 80)}
    mf = markers(Pf, 40)
    pd.concat([m.assign(set=f"coarse_m{n}") for n, m in mk.items()] + [mf.assign(set="fine_m40")]) \
        .to_csv(S / "markers.tsv", sep="\t", index=False)
    refs = ((Pc, Pf), (mk[20], mk[40], mk[80], mf))
    secs = sorted(x.stem for x in (INT / "kpmp_spots").glob("*.parquet"))
    done = Parallel(n_jobs=24)(delayed(kpmp_one)(s, refs) for s in secs)
    mm = mmread(INT / "gse211785_counts" / "counts.mtx").tocsr().T
    genes = pd.read_csv(INT / "gse211785_counts" / "genes.tsv", header=None)[0].values
    bcs = pd.Index(pd.read_csv(INT / "gse211785_counts" / "barcodes.tsv", header=None)[0].values)
    es = pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t").section
    for s in es:
        t = pd.read_parquet(INT / "gse211785_spots" / f"{s}.parquet")
        ad = sc.AnnData(mm[bcs.get_indexer(t.index)].astype(np.float32))
        ad.obs_names, ad.var_names = t.index.values, genes
        v = variants(normalise(ad), refs)
        v.index.name = "barcode"
        v.to_parquet(OUT / "gse211785" / f"{s}.parquet")
    pd.DataFrame({"section": list(done) + list(es)}).to_csv(S / "sections.tsv", sep="\t", index=False)
    write_provenance("16e_composition_alt", cfg, [], [S / "markers.tsv", S / "sections.tsv"],
                     extra={"fine_classes": list(Pf.columns)})


if __name__ == "__main__":
    main()
