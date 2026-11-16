"""Split-half reliability of the measured programme scores in the external cohort (as in 16b)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import PROGS  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
R = cfg["region"]
S = stage_dir("18b_external_reliability")


def main() -> None:
    INT = p(cfg["paths"]["interim"])
    mm = mmread(INT / "gse211785_counts" / "counts.mtx").tocsr().T
    genes = pd.Index(pd.read_csv(INT / "gse211785_counts" / "genes.tsv", header=None)[0].values)
    bcs = pd.Index(pd.read_csv(INT / "gse211785_counts" / "barcodes.tsv", header=None)[0].values)
    lib = np.asarray(mm.sum(1)).ravel()
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for sid in pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t").section:
        t = pd.read_parquet(INT / "gse211785_spots" / f"{sid}.parquet")
        ii = bcs.get_indexer(t.index)
        tl = (t.array_row // R["tile_rows"]).astype(str) + "_" + (t.array_col // R["tile_cols"]).astype(str)
        for k in PROGS:
            g = [x for x in cfg["programmes"][k] if x in genes]
            X = np.asarray(mm[ii][:, genes.get_indexer(g)].todense()) / lib[ii, None] * 1e4
            Z = np.log1p(X)
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
            rs, rr = [], []
            for _ in range(R["n_splits"]):
                perm = rng.permutation(len(g))
                a, b = Z[:, perm[: len(g) // 2]].mean(1), Z[:, perm[len(g) // 2:]].mean(1)
                rs.append(spearmanr(a, b)[0])
                agg = pd.DataFrame({"a": a, "b": b, "tile": tl.values}).groupby("tile").agg(
                    a=("a", "mean"), b=("b", "mean"), n=("a", "size"))
                agg = agg[agg.n >= R["min_spots"]]
                rr.append(spearmanr(agg.a, agg.b)[0])
            sb = lambda r: 2 * r / (1 + r)
            rows.append(dict(section=sid, programme=k, rel_spot=sb(np.nanmean(rs)), rel_region=sb(np.nanmean(rr))))
    df = pd.DataFrame(rows)
    df.to_csv(S / "reliability_sections.tsv", sep="\t", index=False)
    out = df.groupby("programme")[["rel_spot", "rel_region"]].agg(lambda v: np.sqrt(v.clip(lower=0)).mean())
    out.to_csv(S / "ceiling.tsv", sep="\t")
    print(out.round(3))
    write_provenance("18b_external_reliability", cfg, [], [S / "ceiling.tsv"])


if __name__ == "__main__":
    main()
