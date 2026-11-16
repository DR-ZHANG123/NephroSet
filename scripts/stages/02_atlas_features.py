"""Neighbourhood descriptors and programme scores for proximal tubule cells in the atlas.

For each section, up to `n_target` PT cells are sampled; for each we compute the four levels
of neighbourhood descriptors (lib/neighbourhood.py) on the observed labels and on one
interstitial-label permutation (tile-preserving null), plus the cell's programme scores from
directly measured genes of that platform.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.neighbourhood import features, permute_nonepithelial  # noqa: E402
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
set_global_seed(cfg["seed"])
A = cfg["atlas"]
S = stage_dir("02_atlas_features")
IN = p(cfg["paths"]["interim"]) / "atlas_cells"
OUT = p(cfg["paths"]["interim"]) / "atlas_features"
OUT.mkdir(parents=True, exist_ok=True)


def programme_genes() -> dict[str, dict[str, list[str]]]:
    cov = pd.read_csv(p("results/01_atlas_extract/panel_coverage.tsv"), sep="\t")
    ok = cov[cov.frac_cells_detected >= A["min_gene_detection"]]
    return {t: {k: sorted(set(v) & set(ok[ok.platform == t].gene)) for k, v in cfg["programmes"].items()}
            for t in ["CosMx", "Xenium"]}


def one(sample: str, genes: dict, seed: int) -> pd.DataFrame | None:
    df = pd.read_parquet(IN / f"{sample}.parquet")
    pt = np.where(df.coarse.values == "PT")[0]
    if len(pt) < A["min_pt_cells"]:
        return None
    rng = np.random.default_rng(seed)
    # avoid section edges: keep targets with a full regional neighbourhood inside the hull box
    xy = df[["x_um", "y_um"]].values
    lo, hi = np.percentile(xy, 1, axis=0) + A["r_region"], np.percentile(xy, 99, axis=0) - A["r_region"]
    inner = pt[np.all((xy[pt] > lo) & (xy[pt] < hi), axis=1)]
    if len(inner) < A["min_pt_cells"]:
        inner = pt
    tg = np.sort(rng.choice(inner, size=min(A["n_target"], len(inner)), replace=False))
    lab = df.coarse.values.astype(object)
    units: list = []
    real = features(xy, lab, tg, A, units_out=units)
    perm_lab = permute_nonepithelial(xy, lab, A["perm_tile"], rng)
    null = features(xy, perm_lab, tg, A)
    null.columns = ["null_" + c for c in null.columns]
    out = pd.concat([real, null], axis=1)
    t = df.tech.iloc[0]
    lc_all = np.log1p(df.n_counts.values)
    for k, g in genes[t].items():
        if not g:
            out[f"y_{k}"] = out[f"yres_{k}"] = out[f"yunit_{k}"] = np.nan
            continue
        sc = df[g].mean(axis=1).values
        # remove the dependence of a cell's score on its own transcript count (fitted on PT cells)
        coef = np.polyfit(lc_all[pt], sc[pt], 3)
        res = sc - np.polyval(coef, lc_all)
        out[f"y_{k}"] = sc[tg]
        out[f"yres_{k}"] = res[tg]
        # tubular-unit target: mean residual score of the PT cells forming the cell's unit
        out[f"yunit_{k}"] = [res[u].mean() for u in units]
    out["unit_n"] = [len(u) for u in units]
    out["is_iPT"] = (df.annotation_updated.values[tg] == "iPT").astype(int)
    out["log_counts"] = np.log1p(df.n_counts.values[tg])
    out["sample"] = sample
    out["cell"] = tg
    out.to_parquet(OUT / f"{sample}.parquet")
    return out


def main() -> None:
    man = pd.read_csv(p("results/01_atlas_extract/atlas_samples.tsv"), sep="\t")
    genes = programme_genes()
    pd.DataFrame([(t, k, ",".join(v), len(v)) for t, d in genes.items() for k, v in d.items()],
                 columns=["platform", "programme", "genes", "n_genes"]).to_csv(S / "programme_genes_used.tsv",
                                                                                sep="\t", index=False)
    res = Parallel(n_jobs=A["n_jobs"])(delayed(one)(s, genes, cfg["seed"] + i)
                                       for i, s in enumerate(man["sample"]))
    res = [r for r in res if r is not None]
    allf = pd.concat(res, ignore_index=True)
    allf = allf.merge(man[["sample", "donor", "platform", "condition"]], on="sample")
    allf.to_parquet(OUT / "all_targets.parquet")
    summ = allf.groupby(["sample", "platform", "condition"]).size().rename("n_targets").reset_index()
    summ.to_csv(S / "targets_per_section.tsv", sep="\t", index=False)
    print(summ.to_string())
    write_provenance("02_atlas_features", cfg, [p("results/01_atlas_extract/atlas_samples.tsv")],
                     [S / "targets_per_section.tsv", S / "programme_genes_used.tsv"],
                     extra={"n_targets": len(allf)})


if __name__ == "__main__":
    main()
