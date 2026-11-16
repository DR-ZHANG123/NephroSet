"""How much of each measured programme score is reproducible, and how well are regions predicted?

Reliability: programme genes are split at random into two halves (repeated); the within-section
Spearman correlation between the two half-scores, stepped up with the Spearman-Brown formula,
estimates the reliability of the full score. Its square root bounds the correlation any
predictor can reach with the measured score. This is computed at spot level and for regions.

Regions: spots are grouped into tiles of about 400 x 350 um; measured and predicted scores are
averaged within tiles holding at least `min_spots` spots, and agreement is computed across the
tiles of a section.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import PROGS, boot_ci  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402
from lib.visium import normalise  # noqa: E402

cfg = load_config()
R = cfg["region"]
S = stage_dir("16b_reliability_region")
CV = p(cfg["paths"]["cache"]) / "cv_predictions"


def tiles(t: pd.DataFrame) -> pd.Series:
    return (t.array_row // R["tile_rows"]).astype(str) + "_" + (t.array_col // R["tile_cols"]).astype(str)


def reliability(sid: str) -> list[dict]:
    d = p(cfg["paths"]["interim"]) / "kpmp_sections" / sid
    t = pd.read_parquet(p(cfg["paths"]["interim"]) / "kpmp_spots" / f"{sid}.parquet")
    ad = sc.read_10x_h5(d / "filtered_feature_bc_matrix.h5")
    ad.var_names_make_unique()
    ad = normalise(ad[t.index].copy())
    tl = tiles(t)
    rng = np.random.default_rng(cfg["seed"])
    out = []
    for k in PROGS:
        g = [x for x in cfg["programmes"][k] if x in ad.var_names]
        X = ad[:, g].X
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
        rs, rr = [], []
        for _ in range(R["n_splits"]):
            perm = rng.permutation(len(g))
            a, b = Z[:, perm[: len(g) // 2]].mean(1), Z[:, perm[len(g) // 2:]].mean(1)
            rs.append(spearmanr(a, b)[0])
            df = pd.DataFrame({"a": a, "b": b, "tile": tl.values})
            agg = df.groupby("tile").agg(a=("a", "mean"), b=("b", "mean"), n=("a", "size"))
            agg = agg[agg.n >= R["min_spots"]]
            rr.append(spearmanr(agg.a, agg.b)[0] if len(agg) >= R["min_tiles"] else np.nan)
        sb = lambda r: 2 * r / (1 + r) if r > -1 else np.nan
        out.append(dict(section=sid, programme=k, half_rho_spot=np.nanmean(rs), rel_spot=sb(np.nanmean(rs)),
                        half_rho_region=np.nanmean(rr), rel_region=sb(np.nanmean(rr))))
    return out


def region_metrics(model: str, truth: pd.DataFrame) -> pd.DataFrame:
    pr = pd.concat([pd.read_parquet(f) for f in sorted(CV.glob(f"{model}_fold*.parquet"))])
    pr = pr.reset_index().groupby(["section", "barcode"]).mean(numeric_only=True)
    mg = truth.join(pr[[f"pred_{k}" for k in PROGS]], how="inner").reset_index()
    mg["tile"] = tiles(mg)
    rows = []
    for s, g in mg.groupby("section"):
        agg = g.groupby("tile").agg(**{c: (c, "mean") for c in [f"y_{k}" for k in PROGS] + [f"pred_{k}" for k in PROGS]},
                                    n=("tile", "size"))
        agg = agg[agg.n >= R["min_spots"]]
        for k in PROGS:
            rho = spearmanr(agg[f"y_{k}"], agg[f"pred_{k}"])[0] if len(agg) >= R["min_tiles"] else np.nan
            rows.append(dict(model=model, section=s, programme=k, rho_region=rho, n_tiles=len(agg)))
    return pd.DataFrame(rows)


def main() -> None:
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    part_of = dict(zip(idx.section, idx.participant))
    secs = sorted({f.stem for f in (p(cfg["paths"]["interim"]) / "kpmp_spots").glob("*.parquet")}
                  & set(pd.read_csv(p("results/12_image_features_kpmp/encoded_sections.tsv"), sep="\t").section))
    rel = pd.DataFrame([r for rr in Parallel(n_jobs=32)(delayed(reliability)(s) for s in secs) for r in rr])
    rel["participant"] = rel.section.map(part_of)
    rel.to_csv(S / "reliability_sections.tsv", sep="\t", index=False)
    truth = pd.concat([pd.read_parquet(p(cfg["paths"]["interim"]) / "kpmp_spots" / f"{s}.parquet") for s in secs]) \
        .reset_index().set_index(["section", "barcode"])
    reg = pd.concat([region_metrics(m, truth) for m in cfg["evaluate"]["models"]
                     if (CV / f"{m}_fold0.parquet").exists()])
    reg["participant"] = reg.section.map(part_of)
    reg.to_csv(S / "region_sections.tsv", sep="\t", index=False)
    rng = np.random.default_rng(cfg["seed"])
    summ = []
    rp = rel.groupby(["participant", "programme"]).mean(numeric_only=True).reset_index()
    for k in PROGS:
        q = rp[rp.programme == k]
        for col in ["rel_spot", "rel_region"]:
            v = q[col].clip(lower=0)
            lo, hi = boot_ci(np.sqrt(v.values), 2000, rng)
            summ.append(dict(model="ceiling", programme=k, metric=col, mean=float(np.sqrt(v).mean()), lo=lo, hi=hi))
    pp = reg.groupby(["model", "participant", "programme"]).rho_region.mean().reset_index()
    for (m, k), q in pp.groupby(["model", "programme"]):
        lo, hi = boot_ci(q.rho_region.values, 2000, rng)
        summ.append(dict(model=m, programme=k, metric="rho_region", mean=q.rho_region.mean(), lo=lo, hi=hi))
    Sm = pd.DataFrame(summ)
    Sm.to_csv(S / "summary.tsv", sep="\t", index=False)
    print(Sm.round(3).to_string())
    write_provenance("16b_reliability_region", cfg, [], [S / "summary.tsv"])


if __name__ == "__main__":
    main()
