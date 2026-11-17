"""External cohort: regional agreement and coverage of intervals calibrated on KPMP.

Regional agreement uses the same tiles as in KPMP. Split-conformal intervals are calibrated on the
KPMP validation participants of the model trained on all KPMP participants and applied unchanged
to its external predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import PRIMARY_SEEDS, read_ext, ext_file, PROGS, boot_ci  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
R, C = cfg["region"], cfg["confidence"]
S = stage_dir("18c_external_region_coverage")
EXT = p(cfg["paths"]["cache"]) / "external_predictions"


def main() -> None:
    sec = pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t")
    sp = p(cfg["paths"]["interim"]) / "gse211785_spots"
    truth = pd.concat([pd.read_parquet(sp / f"{s}.parquet") for s in sec.section]).reset_index() \
        .set_index(["section", "barcode"])
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for m in cfg["evaluate"]["models"]:
        f = ext_file(EXT, m)
        if not f.exists():
            continue
        pr = read_ext(EXT, m).reset_index().groupby(["section", "barcode"]).mean(numeric_only=True)
        mg = truth.join(pr[[f"pred_{k}" for k in PROGS]], how="inner").reset_index()
        mg["tile"] = (mg.array_row // R["tile_rows"]).astype(str) + "_" + (mg.array_col // R["tile_cols"]).astype(str)
        for s, g in mg.groupby("section"):
            agg = g.groupby("tile").agg(**{c: (c, "mean") for c in [f"y_{k}" for k in PROGS] + [f"pred_{k}" for k in PROGS]},
                                        n=("tile", "size"))
            agg = agg[agg.n >= R["min_spots"]]
            for k in PROGS:
                rows.append(dict(model=m, section=s, programme=k, n_tiles=len(agg),
                                 rho_region=spearmanr(agg[f"y_{k}"], agg[f"pred_{k}"])[0]))
    reg = pd.DataFrame(rows)
    reg.to_csv(S / "region_sections.tsv", sep="\t", index=False)
    summ = []
    for (m, k), q in reg.groupby(["model", "programme"]):
        lo, hi = boot_ci(q.rho_region.values, 2000, rng)
        summ.append(dict(model=m, programme=k, metric="rho_region", mean=q.rho_region.mean(), lo=lo, hi=hi))
    # split-conformal intervals: the model trained on all KPMP participants is calibrated on its own
    # KPMP validation participants (held out from fitting) and applied unchanged to the external
    # sections. Calibration and test use the same three-seed ensemble as the primary external result.
    m = C["model"]
    seeds = list(PRIMARY_SEEDS)
    kp = p(cfg["paths"]["interim"]) / "kpmp_spots"
    va = pd.read_parquet(EXT / "validation" / f"{m}_s10.parquet").reset_index()
    va = va[va.seed.isin(seeds)]
    vy = pd.concat([pd.read_parquet(kp / f"{s}.parquet").assign(section=s) for s in va.section.unique()]) \
        .reset_index().set_index(["section", "barcode"])
    ext = read_ext(EXT, m).reset_index()
    cov, per = [], []
    for k in PROGS:
        v = vy.join(va.groupby(["section", "barcode"])[f"pred_{k}"].agg(["mean", "std"]), how="inner")
        sc = np.abs(v[f"y_{k}"] - v["mean"]) / (v["std"] + C["sd_floor"])
        q = np.quantile(sc, np.ceil((len(sc) + 1) * (1 - C["alpha"])) / len(sc))
        e = truth.join(ext.groupby(["section", "barcode"])[f"pred_{k}"].agg(["mean", "std"]), how="inner")
        half = q * (e["std"] + C["sd_floor"])
        inside = np.abs(e[f"y_{k}"] - e["mean"]) <= half
        pp = inside.groupby(level=0).mean()
        lo, hi = boot_ci(pp.values, 2000, rng)
        cov.append(dict(programme=k, nominal=1 - C["alpha"], q=q, n_cal_spots=len(sc),
                        n_cal_sections=int(v.index.get_level_values(0).nunique()),
                        external_coverage_spots=inside.mean(), external_coverage=pp.mean(), lo=lo, hi=hi,
                        donor_min=pp.min(), donor_max=pp.max(), frac_donors_ge80=(pp >= 0.8).mean(),
                        median_half_width_external=float(half.median())))
        per.append(pd.DataFrame(dict(programme=k, section=pp.index, coverage=pp.values)))
    pd.concat(per).to_csv(S / "coverage_sections.tsv", sep="\t", index=False)
    pd.DataFrame(summ).to_csv(S / "region_summary.tsv", sep="\t", index=False)
    pd.DataFrame(cov).to_csv(S / "coverage.tsv", sep="\t", index=False)
    print(pd.DataFrame(summ).round(3).to_string())
    print(pd.DataFrame(cov).round(3).to_string())
    write_provenance("18c_external_region_coverage", cfg, [], [S / "region_summary.tsv", S / "coverage.tsv"])


if __name__ == "__main__":
    main()
