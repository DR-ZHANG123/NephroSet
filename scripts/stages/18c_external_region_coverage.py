"""External cohort: regional agreement and coverage of intervals calibrated on KPMP.

Regional agreement uses the same tiles as in KPMP. Conformal intervals use the quantile of
disagreement-scaled residuals over all KPMP held-out spots (cross-validation predictions) and
are applied unchanged to the external predictions of the model trained on all participants.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import ext_file, PROGS, boot_ci  # noqa: E402
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
        pr = pd.read_parquet(f).reset_index().groupby(["section", "barcode"]).mean(numeric_only=True)
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
    # coverage of KPMP-calibrated intervals
    m = C["model"]
    cv = pd.read_parquet(p(cfg["paths"]["cache"]) / "cv_confidence_spots.parquet")
    # interval widths use the three seeds of cross-validation, so disagreement is on the same scale
    ext = pd.read_parquet(ext_file(EXT, m)).reset_index()
    ext = ext[ext.seed.isin(cfg["seeds"])]
    cov = []
    for k in PROGS:
        sd_cv = cv[f"sd_{k}"] + C["sd_floor"]
        q = np.quantile(np.abs(cv[f"y_{k}"] - cv[f"pred_{k}"]) / sd_cv, 1 - C["alpha"])
        e = ext.groupby(["section", "barcode"])[f"pred_{k}"].agg(["mean", "std"])
        e = truth.join(e, how="inner")
        half = q * (e["std"] + C["sd_floor"])
        inside = (np.abs(e[f"y_{k}"] - e["mean"]) <= half)
        per = inside.groupby(level=0).mean()
        lo, hi = boot_ci(per.values, 2000, rng)
        # the same intervals re-centred on each section's own mean offset (level shift removed)
        off = (e[f"y_{k}"] - e["mean"]).groupby(level=0).transform("mean")
        inside_c = (np.abs(e[f"y_{k}"] - e["mean"] - off) <= half).groupby(level=0).mean()
        cv_cov = (np.abs(cv[f"y_{k}"] - cv[f"pred_{k}"]) <= q * sd_cv).groupby(cv.participant).mean()
        cov.append(dict(programme=k, nominal=1 - C["alpha"], q=q, kpmp_coverage=cv_cov.mean(),
                        external_coverage=per.mean(), lo=lo, hi=hi, external_coverage_level_corrected=inside_c.mean(),
                        median_half_width_kpmp=float(np.median(q * sd_cv)), median_half_width_external=float(half.median())))
    pd.DataFrame(summ).to_csv(S / "region_summary.tsv", sep="\t", index=False)
    pd.DataFrame(cov).to_csv(S / "coverage.tsv", sep="\t", index=False)
    print(pd.DataFrame(summ).round(3).to_string())
    print(pd.DataFrame(cov).round(3).to_string())
    write_provenance("18c_external_region_coverage", cfg, [], [S / "region_summary.tsv", S / "coverage.tsv"])


if __name__ == "__main__":
    main()
