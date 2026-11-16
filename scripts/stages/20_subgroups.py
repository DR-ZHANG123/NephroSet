"""Held-out agreement of the chosen model by diagnosis and by tissue source (KPMP)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import PROGS, boot_ci  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
S = stage_dir("20_subgroups")
DX = {"Diabetic Kidney Disease": "Diabetic kidney disease", "Hypertensive Kidney Disease": "Hypertensive kidney disease",
      "Acute Tubular Injury": "Acute tubular injury", "Acute Interstitial Nephritis": "Acute interstitial nephritis",
      "Healthy Reference": "Healthy reference", "Diabetes Mellitus - Resilient": "Diabetes without kidney disease",
      "DM-R": "Diabetes without kidney disease"}
SRC = {"Percutaneous Needle Biopsy": "Needle biopsy", "Transplant Pre-perfusion Biopsy": "Donor kidney biopsy"}


def main() -> None:
    m = cfg["confidence"]["model"]
    pm = pd.read_csv(p("results/16_evaluate_cv/participant_metrics.tsv"), sep="\t")
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t").drop_duplicates("participant")
    q = pm[pm.model == m].merge(idx[["participant", "primary_adjudicated_category", "enrollment_category",
                                     "sample_type"]], on="participant")
    q["diagnosis"] = q.primary_adjudicated_category.fillna(q.enrollment_category).map(DX).fillna("Other or undetermined")
    q["source"] = q.sample_type.map(SRC).fillna("Nephrectomy or intra-operative")
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for by in ["diagnosis", "source"]:
        for k in PROGS:
            z = q[q.programme == k]
            groups = [g.rho.dropna().values for _, g in z.groupby(by)]
            pk = kruskal(*groups).pvalue if len(groups) > 1 else np.nan
            for gname, g in z.groupby(by):
                lo, hi = boot_ci(g.rho.values, 2000, rng)
                rows.append(dict(grouping=by, group=gname, programme=k, n=len(g), rho=g.rho.mean(), lo=lo, hi=hi,
                                 rho_beyond_comp=g.rho_beyond_comp.mean(), mean_measured=g.mean_y.mean(),
                                 p_kruskal=pk))
    out = pd.DataFrame(rows)
    out.to_csv(S / "subgroups.tsv", sep="\t", index=False)
    # agreement versus how much the measured programme varies within the section, both cohorts
    from scipy.stats import spearmanr
    sd_rows = []
    for cohort, spots, sec_metrics in [
            ("KPMP", "kpmp_spots", pd.read_csv(p("results/16_evaluate_cv/participant_metrics.tsv"), sep="\t")
             .query("model == @m").merge(idx[["participant", "section"]], on="participant")),
            ("External", "gse211785_spots", pd.read_csv(p("results/18_evaluate_external/section_metrics.tsv"), sep="\t")
             .query("model == @m"))]:
        for s, g in sec_metrics.groupby("section"):
            t = pd.read_parquet(p(cfg["paths"]["interim"]) / spots / f"{s}.parquet")
            for _, r in g.iterrows():
                sd_rows.append(dict(cohort=cohort, section=s, programme=r.programme, rho=r.rho,
                                    sd_measured=t[f"y_{r.programme}"].std(), n_spots=len(t)))
    sd = pd.DataFrame(sd_rows)
    sd.to_csv(S / "agreement_vs_spread.tsv", sep="\t", index=False)
    rel = []
    for (c, k), g in sd.groupby(["cohort", "programme"]):
        rel.append(dict(cohort=c, programme=k, rho_agreement_vs_sd=spearmanr(g.sd_measured, g.rho)[0],
                        p=spearmanr(g.sd_measured, g.rho)[1], median_sd=g.sd_measured.median(), n=len(g)))
    # KPMP sections matched to the external range of within-section spread
    for k in PROGS:
        e = sd[(sd.cohort == "External") & (sd.programme == k)]
        kk = sd[(sd.cohort == "KPMP") & (sd.programme == k)]
        lo_, hi_ = e.sd_measured.min(), e.sd_measured.max()
        mm = kk[kk.sd_measured.between(lo_, hi_)]
        rel.append(dict(cohort="KPMP within external spread range", programme=k, rho_agreement_vs_sd=np.nan, p=np.nan,
                        median_sd=mm.sd_measured.median(), n=len(mm), mean_rho=mm.rho.mean()))
        rel.append(dict(cohort="External mean", programme=k, rho_agreement_vs_sd=np.nan, p=np.nan,
                        median_sd=e.sd_measured.median(), n=len(e), mean_rho=e.rho.mean()))
    pd.DataFrame(rel).to_csv(S / "spread_summary.tsv", sep="\t", index=False)
    print(pd.DataFrame(rel).round(3).to_string())
    print(out.round(3).to_string())
    write_provenance("20_subgroups", cfg, [], [S / "subgroups.tsv"])


if __name__ == "__main__":
    main()
