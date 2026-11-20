"""Participant-level programme burden against kidney function and albuminuria in KPMP.

For each held-out participant, the mean measured and the mean cross-validated NephroSet estimate of each programme
(results/16_evaluate_cv/participant_metrics.tsv) are related to the binned clinical covariates released with the
KPMP open data (metadata/kpmp_spatial_manifest.tsv): baseline eGFR (10 ml/min/1.73 m2 bins, bin midpoints),
albuminuria and proteinuria (ordered categories). Spearman correlations with 95% bootstrap intervals over
participants; the difference estimated minus measured is bootstrapped on the same resamples.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
S = stage_dir("16i_clinical")
MODEL = "tok_abmil_ctx"
ALB = ["<30 mg/g cr (alb)", "30 to <300 mg/g cr (alb)", "300 to <500 mg/g cr (alb)", "500 to <1000 mg/g cr (alb)",
       ">=1000 mg/g cr (alb)"]
PROT = ["<150 mg/g cr (prot)", "150 to <500 mg/g cr (prot)", "500 to <1000 mg/g cr (prot)", ">=1000 mg/g cr (prot)"]
LABEL = {"egfr": "Baseline eGFR", "albuminuria": "Albuminuria", "proteinuria": "Proteinuria"}


def egfr_mid(s) -> float:
    if not isinstance(s, str):
        return np.nan
    a, b = (float(x) for x in re.findall(r"\d+", s)[:2])
    return (a + b + 1) / 2


def clinical() -> pd.DataFrame:
    m = pd.read_csv(p("metadata/kpmp_spatial_manifest.tsv"), sep="\t").drop_duplicates("redcap_id")
    return pd.DataFrame({"egfr": m.baseline_egfr.map(egfr_mid).values,
                         "albuminuria": m.albuminuria.map({k: i for i, k in enumerate(ALB)}).values,
                         "proteinuria": m.proteinuria.map({k: i for i, k in enumerate(PROT)}).values},
                        index=m.redcap_id.values)


def main() -> None:
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    n_boot = cfg["evaluate"]["n_boot"]
    P = pd.read_csv(p("results/16_evaluate_cv/participant_metrics.tsv"), sep="\t")
    P = P[P.model == MODEL]
    C = clinical()
    rows = []
    for k in ["pt_injury", "inflammation", "pt_transport"]:
        q = P[P.programme == k].set_index("participant").join(C, how="inner")
        for v in LABEL:
            d = q.dropna(subset=[v])
            y, e, x = d.mean_y.values, d.mean_pred.values, d[v].values
            rm, re_ = spearmanr(y, x)[0], spearmanr(e, x)[0]
            bm, be = [], []
            for _ in range(n_boot):
                i = rng.integers(0, len(d), len(d))
                bm.append(spearmanr(y[i], x[i])[0])
                be.append(spearmanr(e[i], x[i])[0])
            bm, be = np.array(bm), np.array(be)
            ci = lambda a: np.nanpercentile(a, [2.5, 97.5])  # noqa: E731
            rows.append(dict(programme=k, covariate=LABEL[v], n_participants=len(d),
                             rho_measured=rm, measured_lo=ci(bm)[0], measured_hi=ci(bm)[1],
                             rho_estimated=re_, estimated_lo=ci(be)[0], estimated_hi=ci(be)[1],
                             diff=re_ - rm, diff_lo=ci(be - bm)[0], diff_hi=ci(be - bm)[1],
                             p_estimated=spearmanr(e, x)[1]))
    out = pd.DataFrame(rows)
    out.to_csv(S / "clinical_correlations.tsv", sep="\t", index=False)
    print(out.round(3).to_string())
    write_provenance("16i_clinical", cfg, [p("results/16_evaluate_cv/participant_metrics.tsv"),
                                           p("metadata/kpmp_spatial_manifest.tsv")], [S / "clinical_correlations.tsv"])


if __name__ == "__main__":
    main()
