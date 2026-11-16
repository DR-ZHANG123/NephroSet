"""Second-centre results donor by donor, and their stability over training seeds.

Usage: python 18d_external_donors.py

donor_metrics.tsv  one row per donor and programme: within-section agreement of the main image
                   models, composition-adjusted agreement of NephroSet, spread of the measured
                   score and diagnosis.
seed_stability.tsv for each model with ten-seed external predictions: donor-mean agreement of
                   every single seed, of the first three seeds averaged and of all ten averaged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import PROGS, ext_file, section_metrics  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
E = cfg["evaluate"]
S = stage_dir("18d_external_donors")
EXT = p(cfg["paths"]["cache"]) / "external_predictions"
MODELS = ["ridge_image", "mlp", "deepsets", "tok_abmil", "tok_abmil_ctx", "knn_encoder", "istar", "bleep"]


def metrics(truth, pr):
    pr = pr.reset_index().groupby(["section", "barcode"]).mean(numeric_only=True)
    mg = truth.join(pr[[f"pred_{k}" for k in PROGS]], how="inner").reset_index()
    return section_metrics(mg, E["pt_rich_min"])


def main() -> None:
    sec = pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t")
    sp = p(cfg["paths"]["interim"]) / "gse211785_spots"
    truth = pd.concat([pd.read_parquet(sp / f"{s}.parquet") for s in sec.section]).reset_index() \
        .set_index(["section", "barcode"])
    sd = truth.groupby(level=0)[[f"y_{k}" for k in PROGS]].std()
    rows = []
    for m in MODELS:
        f = ext_file(EXT, m)
        if f.exists():
            rows.append(metrics(truth, pd.read_parquet(f)).assign(model=m))
    D = pd.concat(rows)
    wide = D.pivot_table(index=["section", "programme"], columns="model", values="rho").add_prefix("rho_")
    adj = D[D.model == "tok_abmil_ctx"].set_index(["section", "programme"])[["rho_beyond_comp", "n_spots"]] \
        .rename(columns={"rho_beyond_comp": "rho_beyond_comp_tok_abmil_ctx"})
    out = wide.join(adj).reset_index().merge(sec[["section", "status_label"]], on="section")
    out["sd_measured"] = [sd.loc[s, f"y_{k}"] for s, k in zip(out.section, out.programme)]
    out.to_csv(S / "donor_metrics.tsv", sep="\t", index=False)
    stab = []
    for m in ["mlp", "tok_abmil", "tok_abmil_ctx", "tok_hyper_anat_ctx", "deepsets", "tok_pred_mean"]:
        f = EXT / f"{m}_s10.parquet"
        if not f.exists():
            continue
        pr = pd.read_parquet(f)
        seeds = sorted(pr.seed.unique())
        sets = {f"seed_{s}": [s] for s in seeds}
        sets.update({"mean_first3": seeds[:3], "mean_all10": seeds})
        for lab, ss in sets.items():
            q = metrics(truth, pr[pr.seed.isin(ss)])
            for k in PROGS:
                stab.append(dict(model=m, predictor=lab, programme=k, n_seeds=len(ss),
                                 rho=q[q.programme == k].rho.mean()))
    st = pd.DataFrame(stab)
    st.to_csv(S / "seed_stability.tsv", sep="\t", index=False)
    single = st[st.n_seeds == 1].groupby(["model", "programme"]).rho.agg(["mean", "std", "min", "max"])
    print(single.round(3).to_string())
    print(st[st.n_seeds > 1].pivot_table(index=["model", "programme"], columns="predictor", values="rho").round(3))
    write_provenance("18d_external_donors", cfg, [], [S / "donor_metrics.tsv", S / "seed_stability.tsv"])


if __name__ == "__main__":
    main()
