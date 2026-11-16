"""Evaluate predictions on the external cohort (GSE211785), one section per donor."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import ext_file, PROGS, boot_ci, paired, section_metrics  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
E = cfg["evaluate"]
S = stage_dir("18_evaluate_external")
EXT = p(cfg["paths"]["cache"]) / "external_predictions"


def main() -> None:
    sec = pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t")
    sp = p(cfg["paths"]["interim"]) / "gse211785_spots"
    truth = pd.concat([pd.read_parquet(sp / f"{s}.parquet") for s in sec.section]).reset_index() \
        .set_index(["section", "barcode"])
    rng = np.random.default_rng(cfg["seed"])
    per, summ = [], []
    models = [m for m in E["models"] if ext_file(EXT, m).exists()]
    for m in models:
        pr = pd.read_parquet(ext_file(EXT, m)).reset_index()
        pr = pr.groupby(["section", "barcode"]).mean(numeric_only=True)
        mg = truth.join(pr[[f"pred_{k}" for k in PROGS]], how="inner").reset_index()
        sm = section_metrics(mg, E["pt_rich_min"]).assign(model=m)
        sm = sm.merge(sec[["section", "status_label"]], on="section")
        per.append(sm)
        for k in PROGS:
            q = sm[sm.programme == k]
            for col in ["rho", "rho_pt_rich", "rho_beyond_comp", "nmae"]:
                lo, hi = boot_ci(q[col].values, E["n_boot"], rng)
                summ.append(dict(model=m, programme=k, metric=col, mean=q[col].mean(), lo=lo, hi=hi,
                                 n_donors=int(q[col].notna().sum())))
            summ.append(dict(model=m, programme=k, metric="between_donor_r",
                             mean=np.corrcoef(q.mean_y, q.mean_pred)[0, 1], lo=np.nan, hi=np.nan, n_donors=len(q)))
    P = pd.concat(per)
    P.to_csv(S / "section_metrics.tsv", sep="\t", index=False)
    Sm = pd.DataFrame(summ)
    Sm.to_csv(S / "summary.tsv", sep="\t", index=False)
    cons = []
    for a, b in E["contrasts"]:
        if a in models and b in models:
            for k in PROGS:
                for col in ["rho", "rho_beyond_comp"]:
                    x = P[(P.model == a) & (P.programme == k)].set_index("section")[col]
                    y = P[(P.model == b) & (P.programme == k)].set_index("section")[col]
                    cons.append(dict(a=a, b=b, programme=k, metric=col, **paired(x, y, E["n_boot"], rng)))
    pd.DataFrame(cons).to_csv(S / "contrasts.tsv", sep="\t", index=False)
    print(Sm[Sm.metric.isin(["rho", "rho_beyond_comp"])].pivot_table(index="model", columns=["programme", "metric"],
                                                                     values="mean").round(3).to_string())
    write_provenance("18_evaluate_external", cfg, [], [S / "summary.tsv", S / "contrasts.tsv"])


if __name__ == "__main__":
    main()
