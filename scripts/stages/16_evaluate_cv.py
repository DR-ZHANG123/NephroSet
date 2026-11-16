"""Summarise cross-validated predictions on KPMP at the participant level.

Predictions of the three training seeds are averaged per spot (an ensemble). Metrics are
computed within each section, averaged within participant, then over participants with equal
weight. Model contrasts are paired by participant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import PROGS, boot_ci, paired, participant_table, section_metrics  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
E = cfg["evaluate"]
S = stage_dir("16_evaluate_cv")
CV = p(cfg["paths"]["cache"]) / "cv_predictions"


def load_truth(sections) -> pd.DataFrame:
    sp = p(cfg["paths"]["interim"]) / "kpmp_spots"
    return pd.concat([pd.read_parquet(sp / f"{s}.parquet") for s in sections])


def main() -> None:
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    part_of = dict(zip(idx.section, idx.participant))
    models = [m for m in E["models"] if (CV / f"{m}_fold0.parquet").exists()]
    rng = np.random.default_rng(cfg["seed"])
    truth = None
    allpart, summ = [], []
    for m in models:
        pr = pd.concat([pd.read_parquet(f) for f in sorted(CV.glob(f"{m}_fold*.parquet"))])
        pr = pr.reset_index().groupby(["section", "barcode"]).mean(numeric_only=True)
        if truth is None:
            truth = load_truth(pr.index.get_level_values(0).unique()).reset_index().set_index(["section", "barcode"])
        mg = truth.join(pr[[f"pred_{k}" for k in PROGS]], how="inner").reset_index()
        sec = section_metrics(mg, E["pt_rich_min"])
        pt = participant_table(sec, part_of).assign(model=m)
        allpart.append(pt)
        for k in PROGS:
            q = pt[pt.programme == k]
            for col in ["rho", "rho_pt_rich", "rho_beyond_comp", "nmae"]:
                lo, hi = boot_ci(q[col].values, E["n_boot"], rng)
                summ.append(dict(model=m, programme=k, metric=col, mean=q[col].mean(), lo=lo, hi=hi,
                                 n_participants=int(q[col].notna().sum())))
            # between-participant agreement of mean levels
            r = np.corrcoef(q.mean_y, q.mean_pred)[0, 1]
            summ.append(dict(model=m, programme=k, metric="between_participant_r", mean=r, lo=np.nan, hi=np.nan,
                             n_participants=len(q)))
    P = pd.concat(allpart)
    P.to_csv(S / "participant_metrics.tsv", sep="\t", index=False)
    Sm = pd.DataFrame(summ)
    Sm.to_csv(S / "summary.tsv", sep="\t", index=False)
    cons = []
    for a, b in E["contrasts"]:
        if a not in models or b not in models:
            continue
        for k in PROGS:
            for col in ["rho", "rho_pt_rich", "rho_beyond_comp"]:
                x = P[(P.model == a) & (P.programme == k)].set_index("participant")[col]
                y = P[(P.model == b) & (P.programme == k)].set_index("participant")[col]
                cons.append(dict(a=a, b=b, programme=k, metric=col, **paired(x, y, E["n_boot"], rng)))
    pd.DataFrame(cons).to_csv(S / "contrasts.tsv", sep="\t", index=False)
    print(Sm[Sm.metric.isin(["rho", "rho_beyond_comp"])].pivot_table(index="model", columns=["programme", "metric"],
                                                                     values="mean").round(3).to_string())
    print(pd.DataFrame(cons).query("metric=='rho' and programme=='pt_injury'").round(4).to_string())
    write_provenance("16_evaluate_cv", cfg, [], [S / "summary.tsv", S / "contrasts.tsv"])


if __name__ == "__main__":
    main()
