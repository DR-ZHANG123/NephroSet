"""Token-region models: is the gap between the capture area and the crop edge due to tissue coverage?

For every held-out KPMP spot, the number of tissue tokens (tissue fraction >= 0.3, the mask used by all
token models) inside the 55 um capture area and among the 32 outermost tokens of the crop is counted.
Agreement of the inside-only and outermost-only attention models is then recomputed on spots in which
both regions hold at least 24 of their 32 tokens on tissue, per participant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import boot_ci, paired  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
S = stage_dir("16h_region_tissue")
CV = p(cfg["paths"]["cache"]) / "cv_predictions"
MODELS = ["tok_abmil_disc", "tok_abmil_outer32", "mlp"]
MIN_TOK = 24


def masks():
    r, c = np.divmod(np.arange(196), 14)
    d = np.hypot((r - 6.5) * 8, (c - 6.5) * 8)
    return d <= 27.5, d >= np.sort(d)[-32]


def rho(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b)[0] if len(a) >= 30 and np.std(a) > 0 and np.std(b) > 0 else np.nan


def main() -> None:
    disc, outer = masks()
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    part_of = dict(zip(idx.section, idx.participant))
    pr = {m: pd.concat([pd.read_parquet(f) for f in sorted(CV.glob(f"{m}_fold*.parquet"))]).reset_index()
          .groupby(["section", "barcode"]).mean(numeric_only=True) for m in MODELS}
    rows, cnt = [], []
    for s in pr["tok_abmil_disc"].index.get_level_values(0).unique():
        fold = int(pr["tok_abmil_disc"].loc[s].fold.iloc[0])
        z = np.load(p(cfg["paths"]["cache"]) / f"tok_kpmp_fold{fold}" / f"{s}.npz", allow_pickle=True)
        tis = z["col"][..., 0].astype(float) >= 0.3
        n_in, n_out = tis[:, disc].sum(1), tis[:, outer].sum(1)
        cnt.append(dict(section=s, participant=part_of[s], n_spots=len(n_in), mean_tissue_tokens_inside=n_in.mean(),
                        mean_tissue_tokens_outermost=n_out.mean(), frac_both_ge=((n_in >= MIN_TOK) & (n_out >= MIN_TOK)).mean()))
        keep = pd.Index(z["barcode"][(n_in >= MIN_TOK) & (n_out >= MIN_TOK)], name="barcode")
        t = pd.read_parquet(p(cfg["paths"]["interim"]) / "kpmp_spots" / f"{s}.parquet")
        for sub, bc in [("all spots", t.index), (f"both regions >= {MIN_TOK} tissue tokens", keep)]:
            for m in MODELS:
                q = t.loc[t.index.intersection(bc)].join(pr[m].loc[s][["pred_pt_injury", "pred_pt_transport"]], how="inner")
                for k in ["pt_injury", "pt_transport"]:
                    rows.append(dict(section=s, participant=part_of[s], subset=sub, model=m, programme=k,
                                     n_spots=len(q), rho=rho(q[f"y_{k}"].values, q[f"pred_{k}"].values)))
    C = pd.DataFrame(cnt)
    C.to_csv(S / "tissue_tokens.tsv", sep="\t", index=False)
    R = pd.DataFrame(rows).groupby(["participant", "subset", "model", "programme"]).rho.mean().reset_index()
    rng = np.random.default_rng(cfg["seed"])
    out = []
    for (sub, k), g in R.groupby(["subset", "programme"]):
        w = g.pivot(index="participant", columns="model", values="rho")
        for a, b in [("tok_abmil_disc", "tok_abmil_outer32"), ("tok_abmil_disc", "mlp"), ("tok_abmil_outer32", "mlp")]:
            out.append(dict(subset=sub, programme=k, a=a, b=b, mean_a=w[a].mean(), mean_b=w[b].mean(),
                            **paired(w[a], w[b], 2000, rng)))
    O = pd.DataFrame(out)
    O.to_csv(S / "contrasts.tsv", sep="\t", index=False)
    print(C[["mean_tissue_tokens_inside", "mean_tissue_tokens_outermost", "frac_both_ge"]].mean().round(2))
    print(O.round(3).to_string())
    write_provenance("16h_region_tissue", cfg, [], [S / "tissue_tokens.tsv", S / "contrasts.tsv"])


if __name__ == "__main__":
    main()
