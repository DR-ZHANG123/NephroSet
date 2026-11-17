"""Second centre: is lower agreement explained by image-to-array registration or tissue coverage?

For every external spot, the tissue fraction inside the 55 um capture spot is read from the image
tokens (mean tissue fraction of the 32 tokens whose centres lie inside the spot). Agreement of
NephroSet and the spot-only network, and their paired difference, are recomputed (i) on all spots,
(ii) on spots with at least half of the capture area covered by tissue in the image, (iii) on
sections in which at least 85% of the in-tissue array spots fall on tissue in the image, and
(iv) on both, and (v) on the donors who do not also contribute sections to the single-cell atlas.
Primary three-seed ensembles throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import PROGS, boot_ci, paired, read_ext, section_metrics  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
E = cfg["evaluate"]
S = stage_dir("18e_external_tissue")
EXT = p(cfg["paths"]["cache"]) / "external_predictions"
MODELS = ["tok_abmil_ctx", "mlp"]
SPOT_MIN, SECTION_MIN = 0.5, 0.85


def disc_tissue(section: str) -> pd.Series:
    z = np.load(p(cfg["paths"]["cache"]) / "tok_gse211785" / f"{section}.npz", allow_pickle=True)
    r, c = np.divmod(np.arange(196), 14)
    disc = np.hypot((r - 6.5) * 8, (c - 6.5) * 8) <= 27.5
    tis = z["col"][..., 0].astype(float)[:, disc].mean(1)
    return pd.Series(tis, index=pd.Index(z["barcode"], name="barcode"))


def main() -> None:
    sec = pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t")
    sp = p(cfg["paths"]["interim"]) / "gse211785_spots"
    truth = pd.concat([pd.read_parquet(sp / f"{s}.parquet").assign(section=s, disc_tissue=disc_tissue(s))
                       for s in sec.section]).reset_index().set_index(["section", "barcode"])
    good = set(sec.section[sec.frac_spots_on_tissue >= SECTION_MIN])
    atlas = set(pd.read_csv(p("results/01_atlas_extract/atlas_samples.tsv"), sep="\t").donor.astype(str))
    pd.DataFrame(dict(section=sorted(set(sec.section) & atlas))).to_csv(S / "donors_shared_with_atlas.tsv",
                                                                       sep="\t", index=False)
    filters = {"all spots": lambda t: t,
               f"spots with image tissue >= {SPOT_MIN}": lambda t: t[t.disc_tissue >= SPOT_MIN],
               f"sections with >= {SECTION_MIN:.0%} spots on tissue": lambda t: t[t.index.get_level_values(0).isin(good)],
               "both": lambda t: t[(t.disc_tissue >= SPOT_MIN) & t.index.get_level_values(0).isin(good)],
               "donors not in the single-cell atlas": lambda t: t[~t.index.get_level_values(0).isin(atlas)]}
    rng = np.random.default_rng(cfg["seed"])
    preds = {m: read_ext(EXT, m).reset_index().groupby(["section", "barcode"]).mean(numeric_only=True) for m in MODELS}
    rows, cons, spots = [], [], []
    for lab, fn in filters.items():
        t = fn(truth)
        spots.append(dict(filter=lab, n_spots=len(t), n_sections=t.index.get_level_values(0).nunique(),
                          frac_spots=len(t) / len(truth)))
        sm = {}
        for m in MODELS:
            mg = t.join(preds[m][[f"pred_{k}" for k in PROGS]], how="inner").reset_index()
            sm[m] = section_metrics(mg, E["pt_rich_min"])
            for k in PROGS:
                q = sm[m][sm[m].programme == k]
                for col in ["rho", "rho_beyond_comp"]:
                    lo, hi = boot_ci(q[col].values, E["n_boot"], rng)
                    rows.append(dict(filter=lab, model=m, programme=k, metric=col, mean=q[col].mean(), lo=lo, hi=hi,
                                     n_donors=int(q[col].notna().sum())))
        for k in PROGS:
            a = sm["tok_abmil_ctx"][sm["tok_abmil_ctx"].programme == k].set_index("section").rho
            b = sm["mlp"][sm["mlp"].programme == k].set_index("section").rho
            cons.append(dict(filter=lab, programme=k, a="tok_abmil_ctx", b="mlp", **paired(a, b, E["n_boot"], rng)))
    pd.DataFrame(rows).to_csv(S / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(cons).to_csv(S / "contrasts.tsv", sep="\t", index=False)
    pd.DataFrame(spots).to_csv(S / "spots_kept.tsv", sep="\t", index=False)
    print(pd.DataFrame(spots).round(3).to_string())
    print(pd.DataFrame(rows).pivot_table(index=["filter", "model"], columns=["programme", "metric"], values="mean").round(3))
    print(pd.DataFrame(cons).round(3).to_string())
    write_provenance("18e_external_tissue", cfg, [], [S / "summary.tsv", S / "contrasts.tsv"])


if __name__ == "__main__":
    main()
