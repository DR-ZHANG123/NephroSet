"""Measured programme scores and estimated composition for every KPMP Visium spot on tissue."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402
from lib.visium import composition, normalise, programme_scores  # noqa: E402

cfg = load_config()
S = stage_dir("11b_kpmp_targets")
SEC = p(cfg["paths"]["interim"]) / "kpmp_sections"
OUT = p(cfg["paths"]["interim"]) / "kpmp_spots"
OUT.mkdir(parents=True, exist_ok=True)


def one(sid: str, prof: pd.DataFrame, mk: pd.DataFrame) -> dict:
    d = SEC / sid
    ad = sc.read_10x_h5(d / "filtered_feature_bc_matrix.h5")
    pos = pd.read_csv(d / "tissue_positions.csv").set_index("barcode")
    ad = ad[ad.obs_names.isin(pos.index[pos.in_tissue == 1])].copy()
    umi = np.asarray(ad.X.sum(1)).ravel()
    ad = ad[umi >= cfg["visium"]["min_umi"]].copy()
    if ad.n_obs < cfg["kpmp_qc"]["min_spots"]:
        return dict(section=sid, n_spots=int(ad.n_obs), status="too_few_spots")
    ad.obs["umi"] = np.asarray(ad.X.sum(1)).ravel()
    ad = normalise(ad)
    y = programme_scores(ad, cfg["programmes"], cfg["visium"]["ctrl_size"], cfg["seed"])
    comp = composition(ad, prof, mk)
    t = pd.concat([pos.loc[ad.obs_names, ["array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]],
                   ad.obs[["umi"]], y, comp], axis=1)
    t.insert(0, "section", sid)
    t.index.name = "barcode"
    t.to_parquet(OUT / f"{sid}.parquet")
    return dict(section=sid, n_spots=int(ad.n_obs), status="ok", median_umi=float(t.umi.median()),
                mean_pt=float(t.comp_PT.mean()))


def main() -> None:
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    idx = idx[idx.status == "ok"]
    prof = pd.read_csv(p("results/11a_reference/reference_profiles_cp10k.tsv.gz"), sep="\t", index_col=0)
    mk = pd.read_csv(p("results/11a_reference/markers.tsv"), sep="\t")
    res = Parallel(n_jobs=24)(delayed(one)(s, prof, mk) for s in idx.section)
    r = pd.DataFrame(res)
    r.to_csv(S / "spot_summary.tsv", sep="\t", index=False)
    print(r.status.value_counts(), r.n_spots.describe())
    write_provenance("11b_kpmp_targets", cfg, [p("results/10_kpmp_index/sections.tsv")], [S / "spot_summary.tsv"])


if __name__ == "__main__":
    main()
