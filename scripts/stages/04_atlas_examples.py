"""Example window of a DKD Xenium section with one PT cell's tubular unit and halo (for display)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.neighbourhood import features  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
X = cfg["examples"]
S = stage_dir("04_atlas_examples")


def main() -> None:
    df = pd.read_parquet(p(cfg["paths"]["interim"]) / "atlas_cells" / f"{X['atlas_section']}.parquet")
    feats = pd.read_parquet(p(cfg["paths"]["interim"]) / "atlas_features" / f"{X['atlas_section']}.parquet")
    # choose a PT cell whose unit has a mixed halo and whose window lies inside the section
    cand = feats[(feats.D_halo_n_types >= 3) & (feats.D_unit_size.between(8, 25))]
    t = int(cand.sort_values("yunit_pt_injury", ascending=False).cell.iloc[len(cand) // 10])
    xy = df[["x_um", "y_um"]].values
    units: list = []
    features(xy, df.coarse.values.astype(object), np.array([t]), cfg["atlas"], units_out=units)
    unit = set(units[0].tolist())
    c = xy[t]
    w = X["window_um"] / 2
    m = np.all(np.abs(xy - c) <= w, axis=1)
    sub = df.loc[m, ["x_um", "y_um", "coarse"]].copy()
    sub["x_um"] -= c[0]
    sub["y_um"] -= c[1]
    sub["in_unit"] = [i in unit for i in np.where(m)[0]]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy[list(unit)]).query(xy[m], k=1)
    sub["in_halo"] = (d <= cfg["atlas"]["d_halo"]) & sub.coarse.isin(["Endothelial", "Fibroblast", "Mural", "Immune"]).values
    sub["is_target"] = np.where(m)[0] == t
    sub.to_csv(S / "example_window.tsv", sep="\t", index=False)
    write_provenance("04_atlas_examples", cfg, [], [S / "example_window.tsv"], extra={"cell": t})


if __name__ == "__main__":
    main()
