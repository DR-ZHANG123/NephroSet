"""Which predictions can be trusted? Ensemble disagreement and distance from the training images.

For the chosen model, every held-out spot has three predictions (one per training seed).
Disagreement is their standard deviation. Image novelty is the Mahalanobis distance of the
spot's encoder features (first principal components fitted on the fold's training sections)
from the training distribution. For both scores we report how agreement with the measured
programme changes as the least certain spots are set aside, within each participant, and
the coverage of intervals calibrated on validation participants.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import data as D  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
C = cfg["confidence"]
S = stage_dir("17_confidence")
CV = p(cfg["paths"]["cache"]) / "cv_predictions"
EXT = p(cfg["paths"]["cache"]) / "external_predictions"


def novelty(train_x: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    pca = PCA(n_components=k, random_state=cfg["seed"]).fit(train_x)
    z = pca.transform(train_x)
    mu, cov = z.mean(0), np.cov(z, rowvar=False) + 1e-6 * np.eye(k)
    ic = np.linalg.inv(cov)
    q = pca.transform(x) - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", q, ic, q))


def retained_curve(df: pd.DataFrame, score: str, prog: str) -> pd.DataFrame:
    rows = []
    for keep in C["keep_fracs"]:
        rs = []
        for s, g in df.groupby("section"):
            thr = g[score].quantile(keep)
            h = g[g[score] <= thr]
            if len(h) >= 30 and h[f"y_{prog}"].std() > 0:
                rs.append((g.participant.iloc[0], spearmanr(h[f"y_{prog}"], h[f"pred_{prog}"])[0],
                           np.mean(np.abs(h[f"y_{prog}"] - h[f"pred_{prog}"]))))
        r = pd.DataFrame(rs, columns=["participant", "rho", "mae"]).groupby("participant").mean()
        rows.append(dict(score=score, programme=prog, keep_frac=keep, rho=r.rho.mean(), mae=r.mae.mean(),
                         n_participants=len(r)))
    return pd.DataFrame(rows)


def main() -> None:
    m = C["model"]
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    part_of = dict(zip(idx.section, idx.participant))
    pr = pd.concat([pd.read_parquet(f) for f in sorted(CV.glob(f"{m}_fold*.parquet"))]).reset_index()
    cols = [c for c in pr.columns if c.startswith("pred_")]
    agg = pr.groupby(["section", "barcode"])[cols].agg(["mean", "std"])
    agg.columns = [f"{a}" if b == "mean" else f"sd_{a[5:]}" for a, b in agg.columns]
    fold_of = pr.groupby("section").fold.first()
    secs = list(agg.index.get_level_values(0).unique())
    data = D.load_cohort(p(cfg["paths"]["interim"]) / "kpmp_spots", p(cfg["paths"]["cache"]) / "img_kpmp", secs)
    rows = []
    for fi in sorted(fold_of.unique()):
        te = [s for s in secs if fold_of[s] == fi]
        tr = [s for s in secs if fold_of[s] != fi]
        rng = np.random.default_rng(cfg["seed"] + fi)
        sub = rng.choice(np.concatenate([data[s]["x"] for s in tr]), size=C["n_pca_fit"], replace=False)
        for s in te:
            t = data[s]["table"].copy()
            t["novelty"] = novelty(sub, data[s]["x"], C["n_pcs"])
            t["section"], t["participant"], t["fold"] = s, part_of[s], fi
            rows.append(t)
    T = pd.concat(rows).reset_index().set_index(["section", "barcode"]).join(agg).reset_index()
    curves = []
    for prog in ["pt_injury", "inflammation", "pt_transport"]:
        for score in [f"sd_{prog}", "novelty"]:
            curves.append(retained_curve(T, score, prog))
        T[f"abs_err_{prog}"] = np.abs(T[f"y_{prog}"] - T[f"pred_{prog}"])
    curves = pd.concat(curves)
    curves.to_csv(S / "retained_curves.tsv", sep="\t", index=False)
    # error versus disagreement, within participants
    rel = []
    for prog in ["pt_injury", "inflammation", "pt_transport"]:
        for score in [f"sd_{prog}", "novelty"]:
            r = T.groupby("participant").apply(lambda g: spearmanr(g[score], g[f"abs_err_{prog}"])[0])
            rel.append(dict(programme=prog, score=score, mean_rho_error=r.mean(), frac_positive=(r > 0).mean(),
                            n=len(r)))
    pd.DataFrame(rel).to_csv(S / "error_vs_score.tsv", sep="\t", index=False)
    # split-conformal intervals scaled by disagreement, calibrated per fold on the other folds
    cov = []
    for prog in ["pt_injury"]:
        sdc = f"sd_{prog}"
        for fi in sorted(T.fold.unique()):
            cal = T[T.fold != fi]
            te = T[T.fold == fi]
            sc_cal = np.abs(cal[f"y_{prog}"] - cal[f"pred_{prog}"]) / (cal[sdc] + C["sd_floor"])
            q = np.quantile(sc_cal, 1 - C["alpha"])
            half = q * (te[sdc] + C["sd_floor"])
            inside = np.abs(te[f"y_{prog}"] - te[f"pred_{prog}"]) <= half
            cov.append(dict(programme=prog, fold=fi, nominal=1 - C["alpha"], coverage=inside.mean(),
                            median_half_width=half.median(), q=q))
    pd.DataFrame(cov).to_csv(S / "conformal_cv.tsv", sep="\t", index=False)
    keep = ["section", "barcode", "participant", "fold", "novelty", "array_row", "array_col",
            "pxl_row_in_fullres", "pxl_col_in_fullres"] + \
        [c for c in T.columns if c.startswith(("y_", "pred_", "sd_", "comp_", "abs_err_"))]
    T[keep].to_parquet(p(cfg["paths"]["cache"]) / "cv_confidence_spots.parquet")
    print(curves.round(3).to_string())
    print(pd.DataFrame(rel).round(3).to_string())
    print(pd.DataFrame(cov).round(3).to_string())
    write_provenance("17_confidence", cfg, [], [S / "retained_curves.tsv", S / "conformal_cv.tsv"])


if __name__ == "__main__":
    main()
