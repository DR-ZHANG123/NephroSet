"""Example sections for display: H&E thumbnail, measured and held-out predicted programme maps.

KPMP: the diabetic kidney disease section whose within-section agreement for the chosen model
is closest to the median of diabetic kidney disease participants. External: the donor closest
to the external median. Writes thumbnails (PNG) and spot tables for the figure scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evaluate import ext_file  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
X = cfg["examples"]
S = stage_dir("19_examples")
CV = p(cfg["paths"]["cache"]) / "cv_predictions"
EXT = p(cfg["paths"]["cache"]) / "external_predictions"


def thumb(sid: str, cohort: str, t: pd.DataFrame, tag: str) -> pd.DataFrame:
    sec = p(cfg["paths"]["interim"]) / f"{cohort}_sections" / sid
    img = tifffile.imread((sec / "image_path.txt").read_text().strip())
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    img = img[..., :3]
    r, c = t.pxl_row_in_fullres.values, t.pxl_col_in_fullres.values
    pad = 300
    r0, r1 = max(int(r.min()) - pad, 0), min(int(r.max()) + pad, img.shape[0])
    c0, c1 = max(int(c.min()) - pad, 0), min(int(c.max()) + pad, img.shape[1])
    sub = img[r0:r1, c0:c1]
    f = X["thumb_px"] / max(sub.shape[:2])
    th = cv2.resize(sub, (int(sub.shape[1] * f), int(sub.shape[0] * f)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(S / f"{tag}_he.png"), cv2.cvtColor(th, cv2.COLOR_RGB2BGR))
    out = t.copy()
    out["tx"] = (c - c0) * f
    out["ty"] = (r - r0) * f
    um_px = float((sec / "um_per_px.txt").read_text()) if (sec / "um_per_px.txt").exists() else \
        55.0 / __import__("json").loads((sec / "scalefactors_json.json").read_text())["spot_diameter_fullres"]
    out["um_per_thumb_px"] = um_px / f
    return out


def main() -> None:
    m = X["model"]
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    pm = pd.read_csv(p("results/16_evaluate_cv/participant_metrics.tsv"), sep="\t")
    q = pm[(pm.model == m) & (pm.programme == "pt_injury")].merge(
        idx[["participant", "section", "primary_adjudicated_category"]], on="participant")
    q = q[(q.primary_adjudicated_category == "Diabetic Kidney Disease") & (q.n_spots >= X["min_spots"])]
    sid = q.iloc[(q.rho - q.rho.median()).abs().argsort().iloc[0]].section
    pr = pd.concat([pd.read_parquet(f) for f in sorted(CV.glob(f"{m}_fold*.parquet"))])
    pr = pr[pr.section == sid].reset_index().groupby("barcode").mean(numeric_only=True)
    t = pd.read_parquet(p(cfg["paths"]["interim"]) / "kpmp_spots" / f"{sid}.parquet").join(pr, how="inner")
    k = thumb(sid, "kpmp", t, "kpmp")
    k["section"] = sid
    k.reset_index().to_csv(S / "kpmp_spots.tsv", sep="\t", index=False)
    em = pd.read_csv(p("results/18_evaluate_external/section_metrics.tsv"), sep="\t")
    e = em[(em.model == m) & (em.programme == "pt_injury")]
    esid = e.iloc[(e.rho - e.rho.median()).abs().argsort().iloc[0]].section
    pr = pd.read_parquet(ext_file(EXT, m))
    pr = pr[pr.section == esid].reset_index().groupby("barcode").mean(numeric_only=True)
    t = pd.read_parquet(p(cfg["paths"]["interim"]) / "gse211785_spots" / f"{esid}.parquet").join(pr, how="inner")
    e2 = thumb(esid, "gse211785", t, "external")
    e2["section"] = esid
    e2.reset_index().to_csv(S / "external_spots.tsv", sep="\t", index=False)
    write_provenance("19_examples", cfg, [], [S / "kpmp_spots.tsv", S / "external_spots.tsv"],
                     extra={"kpmp_section": sid, "external_section": esid})


if __name__ == "__main__":
    main()
