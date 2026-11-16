"""External cohort (GSE211785, Visium on fresh-frozen nephrectomy tissue): spot tables and images.

Counts come from the deposited Seurat count matrix (exported to Matrix Market by
13a_export_counts.R); spot centres come from the Loupe alignment JSON of each section, whose
array row/column are matched to barcodes through the Visium v1 barcode layout. The layout is
read from any KPMP v1 tissue_positions file (identical for all v1 slides). Sections without
a deposited image (HK2671) are left out.
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
from scipy.io import mmread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402
from lib.visium import composition, normalise, programme_scores  # noqa: E402

cfg = load_config()
RAW = p(cfg["paths"]["raw"]) / "GSE211785"
INT = p(cfg["paths"]["interim"])
SEC = INT / "gse211785_sections"
SPOTS = INT / "gse211785_spots"
SEC.mkdir(parents=True, exist_ok=True)
SPOTS.mkdir(parents=True, exist_ok=True)
S = stage_dir("13_gse211785_prep")


def v1_layout() -> pd.DataFrame:
    for f in sorted((INT / "kpmp_sections").glob("*/tissue_positions.csv")):
        t = pd.read_csv(f)
        if len(t) == 4992 and re.search(r"V1\d", f.parent.name):
            t["bc"] = t.barcode.str.replace(r"-\d+$", "", regex=True)
            return t[["bc", "array_row", "array_col"]]
    raise RuntimeError("no v1 layout found")


def pairs() -> dict[str, tuple[Path, Path]]:
    tifs = {re.search(r"HK\d{4}", f.name).group(0): f for f in RAW.glob("*.tif.gz")}
    js = {}
    for f in RAW.glob("*.json.gz"):
        m = re.search(r"HK\d{4}", f.name)
        if m:
            js[m.group(0)] = f
    # series-level JSONs carry the slide/area only; match them to the TIFF of the same slide/area
    for f in RAW.glob("GSE211785_V1*.json.gz"):
        key = re.sub(r"[-_]", "", re.search(r"V\d{2}Y\d{2}[-_]0?\d{2,3}[-_][A-D]1", f.name).group(0)).replace("079", "79")
        for hk, t in tifs.items():
            tk = re.search(r"V\d{2}Y\d{2}[-_]0?\d{2,3}[-_][A-D]1", t.name)
            if tk and re.sub(r"[-_]", "", tk.group(0)).replace("079", "79") == key:
                js.setdefault(hk, f)
    return {hk: (tifs[hk], js[hk]) for hk in tifs if hk in js}


def main() -> None:
    mm = mmread(INT / "gse211785_counts" / "counts.mtx").tocsr().T  # spots x genes
    genes = pd.read_csv(INT / "gse211785_counts" / "genes.tsv", header=None)[0].values
    bcs = pd.read_csv(INT / "gse211785_counts" / "barcodes.tsv", header=None)[0].values
    meta = pd.read_csv(RAW / "GSE211785_ST_metadata.txt.gz", sep="\t", index_col=0)
    meta = meta.loc[[b for b in bcs if b in meta.index]]
    lay = v1_layout()
    prof = pd.read_csv(p("results/11a_reference/reference_profiles_cp10k.tsv.gz"), sep="\t", index_col=0)
    mk = pd.read_csv(p("results/11a_reference/markers.tsv"), sep="\t")
    rows = []
    for hk, (tifgz, jsf) in sorted(pairs().items()):
        sid = hk
        d = SEC / sid
        d.mkdir(exist_ok=True)
        tif = INT / "gse211785_images" / tifgz.name.replace(".gz", "")
        tif.parent.mkdir(exist_ok=True)
        if not tif.exists():
            with gzip.open(tifgz, "rb") as a, open(tif, "wb") as b:
                shutil.copyfileobj(a, b, 1 << 24)
        j = json.load(gzip.open(jsf))
        ol = pd.DataFrame(j["oligo"]).rename(columns={"row": "array_row", "col": "array_col"})
        ol = ol.merge(lay, on=["array_row", "array_col"])
        dia = float(np.median(ol.dia))
        sub = meta[meta["orig.ident"] == f"{hk}_ST"]
        sub_bc = sub.index.str.replace(r"-\d+_\d+$", "", regex=True)
        pos = ol.set_index("bc").reindex(sub_bc)
        ok = pos.imageX.notna().values
        img = tifffile.imread(tif)
        if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        h, w = img.shape[:2]
        best = None
        for swap in (False, True):
            r = (pos.imageX if swap else pos.imageY).values[ok]
            c = (pos.imageY if swap else pos.imageX).values[ok]
            inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            px = img[r[inside].astype(int), c[inside].astype(int), :3].astype(float)
            on = ((px.mean(1) < 220) | ((px.max(1) - px.min(1)) > 20)).mean() * inside.mean()
            if best is None or on > best[0]:
                best = (on, swap)
        frac_on, swap = best
        idx = np.where(np.isin(bcs, sub.index[ok]))[0]
        ad = sc.AnnData(mm[idx].astype(np.float32))
        ad.obs_names = bcs[idx]
        ad.var_names = genes
        umi = np.asarray(ad.X.sum(1)).ravel()
        ad = ad[umi >= cfg["visium"]["min_umi"]].copy()
        ad.obs["umi"] = np.asarray(ad.X.sum(1)).ravel()
        ad = normalise(ad)
        y = programme_scores(ad, cfg["programmes"], cfg["visium"]["ctrl_size"], cfg["seed"])
        comp = composition(ad, prof, mk)
        pp = pos.copy()
        pp.index = sub.index
        pp = pp.loc[ad.obs_names]
        t = pd.DataFrame({"section": sid, "array_row": pp.array_row.values, "array_col": pp.array_col.values,
                          "pxl_row_in_fullres": (pp.imageX if swap else pp.imageY).values,
                          "pxl_col_in_fullres": (pp.imageY if swap else pp.imageX).values},
                         index=ad.obs_names)
        t = pd.concat([t, ad.obs[["umi"]], y, comp], axis=1)
        t.index.name = "barcode"
        t.to_parquet(SPOTS / f"{sid}.parquet")
        (d / "image_path.txt").write_text(str(tif.resolve()))
        (d / "um_per_px.txt").write_text(str(55.0 / dia))
        rows.append(dict(section=sid, status_label=sub.Status.iloc[0], n_spots=len(t), um_per_px=55.0 / dia,
                         frac_spots_on_tissue=frac_on, swapped=swap, img_h=h, img_w=w))
        print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(S / "sections.tsv", sep="\t", index=False)
    write_provenance("13_gse211785_prep", cfg, [RAW / "GSE211785_ST_metadata.txt.gz"], [S / "sections.tsv"])


if __name__ == "__main__":
    main()
