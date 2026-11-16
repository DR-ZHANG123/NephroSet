"""Encode every spot of every section with the frozen encoder at two scales.

Usage: python 12_image_features.py <cohort>   (cohort = kpmp | gse211785)
Writes one .npz per section with encoder features (scale 1, scale 2) and morphology
descriptors, ordered as the section's spot table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.imagefeat import Encoder, crop, morphology  # noqa: E402
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
set_global_seed(cfg["seed"])
I = cfg["image"]
COHORT = sys.argv[1]
S = stage_dir(f"12_image_features_{COHORT}")
SPOTS = p(cfg["paths"]["interim"]) / f"{COHORT}_spots"
SEC = p(cfg["paths"]["interim"]) / f"{COHORT}_sections"
OUT = p(cfg["paths"]["cache"]) / f"img_{COHORT}"
OUT.mkdir(parents=True, exist_ok=True)


def crops_for(sid: str):
    cv2_threads()
    t = pd.read_parquet(SPOTS / f"{sid}.parquet")
    img_path = (SEC / sid / "image_path.txt").read_text().strip()
    um_per_px = float((SEC / sid / "um_per_px.txt").read_text()) if (SEC / sid / "um_per_px.txt").exists() \
        else None
    if um_per_px is None:
        import json
        sf = json.loads((SEC / sid / "scalefactors_json.json").read_text())
        um_per_px = 55.0 / sf["spot_diameter_fullres"]
    img = tifffile.imread(img_path)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    r, c = t.pxl_row_in_fullres.values, t.pxl_col_in_fullres.values
    h1 = I["scale1_um"] / 2 / um_per_px
    h2 = I["scale2_um"] / 2 / um_per_px
    c1 = np.stack([crop(img, a, b, h1) for a, b in zip(r, c)])
    c2 = np.stack([crop(img, a, b, h2) for a, b in zip(r, c)])
    mor = np.stack([morphology(x) for x in c1])
    return sid, c1, c2, mor, t.index.values


def cv2_threads():
    import cv2
    cv2.setNumThreads(1)


def main() -> None:
    sections = sorted(x.stem for x in SPOTS.glob("*.parquet"))
    todo = [s for s in sections if not (OUT / f"{s}.npz").exists()]
    enc = Encoder(I["encoder"], I["device"])
    bs = I["batch"]
    for i in range(0, len(todo), I["prefetch"]):
        chunk = Parallel(n_jobs=I["prefetch"], max_nbytes=None)(delayed(crops_for)(s) for s in todo[i:i + I["prefetch"]])
        for sid, c1, c2, mor, bc in chunk:
            f1 = np.concatenate([enc(c1[j:j + bs]) for j in range(0, len(c1), bs)])
            f2 = np.concatenate([enc(c2[j:j + bs]) for j in range(0, len(c2), bs)])
            np.savez_compressed(OUT / f"{sid}.npz", f1=f1.astype(np.float16), f2=f2.astype(np.float16),
                                morph=mor, barcode=bc)
            print(sid, len(bc), flush=True)
    done = sorted(OUT.glob("*.npz"))
    pd.DataFrame({"section": [d.stem for d in done]}).to_csv(S / "encoded_sections.tsv", sep="\t", index=False)
    write_provenance(f"12_image_features_{COHORT}", cfg, [], [S / "encoded_sections.tsv"],
                     extra={"encoder": I["encoder"], "n_sections": len(done)})


if __name__ == "__main__":
    main()
