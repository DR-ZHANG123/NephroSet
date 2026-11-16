"""Patch-token features inside each spot crop (8 um tokens), for structure models within the spot.

Usage: python 12b_token_features.py <cohort>   (cohort = kpmp | gse211785)

The scale-1 crop (112 um, 224 px, 0.5 um/px) is encoded with the same frozen encoder; the
14 x 14 patch tokens (16 px = 8 um each) are projected onto principal components fitted once on
tokens from a random subset of KPMP sections (unsupervised, no targets), and stored with three
colour descriptors per token (tissue fraction, haematoxylin and eosin optical density).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from skimage.color import rgb2hed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.imagefeat import MEAN, STD, Encoder  # noqa: E402
from lib.repro import load_config, p, set_global_seed, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
set_global_seed(cfg["seed"])
I, T = cfg["image"], cfg["tokens"]
COHORT = sys.argv[1]
S = stage_dir(f"12b_token_features_{COHORT}")
SPOTS = p(cfg["paths"]["interim"]) / f"{COHORT}_spots"
OUT = p(cfg["paths"]["cache"]) / f"tok_{COHORT}"
OUT.mkdir(parents=True, exist_ok=True)
PCA_F = p(cfg["paths"]["cache"]) / "tok_pca_kpmp.npz"

SEC = p(cfg["paths"]["interim"]) / f"{COHORT}_sections"


def crops(sid: str):
    """Scale-1 crops exactly as in the spot-level encoding (same centres, size and resampling)."""
    import json

    import cv2
    import tifffile
    from lib.imagefeat import crop
    cv2.setNumThreads(1)
    t = pd.read_parquet(SPOTS / f"{sid}.parquet")
    img_path = (SEC / sid / "image_path.txt").read_text().strip()
    if (SEC / sid / "um_per_px.txt").exists():
        um_per_px = float((SEC / sid / "um_per_px.txt").read_text())
    else:
        sf = json.loads((SEC / sid / "scalefactors_json.json").read_text())
        um_per_px = 55.0 / sf["spot_diameter_fullres"]
    img = tifffile.imread(img_path)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    h1 = I["scale1_um"] / 2 / um_per_px
    c1 = np.stack([crop(img, a, b, h1) for a, b in zip(t.pxl_row_in_fullres.values, t.pxl_col_in_fullres.values)])
    return sid, c1, t.index.values


def colour(c1: np.ndarray) -> np.ndarray:
    """Per-token tissue fraction, haematoxylin and eosin OD (n, 196, 3)."""
    f = c1.astype(np.float32) / 255.0
    tissue = ((f.mean(-1) < 0.85) | ((f.max(-1) - f.min(-1)) > 0.08)).astype(np.float32)
    hed = np.stack([rgb2hed(x)[..., :2] for x in f])
    arr = np.concatenate([tissue[..., None], hed], -1)  # n,224,224,3
    n = arr.shape[0]
    return arr.reshape(n, 14, 16, 14, 16, 3).mean((2, 4)).reshape(n, 196, 3)


@torch.no_grad()
def tokens(enc: Encoder, batch: np.ndarray) -> np.ndarray:
    x = (batch.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(enc.device).half()
    h = enc.model(pixel_values=x).last_hidden_state
    assert h.shape[1] >= 197, h.shape
    return h[:, -196:].float()


def main() -> None:
    sections = sorted(x.stem for x in SPOTS.glob("*.parquet"))
    enc = Encoder(I["encoder"], I["device"])
    bs = I["batch"]
    if not PCA_F.exists():
        assert COHORT == "kpmp", "fit the token projection on KPMP first"
        rng = np.random.default_rng(cfg["seed"])
        fit = list(rng.choice(sections, T["n_fit_sections"], replace=False))
        chunk = Parallel(n_jobs=I["prefetch"], max_nbytes=None)(delayed(crops)(s) for s in fit)
        acc = []
        for _, c1, _ in chunk:
            idx = rng.choice(len(c1), min(len(c1), T["n_fit_spots"]), replace=False)
            for j in range(0, len(idx), bs):
                h = tokens(enc, c1[idx[j:j + bs]]).reshape(-1, 1024)
                acc.append(h[torch.randperm(h.shape[0])[:4000]].cpu())
        X = torch.cat(acc).double()
        mu = X.mean(0)
        U, Sv, V = torch.linalg.svd(X - mu, full_matrices=False)
        k = T["n_pcs"]
        np.savez(PCA_F, mu=mu.numpy().astype(np.float32), W=V[:k].T.numpy().astype(np.float32),
                 var=(Sv[:k] ** 2 / (Sv ** 2).sum()).numpy(), fit_sections=np.array(fit))
        print("pca fitted on", X.shape[0], "tokens; variance kept", float((Sv[:k] ** 2).sum() / (Sv ** 2).sum()))
    pz = np.load(PCA_F)
    mu = torch.as_tensor(pz["mu"], device=enc.device)
    W = torch.as_tensor(pz["W"], device=enc.device)
    todo = [s for s in sections if not (OUT / f"{s}.npz").exists()]
    for i in range(0, len(todo), I["prefetch"]):
        chunk = Parallel(n_jobs=I["prefetch"], max_nbytes=None)(delayed(crops)(s) for s in todo[i:i + I["prefetch"]])
        for sid, c1, bc in chunk:
            z = np.concatenate([((tokens(enc, c1[j:j + bs]) - mu) @ W).cpu().numpy() for j in range(0, len(c1), bs)])
            col = colour(c1)
            np.savez(OUT / f"{sid}.npz", tok=z.astype(np.float16), col=col.astype(np.float16), barcode=bc)
            print(sid, len(bc), flush=True)
    done = sorted(OUT.glob("*.npz"))
    pd.DataFrame({"section": [d.stem for d in done]}).to_csv(S / "encoded_sections.tsv", sep="\t", index=False)
    write_provenance(f"12b_token_features_{COHORT}", cfg, [], [S / "encoded_sections.tsv"],
                     extra={"encoder": I["encoder"], "n_sections": len(done), "n_pcs": T["n_pcs"]})


if __name__ == "__main__":
    main()
