"""BLEEP baseline (Xie et al., NeurIPS 2023): contrastive image-expression embedding + retrieval.

The model class, image encoder (timm ResNet-50, ImageNet weights, fully trainable), projection
heads, loss, optimiser settings (AdamW, lr 1e-3, weight decay 1e-3), batch size (256), training
augmentation (random h/v flips and 90-degree rotations, ImageNet normalisation) and the
retrieval rule (cosine top-50 nearest training spots, unweighted average) follow the official
code fetched into scripts/third_party/bleep by scripts/third_party/fetch.sh (Apache-2.0), which is imported unchanged.

Adaptations for this study (documented in the stage scripts):
  * crops are the 112 um spot-centred crops resized to 224 px (the official code cuts 224 px at
    full resolution of ~0.5 um/px, i.e. the same physical field); RGB order is used throughout;
  * the expression input is log-normalised expression (counts per 10,000, log1p) of the union of
    the programme genes and 3,000 highly variable genes ranked across the training sections only
    (scanpy batch-aware ranking: number of training sections in which a gene is among that
    section's top-2,000 HVGs, ties broken by median normalised dispersion); Harmony correction of
    the original preprocessing is not applied (expression enters only as a training-side key);
  * the retrieved quantity is the measured programme scores of the top-50 training spots
    (averaged), i.e. the same targets as every other model; BLEEP's own rule averages the
    retrieved spots' expression, and averaging a per-spot linear readout is the same operation;
  * the number of epochs is chosen on the held-aside validation participants (within-section
    Spearman of the injury score, as in the project's other models) instead of validation loss;
  * bf16 autocast for speed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
BLEEP_DIR = ROOT / "scripts" / "third_party" / "bleep"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TOP_K = 50          # official inference notebook: method "average", top_k=50
BATCH = 256         # official BLEEP_main.py default
N_HVG_SECTION = 2000
N_HVG = 3000


def official():
    """Import the official BLEEP modules (config.py, modules.py, models.py) unchanged."""
    if str(BLEEP_DIR) not in sys.path:
        sys.path.insert(0, str(BLEEP_DIR))
    import config as CFG  # noqa: F401
    import models as M
    return CFG, M


# ----------------------------------------------------------------------------- crops
def crop_section(spots_dir: Path, sec_dir: Path, out_dir: Path, sid: str, scale_um: float) -> str:
    """Cache uint8 spot crops (n, 224, 224, 3) ordered as the spot table."""
    import json

    import cv2
    import tifffile
    cv2.setNumThreads(1)
    sys.path.insert(0, str(ROOT / "scripts"))
    from lib.imagefeat import crop
    f = out_dir / f"{sid}.npy"
    if f.exists():
        return sid
    t = pd.read_parquet(spots_dir / f"{sid}.parquet")
    d = sec_dir / sid
    img_path = (d / "image_path.txt").read_text().strip()
    if (d / "um_per_px.txt").exists():
        um_per_px = float((d / "um_per_px.txt").read_text())
    else:
        sf = json.loads((d / "scalefactors_json.json").read_text())
        um_per_px = 55.0 / sf["spot_diameter_fullres"]
    img = tifffile.imread(img_path)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    h = scale_um / 2 / um_per_px
    c = np.stack([crop(img, a, b, h) for a, b in zip(t.pxl_row_in_fullres.values, t.pxl_col_in_fullres.values)])
    tmp = out_dir / f"{sid}.tmp.npy"
    np.save(tmp, c)
    tmp.rename(f)
    return sid


def load_crops(crop_dir: Path, sids: list[str]) -> dict:
    """All cached crops in one preallocated array (one host allocation; views per section).
    Files are read by parallel threads straight into their slice."""
    from concurrent.futures import ThreadPoolExecutor
    mm = {s: np.load(crop_dir / f"{s}.npy", mmap_mode="r") for s in sids}
    off = np.cumsum([0] + [len(mm[s]) for s in sids])
    allc = np.empty((off[-1], *mm[sids[0]].shape[1:]), np.uint8)

    def rd(i):
        allc[off[i]:off[i + 1]] = mm[sids[i]]

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(rd, range(len(sids))))
    return {s: allc[off[i]:off[i + 1]] for i, s in enumerate(sids)}


# ----------------------------------------------------------------------------- expression
def section_expression(sec_dir: Path, spots_dir: Path, sid: str):
    """Log-normalised expression (CSR, spots in spot-table order) and per-section HVG statistics."""
    import scanpy as sc
    sys.path.insert(0, str(ROOT / "scripts"))
    from lib.visium import normalise
    t = pd.read_parquet(spots_dir / f"{sid}.parquet")
    ad = sc.read_10x_h5(sec_dir / sid / "filtered_feature_bc_matrix.h5")
    ad.var_names_make_unique()
    ad = ad[t.index].copy()
    ad = normalise(ad)
    del ad.layers["counts"]
    sc.pp.highly_variable_genes(ad, n_top_genes=N_HVG_SECTION, flavor="seurat")
    hv = ad.var["highly_variable"].values.astype(bool)
    disp = ad.var["dispersions_norm"].values.astype(np.float32)
    X = ad.X.tocsr().astype(np.float32)
    return sid, X, np.asarray(ad.var_names), hv, disp


def select_genes(expr: dict, train: list[str], prog_genes: list[str]) -> np.ndarray:
    """Indices of programme genes plus the top HVGs ranked across training sections only."""
    names = expr[train[0]][1]
    hv = np.stack([expr[s][2] for s in train]).sum(0)
    disp = np.nanmedian(np.stack([expr[s][3] for s in train]), 0)
    disp = np.nan_to_num(disp, nan=-np.inf)
    order = np.lexsort((-disp, -hv))
    top = set(order[:N_HVG].tolist())
    pg = set(np.where(np.isin(names, prog_genes))[0].tolist())
    return np.array(sorted(top | pg))


# ----------------------------------------------------------------------------- model
def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Official training augmentation on GPU: random h/v flip, random rotation in {0, 90, 180, -90}."""
    n = x.shape[0]
    r = torch.rand(n, 2, generator=gen).to(x.device)
    k = torch.randint(0, 4, (n,), generator=gen).to(x.device)
    out = x.clone()
    hf, vf = r[:, 0] > 0.5, r[:, 1] > 0.5
    out[hf] = out[hf].flip(-1)
    out[vf] = out[vf].flip(-2)
    for kk in range(1, 4):
        m = k == kk
        if m.any():
            out[m] = torch.rot90(out[m], kk, dims=(-2, -1))
    return out


def to_input(crops_u8: np.ndarray, dev) -> torch.Tensor:
    x = torch.from_numpy(crops_u8).to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN.to(dev)) / STD.to(dev)


@torch.no_grad()
def image_embeddings(model, crops: np.ndarray, dev, bs: int = 512) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(crops), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model.image_projection(model.image_encoder(to_input(crops[i:i + bs], dev)))
        out.append(e.float())
    return torch.cat(out)


@torch.no_grad()
def spot_embeddings(model, expr: torch.Tensor, bs: int = 4096) -> torch.Tensor:
    model.eval()
    return torch.cat([model.spot_projection(expr[i:i + bs]).float() for i in range(0, len(expr), bs)])


@torch.no_grad()
def retrieve(query: torch.Tensor, key: torch.Tensor, values: torch.Tensor, k: int = TOP_K, bs: int = 2048):
    """Official find_matches (L2-normalise, dot product, top-k) + unweighted average of values."""
    q = F.normalize(query, p=2, dim=-1)
    kk = F.normalize(key, p=2, dim=-1)
    out = []
    for i in range(0, len(q), bs):
        idx = torch.topk(q[i:i + bs] @ kk.T, k=k, dim=1).indices
        out.append(values[idx].mean(1))
    return torch.cat(out).cpu().numpy()


class Pool:
    """Training pool: crops (CPU uint8) and expression (GPU float) concatenated over sections."""

    def __init__(self, crops: dict, expr_rows: dict, y: dict, sections: list[str], dev):
        # crops are views into one shared array: keep row indices into it, no copy
        base = crops[sections[0]].base
        assert all(crops[s].base is base for s in sections)
        start = {s: (crops[s].__array_interface__["data"][0] - base.__array_interface__["data"][0])
                 // crops[s][0].nbytes for s in sections}
        self.rows = np.concatenate([start[s] + np.arange(len(crops[s])) for s in sections])
        self.crops = base
        # expr_rows: section -> dense array, or a callable building it (moved to the GPU one at a time)
        get = expr_rows if callable(expr_rows) else expr_rows.__getitem__
        self.expr = torch.cat([torch.from_numpy(get(s)).to(dev) for s in sections])
        self.y = torch.from_numpy(np.concatenate([y[s] for s in sections])).float().to(dev)
        self.n = len(self.rows)
        # reusable pinned batch buffer: fresh host allocations are very slow on the shared server
        self.buf = torch.empty((BATCH, *self.crops.shape[1:]), dtype=torch.uint8).pin_memory()
        self.buf_np = self.buf.numpy()


def train_epoch(model, opt, pool: Pool, rng: np.random.Generator, gen: torch.Generator, dev) -> float:
    model.train()
    perm = rng.permutation(pool.n)
    losses = []
    nb = pool.n // BATCH  # drop_last=True as in the official loader
    for b in range(nb):
        idx = np.sort(perm[b * BATCH:(b + 1) * BATCH])
        np.take(pool.crops, pool.rows[idx], axis=0, out=pool.buf_np)
        x = augment(pool.buf.to(dev).permute(0, 3, 1, 2).float().div_(255.0).sub_(MEAN.to(dev)).div_(STD.to(dev)), gen)
        e = pool.expr[torch.from_numpy(idx).to(dev)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model({"image": x, "reduced_expression": e})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def predict(model, pool: Pool, crops: np.ndarray, dev) -> np.ndarray:
    """Retrieve the top-k training spots for each query image and average their programme scores."""
    key = spot_embeddings(model, pool.expr)
    return retrieve(image_embeddings(model, crops, dev), key, pool.y)


def fit(pool: Pool, val_crops: dict, val_y: dict, seed: int, dev, max_epochs: int, patience: int, log=print):
    """Train the official CLIPModel; keep the epoch with the best validation within-section Spearman
    of the injury score (programme 0) for retrieval from the training pool."""
    from scipy.stats import spearmanr
    import random
    CFG, M = official()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = M.CLIPModel(spot_embedding=pool.expr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, bad, hist = -np.inf, None, 0, 0, []
    for ep in range(1, max_epochs + 1):
        tl = train_epoch(model, opt, pool, rng, gen, dev)
        key = spot_embeddings(model, pool.expr)
        rs = []
        for s, c in val_crops.items():
            yp = retrieve(image_embeddings(model, c, dev), key, pool.y)
            yt = val_y[s]
            rs.append([spearmanr(yp[:, j], yt[:, j])[0] if np.std(yp[:, j]) > 0 and np.std(yt[:, j]) > 0
                       else np.nan for j in range(yt.shape[1])])
        rv = np.nanmean(np.array(rs, float), 0)
        hist.append(dict(epoch=ep, train_loss=tl, **{f"val_rho_{j}": float(v) for j, v in enumerate(rv)}))
        log(f"  seed {seed} epoch {ep} loss {tl:.4f} val_rho {np.round(rv, 3).tolist()}")
        if rv[0] > best + 1e-4:
            best, best_ep, bad = float(rv[0]), ep, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_ep, best, hist
