"""iStar baseline (Zhang et al., Nat Biotechnol 2024), run inside the project's participant folds.

Official code is fetched into scripts/third_party/istar by scripts/third_party/fetch.sh (GPL-3.0, not redistributed here). What is used unchanged:
* histology features: extract_features.get_embeddings_shift (HIPT ViT-256 + ViT-4K, DINO weights
  vit256_small_dino.pth / vit4k_xs_dino.pth, 16 shifted passes), the RGB channels and the
  uniform smoothing of the cls (16) and sub (4) groups exactly as extract_features.main();
  579 channels per 16 x 16 px superpixel at 0.5 um/px (8 um);
* image preparation: rescaling to 0.5 um/px (cv2 area/bilinear resampling instead of
  skimage.transform.rescale, for speed) and preprocess.adjust_margins (white pad to a multiple of 256);
* spot footprint: image.get_disk_mask(radius / 16) and impute.get_patches_flat on locations
  rounded to the superpixel grid (impute_by_basic.get_locs), radius = 27.5 um at 0.5 um/px;
* model: impute.ForwardSumModel (4 x 256 LeakyReLU MLP per superpixel, ELU(0.01)+0.01 output),
  spot prediction = mean of superpixel predictions in the disk, MSE on min-max normalised targets
  (impute.normalize), Adam lr 1e-4, batch 128, 5 states combined by the median at superpixel level.

Adaptations to the cross-cohort setting (iStar is fitted per section): the image is cropped to the
spot bounding box plus a 512 um white-padded margin before encoding; one model is fitted on all
training participants; training length is set by early stopping on the validation participants
(within-section Spearman of injury, max_epochs/patience from config train) instead of 400 epochs
on one section; targets are the three programme scores (min-max normalised on the training
participants, de-normalised afterwards) instead of per-gene counts.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
ISTAR = ROOT / "scripts" / "third_party" / "istar"
PIX_UM = 0.5          # iStar analysis pixel size
SPOT_RADIUS_UM = 27.5  # Visium spot radius
MARGIN_UM = 512.0
FACTOR = 16           # superpixel size in pixels


def _istar_path():
    if str(ISTAR) not in sys.path:
        sys.path.insert(0, str(ISTAR))


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_section_image(sec_dir: Path, spots: pd.DataFrame):
    """White-padded crop of the full-resolution image around the spots, rescaled to 0.5 um/px.

    Returns (image uint8, spot locations in the rescaled crop (row, col) float)."""
    import cv2
    import tifffile
    _istar_path()
    from preprocess import adjust_margins
    img_path = (sec_dir / "image_path.txt").read_text().strip()
    if (sec_dir / "um_per_px.txt").exists():
        um = float((sec_dir / "um_per_px.txt").read_text())
    else:
        sf = json.loads((sec_dir / "scalefactors_json.json").read_text())
        um = 55.0 / sf["spot_diameter_fullres"]
    try:  # read only the needed region when the TIFF is uncompressed
        img = tifffile.memmap(img_path, mode="r")
    except ValueError:
        img = tifffile.imread(img_path)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    r, c = spots.pxl_row_in_fullres.values, spots.pxl_col_in_fullres.values
    m = MARGIN_UM / um
    r0, r1 = int(np.floor(r.min() - m)), int(np.ceil(r.max() + m))
    c0, c1 = int(np.floor(c.min() - m)), int(np.ceil(c.max() + m))
    h, w = img.shape[:2]
    pad = np.full((r1 - r0, c1 - c0, 3), 255, np.uint8)
    rr0, rr1, cc0, cc1 = max(r0, 0), min(r1, h), max(c0, 0), min(c1, w)
    reg = np.asarray(img[rr0:rr1, cc0:cc1, :3])
    if reg.dtype != np.uint8:  # same conversion as the project's crops (whole-image maximum)
        reg = (reg / np.asarray(img).max() * 255).astype(np.uint8)
    pad[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = reg
    del img, reg
    scale = um / PIX_UM
    # iStar rescales with skimage.transform.rescale (anti-aliased bilinear); cv2 area averaging
    # (downscaling) / bilinear (upscaling) is used instead because it is ~50x faster
    size = (int(round(pad.shape[1] * scale)), int(round(pad.shape[0] * scale)))
    out = cv2.resize(pad, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    out = adjust_margins(out, pad=256, pad_value=255)
    locs = np.stack([(r - r0) * scale, (c - c0) * scale], 1)
    return out, locs


_HIPT = {}


def _hipt(device):
    """Official HIPT_4K with the iStar checkpoints, built once per process (the official
    get_embeddings rebuilds it for every pass)."""
    if device not in _HIPT:
        _istar_path()
        from hipt_4k import HIPT_4K
        with _cwd(ISTAR):
            m = HIPT_4K(model256_path="checkpoints/vit256_small_dino.pth",
                        model4k_path="checkpoints/vit4k_xs_dino.pth", device256=device, device4k=device)
        _HIPT[device] = m.eval()
    return _HIPT[device]


@torch.no_grad()
def get_embeddings(img, device="cuda"):
    """Same computation as extract_features.get_embeddings (official), returning torch tensors
    (192, H/16, W/16) and (384, H/16, W/16); the per-channel numpy loops are replaced by
    equivalent tensor rearrangements."""
    _istar_path()
    import extract_features as ef
    from einops import rearrange, repeat
    model = _hipt(device)
    tiles, shapes = ef.patchify(img, patch_size=4096)
    mids, subs = [], []
    for t in tiles:
        x_cls, x_sub = ef.get_embeddings_sub(model, t)  # (16,16,384), (16,16,16,16,384)
        mids.append(x_cls)
        subs.append(torch.from_numpy(x_sub))
    h1, w1 = shapes["tiles"]
    emb_mid = rearrange(mids, "(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k", h1=h1, w1=w1)
    emb_cls = ef.get_embeddings_cls(model, emb_mid)  # (H/256, W/256, 192)
    so = np.array(shapes["original"]) // 16
    sub = rearrange(torch.stack(subs), "(h1 w1) h2 w2 h3 w3 k -> k (h1 h2 h3) (w1 w2 w3)", h1=h1, w1=w1)
    cls = repeat(torch.from_numpy(np.ascontiguousarray(emb_cls)), "h w k -> k (h a) (w b)", a=16, b=16)
    return cls[:, :so[0], :so[1]].contiguous(), sub[:, :so[0], :so[1]].contiguous()


@torch.no_grad()
def get_embeddings_gpu(img_t: torch.Tensor, model, tile: int = 4096):
    """GPU version of extract_features.get_embeddings for an image tensor (H, W, 3) uint8 on the
    device: edge padding to a multiple of 4096 (patchify), (x/255-0.5)/0.5 (eval_transforms), ViT-256
    on every 256-px patch (the patch grid of the padded image equals the per-tile grids), ViT-4K on
    the full grid of ViT-256 cls tokens (get_embeddings_cls), sub4k repeated to 16-px superpixels."""
    from einops import rearrange
    H, W = img_t.shape[:2]
    Hp, Wp = -(-H // tile) * tile, -(-W // tile) * tile
    dev = img_t.device
    # edge padding ('edge' mode of np.pad) by index clamping, kept as uint8 to save memory
    ri = torch.arange(Hp, device=dev).clamp(max=H - 1)
    ci = torch.arange(Wp, device=dev).clamp(max=W - 1)
    xp = img_t[ri][:, ci]
    p1, p2 = Hp // 256, Wp // 256
    patches = rearrange(xp, "(p1 a) (p2 b) c -> (p1 p2) c a b", a=256, b=256)  # uint8
    del xp
    cls_l, sub_l = [], []
    for i in range(0, len(patches), 128):
        xb = (patches[i:i + 128].float() / 255.0 - 0.5) / 0.5
        f = model.model256.forward_all(xb)
        cls_l.append(f[:, 0])
        sub_l.append(f[:, 1:])
    del patches
    cls256 = rearrange(torch.cat(cls_l), "(p1 p2) k -> k p1 p2", p1=p1, p2=p2)[None]
    sub = rearrange(torch.cat(sub_l), "(p1 p2) (h3 w3) k -> k (p1 h3) (p2 w3)", p1=p1, p2=p2, h3=16, w3=16)
    _, sub4k = model.forward_all4k(cls256)  # (1, 192, p1, p2)
    cls = sub4k[0].repeat_interleave(16, 1).repeat_interleave(16, 2)
    so = (H // 16, W // 16)
    return cls[:, :so[0], :so[1]], sub[:, :so[0], :so[1]]


def get_embeddings_shift(img, margin=256, stride=64, device="cuda"):
    """extract_features.get_embeddings_shift (official), computed on the GPU."""
    factor = 16
    model = _hipt(device)
    img_t = torch.as_tensor(np.ascontiguousarray(img), device=device)
    shape_emb = np.array(img.shape[:2]) // factor
    cls_acc = torch.zeros((192, *shape_emb), dtype=torch.float32, device=device)
    sub_acc = torch.zeros((384, *shape_emb), dtype=torch.float32, device=device)
    starts = list(range(0, margin, stride))
    n = 0
    for s0 in starts:
        for s1 in starts:
            e0, e1 = -margin + s0, -margin + s1
            cls, sub = get_embeddings_gpu(img_t[s0:e0, s1:e1], model)
            cls_acc[:, s0 // factor:e0 // factor, s1 // factor:e1 // factor] += cls
            sub_acc[:, s0 // factor:e0 // factor, s1 // factor:e1 // factor] += sub
            del cls, sub
            n += 1
    mar = margin // factor
    for acc in (cls_acc, sub_acc):
        acc /= n
        acc[:, -mar:] = 0.0
        acc[:, :, -mar:] = 0.0
    return list(cls_acc.cpu().numpy()), list(sub_acc.cpu().numpy())


def hist_embeddings(img: np.ndarray, device: str = "cuda") -> np.ndarray:
    """extract_features.main() of iStar (default arguments) -> (H/16, W/16, 579) float32."""
    _istar_path()
    import extract_features as ef
    from einops import reduce
    np.random.seed(0)
    torch.manual_seed(0)
    emb_cls, emb_sub = get_embeddings_shift(img, device=device)
    embs = dict(cls=emb_cls, sub=emb_sub)
    embs["rgb"] = np.stack([reduce(img[..., i].astype(np.float16) / 255.0, "(h1 h) (w1 w) -> h1 w1", "mean",
                                   h=16, w=16).astype(np.float32) for i in range(3)])
    embs = ef.smoothen_embeddings(embs, size=16, kernel="uniform", groups=["cls"], method="cv", device=device)
    embs = ef.smoothen_embeddings(embs, size=4, kernel="uniform", groups=["sub"], method="cv", device=device)
    x = np.concatenate([np.stack(embs["cls"]), np.stack(embs["sub"]), embs["rgb"]])
    return x.transpose(1, 2, 0)


def disk_mask() -> np.ndarray:
    _istar_path()
    from image import get_disk_mask
    radius = int(np.round(SPOT_RADIUS_UM / PIX_UM)) / FACTOR
    return get_disk_mask(radius)


def spot_patches(embs: np.ndarray, locs: np.ndarray) -> np.ndarray:
    """(n_spots, n_superpixels_in_disk, 579) as in impute.SpotDataset."""
    _istar_path()
    from impute import get_patches_flat
    loc = (locs / FACTOR).round().astype(int)
    return get_patches_flat(embs, loc, disk_mask())


# ----------------------------------------------------------------------------- training

def load_features(cache_dir: Path, sections, tables: dict) -> dict:
    out = {}
    for s in sections:
        z = np.load(cache_dir / f"{s}.npz", allow_pickle=True)
        assert (z["barcode"] == tables[s].index.values).all(), s
        out[s] = torch.as_tensor(z["x"])  # float16, n x m x 579
    return out


def _model(n_inp: int, n_out: int):
    _istar_path()
    from impute import ForwardSumModel
    return ForwardSumModel(lr=1e-4, n_inp=n_inp, n_out=n_out)


@torch.no_grad()
def superpixel_pred(model, x: torch.Tensor, device, bs: int = 2048) -> torch.Tensor:
    model.eval()
    out = [model(x[i:i + bs].to(device).float()) for i in range(0, len(x), bs)]
    return torch.cat(out)  # n x m x k (normalised scale)


def within_rho(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan


def train_state(X, Ytr, tr, va, Y, seed, T, device, log=None):
    """One iStar state: ForwardSumModel trained with iStar's loss/optimiser, early-stopped on
    the mean within-section Spearman of injury over the validation sections."""
    from .repro import set_global_seed
    set_global_seed(seed)
    Xtr = torch.cat([X[s] for s in tr]).to(device)  # fp16 on device
    ytr = torch.as_tensor(Ytr, dtype=torch.float32, device=device)
    model = _model(Xtr.shape[-1], ytr.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=model.lr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(Xtr)
    best, best_state, bad, ep = -np.inf, None, 0, 0
    for ep in range(T["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            y_pred = model(Xtr[idx].float())
            loss = ((y_pred.mean(-2) - ytr[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        r = [within_rho(superpixel_pred(model, X[s], device).mean(1)[:, 0].cpu().numpy(), Y[s][:, 0]) for s in va]
        score = float(np.nanmean(r))
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= T["patience"]:
                break
    model.load_state_dict(best_state)
    del Xtr, ytr
    return model, best, ep + 1


def fit_predict(X, Y, tr, va, te, seed, T, device, n_states=5, tag=""):
    """iStar for one seed: n_states models, median over states at superpixel level, spot = mean
    of its superpixels, de-normalised with the training min/max. Returns {section: (n,3)}, log."""
    yy = np.concatenate([Y[s] for s in tr])
    ymin, ymax = yy.min(0), yy.max(0)
    Ytr = (yy - ymin) / (ymax - ymin + 1e-12)
    log, models = [], []
    for k in range(n_states):
        state_seed = seed * 100 + k
        for attempt in range(1000):
            try:
                m, val, ep = train_state(X, Ytr, tr, va, Y, state_seed, T, device)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("OOM, retrying in 120 s", tag, k, flush=True)
                time.sleep(120)
        models.append(m)
        log.append(dict(seed=seed, state=k, val_rho=val, epochs=ep))
        print(tag, "seed", seed, "state", k, round(val, 3), ep, time.strftime("%H:%M"), flush=True)
    out = {}
    for s in te:
        sp = torch.stack([superpixel_pred(m, X[s], device) for m in models]).median(0).values
        out[s] = sp.mean(1).cpu().numpy() * (ymax - ymin + 1e-12) + ymin
    del models
    torch.cuda.empty_cache()
    return out, log


# ----------------------------------------------------------------------------- encoder-matched variant

def token_disk_mask() -> np.ndarray:
    """Tokens of the 14 x 14 grid (8 um, 112 um crop centred on the spot) whose centres lie inside the
    55 um spot: the same footprint rule as iStar's disk mask on 16-px (8 um) superpixels."""
    g = (np.arange(14) - 6.5) * 8.0
    d2 = g[:, None] ** 2 + g[None, :] ** 2
    return (d2 <= SPOT_RADIUS_UM ** 2).ravel()


def load_token_features(tok_dir: Path, sections, tables: dict) -> dict:
    """istar_phikon: Phikon-v2 patch tokens (128 PCs) + 3 colour channels per 8 um token, restricted
    to the spot disk -> (n, m, 131) float16, used in place of the HIPT superpixel features."""
    m = token_disk_mask()
    out = {}
    for s in sections:
        z = np.load(tok_dir / f"{s}.npz", allow_pickle=True)
        assert (z["barcode"] == tables[s].index.values).all(), s
        x = np.concatenate([z["tok"], z["col"]], -1)[:, m, :]
        out[s] = torch.as_tensor(x.astype(np.float16))
    return out
