"""Structures inside a spot crop: 14 x 14 grid of 8 um image tokens.

At Visium resolution a tubular cross-section (40-60 um) and its peritubular interstitium lie
inside one spot, so tubular units and their surroundings have to be represented below the
spot, on image tokens. Token roles come from a linear head trained so that the mean token role
of a spot matches its sequencing-derived composition (multiple-instance supervision).

Hyperedge families (all within one spot):
* profile  : connected token regions of one compartment (4-connectivity), size >= 2
* halo     : an epithelial profile together with the adjacent tokens of other compartments
* rolebag  : all tokens of one compartment in the spot, without contiguity (composition null)
* rewired  : random tokens of the same spot, matching the sizes of profile and halo edges
* window   : generic 3 x 3 token windows with stride 2 (geometry without anatomy)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .structure import CLASSES, COMPARTMENTS

G = 14
NT = G * G
COMP_OF_CLASS = np.zeros(len(CLASSES), int)
for k, v in enumerate(COMPARTMENTS.values()):
    for c in v:
        COMP_OF_CLASS[CLASSES.index(c)] = k
EPI = [0, 1]  # proximal, distal
TISSUE_MIN = 0.3
_S4 = np.zeros((3, 3, 3), bool)
_S4[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]


def token_pos() -> np.ndarray:
    r, c = np.divmod(np.arange(NT), G)
    return (np.stack([r, c], 1) - (G - 1) / 2) / G


def grid_edges(n_spots: int) -> np.ndarray:
    """8-neighbour edges inside every spot (no edges between spots)."""
    r, c = np.divmod(np.arange(NT), G)
    src, dst = [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            ok = (r + dr >= 0) & (r + dr < G) & (c + dc >= 0) & (c + dc < G)
            src.append(np.where(ok)[0])
            dst.append(((r + dr) * G + (c + dc))[ok])
    s, d = np.concatenate(src), np.concatenate(dst)
    off = (np.arange(n_spots) * NT)[:, None]
    return np.stack([(s + off).ravel(), (d + off).ravel()]).astype(np.int64)


def train_role_head(tok_list, col_list, comp_list, device, epochs=8, seed=0):
    """Linear token head; mean of token class probabilities over tissue tokens ~ spot composition."""
    torch.manual_seed(seed)
    d = tok_list[0].shape[-1] + col_list[0].shape[-1]
    head = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, len(CLASSES))).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(len(tok_list)):
            x = torch.cat([tok_list[i], col_list[i]], -1).to(device).float()  # n,196,d
            w = (col_list[i][..., 0].to(device).float() >= TISSUE_MIN).float()
            pr = F.softmax(head(x), -1)
            m = (pr * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)
            loss = F.kl_div(m.clamp(min=1e-6).log(), comp_list[i], reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def token_roles(head, tok, col) -> np.ndarray:
    dev = next(head.parameters()).device
    pr = F.softmax(head(torch.cat([tok, col], -1).to(dev).float()), -1)
    return pr.cpu().numpy()


def _groups(key: np.ndarray, val: np.ndarray):
    """Split val by key (both 1-d); yields (key, members) for each distinct key."""
    o = np.argsort(key, kind="stable")
    k, v = key[o], val[o]
    cut = np.flatnonzero(np.diff(k)) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(k)]):
        yield k[a], v[a:b]


def hyperedges(roles: np.ndarray, tissue: np.ndarray, rng) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """roles (n,196,9) probabilities, tissue (n,196) fraction -> incidence per family.

    Returns {family: (node, he, type)} with global token indices spot*196 + token.
    """
    n = roles.shape[0]
    comp = COMP_OF_CLASS[roles.argmax(-1)]
    comp = np.where(tissue >= TISSUE_MIN, comp, -1).reshape(n, G, G)
    cf = comp.reshape(n, NT)
    fam = {k: [[], [], []] for k in ["profile", "halo", "rolebag", "rewired", "window"]}

    def add(f, members, typ):
        fam[f][0].append(members)
        fam[f][1].append(np.full(len(members), len(fam[f][2])))
        fam[f][2].append(typ)

    sizes = []
    shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for k in range(4):
        lab, nl = ndimage.label(comp == k, structure=_S4)
        if nl == 0:
            continue
        flat = lab.ravel()
        idx = np.flatnonzero(flat)
        comps = dict(_groups(flat[idx], idx))
        halo = {}
        if k in EPI:
            # tokens of another compartment 4-adjacent to a profile
            src, dst = [], []
            for dr, dc in shifts:
                sh = np.zeros_like(lab)
                rs = slice(max(dr, 0), G + min(dr, 0))
                rd = slice(max(-dr, 0), G + min(-dr, 0))
                cs = slice(max(dc, 0), G + min(dc, 0))
                cd = slice(max(-dc, 0), G + min(-dc, 0))
                sh[:, rd, cd] = lab[:, rs, cs]  # label of the neighbour
                ok = (sh > 0) & (comp >= 0) & (comp != k)
                src.append(sh[ok])
                dst.append(np.flatnonzero(ok.ravel()))
            hs, hd = np.concatenate(src), np.concatenate(dst)
            halo = {key: np.unique(v) for key, v in _groups(hs, hd)}
        for j, mem in comps.items():
            if len(mem) < 2:
                continue
            add("profile", mem, k)
            sizes.append((mem[0] // NT, len(mem), k))
            if j in halo:
                hal = np.concatenate([mem, halo[j]])
                add("halo", hal, 4)
                sizes.append((mem[0] // NT, len(hal), 4))
        sp, tk = np.where(cf == k)
        for s, mem in _groups(sp, sp * NT + tk):
            if len(mem) >= 2:
                add("rolebag", mem, k)
    tis = [np.flatnonzero(cf[i] >= 0) for i in range(n)]
    for s, size, k in sizes:
        pool = tis[s] if len(tis[s]) >= size else np.arange(NT)
        add("rewired", s * NT + rng.choice(pool, size=size, replace=False), k)
    base = np.array([[(r0 + a) * G + c0 + b for a in range(3) for b in range(3)]
                     for r0 in range(0, G - 2, 2) for c0 in range(0, G - 2, 2)])
    okw = ((cf[:, base] >= 0).sum(-1) >= 3)  # n, n_windows
    for s, w in zip(*np.where(okw)):
        add("window", s * NT + base[w], 0)
    out = {}
    for f, (nodes, hes, typ) in fam.items():
        if nodes:
            out[f] = (np.concatenate(nodes).astype(np.int64), np.concatenate(hes).astype(np.int64),
                      np.array(typ, np.int64))
        else:
            out[f] = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64))
    return out
