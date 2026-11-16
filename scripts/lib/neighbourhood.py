"""Neighbourhood descriptors of single cells at four levels of spatial information.

A  composition : coarse-class fractions and density around the cell and in its region
B  pairwise    : nearest distance to each class and class counts in distance rings
C  local group : co-occurrence of class pairs inside a 30 um ball (higher order, no anatomy)
D  anatomical  : the cell's tubular unit (same-class epithelial cells connected through
                 short contacts) and the non-epithelial cells that surround that unit

Only coarse class labels and coordinates are used. The expression of the target cell and of
its neighbours never enters any descriptor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

CLASSES = ["PT", "LOH", "DCT_CNT", "CD", "Glomerular", "Endothelial", "Fibroblast", "Mural", "Immune"]
EPI = {"PT", "LOH", "DCT_CNT", "CD"}
NONEPI = ["Endothelial", "Fibroblast", "Mural", "Immune"]


def _onehot(labels: np.ndarray) -> np.ndarray:
    idx = pd.Categorical(labels, categories=CLASSES).codes
    out = np.zeros((len(labels), len(CLASSES)), np.float32)
    out[np.arange(len(labels)), idx] = 1
    return out


def features(xy: np.ndarray, labels: np.ndarray, targets: np.ndarray, p: dict, units_out: list | None = None) -> pd.DataFrame:
    """Descriptors for cells `targets` (indices into xy) of one tissue section."""
    tree = cKDTree(xy)
    oh = _onehot(labels)
    is_epi = np.isin(labels, list(EPI))
    F: dict[str, list] = {}

    def put(k, v):
        F.setdefault(k, []).append(v)

    r_local, r_region = p["r_local"], p["r_region"]
    rings = p["rings"]
    r_ball, r_pair = p["r_ball"], p["r_pair"]
    r_unit, d_contact, d_halo = p["r_unit"], p["d_contact"], p["d_halo"]
    region_nb = tree.query_ball_point(xy[targets], r_region)
    # nearest distance per class
    nearest = {}
    for ci, c in enumerate(CLASSES):
        m = np.where(labels == c)[0]
        if len(m) < 2:
            nearest[c] = np.full(len(targets), r_region, np.float32)
            continue
        t = cKDTree(xy[m])
        d, j = t.query(xy[targets], k=2)
        # exclude the target itself when it belongs to class c
        self_hit = m[j[:, 0]] == targets
        nearest[c] = np.minimum(np.where(self_hit, d[:, 1], d[:, 0]), r_region).astype(np.float32)

    for ti, (t, reg) in enumerate(zip(targets, region_nb)):
        reg = np.asarray([j for j in reg if j != t], dtype=np.int64)
        d = np.linalg.norm(xy[reg] - xy[t], axis=1) if len(reg) else np.zeros(0)
        loc = reg[d <= r_local]
        # A: composition
        put("A_density_local", len(loc))
        put("A_density_region", len(reg))
        cl = oh[loc].sum(0) / max(len(loc), 1)
        cr = oh[reg].sum(0) / max(len(reg), 1)
        for ci, c in enumerate(CLASSES):
            put(f"A_frac_local_{c}", cl[ci])
            put(f"A_frac_region_{c}", cr[ci])
        # B: pairwise geometry
        for c in CLASSES:
            put(f"B_nearest_{c}", nearest[c][ti])
        lo = 0.0
        for hi in rings:
            sel = reg[(d > lo) & (d <= hi)]
            cnt = oh[sel].sum(0)
            for ci, c in enumerate(CLASSES):
                put(f"B_ring{int(hi)}_{c}", cnt[ci])
            lo = hi
        # C: higher-order co-occurrence in a ball (no anatomy)
        ball = reg[d <= r_ball]
        if len(ball) > 1:
            bt = cKDTree(xy[ball])
            pairs = np.array(list(bt.query_pairs(r_pair)), dtype=int).reshape(-1, 2)
        else:
            pairs = np.zeros((0, 2), int)
        lab_b = labels[ball]
        for a_i, a in enumerate(NONEPI):
            for b in NONEPI[a_i:]:
                if len(pairs):
                    la, lb = lab_b[pairs[:, 0]], lab_b[pairs[:, 1]]
                    v = int(((la == a) & (lb == b)).sum() + ((la == b) & (lb == a)).sum()) if a != b \
                        else int(((la == a) & (lb == a)).sum())
                else:
                    v = 0
                put(f"C_pair_{a}_{b}", v)
        types_ball = {x for x in lab_b if x in NONEPI}
        put("C_n_types_ball", len(types_ball))
        # D: anatomical unit and its halo
        same = reg[(labels[reg] == labels[t]) & (d <= r_unit)]
        members = np.concatenate([[t], same])
        if len(members) > 1:
            mt = cKDTree(xy[members])
            pr = np.array(list(mt.query_pairs(d_contact)), dtype=int).reshape(-1, 2)
            g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(members),) * 2)
            _, comp = connected_components(g, directed=False)
            unit = members[comp == comp[0]]
        else:
            unit = members
        cand = reg[~is_epi[reg]]
        if len(cand):
            dh, _ = cKDTree(xy[unit]).query(xy[cand], k=1)
            halo = cand[dh <= d_halo]
        else:
            halo = cand
        if units_out is not None:
            units_out.append(unit)
        put("D_unit_size", len(unit))
        ext = np.ptp(xy[unit], axis=0).max() if len(unit) > 1 else 0.0
        put("D_unit_extent", ext)
        hc = oh[halo].sum(0)
        for c in NONEPI:
            put(f"D_halo_{c}", hc[CLASSES.index(c)])
        put("D_halo_total", len(halo))
        present = [c for c in NONEPI if hc[CLASSES.index(c)] > 0]
        put("D_halo_n_types", len(present))
        frac = hc[[CLASSES.index(c) for c in NONEPI]] / max(len(halo), 1)
        nz = frac[frac > 0]
        put("D_halo_entropy", float(-(nz * np.log(nz)).sum()))
        put("D_halo_immune_and_fibro", int(hc[CLASSES.index("Immune")] > 0 and hc[CLASSES.index("Fibroblast")] > 0))
        put("D_halo_immune_x_fibro", float(hc[CLASSES.index("Immune")] * hc[CLASSES.index("Fibroblast")]))
        put("D_halo_per_unit_cell", len(halo) / len(unit))
        # other epithelial classes touching the unit (mixed / abutting tubules)
        oe = reg[is_epi[reg] & (labels[reg] != labels[t])]
        if len(oe):
            de, _ = cKDTree(xy[unit]).query(xy[oe], k=1)
            put("D_unit_other_epi_contacts", int((de <= d_contact).sum()))
        else:
            put("D_unit_other_epi_contacts", 0)
    return pd.DataFrame(F, index=targets)


def permute_nonepithelial(xy: np.ndarray, labels: np.ndarray, tile: float, rng) -> np.ndarray:
    """Shuffle interstitial labels (endothelial, fibroblast, mural, immune) within square tiles.

    Positions, epithelial labels, cell density and tile-level composition are preserved;
    which tubule a given immune or stromal cell sits against is randomised.
    """
    out = labels.copy()
    ne = np.where(np.isin(labels, NONEPI))[0]
    key = (np.floor(xy[ne, 0] / tile).astype(np.int64) * 1_000_003
           + np.floor(xy[ne, 1] / tile).astype(np.int64))
    for k in np.unique(key):
        m = ne[key == k]
        out[m] = labels[rng.permutation(m)]
    return out
