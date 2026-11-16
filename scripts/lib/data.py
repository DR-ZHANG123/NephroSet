"""Load spot tables and image features per section; build per-fold section graphs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from . import structure as st

PROGS = ["y_pt_injury", "y_inflammation", "y_pt_transport"]
COMP = [f"comp_{c}" for c in st.CLASSES]


def load_cohort(spots_dir, feat_dir, sections, section_norm: bool = False) -> dict:
    """section_norm: standardise encoder features and morphology within each section, which
    removes slide-level appearance offsets using only the section's own unlabelled image."""
    out = {}
    for s in sections:
        t = pd.read_parquet(spots_dir / f"{s}.parquet")
        z = np.load(feat_dir / f"{s}.npz", allow_pickle=True)
        assert (z["barcode"] == t.index.values).all()
        x = np.concatenate([z["f1"].astype(np.float32), z["f2"].astype(np.float32)], 1)
        morph = z["morph"].astype(np.float32)
        if section_norm:
            x = (x - x.mean(0)) / (x.std(0) + 1e-6)
            morph = (morph - morph.mean(0)) / (morph.std(0) + 1e-6)
        pos = np.stack([t.array_col.values * 50.0, t.array_row.values * 86.6], 1).astype(np.float32)
        nb = st.hex_neighbours(t.array_row.values, t.array_col.values)
        out[s] = dict(table=t, x=x, morph=morph, pos=pos, nb=nb,
                      ring1=st.ring_sets(nb, 1), ring2=st.ring_sets(nb, 2))
    return out


def role_model(data: dict, train: list[str], alpha: float) -> Ridge:
    X = np.concatenate([np.concatenate([data[s]["x"], data[s]["morph"]], 1) for s in train])
    Y = np.concatenate([data[s]["table"][COMP].values for s in train])
    return Ridge(alpha=alpha).fit(X, Y)


def predict_roles(model: Ridge, d: dict) -> np.ndarray:
    r = model.predict(np.concatenate([d["x"], d["morph"]], 1)).clip(0, None)
    return (r / np.maximum(r.sum(1, keepdims=True), 1e-6)).astype(np.float32)


def oof_roles(data: dict, train: list[str], alpha: float, n_split: int, rng) -> dict[str, np.ndarray]:
    """Roles for training sections from models that did not see them."""
    tr = np.array(train)
    grp = rng.permutation(len(tr)) % n_split
    out = {}
    for k in range(n_split):
        m = role_model(data, list(tr[grp != k]), alpha)
        for s in tr[grp == k]:
            out[s] = predict_roles(m, data[s])
    return out


def build_graph(d: dict, roles: np.ndarray, morph_mu, morph_sd, ymu, ysd, max_unit: int, ring_k: int,
                rng, device) -> dict:
    t = d["table"]
    morph = (d["morph"] - morph_mu) / morph_sd
    x = np.concatenate([d["x"], morph, roles], 1)
    ring = d[f"ring{ring_k}"]
    comp_idx = st.compartment(roles)
    ana, etype = st.anatomical_hyperedges(d["nb"], comp_idx, max_unit, rng)
    rew = st.rewire(ana, len(t), rng)
    g = {"x": x, "pos": d["pos"], "roles": roles,
         "y": ((t[PROGS].values - ymu) / ysd).astype(np.float32),
         "comp": t[COMP].values.astype(np.float32),
         "hex_ei": st.edge_index(d["nb"]), "clique_ei": st.clique_edges(ana)}
    # deep-set neighbourhoods exclude the centre spot
    sets = [r[r != i] for i, r in enumerate(ring)]
    g["set_node"] = np.concatenate(sets).astype(np.int64)
    g["set_he"] = np.concatenate([np.full(len(r), i) for i, r in enumerate(sets)]).astype(np.int64)
    for pre, edges, typ in [("ring", ring, np.zeros(len(ring), int)), ("ana", ana, etype),
                            ("rew", rew, etype)]:
        node, he = st.incidence(edges)
        g[f"{pre}_node"], g[f"{pre}_he"], g[f"{pre}_n"] = node, he, len(edges)
        g[f"{pre}_type"] = typ.astype(np.int64)
    tg = {}
    for k, v in g.items():
        if isinstance(v, np.ndarray):
            tg[k] = torch.as_tensor(v, device=device)
        else:
            tg[k] = v
    tg["n_ana_unit"] = int((etype == 0).sum())
    tg["n_ana_interface"] = int((etype == 1).sum())
    return tg


def participant_folds(sections: pd.DataFrame, n_folds: int, seed: int) -> list[tuple[list[str], list[str]]]:
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(list(sections.section.values[a]), list(sections.section.values[b]))
            for a, b in gkf.split(sections, groups=sections.participant)]


def split_validation(sections: pd.DataFrame, tr_all: list[str], frac: float, rng) -> tuple[list[str], list[str]]:
    part = sections.set_index("section").participant
    parts = part.loc[tr_all].unique()
    va_p = set(rng.choice(parts, size=max(2, int(len(parts) * frac)), replace=False))
    va = [s for s in tr_all if part[s] in va_p]
    return [s for s in tr_all if s not in va], va
