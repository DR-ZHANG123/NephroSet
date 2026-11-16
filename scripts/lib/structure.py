"""Spot graphs and hypergraphs on the Visium hexagonal lattice.

All structures are built from spot positions and image-derived quantities only.

* hex graph          : the six lattice neighbours of each spot (about 100 um apart)
* ring sets          : each spot with all spots within k lattice steps (generic local hyperedges)
* anatomical units   : contiguous spots sharing the same image-predicted dominant tissue
                       compartment, cut into compact pieces of at most `max_unit` spots
* interface edges    : spots on both sides of a boundary between two adjacent units of
                       different compartments (tubule-interstitium and glomerular borders)
"""
from __future__ import annotations

from collections import deque

import numpy as np

COMPARTMENTS = {"proximal": ["PT"], "distal": ["LOH", "DCT_CNT", "CD"],
                "interstitial": ["Fibroblast", "Immune", "Endothelial", "Mural"], "glomerular": ["Glomerular"]}
CLASSES = ["PT", "LOH", "DCT_CNT", "CD", "Glomerular", "Endothelial", "Fibroblast", "Mural", "Immune"]


def hex_neighbours(row: np.ndarray, col: np.ndarray) -> list[np.ndarray]:
    key = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(row, col))}
    off = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    return [np.array([key[(int(r) + a, int(c) + b)] for a, b in off if (int(r) + a, int(c) + b) in key], int)
            for r, c in zip(row, col)]


def edge_index(nb: list[np.ndarray]) -> np.ndarray:
    src = np.concatenate([np.full(len(n), i) for i, n in enumerate(nb)]) if nb else np.zeros(0, int)
    dst = np.concatenate(nb) if nb else np.zeros(0, int)
    return np.stack([src, dst]).astype(np.int64)


def ring_sets(nb: list[np.ndarray], k: int) -> list[np.ndarray]:
    out = []
    for i in range(len(nb)):
        seen, frontier = {i}, {i}
        for _ in range(k):
            frontier = {j for f in frontier for j in nb[f]} - seen
            seen |= frontier
        out.append(np.array(sorted(seen), int))
    return out


def compartment(comp: np.ndarray) -> np.ndarray:
    """Dominant compartment index per spot from (predicted) class fractions."""
    agg = np.stack([comp[:, [CLASSES.index(c) for c in v]].sum(1) for v in COMPARTMENTS.values()], 1)
    return agg.argmax(1)


def anatomical_units(nb: list[np.ndarray], comp_idx: np.ndarray, max_unit: int, rng) -> np.ndarray:
    """Partition spots into compact same-compartment units by seeded breadth-first growth."""
    n = len(nb)
    unit = np.full(n, -1)
    order = rng.permutation(n)
    u = 0
    for s in order:
        if unit[s] >= 0:
            continue
        q, unit[s], size = deque([s]), u, 1
        while q and size < max_unit:
            x = q.popleft()
            for y in nb[x]:
                if unit[y] < 0 and comp_idx[y] == comp_idx[s] and size < max_unit:
                    unit[y] = u
                    size += 1
                    q.append(y)
        u += 1
    return unit


def anatomical_hyperedges(nb: list[np.ndarray], comp_idx: np.ndarray, max_unit: int, rng):
    """Unit hyperedges plus interface hyperedges; returns (list of member arrays, edge type)."""
    unit = anatomical_units(nb, comp_idx, max_unit, rng)
    edges, etype = [], []
    for u in range(unit.max() + 1):
        m = np.where(unit == u)[0]
        edges.append(m)
        etype.append(0)
    seen = set()
    for i in range(len(nb)):
        for j in nb[i]:
            a, b = unit[i], unit[j]
            if a == b or comp_idx[i] == comp_idx[j]:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            ma, mb = np.where(unit == a)[0], np.where(unit == b)[0]
            # border spots of each unit that touch the other unit
            ba = [x for x in ma if np.isin(nb[x], mb).any()]
            bb = [x for x in mb if np.isin(nb[x], ma).any()]
            edges.append(np.array(sorted(set(ba) | set(bb)), int))
            etype.append(1)
    return edges, np.array(etype, int)


def rewire(edges: list[np.ndarray], n: int, rng) -> list[np.ndarray]:
    """Random hyperedges with the same size distribution (structure null)."""
    return [np.sort(rng.choice(n, size=len(e), replace=False)) for e in edges]


def incidence(edges: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    node = np.concatenate(edges) if edges else np.zeros(0, int)
    he = np.concatenate([np.full(len(e), k) for k, e in enumerate(edges)]) if edges else np.zeros(0, int)
    return node.astype(np.int64), he.astype(np.int64)


def clique_edges(edges: list[np.ndarray]) -> np.ndarray:
    pairs = set()
    for e in edges:
        for a in e:
            for b in e:
                if a != b:
                    pairs.add((int(a), int(b)))
    if not pairs:
        return np.zeros((2, 0), np.int64)
    return np.array(sorted(pairs), np.int64).T
