"""Spot-level predictors that share one input representation and differ only in how context is used.

Every model sees the same per-spot input: frozen encoder features at two scales, the
morphology descriptors and image-predicted coarse class fractions ("roles"). Every model has
the same output heads: three programme scores and nine class fractions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

N_PROG, N_COMP = 3, 9


def scatter_mean(src, index, n):
    out = torch.zeros(n, src.shape[1], device=src.device, dtype=src.dtype)
    cnt = torch.zeros(n, 1, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt.index_add_(0, index, torch.ones_like(src[:, :1]))
    return out / cnt.clamp(min=1)


def scatter_max(src, index, n):
    out = torch.full((n, src.shape[1]), -1e4, device=src.device, dtype=src.dtype)
    out = out.scatter_reduce(0, index[:, None].expand_as(src), src, reduce="amax", include_self=True)
    return torch.where(out <= -1e4 + 1, torch.zeros_like(out), out)


def scatter_softmax(score, index, n):
    mx = torch.full((n,), -1e9, device=score.device, dtype=score.dtype)
    mx = mx.scatter_reduce(0, index, score, reduce="amax", include_self=True)
    e = torch.exp(score - mx[index])
    s = torch.zeros(n, device=score.device, dtype=score.dtype).index_add_(0, index, e)
    return e / s[index].clamp(min=1e-12)


class Stem(nn.Module):
    def __init__(self, d_in, d, p):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(p))

    def forward(self, x):
        return self.net(x)


class Heads(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.prog = nn.Linear(d, N_PROG)
        self.comp = nn.Linear(d, N_COMP)

    def forward(self, h):
        return self.prog(h), F.log_softmax(self.comp(h), -1)


class MLP(nn.Module):
    def __init__(self, d_in, d=256, p=0.2, **_):
        super().__init__()
        self.stem = Stem(d_in, d, p)
        self.body = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d), nn.GELU())
        self.heads = Heads(d)

    def forward(self, g):
        return self.heads(self.body(self.stem(g["x"])))


class DeepSets(nn.Module):
    """Spot plus an unordered set of its lattice neighbours with relative positions."""

    def __init__(self, d_in, d=256, p=0.2, **_):
        super().__init__()
        self.stem = Stem(d_in, d, p)
        self.phi = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, d))
        self.rho = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d), nn.GELU())
        self.heads = Heads(d)

    def forward(self, g):
        h = self.stem(g["x"])
        node, he = g["set_node"], g["set_he"]  # he indexes the centre spot
        rel = (g["pos"][node] - g["pos"][he]) / 300.0
        m = self.phi(torch.cat([h[node], rel], -1))
        n = h.shape[0]
        return self.heads(self.rho(torch.cat([h, scatter_mean(m, he, n), scatter_max(m, he, n)], -1)))


class GCNLayer(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.drop = nn.Dropout(p)

    def forward(self, h, ei):
        n = h.shape[0]
        loops = torch.arange(n, device=h.device)
        src = torch.cat([ei[0], loops])
        dst = torch.cat([ei[1], loops])
        deg = torch.zeros(n, device=h.device).index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        w = (deg[src] * deg[dst]).rsqrt()
        msg = self.lin(h)[src] * w[:, None]
        out = torch.zeros_like(h).index_add_(0, dst, msg)
        return h + self.drop(F.gelu(out))


class GCN(nn.Module):
    def __init__(self, d_in, d=256, p=0.2, edge_key="hex_ei", **_):
        super().__init__()
        self.stem = Stem(d_in, d, p)
        self.layers = nn.ModuleList([GCNLayer(d, p) for _ in range(2)])
        self.heads = Heads(d)
        self.edge_key = edge_key

    def forward(self, g):
        h = self.stem(g["x"])
        for layer in self.layers:
            h = layer(h, g[self.edge_key])
        return self.heads(h)


class GATLayer(nn.Module):
    def __init__(self, d, p, heads=4):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(p)

    def forward(self, h, ei):
        n = h.shape[0]
        loops = torch.arange(n, device=h.device)
        src, dst = torch.cat([ei[0], loops]), torch.cat([ei[1], loops])
        q = self.q(h).view(n, self.h, self.dk)
        k = self.k(h).view(n, self.h, self.dk)
        v = self.v(h).view(n, self.h, self.dk)
        s = (q[dst] * k[src]).sum(-1) / self.dk ** 0.5
        a = torch.stack([scatter_softmax(s[:, j], dst, n) for j in range(self.h)], 1)
        out = torch.zeros(n, self.h, self.dk, device=h.device, dtype=h.dtype)
        out.index_add_(0, dst, a[..., None] * v[src])
        return h + self.drop(F.gelu(self.o(out.reshape(n, -1))))


class GAT(GCN):
    def __init__(self, d_in, d=256, p=0.2, **kw):
        super().__init__(d_in, d, p, **kw)
        self.layers = nn.ModuleList([GATLayer(d, p) for _ in range(2)])


class HGNNLayer(nn.Module):
    """Linear hypergraph convolution (node -> hyperedge mean -> node mean)."""

    def __init__(self, d, p):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.drop = nn.Dropout(p)

    def forward(self, h, node, he, n_he):
        z = scatter_mean(self.lin(h)[node], he, n_he)
        out = scatter_mean(z[he], node, h.shape[0])
        return h + self.drop(F.gelu(out))


class HGNN(nn.Module):
    def __init__(self, d_in, d=256, p=0.2, edge_prefix="ring", **_):
        super().__init__()
        self.stem = Stem(d_in, d, p)
        self.layers = nn.ModuleList([HGNNLayer(d, p) for _ in range(2)])
        self.heads = Heads(d)
        self.pre = edge_prefix

    def forward(self, g):
        h = self.stem(g["x"])
        node, he, n_he = g[f"{self.pre}_node"], g[f"{self.pre}_he"], g[f"{self.pre}_n"]
        for layer in self.layers:
            h = layer(h, node, he, n_he)
        return self.heads(h)


class SetHyperLayer(nn.Module):
    """Non-linear hyperedge aggregation with roles and geometry, then attention back to nodes."""

    def __init__(self, d, p, n_types=2):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d + N_COMP + 2, d), nn.GELU(), nn.Linear(d, d))
        self.psi = nn.Sequential(nn.Linear(2 * d + N_COMP + 1 + n_types, d), nn.GELU(), nn.Linear(d, d))
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.n_types = n_types

    def forward(self, h, g, pre, use_roles=True, use_geom=True):
        node, he, n_he = g[f"{pre}_node"], g[f"{pre}_he"], g[f"{pre}_n"]
        roles = g["roles"] if use_roles else torch.zeros_like(g["roles"])
        cen = scatter_mean(g["pos"][node], he, n_he)
        rel = (g["pos"][node] - cen[he]) / 300.0
        if not use_geom:
            rel = torch.zeros_like(rel)
        m = self.phi(torch.cat([h[node], roles[node], rel], -1))
        size = torch.zeros(n_he, 1, device=h.device).index_add_(0, he, torch.ones_like(he, dtype=h.dtype)[:, None])
        rmean = scatter_mean(roles[node], he, n_he)
        et = F.one_hot(g[f"{pre}_type"], self.n_types).to(h.dtype)
        z = self.psi(torch.cat([scatter_mean(m, he, n_he), scatter_max(m, he, n_he), rmean,
                                torch.log1p(size) / 3.0, et], -1))
        s = (self.q(h)[node] * self.k(z)[he]).sum(-1) / h.shape[1] ** 0.5
        a = scatter_softmax(s, node, h.shape[0])
        agg = torch.zeros_like(h).index_add_(0, node, a[:, None] * z[he])
        return self.norm(h + self.upd(torch.cat([h, agg], -1)))


class SetHyper(nn.Module):
    def __init__(self, d_in, d=256, p=0.2, edge_prefix="ana", use_roles=True, use_geom=True, **_):
        super().__init__()
        self.stem = Stem(d_in, d, p)
        self.layers = nn.ModuleList([SetHyperLayer(d, p) for _ in range(2)])
        self.heads = Heads(d)
        self.pre, self.use_roles, self.use_geom = edge_prefix, use_roles, use_geom

    def forward(self, g):
        h = self.stem(g["x"])
        for layer in self.layers:
            h = layer(h, g, self.pre, self.use_roles, self.use_geom)
        return self.heads(h)


REGISTRY = {
    "mlp": (MLP, {}),
    "deepsets": (DeepSets, {}),
    "gcn": (GCN, {"edge_key": "hex_ei"}),
    "gat": (GAT, {"edge_key": "hex_ei"}),
    "clique_gcn": (GCN, {"edge_key": "clique_ei"}),
    "hgnn_ring": (HGNN, {"edge_prefix": "ring"}),
    "hgnn_anat": (HGNN, {"edge_prefix": "ana"}),
    "set_ring": (SetHyper, {"edge_prefix": "ring"}),
    "set_anat": (SetHyper, {"edge_prefix": "ana"}),
    "set_anat_rewired": (SetHyper, {"edge_prefix": "rew"}),
    "set_anat_noroles": (SetHyper, {"edge_prefix": "ana", "use_roles": False}),
    "set_anat_nogeom": (SetHyper, {"edge_prefix": "ana", "use_geom": False}),
}


# ---------------------------------------------------------------------------------------------
# Structure inside the spot: 14 x 14 image tokens of 8 um (see lib/tokens.py)
# ---------------------------------------------------------------------------------------------
D_TOK = 128 + 3


class TokGATLayer(GATLayer):
    pass


class TokHyperLayer(nn.Module):
    """Set aggregation inside each hyperedge (members, their position relative to the hyperedge
    centroid, hyperedge size and type), then attention of each token over its hyperedges."""

    def __init__(self, d, p, n_types=5):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, d))
        self.psi = nn.Sequential(nn.Linear(2 * d + 1 + n_types, d), nn.GELU(), nn.Linear(d, d))
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.n_types = n_types

    def forward(self, h, pos, node, he, n_he, typ):
        cen = scatter_mean(pos[node], he, n_he)
        m = self.phi(torch.cat([h[node], (pos[node] - cen[he]) * 4], -1))
        size = torch.zeros(n_he, 1, device=h.device, dtype=h.dtype).index_add_(
            0, he, torch.ones(len(he), 1, device=h.device, dtype=h.dtype))
        z = self.psi(torch.cat([scatter_mean(m, he, n_he), scatter_max(m, he, n_he), torch.log1p(size) / 3.0,
                                F.one_hot(typ, self.n_types).to(h.dtype)], -1))
        s = (self.q(h)[node] * self.k(z)[he]).sum(-1) / h.shape[1] ** 0.5
        a = scatter_softmax(s, node, h.shape[0])
        agg = torch.zeros_like(h).index_add_(0, node, a[:, None] * z[he])
        return self.norm(h + self.upd(torch.cat([h, agg], -1)))


class TokenNet(nn.Module):
    """Spot input (as in the MLP) plus a pooled summary of the spot's image tokens.

    mode: mean (token average), abmil (gated attention pooling, an unordered set),
          gat (pairwise attention on the 8-neighbour token grid, then attention pooling),
          hyper (hypergraph layers on the hyperedge family `fam`, then attention pooling).
    ctx:  also aggregate the spot embeddings of lattice neighbours as in NephroSet.
    """

    def __init__(self, d_in, d=256, p=0.2, mode="abmil", fam=None, ctx=False, dt=128, **_):
        super().__init__()
        self.mode, self.fam, self.ctx = mode, fam, ctx
        self.stem = Stem(d_in, d, p)
        self.tstem = nn.Sequential(nn.LayerNorm(D_TOK + 2), nn.Linear(D_TOK + 2, dt), nn.GELU(), nn.Dropout(p))
        if mode == "gat":
            self.tlayers = nn.ModuleList([TokGATLayer(dt, p) for _ in range(2)])
        elif mode == "hyper":
            self.tlayers = nn.ModuleList([TokHyperLayer(dt, p) for _ in range(2)])
        self.att_v, self.att_u, self.att_w = nn.Linear(dt, dt), nn.Linear(dt, dt), nn.Linear(dt, 1)
        if ctx:
            self.phi = nn.Sequential(nn.Linear(d + 2, d), nn.GELU(), nn.Linear(d, d))
        din = d + dt + (2 * d if ctx else 0)
        self.body = nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d), nn.GELU())
        self.heads = Heads(d)

    @staticmethod
    def grid(n, nt, device):
        """8-neighbour token edges inside every spot, built on the fly (not stored per section)."""
        g = int(round(nt ** 0.5))
        r, c = torch.div(torch.arange(nt, device=device), g, rounding_mode="floor"), torch.arange(nt, device=device) % g
        src, dst = [], []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                ok = (r + dr >= 0) & (r + dr < g) & (c + dc >= 0) & (c + dc < g)
                src.append(torch.nonzero(ok).squeeze(1))
                dst.append(((r + dr) * g + (c + dc))[ok])
        s, d = torch.cat(src), torch.cat(dst)
        off = (torch.arange(n, device=device) * nt)[:, None]
        return torch.stack([(s + off).reshape(-1), (d + off).reshape(-1)])

    def tokens(self, g):
        dev = g["x"].device
        g = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in g.items()}
        n = g["x"].shape[0]
        nt = g["tok"].shape[0] // n
        pos = g["tpos"].repeat(n, 1)
        t = self.tstem(torch.cat([g["tok"].float(), pos], -1))
        ck = (lambda fn, *a: checkpoint(fn, *a, use_reentrant=False)) if self.training else (lambda fn, *a: fn(*a))
        if self.mode == "gat":
            for layer in self.tlayers:
                t = ck(layer, t, self.grid(n, nt, t.device))
        elif self.mode == "hyper":
            f = self.fam
            for layer in self.tlayers:
                t = ck(layer, t, pos, g[f"t_{f}_node"], g[f"t_{f}_he"], g[f"t_{f}_n"], g[f"t_{f}_type"])
        t = t.view(n, nt, -1)
        w = g["tmask"].view(n, nt)
        if self.mode == "mean":
            return (t * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)
        s = self.att_w(torch.tanh(self.att_v(t)) * torch.sigmoid(self.att_u(t))).squeeze(-1)
        s = s.masked_fill(w < 0.5, -1e4)
        return (torch.softmax(s, 1)[..., None] * t).sum(1)

    def forward(self, g):
        h = self.stem(g["x"])
        parts = [h, self.tokens(g)]
        if self.ctx:
            node, he = g["set_node"], g["set_he"]
            rel = (g["pos"][node] - g["pos"][he]) / 300.0
            m = self.phi(torch.cat([h[node], rel], -1))
            parts += [scatter_mean(m, he, h.shape[0]), scatter_max(m, he, h.shape[0])]
        return self.heads(self.body(torch.cat(parts, -1)))


TOKEN_REGISTRY = {
    "tok_mean": (TokenNet, {"mode": "mean"}),
    "tok_abmil": (TokenNet, {"mode": "abmil"}),
    "tok_gat": (TokenNet, {"mode": "gat"}),
    "tok_hyper_window": (TokenNet, {"mode": "hyper", "fam": "window"}),
    "tok_hyper_rolebag": (TokenNet, {"mode": "hyper", "fam": "rolebag"}),
    "tok_hyper_rewired": (TokenNet, {"mode": "hyper", "fam": "rewired"}),
    "tok_hyper_profile": (TokenNet, {"mode": "hyper", "fam": "profile"}),
    "tok_hyper_anat": (TokenNet, {"mode": "hyper", "fam": "anat"}),
    "tok_abmil_ctx": (TokenNet, {"mode": "abmil", "ctx": True}),
    "tok_hyper_anat_ctx": (TokenNet, {"mode": "hyper", "fam": "anat", "ctx": True}),
}
REGISTRY.update(TOKEN_REGISTRY)
