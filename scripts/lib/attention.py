"""Token attention weights of the token models, exposed without changing their computation.

`AttnTokenNet` is a TokenNet whose `tokens` step additionally keeps the softmax attention over
the 196 tokens of every spot (`last_att`, shape n x 196; zero weight on masked tokens). Its
forward pass is numerically identical to TokenNet; a trained TokenNet state dict loads as is.
An optional `keep` tensor (n*196, 0/1) removes tokens from the attention pool in addition to
the tissue mask (used for deletion tests).
"""
from __future__ import annotations

import torch

from .models import TOKEN_REGISTRY, TokenNet


class AttnTokenNet(TokenNet):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last_att = None
        self.keep = None

    def tokens(self, g):
        if self.mode in ("mean",):
            raise ValueError("attention weights exist only for attention pooling")
        if self.keep is not None:
            g = dict(g)
            g["tmask"] = g["tmask"].to(self.keep.device) * self.keep
        dev = g["x"].device
        g = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in g.items()}
        n = g["x"].shape[0]
        nt = g["tok"].shape[0] // n
        pos = g["tpos"].repeat(n, 1)
        t = self.tstem(torch.cat([g["tok"].float(), pos], -1))
        if self.mode == "gat":
            for layer in self.tlayers:
                t = layer(t, self.grid(n, nt, t.device))
        elif self.mode == "hyper":
            f = self.fam
            for layer in self.tlayers:
                t = layer(t, pos, g[f"t_{f}_node"], g[f"t_{f}_he"], g[f"t_{f}_n"], g[f"t_{f}_type"])
        t = t.view(n, nt, -1)
        w = g["tmask"].view(n, nt)
        s = self.att_w(torch.tanh(self.att_v(t)) * torch.sigmoid(self.att_u(t))).squeeze(-1)
        s = s.masked_fill(w < 0.5, -1e4)
        a = torch.softmax(s, 1)
        self.last_att = a.detach()
        return (a[..., None] * t).sum(1)


def from_trained(model: TokenNet, name: str, d_in: int, d: int, p: float) -> AttnTokenNet:
    """Copy of a trained TokenNet of registry entry `name` that also records attention."""
    _, kw = TOKEN_REGISTRY[name]
    m = AttnTokenNet(d_in, d=d, p=p, **kw).to(next(model.parameters()).device)
    m.load_state_dict(model.state_dict())
    return m.eval()
