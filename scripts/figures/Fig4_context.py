"""Figure 4 (components, scale, capture area, cost; panels in fig4_panels.py) and Supplementary Fig. S3 (every model)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import fig4_panels as P
from figstyle import PAL, ROOT, W2, label, save

R = ROOT / "results"
PROG_LAB = {"pt_injury": "PT injury", "inflammation": "Inflammation", "pt_transport": "PT transport"}
PCOL = {"pt_injury": PAL["anatomy"], "inflammation": PAL["group"], "pt_transport": PAL["ref"]}
BETWEEN = [("gcn", "Lattice graph (GCN)"), ("gat", "Lattice graph (GAT)"), ("hgnn_ring", "Local hypergraph"),
           ("hgnn_anat", "Unit hypergraph"), ("clique_gcn", "Clique expansion"), ("set_ring", "Local set hypergraph"),
           ("set_anat", "Anatomical set hypergraph"), ("deepsets", "Neighbourhood set")]
WITHIN = [("tok_mean", "Token mean"), ("tok_abmil", "Token attention"), ("tok_gat", "Token graph (GAT)"),
          ("tok_hyper_anat", "Anatomical token hypergraph"), ("tok_abmil_ctx", "NephroSet")]
ABL = [("tok_hyper_rewired", "Random tokens, same size"), ("tok_hyper_rolebag", "Compartment, no contiguity"),
       ("tok_hyper_window", "3 × 3 windows"), ("tok_hyper_profile", "Profiles without halo"),
       ("tok_abmil", "No hyperedges (attention)")]


def paired(pm, a, b, k, metric, rng, n=2000):
    x = pm[(pm.model == a) & (pm.programme == k)].set_index("participant")[metric]
    y = pm[(pm.model == b) & (pm.programme == k)].set_index("participant")[metric]
    d = (x - y).dropna().values
    bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (d > 0).mean()


def forest(ax, pm, items, ref, rng, metric="rho"):
    rows, yt, yl = [], [], []
    y = 0
    for m, lab in items:
        for j, k in enumerate(PROG_LAB):
            mu, lo, hi, fp = paired(pm, m, ref, k, metric, rng)
            ax.errorbar(mu, -(y + j * 0.28), xerr=[[mu - lo], [hi - mu]], fmt="o", ms=2.6, color=PCOL[k], lw=0.8)
            rows.append(dict(model=m, reference=ref, programme=k, metric=metric, diff=mu, lo=lo, hi=hi,
                             frac_participants_positive=fp))
        yt.append(-(y + 0.28))
        yl.append(lab)
        y += 1.3
    ln = ax.axvline(0, c="0.6", lw=0.5)
    ln._is_reference = True
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    return rows


def supplementary_forests(pm, rng) -> None:
    """Supplementary figure: every model with context between spots, within the spot, and other hyperedges."""
    fig = plt.figure(figsize=(W2, 3.1))
    gs = fig.add_gridspec(1, 3, wspace=1.3, left=0.14, right=0.93, top=0.84, bottom=0.2)
    out = {}
    for j, (items, ref, xlab, title, xl, key) in enumerate([
            (BETWEEN, "mlp", "Δ ρ vs spot only", "Context between spots", (-0.03, 0.05), "a_between"),
            (WITHIN, "mlp", "Δ ρ vs spot only", "Token resolution", (-0.03, 0.05), "b_within"),
            (ABL, "tok_hyper_anat", "Δ ρ vs anatomical\ntoken hypergraph", "Other hyperedges", (-0.015, 0.016),
             "c_hyperedges")]):
        ax = fig.add_subplot(gs[j])
        out[key] = pd.DataFrame(forest(ax, pm, items, ref, rng))
        ax.set_xlabel(xlab + ("\n(95% CI, 201 participants)" if j < 2 else "\n(95% CI)"))
        ax.set_title(title, pad=3)
        ax.set_xlim(*xl)
        ax.tick_params(axis="y", labelsize=5.6)
        if j == 0:
            hl = [Line2D([], [], marker="o", ls="", ms=3, color=PCOL[k], label=PROG_LAB[k]) for k in PROG_LAB]
            fig.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.56, 0.995), ncol=3, fontsize=5.6,
                       handletextpad=0.1, columnspacing=1.2, frameon=False)
        label(ax, "abc"[j])
    save(fig, "FigS3", out)


def main() -> None:
    pm = pd.read_csv(R / "16_evaluate_cv" / "participant_metrics.tsv", sep="\t")
    rng = np.random.default_rng(2026)
    supplementary_forests(pm, rng)
    fig = plt.figure(figsize=(W2, 5.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1], hspace=0.78, wspace=0.42,
                          left=0.1, right=0.98, top=0.95, bottom=0.09)
    out = {}
    ax = fig.add_subplot(gs[0, 0])
    out["a_components"] = P.ladder_panel(ax)
    label(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    out["b_scale"] = P.scale_panel(ax)
    label(ax, "b")
    sub = gs[1, 0].subgridspec(1, 2, width_ratios=[0.42, 1], wspace=0.75)
    sch = fig.add_subplot(sub[0])
    ax = fig.add_subplot(sub[1])
    out["c_region"] = P.region_panel(ax, sch)
    label(sch, "c")
    ax = fig.add_subplot(gs[1, 1])
    out["d_cost"] = P.bubble_panel(ax)
    label(ax, "d")
    save(fig, "Fig4", out)


if __name__ == "__main__":
    main()
