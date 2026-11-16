"""Figure 4: context between spots versus structure within the spot (paired by participant)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figstyle import MODEL_LABEL, PAL, ROOT, W2, label, save

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


SCALE = [("deepsets", "mlp", "Between spots\n(neighbourhood set)"),
         ("tok_abmil", "mlp", "Within the spot\n(token attention)"),
         ("tok_hyper_anat", "tok_abmil", "Anatomical prior\n(token hypergraph)")]
LADDER = ["ridge_image", "mlp", "deepsets", "tok_abmil", "tok_abmil_ctx"]
LAD_LAB = {"ridge_image": "Linear\nprobe", "mlp": "Spot\nonly", "deepsets": "Neighbour-\nhood set",
           "tok_abmil": "Token\nattention", "tok_abmil_ctx": "Nephro-\nSet"}
COH = [("KPMP", "16_evaluate_cv", PAL["anatomy"], "o"), ("Second centre", "18_evaluate_external", PAL["ref"], "s")]


def scale_panel(ax):
    rows = []
    for c, (coh, d, col, mk) in enumerate(COH):
        con = pd.read_csv(R / d / "contrasts.tsv", sep="\t")
        for i, (a, b, _) in enumerate(SCALE):
            for j, k in enumerate(["pt_injury", "pt_transport"]):
                q = con[(con.a == a) & (con.b == b) & (con.programme == k) & (con.metric == "rho")].iloc[0]
                x = i + (j - 0.5) * 0.42 + (c - 0.5) * 0.17
                ax.errorbar(x, q["diff"], yerr=[[q["diff"] - q.lo], [q.hi - q["diff"]]], fmt=mk, ms=2.8, lw=0.8,
                            color=col, mfc=col if k == "pt_injury" else "white", mew=0.8)
                rows.append(dict(cohort=coh, comparison=f"{a} vs {b}", programme=k, diff=q["diff"], lo=q.lo, hi=q.hi,
                                 n=q.n, frac_positive=q.frac_positive))
    ln = ax.axhline(0, c="0.6", lw=0.5)
    ln._is_reference = True
    ax.set_xticks(range(3))
    ax.set_xticklabels([t for *_, t in SCALE])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Δ ρ (95% CI)")
    ax.set_title("Gain from context at each scale", pad=3)
    hl = [Line2D([], [], marker=mk, ls="", ms=3, color=col, label=coh) for coh, _, col, mk in COH]
    hl += [Line2D([], [], marker="o", ls="", ms=3, color="0.3", label="PT injury"),
           Line2D([], [], marker="o", ls="", ms=3, color="0.3", mfc="white", label="PT transport")]
    lo = min(r["lo"] for r in rows)
    ax.set_ylim(lo - 0.01, 0.1)
    ax.legend(handles=hl, loc="upper left", fontsize=5.2, ncol=4, columnspacing=0.9, handletextpad=0.2,
              borderaxespad=0.2)
    return pd.DataFrame(rows)


def ladder_panel(ax):
    rows = []
    for c, (coh, d, col, mk) in enumerate(COH):
        sm = pd.read_csv(R / d / "summary.tsv", sep="\t")
        for j, k in enumerate(["pt_injury", "pt_transport"]):
            ys = []
            for i, m in enumerate(LADDER):
                q = sm[(sm.model == m) & (sm.programme == k) & (sm.metric == "rho")].iloc[0]
                ys.append(q["mean"])
                rows.append(dict(cohort=coh, programme=k, model=m, rho=q["mean"], lo=q.lo, hi=q.hi))
            ax.plot(range(len(LADDER)), ys, marker=mk, ms=2.8, lw=0.9, color=col,
                    ls="-" if k == "pt_injury" else (0, (2, 1.5)), mfc=col if k == "pt_injury" else "white", mew=0.8)
    ax.set_xticks(range(len(LADDER)))
    ax.set_xticklabels([LAD_LAB[m] for m in LADDER], fontsize=5.4)
    ax.set_xlim(-0.4, 5.3)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Within-section Spearman ρ")
    ax.set_title("Absolute agreement", pad=3)
    for k, y in [("PT\ntransport", 0.57), ("PT\ninjury", 0.2)]:
        ax.text(4.25, y, k, fontsize=5.4, ha="left", va="center")
    return pd.DataFrame(rows)


def main() -> None:
    pm = pd.read_csv(R / "16_evaluate_cv" / "participant_metrics.tsv", sep="\t")
    rng = np.random.default_rng(2026)
    fig = plt.figure(figsize=(W2, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 0.85], hspace=0.62, left=0.14, right=0.96, top=0.91, bottom=0.1)
    gs = outer[0].subgridspec(1, 3, wspace=1.25)
    out = {}
    for j, (items, ref, xlab, title, xl, key) in enumerate([
            (BETWEEN, "mlp", "Δ ρ vs spot only", "Context between spots", (-0.03, 0.05), "a_between"),
            (WITHIN, "mlp", "Δ ρ vs spot only", "Structure within the spot", (-0.03, 0.05), "b_within"),
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
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.2, 1.15], wspace=0.3)
    ax = fig.add_subplot(bot[0])
    out["d_scale"] = scale_panel(ax)
    label(ax, "d")
    ax = fig.add_subplot(bot[1])
    out["e_ladder"] = ladder_panel(ax)
    label(ax, "e")
    save(fig, "Fig4", out)


if __name__ == "__main__":
    main()
