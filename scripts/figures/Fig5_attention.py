"""Figure 5: what NephroSet attends to inside the spot (token attention, held-out participants).

Reads only figures/source_data/Fig5_attention_* written by scripts/stages/16d_token_attention.py.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figstyle import PAL, SRC, W2, label, save

CLS_COL = {"PT": "#D55E00", "LOH": "#56B4E9", "DCT_CNT": "#0072B2", "CD": "#332288", "Glomerular": "#CC79A7",
           "Endothelial": "#E69F00", "Fibroblast": "#009E73", "Mural": "#F0E442", "Immune": "#222222"}
CLS_LAB = {"PT": "PT", "LOH": "LOH", "DCT_CNT": "DCT/CNT", "CD": "CD", "Glomerular": "Glomerular",
           "Endothelial": "Endothelial", "Fibroblast": "Fibroblast", "Mural": "Mural", "Immune": "Immune"}
COMP_LAB = {"proximal": "Proximal epithelium", "distal": "Distal epithelium", "interstitial": "Interstitium",
            "glomerular": "Glomerulus", "pale": "Pale (lumen/space)", "lining": "Lining pale tokens",
            "dense": "Dense tissue"}
FEAT_LAB = {"haematoxylin_od": "Haematoxylin OD", "eosin_od": "Eosin OD", "pale": "Pale token",
            "lining": "Lining token"}
INJ_LAB = {"excess_nonPT": "Excess attention,\nnon-PT tokens", "excess_interstitial": "Excess attention,\ninterstitium",
           "excess_pale": "Excess attention,\npale tokens", "entropy_norm": "Attention entropy",
           "share_nonPT": "Non-PT token share\n(composition)"}


def forest(ax, rows, ylab, xcol="log2_ratio", col="0.2", dy=0.0, ms=2.8, marker="o", lab=None):
    y = np.arange(len(rows))[::-1] + dy
    ax.errorbar(rows[xcol], y, xerr=[rows[xcol] - rows.lo, rows.hi - rows[xcol]], fmt=marker, ms=ms, color=col,
                lw=0.8, capsize=0, label=lab)
    ax.set_yticks(np.arange(len(rows))[::-1])
    ax.set_yticklabels(ylab)
    return y


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def main() -> None:
    enr = pd.read_csv(SRC / "Fig5_attention_a_enrichment.tsv", sep="\t")
    mor = pd.read_csv(SRC / "Fig5_attention_b_morphology.tsv", sep="\t")
    inj = pd.read_csv(SRC / "Fig5_attention_c_injury.tsv", sep="\t")
    dsum = pd.read_csv(SRC / "Fig5_attention_d_deletion.tsv", sep="\t")
    ex = pd.read_csv(SRC / "Fig5_attention_e_examples.tsv", sep="\t")
    z = np.load(SRC / "Fig5_attention_e_examples.npz", allow_pickle=True)
    classes = list(z["classes"])

    fig = plt.figure(figsize=(W2, 6.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.32, left=0.12, right=0.99, top=0.96, bottom=0.08)
    top = gs[0].subgridspec(1, 3, width_ratios=[1.05, 0.9, 0.95], wspace=0.95)
    bot = gs[1].subgridspec(1, 2, width_ratios=[0.6, 1.6], wspace=0.28)

    # a  enrichment by token role ------------------------------------------------------------------
    ax = fig.add_subplot(top[0])
    keys = [("compartment", "proximal"), ("compartment", "distal"), ("compartment", "interstitial"),
            ("compartment", "glomerular"), ("position", "pale"), ("position", "lining"), ("position", "dense")]
    for sub, col, dy, mk in [("all", PAL["dark"], 0.15, "o"), ("mixed", PAL["anatomy"], -0.15, "s")]:
        q = enr[enr.subset == sub].set_index(["kind", "group"]).loc[keys].reset_index()
        forest(ax, q, [COMP_LAB[g] for g in q.group], col=col, dy=dy, marker=mk, ms=2.6,
               lab="All spots" if sub == "all" else "Mixed PT–\ninterstitium spots")
    ln = ax.axvline(0, color="0.6", lw=0.6, ls=(0, (2, 1.5)))
    ln._is_reference = True
    ax.set_xlabel("Attention mass / token share (log$_2$)")
    ax.set_title("Attention by token role", pad=3)
    ax.legend(loc="center left", fontsize=5.4, handletextpad=0.2, ncol=1, borderaxespad=0.3)
    label(ax, "a")

    # b  attention vs token morphology -------------------------------------------------------------
    ax = fig.add_subplot(top[1])
    feats = list(FEAT_LAB)
    for scope, col, dy, mk, lab in [("tissue_tokens", PAL["dark"], 0.15, "o", "All tissue tokens"),
                                    ("PT_tokens", PAL["anatomy"], -0.15, "s", "PT tokens only")]:
        q = mor[(mor.subset == "all") & (mor.tokens == scope)].set_index("feature").loc[feats].reset_index()
        q = q.rename(columns={"mean_rho": "r"})
        forest(ax, q, [FEAT_LAB[f] for f in feats], xcol="r", col=col, dy=dy, marker=mk, ms=2.6, lab=lab)
    ln = ax.axvline(0, color="0.6", lw=0.6, ls=(0, (2, 1.5)))
    ln._is_reference = True
    ax.set_xlabel("Spearman ρ with attention\n(within spot, mean over participants)")
    ax.set_title("Attention and token morphology", pad=3)
    ax.legend(loc="upper left", fontsize=5.4, handletextpad=0.2, borderaxespad=0.3)
    label(ax, "b")

    # c  injury dependence ----------------------------------------------------------------------------
    ax = fig.add_subplot(top[2])
    mets = list(INJ_LAB)
    q = inj[inj.target == "y_pt_injury"].set_index("metric").loc[mets].reset_index().rename(columns={"mean_rho": "r"})
    forest(ax, q, [INJ_LAB[m] for m in mets], xcol="r", col=PAL["ref"])
    ln = ax.axvline(0, color="0.6", lw=0.6, ls=(0, (2, 1.5)))
    ln._is_reference = True
    ax.set_xlabel("Spearman ρ with measured PT injury\n(PT-rich spots, within section)")
    ax.set_title("Attention shifts with injury", pad=3)
    label(ax, "c")

    # d  deletion test --------------------------------------------------------------------------------
    ax = fig.add_subplot(bot[0])
    d = dsum[(dsum.programme == "pt_injury") & (dsum.metric == "rho") & (dsum.kind == "difference")]
    fr = [10, 20, 40]
    for how, col, mk, lab in [("top", PAL["anatomy"], "o", "Highest-attention tokens"),
                              ("rand", "0.45", "s", "Random tokens")]:
        q = d.set_index("contrast").loc[[f"pred_{how}{f} - pred" for f in fr]]
        ax.errorbar(fr, q["mean"], yerr=[q["mean"] - q.lo, q.hi - q["mean"]], fmt=mk + "-", ms=2.8, lw=0.8,
                    color=col, label=lab, capsize=0)
    ln = ax.axhline(0, color="0.6", lw=0.6, ls=(0, (2, 1.5)))
    ln._is_reference = True
    ax.set_xticks(fr)
    ax.set_xticklabels([f"{f}%" for f in fr])
    ax.set_xlim(5, 45)
    ax.set_xlabel("Tissue tokens removed from pooling")
    ax.set_ylabel("Change in within-section\nPT injury Spearman ρ")
    ax.set_title("Deletion test", pad=3)
    lo = min(d["lo"].min(), 0)
    ax.set_ylim(lo * 1.15, max(0.02, d["hi"].max()) + abs(lo) * 0.65)
    ax.legend(loc="upper right", fontsize=5.4, handletextpad=0.3, borderaxespad=0.2)
    label(ax, "d")

    # e  examples -------------------------------------------------------------------------------------
    n = len(ex)
    eg = bot[1].subgridspec(3, n + 1, width_ratios=[1] * n + [0.08], wspace=0.06, hspace=0.08)
    cmap_cls = ListedColormap([CLS_COL[c] for c in classes])
    first = None
    for i, r in ex.iterrows():
        img = z["img"][i]
        att = z["att"][i]
        tis = z["tissue"][i] >= float(z["tissue_min"])
        rel = att * tis.sum()  # 1 = uniform attention over tissue tokens
        rel = np.where(tis, rel, np.nan)
        cls = np.where(tis, z["cls"][i], np.nan).astype(float)
        a0 = fig.add_subplot(eg[0, i])
        a0.imshow(img)
        a0.set_title(f"P{i // 2 + 1}, {r.injury_group} injury\n({ordinal(round(r.y_pt_injury_pct))} pct)", pad=2,
                     fontsize=5.6)
        a1 = fig.add_subplot(eg[1, i])
        a1.imshow(img)
        im = a1.imshow(np.log2(np.clip(rel, 0.25, 4)), extent=(0, 224, 224, 0), cmap="magma", vmin=-2, vmax=2,
                       alpha=0.62, interpolation="nearest")
        a2 = fig.add_subplot(eg[2, i])
        a2.imshow(img)
        a2.imshow(cls, extent=(0, 224, 224, 0), cmap=cmap_cls, vmin=-0.5, vmax=len(classes) - 0.5, alpha=0.6,
                  interpolation="nearest")
        for a in (a0, a1, a2):
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
        if i == 0:
            first = a0
            a0.set_ylabel("H&E (112 µm)", fontsize=6)
            a1.set_ylabel("Attention", fontsize=6)
            a2.set_ylabel("Token role", fontsize=6)
            a0.plot([224 - 8 - 50 / 0.5, 224 - 8], [214, 214], c="k", lw=1.0)
    cax = fig.add_subplot(eg[1, n])
    cb = fig.colorbar(im, cax=cax, ticks=[-2, 0, 2])
    cb.ax.set_yticklabels(["¼", "1", "4"])
    cb.set_label("× uniform", fontsize=5.5)
    cax.tick_params(labelsize=5.5)
    leg_ax = fig.add_subplot(eg[2, n])
    leg_ax.axis("off")
    hs = [Patch(facecolor=CLS_COL[c], alpha=0.75, label=CLS_LAB[c]) for c in classes]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.62, 0.035), ncol=9, fontsize=5.4,
               handlelength=0.9, handletextpad=0.3, columnspacing=0.7)
    first.text(1.0, -0.02, "50 µm", transform=first.transAxes, ha="right", va="top", fontsize=5)
    label(first, "e")
    _ = Line2D
    save(fig, "Fig5")


if __name__ == "__main__":
    main()
