"""Figure 2: what spatial organisation adds to PT injury state at single-cell resolution."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from figstyle import PAL, ROOT, W2, label, save

R = ROOT / "results"
CLASS_COL = {"PT": "#9ecae1", "LOH": "#c6dbef", "DCT_CNT": "#dadaeb", "CD": "#e5e5e5", "Glomerular": "#fdd0a2",
             "Endothelial": "#e6550d", "Fibroblast": "#31a354", "Mural": "#756bb1", "Immune": "#de2d26"}
CLASS_LAB = {"PT": "Proximal tubule", "LOH": "Loop of Henle", "DCT_CNT": "Distal/connecting", "CD": "Collecting duct",
             "Glomerular": "Glomerular", "Endothelial": "Endothelial", "Fibroblast": "Fibroblast", "Mural": "Mural",
             "Immune": "Immune"}
TARGETS = {"yunit_pt_injury": "PT injury (unit)", "yres_pt_injury": "PT injury (cell)",
           "yunit_inflammation": "Inflammation (unit)", "is_iPT": "Injured PT label"}
LEVEL_LAB = {"A": "A", "AB": "A+B", "ABC": "A+B+C", "ABD": "A+B+D"}


def main() -> None:
    ex = pd.read_csv(R / "04_atlas_examples" / "example_window.tsv", sep="\t")
    perf = pd.read_csv(R / "03_atlas_increment" / "model_performance.tsv", sep="\t")
    inc = pd.read_csv(R / "03_atlas_increment" / "increments.tsv", sep="\t")

    fig = plt.figure(figsize=(W2, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.25], wspace=0.55, left=0.02, right=0.99, top=0.88,
                          bottom=0.2)
    # a: example window
    ax = fig.add_subplot(gs[0])
    for c, col in CLASS_COL.items():
        q = ex[ex.coarse == c]
        ax.scatter(q.x_um, q.y_um, s=9, c=col, lw=0, label=CLASS_LAB[c])
    u = ex[ex.in_unit]
    ax.scatter(u.x_um, u.y_um, s=16, facecolors="none", edgecolors=PAL["dark"], lw=0.7)
    h = ex[ex.in_halo]
    ax.scatter(h.x_um, h.y_um, s=30, facecolors="none", edgecolors="k", lw=0.9, linestyle=(0, (1, 1)))
    t = ex[ex.is_target]
    ax.scatter(t.x_um, t.y_um, s=40, marker="*", c="k", lw=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.set_ylim(-150, 122)
    ax.plot([60, 110], [-140, -140], c="k", lw=1.2)
    ax.text(85, -134, "50 µm", ha="center", va="bottom", fontsize=6)
    ax.set_title("Tubular unit and its halo", pad=3)
    handles = [Line2D([], [], marker="o", ls="", ms=3.5, mfc=CLASS_COL[c], mec="none", label=CLASS_LAB[c])
               for c in ["PT", "Endothelial", "Fibroblast", "Immune", "Mural", "DCT_CNT", "Glomerular"]]
    handles += [Line2D([], [], marker="o", ls="", ms=4, mfc="none", mec=PAL["dark"], label="Unit member"),
                Line2D([], [], marker="o", ls="", ms=5, mfc="none", mec="k", label="Halo cell"),
                Line2D([], [], marker="*", ls="", ms=6, mfc="k", mec="none", label="Index PT cell")]
    ax.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=5.5,
              handletextpad=0.2, columnspacing=0.6)
    label(ax, "a")

    # b: performance across nested descriptor sets (observed labels)
    ax = fig.add_subplot(gs[1])
    lv = list(LEVEL_LAB)
    rows = []
    for k, (tg, lab) in enumerate(TARGETS.items()):
        q = perf[(perf.programme == tg) & (~perf.null)].set_index("model").loc[lv]
        base = q.donor_mean_rho.iloc[0]
        ax.plot(range(len(lv)), q.donor_mean_rho - base, marker="o", ms=3, lw=1,
                color=[PAL["anatomy"], PAL["image"], PAL["group"], PAL["ref"]][k], label=lab)
        rows += [dict(target=tg, level=m, value=v, gain_over_composition=v - base)
                 for m, v in zip(lv, q.donor_mean_rho)]
    ax.axhline(0, c="0.6", lw=0.5)
    ax.lines[-1]._is_reference = True
    ax.set_xticks(range(len(lv)))
    ax.set_xticklabels([LEVEL_LAB[x] for x in lv])
    ax.set_xlabel("Descriptor set")
    ax.set_ylabel("Gain over composition\n(donor-mean Spearman / AUROC)")
    ax.set_title("Nested descriptor sets", pad=3)
    ax.legend(loc="upper left", fontsize=5.5, handlelength=1.2)
    ax.set_ylim(-0.003, 0.06)
    label(ax, "b")

    # c: increments with donor bootstrap intervals; observed versus permuted interstitial labels
    ax = fig.add_subplot(gs[2])
    con = [("AB-A", "Pairwise\nover composition"), ("ABC-AB", "Local co-occurrence\nover pairwise"),
           ("ABDu-AB", "Unit geometry\nover pairwise"), ("ABDh-AB", "Unit halo\nover pairwise")]
    tg_show = ["yunit_pt_injury", "yunit_inflammation", "is_iPT"]
    cols = [PAL["anatomy"], PAL["group"], PAL["ref"]]
    yrows = []
    for i, (cn, _) in enumerate(con):
        for j, tg in enumerate(tg_show):
            for null in (False, True):
                q = inc[(inc.contrast == cn) & (inc.programme == tg) & (inc.null == null)].iloc[0]
                yy = i * 4 + j + (0.38 if null else 0)
                ax.barh(-yy, q["mean"], 0.36, color="white" if null else cols[j], ec=cols[j], lw=0.6,
                        hatch="//////" if null else None)
                ax.plot([q.lo, q.hi], [-yy, -yy], color="0.15", lw=0.6)
                yrows.append(dict(contrast=cn, target=tg, permuted=null, mean=q["mean"], lo=q.lo, hi=q.hi,
                                  frac_donors_positive=q.frac_donors_positive))
    ax.axvline(0, c="0.6", lw=0.5)
    ax.lines[-1]._is_reference = True
    ax.set_yticks([-(i * 4 + 1) for i in range(len(con))])
    ax.set_yticklabels([c[1] for c in con])
    ax.set_xlabel("Increment (95% CI over donors)")
    ax.set_title("Where the spatial information lies", pad=3)
    hl = [Patch(fc=c, label=TARGETS[t]) for c, t in zip(cols, tg_show)]
    hl += [Patch(fc="0.45", label="Observed labels"), Patch(fc="white", ec="0.45", hatch="//////",
                                                           label="Permuted within 100 µm")]
    ax.legend(handles=hl, loc="lower right", fontsize=5.3, handletextpad=0.2, borderaxespad=0.2)
    ax.set_xlim(-0.006, 0.05)
    label(ax, "c")
    save(fig, "Fig2", {"a_window": ex, "b_levels": pd.DataFrame(rows), "c_increments": pd.DataFrame(yrows)})


if __name__ == "__main__":
    main()
