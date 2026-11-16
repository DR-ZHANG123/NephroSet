"""Figure 1: study design, drawn in code. Cohort numbers are read from result tables."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, RegularPolygon

from figstyle import PAL, ROOT, W2, save

R = ROOT / "results"


def box(ax, x, y, w, h, title, lines, fc, title_fc, step=0.052):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec="0.55", lw=0.6))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.075), w, 0.075, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=title_fc, ec="none"))
    ax.text(x + w / 2, y + h - 0.0375, title, ha="center", va="center", fontsize=6.8, fontweight="bold", color="white")
    for i, ln in enumerate(lines):
        bold = ln.endswith(":")
        ax.text(x + 0.012, y + h - 0.105 - i * step, ln.rstrip(":") if bold else ln, ha="left", va="top",
                fontsize=5.8, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.35"))


def hexes(ax, cx, cy, r, vals, cmap):
    k = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if abs(i + j) > 1 and i * j > 0:
                continue
            x = cx + r * 1.75 * (j + 0.5 * i)
            y = cy + r * 1.52 * i
            ax.add_patch(RegularPolygon((x, y), 6, radius=r, fc=cmap(vals[k % len(vals)]), ec="white", lw=0.4))
            k += 1


def main() -> None:
    atlas = pd.read_csv(R / "01_atlas_extract" / "atlas_samples.tsv", sep="\t")
    tg = pd.read_csv(R / "02_atlas_features" / "targets_per_section.tsv", sep="\t")
    kp = pd.read_csv(R / "11b_kpmp_targets" / "spot_summary.tsv", sep="\t")
    kp = kp[kp.status == "ok"]
    ex = pd.read_csv(R / "13_gse211785_prep" / "sections.tsv", sep="\t")
    nums = dict(atlas_sections=len(atlas), atlas_donors=atlas.donor.nunique(), atlas_cells=int(atlas.n_cells.sum()),
                pt_targets=int(tg.n_targets.sum()), kpmp_participants=len(kp), kpmp_spots=int(kp.n_spots.sum()),
                ext_donors=len(ex), ext_spots=int(ex.n_spots.sum()))

    fig = plt.figure(figsize=(W2, 3.3))
    global AR
    AR = W2 / 3.3
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    blue, green, orange, grey = "#2c6e9b", "#2e8b57", "#c0602a", "#5a5a5a"
    # data column
    box(ax, 0.01, 0.53, 0.22, 0.44, "Public data", [
        "CosMx / Xenium kidney atlas",
        f"  {nums['atlas_sections']} sections, {nums['atlas_donors']} donors",
        f"  {nums['atlas_cells'] / 1e6:.2f} M cells",
        "KPMP Visium + same-section H&E",
        f"  {nums['kpmp_participants']} participants",
        f"  {nums['kpmp_spots']:,} spots",
    ], "#eef4fa", blue)
    box(ax, 0.01, 0.05, 0.22, 0.42, "Independent centre", [
        "GSE211785 Visium + H&E",
        f"  {nums['ext_donors']} donors, {nums['ext_spots']:,} spots",
        "  nephrectomy tissue",
        "Models applied unchanged;",
        "  measurements used only to evaluate",
    ], "#f3f3f3", grey)
    # question 1
    box(ax, 0.27, 0.53, 0.33, 0.44, "1  Spatial organization (single cell)", [
        f"{nums['pt_targets']:,} proximal tubule (PT) cells",
        "Target: PT injury, inflammation, transport",
        "A  composition around the cell",
        "B  + pairwise distances to each class",
        "C  + local co-occurrence of classes",
        "D  + tubular unit and its halo",
        "Donor-held-out; permuted-label null",
    ], "#eef7f1", green, step=0.047)
    # cell sketch for question 1
    rng = np.random.default_rng(3)
    cx, cy = 0.545, 0.64
    for k in range(10):
        a = 2 * np.pi * k / 10
        ax.add_patch(Ellipse((cx + 0.026 * np.cos(a), cy + 0.052 * np.sin(a)), 0.012, 0.012 * AR, fc="#9ecae1", ec="none"))
    for col, n in [("#de2d26", 3), ("#31a354", 3), ("#e6550d", 2)]:
        for _ in range(n):
            a = rng.uniform(0, 2 * np.pi)
            ax.add_patch(Ellipse((cx + 0.045 * np.cos(a), cy + 0.09 * np.sin(a)), 0.010, 0.010 * AR, fc=col, ec="none"))
    # question 2
    box(ax, 0.27, 0.05, 0.33, 0.42, "2  Histology to molecular programmes", [
        "H&E crops 112 µm and 448 µm per spot",
        "Pretrained pathology encoder (Phikon-v2)",
        "Between spots: graph | hypergraph | set",
        "Within spot: 8 µm tokens, attention |",
        "  token graph | token hypergraph",
        "Supervision: same-section Visium",
        "Output: 3 programmes + composition",
    ], "#fdf3ec", orange, step=0.043)
    hexes(ax, 0.545, 0.14, 0.017, [0.2, 0.7, 0.4, 0.9, 0.3, 0.6, 0.8], plt.cm.YlOrRd)
    # evaluation
    box(ax, 0.64, 0.05, 0.35, 0.92, "Evaluation", [
        "Participant-held-out cross-validation:",
        "  within-section agreement",
        "  beyond measured composition",
        "  PT-rich spots; ~400 µm regions",
        "  participant-level burden",
        "  reliability ceiling of each score",
        "Which context helps?:",
        "  between spots vs within the spot",
        "  anatomical vs random, compartment,",
        "  window hyperedges",
        "Unchanged models at a second centre:",
        "  slide-level appearance shift",
        "Confidence:",
        "  ensemble disagreement, image novelty",
        "  conformal 90% intervals",
    ], "#f7f2fa", "#6a3d9a", step=0.051)
    arrow(ax, 0.23, 0.75, 0.27, 0.75)
    arrow(ax, 0.23, 0.70, 0.27, 0.30)
    arrow(ax, 0.23, 0.26, 0.27, 0.26)
    arrow(ax, 0.60, 0.75, 0.64, 0.75)
    arrow(ax, 0.60, 0.26, 0.64, 0.26)
    save(fig, "Fig1", {"numbers": pd.DataFrame([nums])}, check=False)


if __name__ == "__main__":
    main()
