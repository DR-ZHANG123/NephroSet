"""Figure 1: study design, drawn in code. Cohort numbers are read from result tables."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, RegularPolygon

from figstyle import PAL, ROOT, W2, label, save

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


def node(ax, x, y, w, h, text, fc, ec="0.45", ls="-", fs=5.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.015", fc=fc, ec=ec, lw=0.7,
                                ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.25)


def link(ax, x0, y0, x1, y1, ls="-", text=None, tx=0, ty=0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.3", ls=ls,
                                 shrinkA=0, shrinkB=0))
    if text:
        ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text, ha="center", va="center", fontsize=5.2, color="0.25")


def architecture(ax, ar):
    """NephroSet: pretrained encoder (grey), trained modules (orange), training-only output (dashed)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    pre, trn, io = "#e6e6e6", "#fbe3d1", "#eef4fa"
    ax.text(0.5, 0.97, "NephroSet", ha="center", va="top", fontsize=7.2, fontweight="bold")
    # input image: 448 um crop, 112 um crop, 55 um capture spot
    cx, cy, s = 0.075, 0.66, 0.05
    ax.add_patch(plt.Rectangle((cx - s, cy - s * ar), 2 * s, 2 * s * ar, fc="#f2d7e3", ec="0.5", lw=0.6))
    ax.add_patch(plt.Rectangle((cx - s / 4, cy - s * ar / 4), s / 2, s * ar / 2, fc="#e4a9c3", ec="0.3", lw=0.6))
    ax.add_patch(Ellipse((cx, cy), s / 4, s * ar / 4, fc="none", ec="k", lw=0.7))
    ax.text(cx, cy - s * ar - 0.03, "H&E around spot i\n448 µm and 112 µm crops", ha="center", va="top", fontsize=5.4)
    # neighbouring spots
    nx, ny, r = 0.075, 0.17, 0.016
    for k in range(7):
        a = 2 * np.pi * (k - 1) / 6
        px, py = (nx, ny) if k == 0 else (nx + 2.1 * r * np.cos(a), ny + 2.1 * r * ar * np.sin(a))
        ax.add_patch(Ellipse((px, py), 2 * r, 2 * r * ar, fc="#c0602a" if k == 0 else "#9ecae1", ec="white", lw=0.4))
    ax.text(nx, ny - 3.4 * r * ar, "Six neighbouring spots j", ha="center", va="top", fontsize=5.4)
    node(ax, 0.17, 0.40, 0.13, 0.48, "Pathology\nencoder\n(Phikon-v2)\n\npretrained,\nnot trained", pre)
    node(ax, 0.34, 0.66, 0.13, 0.17, "196 tokens\n14 × 14 of 8 µm", io)
    node(ax, 0.34, 0.40, 0.13, 0.19, "Crop embeddings,\nmorphology,\nimage composition", io)
    node(ax, 0.51, 0.66, 0.14, 0.17, "Token stem +\ngated attention → $v$", trn)
    node(ax, 0.51, 0.42, 0.14, 0.15, "Spot stem → $h^0$", trn)
    node(ax, 0.51, 0.08, 0.14, 0.20, "Embed $h^0_j$ with\nposition $p_j - p_i$;\nmean and max", trn)
    node(ax, 0.69, 0.30, 0.12, 0.42, "Concatenate\n[$h^0$; $v$; mean; max]\n\ntwo-layer\nnetwork", trn)
    node(ax, 0.85, 0.52, 0.14, 0.27, "PT injury\nInflammation\nPT transport", io, bold=True)
    node(ax, 0.85, 0.20, 0.14, 0.22, "Composition\n(9 classes)\nauxiliary output", "white", ls=(0, (3, 2)))
    link(ax, cx + s, cy, 0.17, cy)
    link(ax, 0.30, 0.745, 0.34, 0.745)
    link(ax, 0.30, 0.495, 0.34, 0.495)
    link(ax, 0.47, 0.745, 0.51, 0.745)
    link(ax, 0.47, 0.495, 0.51, 0.495)
    link(ax, nx + 3.2 * r, ny, 0.51, ny)
    link(ax, 0.58, 0.42, 0.58, 0.28, text="shared stem", tx=0.045)
    link(ax, 0.65, 0.745, 0.69, 0.62)
    link(ax, 0.65, 0.495, 0.69, 0.51)
    link(ax, 0.65, 0.18, 0.69, 0.38)
    link(ax, 0.81, 0.62, 0.85, 0.655)
    link(ax, 0.81, 0.40, 0.85, 0.31, ls=(0, (3, 2)))
    # key
    ky = 0.015
    for i, (fc, ls, t) in enumerate([(pre, "-", "pretrained, not trained"), (trn, "-", "trained"),
                                      ("white", (0, (3, 2)), "training only (supervised by\nVisium composition)")]):
        x = 0.17 + i * 0.2
        ax.add_patch(FancyBboxPatch((x, ky), 0.025, 0.045, boxstyle="round,pad=0.002", fc=fc, ec="0.45", lw=0.6, ls=ls))
        ax.text(x + 0.032, ky + 0.0225, t, ha="left", va="center", fontsize=5.2)
    ax.text(0.92, 0.88, "Supervision: same-section\nVisium programme scores", ha="center", va="center", fontsize=5.2,
            color="0.25")


def main() -> None:
    atlas = pd.read_csv(R / "01_atlas_extract" / "atlas_samples.tsv", sep="\t")
    tg = pd.read_csv(R / "02_atlas_features" / "targets_per_section.tsv", sep="\t")
    kp = pd.read_csv(R / "11b_kpmp_targets" / "spot_summary.tsv", sep="\t")
    kp = kp[kp.status == "ok"]
    ex = pd.read_csv(R / "13_gse211785_prep" / "sections.tsv", sep="\t")
    nums = dict(atlas_sections=len(atlas), atlas_donors=atlas.donor.nunique(), atlas_cells=int(atlas.n_cells.sum()),
                pt_targets=int(tg.n_targets.sum()), kpmp_participants=len(kp), kpmp_spots=int(kp.n_spots.sum()),
                ext_donors=len(ex), ext_spots=int(ex.n_spots.sum()))

    H = 6.3
    fig = plt.figure(figsize=(W2, H))
    global AR
    AR = W2 / 3.3
    arch = fig.add_axes([0, 3.62 / H, 1, 2.55 / H])
    architecture(arch, W2 / 2.55)
    label(arch, "a")
    ax = fig.add_axes([0, 0, 1, 3.3 / H])
    label(ax, "b")
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
    box(ax, 0.27, 0.53, 0.33, 0.44, "1  Design rationale (single cell)", [
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
    box(ax, 0.27, 0.05, 0.33, 0.42, "2  NephroSet", [
        "H&E crops 112 µm and 448 µm per spot",
        "Phikon-v2 pathology encoder",
        "Fine scale: 8 µm tokens, gated attention",
        "Context: neighbouring spots as a set",
        "No segmentation, no anatomical prior",
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
        "  split-half reliability reference",
        "Design tests (same encoder):",
        "  context between spots",
        "  token mean, graph, hypergraph",
        "  capture area vs crop edge",
        "Published methods:",
        "  iStar, BLEEP, Hist2ST",
        "Second centre:",
        "  model applied unchanged",
        "Confidence:",
        "  ensemble disagreement, image novelty",
        "  calibrated 90% intervals",
    ], "#f7f2fa", "#6a3d9a", step=0.048)
    arrow(ax, 0.23, 0.75, 0.27, 0.75)
    arrow(ax, 0.23, 0.70, 0.27, 0.30)
    arrow(ax, 0.23, 0.26, 0.27, 0.26)
    arrow(ax, 0.60, 0.75, 0.64, 0.75)
    arrow(ax, 0.60, 0.26, 0.64, 0.26)
    save(fig, "Fig1", {"numbers": pd.DataFrame([nums])}, check=False)


if __name__ == "__main__":
    main()
