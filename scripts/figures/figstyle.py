"""Shared figure style, panel-label alignment and overlap checks for all figures.

Figure scripts only read tables under results/ and write the panel plus its plotting input
under figures/source_data/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
SRC = FIG / "source_data"
SRC.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
W1, W2 = 85 * MM, 170 * MM          # single and double column widths

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 400, "figure.dpi": 150,
    "legend.frameon": False,
})

# colour-blind-safe palette (Okabe-Ito based)
PAL = {"composition": "#999999", "image": "#56B4E9", "pairwise": "#E69F00", "group": "#009E73",
       "anatomy": "#D55E00", "null": "#CC79A7", "ref": "#0072B2", "dark": "#333333"}
MODEL_LABEL = {
    "gbm_measured_comp": "Measured composition (reference)",
    "ridge_image": "Linear probe", "gbm_comp_morph": "Image composition + morphology",
    "mlp": "Spot only (MLP)", "deepsets": "Neighbourhood set", "tok_abmil_ctx": "NephroSet", "gcn": "Lattice graph (GCN)",
    "gat": "Lattice graph (GAT)", "clique_gcn": "Clique-expanded units (GCN)",
    "hgnn_ring": "Local hypergraph (HGNN)", "hgnn_anat": "Unit hypergraph (HGNN)",
    "set_ring": "Local set hypergraph", "set_anat": "Anatomical set hypergraph",
    "set_anat_rewired": "Rewired units", "set_anat_noroles": "Without roles",
    "set_anat_nogeom": "Without geometry",
    "tok_mean": "Token mean", "tok_abmil": "Token attention", "tok_gat": "Token graph (GAT)",
    "tok_hyper_window": "Token hypergraph, windows", "tok_hyper_rolebag": "Token hypergraph, compartments",
    "tok_hyper_rewired": "Token hypergraph, random", "tok_hyper_profile": "Token hypergraph, profiles",
    "tok_hyper_anat": "Anatomical token hypergraph", "istar": "iStar", "bleep": "BLEEP", "hist2st": "Hist2ST", "tok_hyper_anat_ctx": "Anatomical token hypergraph + set",
    "knn_encoder": "Retrieval, same encoder", "tok_pred_mean": "Token-level prediction, same encoder",
}


def save(fig, name: str, data: dict[str, pd.DataFrame] | None = None, check: bool = True) -> None:
    if data:
        for k, df in data.items():
            df.to_csv(SRC / f"{name}_{k}.tsv", sep="\t", index=False)
    align_panel_labels(fig)
    if check:
        rep = overlap_report(fig)
        if rep:
            print(f"[{name}] overlap issues:", *rep, sep="\n  ", file=sys.stderr)
            (FIG / f"{name}_overlap.txt").write_text("\n".join(rep))
        else:
            (FIG / f"{name}_overlap.txt").unlink(missing_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def label(ax, letter: str) -> None:
    """Register a panel letter; positions are set together by align_panel_labels."""
    ax._panel_letter = letter


def align_panel_labels(fig, pad_mm: float = 1.5) -> None:
    """Letters of one row share a baseline above the row's tallest content; letters of one
    column share the left edge of the column's leftmost content."""
    axes = [a for a in fig.axes if getattr(a, "_panel_letter", None)]
    if not axes:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = {a: inv.transform(a.get_tightbbox(r).get_points()) for a in axes}
    pad = pad_mm * MM / fig.get_size_inches()[1]
    rows: list[list] = []
    for a in sorted(axes, key=lambda a: -boxes[a][1, 1]):
        top = boxes[a][1, 1]
        for row in rows:
            if abs(np.mean([boxes[b][1, 1] for b in row]) - top) < 0.2:
                row.append(a)
                break
        else:
            rows.append([a])
    cols: list[list] = []
    for a in sorted(axes, key=lambda a: boxes[a][0, 0]):
        left = boxes[a][0, 0]
        for col in cols:
            if abs(np.mean([boxes[b][0, 0] for b in col]) - left) < 0.06:
                col.append(a)
                break
        else:
            cols.append([a])
    ypos = {a: max(boxes[b][1, 1] for b in row) + pad for row in rows for a in row}
    xpos = {a: min(boxes[b][0, 0] for b in col) for col in cols for a in col}
    first = [min(row, key=lambda a: boxes[a][0, 0]) for row in rows]
    if len(first) > 1:  # the first letter of every row sits on the same left edge
        x0 = min(xpos[a] for a in first)
        for a in first:
            xpos[a] = x0
    for a in axes:
        for t in list(fig.texts):
            if getattr(t, "_panel_of", None) is a:
                t.remove()
        t = fig.text(xpos[a], ypos[a], a._panel_letter, fontsize=9, fontweight="bold", ha="left", va="bottom")
        t._panel_of = a


def overlap_report(fig) -> list[str]:
    """Text-text, text-data and legend-data overlaps (bounding boxes in display space)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    issues = []
    texts = []
    for ax in fig.axes:
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        ticks = [t for t in ax.get_xticklabels() if x0 - 1e-9 <= t.get_position()[0] <= x1 + 1e-9] + \
                [t for t in ax.get_yticklabels() if y0 - 1e-9 <= t.get_position()[1] <= y1 + 1e-9]
        for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label] + ticks:
            if t.get_visible() and t.get_text().strip():
                texts.append((ax, t, t.get_window_extent(r)))
    texts += [(None, t, t.get_window_extent(r)) for t in fig.texts if t.get_text().strip()]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            bi, bj = texts[i][2], texts[j][2]
            if bi.overlaps(bj) and bi.width > 0 and bj.width > 0:
                inter = min(bi.x1, bj.x1) - max(bi.x0, bj.x0), min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                if inter[0] > 1 and inter[1] > 1:
                    issues.append(f"text '{texts[i][1].get_text()[:25]}' overlaps text '{texts[j][1].get_text()[:25]}'")
    for ax in fig.axes:
        pts = []      # display coordinates of every data point / line vertex
        boxes = []    # bars and other patches
        for ln in ax.lines:
            if ln.get_visible() and not getattr(ln, "_is_reference", False):
                xy = ln.get_xydata()
                if len(xy):
                    pts.append(ax.transData.transform(xy))
        for col in ax.collections:
            if not col.get_visible() or getattr(col, "_is_reference", False):
                continue
            off = col.get_offsets()
            if len(off):
                pts.append(col.get_offset_transform().transform(off))
        for pa in ax.patches:
            if pa.get_visible() and not getattr(pa, "_is_reference", False):
                bb = pa.get_window_extent(r)
                if np.isfinite(bb.x0) and bb.width * bb.height > 0 and \
                        not (bb.width > 0.8 * ax.bbox.width and bb.height > 0.8 * ax.bbox.height):
                    boxes.append(bb)
        P = np.concatenate(pts) if pts else np.zeros((0, 2))
        anns = [t for t in ax.texts if t.get_visible() and t.get_text().strip()
                and not getattr(t, "_allow_on_data", False)]
        leg = ax.get_legend()
        for t in anns + ([leg] if leg else []):
            tb = t.get_window_extent(r)
            hit = len(P) and np.any((P[:, 0] > tb.x0) & (P[:, 0] < tb.x1) & (P[:, 1] > tb.y0) & (P[:, 1] < tb.y1))
            hit = hit or any(tb.overlaps(b) for b in boxes)
            if hit:
                what = "legend" if t is leg else f"text '{t.get_text()[:25]}'"
                issues.append(f"{what} overlaps data in axis '{ax.get_title()[:25]}'")
    return sorted(set(issues))
