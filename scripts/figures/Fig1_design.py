"""Figure 1: the NephroSet framework (a) and the study workflow (b), drawn as vector graphics.

Pictograms are the traced icons in figures/fig1_icons/ (scripts/figures/fig1_icons.py); every label, arrow,
the token grid, the attention map and the neighbour lattice are drawn here. Cohort numbers are read from
result tables. Coordinates are millimetres.
"""
from __future__ import annotations

import colorsys
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as MPath
from svgpath2mpl import parse_path

from figstyle import ROOT, save

R = ROOT / "results"
ICONS = ROOT / "figures" / "fig1_icons"
W, H, Y0 = 180.0, 150.0, 12.0
INK, SOFT = "#222222", "#555555"
GREY, ORANGE, BLUE, LINE = "#E6E6E6", "#F6D9C4", "#E3EEF8", "#6E6E6E"
TEXTS, BOXES = [], []
_cache: dict[str, tuple] = {}


# ------------------------------------------------------------------ icons
def _load(name: str):
    if name not in _cache:
        root = ET.parse(ICONS / f"{name}.svg").getroot()
        cw, ch = float(root.get("width")), float(root.get("height"))
        items = []
        for el in root.iter("{http://www.w3.org/2000/svg}path"):
            try:
                p = parse_path(el.get("d"))
            except ValueError:
                continue
            if len(p.vertices) < 3:
                continue
            m = re.search(r"translate\(([-\d.]+)[ ,]+([-\d.]+)\)", el.get("transform", ""))
            v = p.vertices + (np.array([float(m.group(1)), float(m.group(2))]) if m else 0.0)
            fill = el.get("fill", "#000000")
            rgb = [int(fill[i:i + 2], 16) for i in (1, 3, 5)]
            edge = v[:, 0].min() <= 2 or v[:, 1].min() <= 2 or v[:, 0].max() >= cw - 2 or v[:, 1].max() >= ch - 2
            if min(rgb) > 225 and edge:  # white page around the pictogram
                continue
            items.append((v, p.codes, fill))
        allv = np.concatenate([v for v, _, _ in items])
        _cache[name] = (items, allv.min(0), allv.max(0))
    return _cache[name]


def hue(fill: str, h: float | None) -> str:
    if h is None:
        return fill
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    _, l, s = colorsys.rgb_to_hls(r, g, b)
    if s < 0.15:
        return fill
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def icon(ax, name, x, y, w, h, z=3, shift=None, clip=None):
    """Draw a traced pictogram fitted and centred in the box (x, y, w, h); returns the box used."""
    items, lo, hi = _load(name)
    span = hi - lo
    k = min(w / span[0], h / span[1])
    ox, oy = x + (w - span[0] * k) / 2, y + (h - span[1] * k) / 2
    for v, codes, fill in items:
        xy = np.column_stack([ox + (v[:, 0] - lo[0]) * k, oy + (hi[1] - v[:, 1]) * k])
        p = PathPatch(MPath(xy, codes), facecolor=hue(fill, shift), edgecolor="none", lw=0, zorder=z)
        ax.add_patch(p)
        if clip is not None:
            p.set_clip_path(clip)
    box = (ox, oy, ox + span[0] * k, oy + span[1] * k)
    BOXES.append((name,) + box)
    return box


# ------------------------------------------------------------------ primitives
def text(ax, x, y, s, size=6.2, weight="normal", color=INK, ha="center", va="center", style="normal", track=True):
    t = ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, style=style, zorder=8,
                linespacing=1.15)
    if track:
        TEXTS.append(t)
    return t


def frame(ax, x0, y0, x1, y1, fc, ec="#BDBDBD", lw=0.6, ls="-", r=2.0, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                                ec=ec, lw=lw, ls=ls, zorder=z))


def arrow(ax, x0, y0, x1, y1, color=LINE, lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw, color=color,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=6))


def vector(ax, x, y, n, color, cell=1.6, vertical=False):
    """A feature vector drawn as a strip of shaded cells; returns its far edge."""
    rng = np.random.default_rng(n * 7 + len(color))
    for i in range(n):
        a = 0.35 + 0.6 * rng.random()
        xy = (x, y + i * cell) if vertical else (x + i * cell, y)
        ax.add_patch(Rectangle(xy, cell, cell, fc=color, alpha=a, ec="white", lw=0.3, zorder=5))
    return (y + n * cell) if vertical else (x + n * cell)


def lattice(ax, cx, cy, r, centre, ring, gap=2.25):
    """The centre spot and its six Visium neighbours."""
    pts = [(cx, cy)] + [(cx + gap * r * np.cos(a), cy + gap * r * np.sin(a)) for a in np.arange(6) * np.pi / 3]
    for i, (x, y) in enumerate(pts):
        ax.add_patch(Circle((x, y), r, fc=centre if i == 0 else ring, ec="white", lw=0.5, zorder=5))
    return pts


# ------------------------------------------------------------------ panel a
def token_grid(ax, x, y, s):
    """112 µm crop with its 14 x 14 tokens, the 55 µm capture spot and a schematic attention map."""
    clip = Rectangle((x, y), s, s, transform=ax.transData)
    icon(ax, "histology", x - s * 0.02, y - s * 0.02, s * 1.04, s * 1.04, z=3, clip=clip)
    BOXES.pop()
    n, c = 14, s / 14
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    att = np.exp(-((xx - 7) ** 2 + (yy - 7) ** 2) / 14.0) * (0.6 + 0.4 * np.random.default_rng(3).random((n, n)))
    att /= att.max()
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x + j * c, y + i * c), c, c, fc="#E8590C", alpha=0.5 * att[i, j] ** 1.5,
                                   ec="white", lw=0.25, zorder=4))
    ax.add_patch(Circle((x + s / 2, y + s / 2), s * 55 / 112 / 2, fc="none", ec=INK, lw=0.8, ls=(0, (2, 1)), zorder=5))
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=INK, lw=0.6, zorder=5))


def panel_a(ax, nums):
    top = 146.0
    frame(ax, 1.0, 50.0, 179.0, top + 1.5, "white", ec="#CFCFCF")
    text(ax, 90, top - 3.0, "NephroSet", size=8, weight="bold")
    # input
    icon(ax, "slide", 4, 112, 30, 18)
    text(ax, 19, 109.5, "H&E section")
    hx, hy, hs = 8.5, 78.5, 22
    icon(ax, "histology", hx, hy, hs, hs)
    ax.add_patch(Circle((hx + hs / 2, hy + hs / 2), hs * 55 / 112 / 2, fc="none", ec=INK, lw=0.8, ls=(0, (2, 1)),
                        zorder=6))
    arrow(ax, 19, 106.8, 19, 101.4, lw=0.7)
    text(ax, 19.5, 74.5, "Crops of 112 and 448 µm\naround each spot")
    # pretrained encoder
    frame(ax, 39, 72, 61, 124, GREY, ec="#9E9E9E")
    icon(ax, "encoder", 41.5, 93, 17, 22)
    text(ax, 50, 88.5, "Pathology\nencoder", weight="bold")
    text(ax, 50, 79.5, "pretrained,\nnot trained", size=5.6, color=SOFT)
    arrow(ax, 31.5, 89.5, 39, 89.5)
    # branch 1: tokens + gated attention
    frame(ax, 67, 106, 121, 141, ORANGE, ec="#D9A47F")
    token_grid(ax, 69.5, 109, 26)
    text(ax, 108.5, 132.5, "196 tokens\nof 8 µm", weight="bold")
    text(ax, 108.5, 123.0, "gated attention\nselects tokens", size=5.6, color=SOFT)
    vend = vector(ax, 101.5, 112.0, 8, "#E8590C")
    text(ax, vend + 1.6, 112.8, r"$v$", size=7, ha="left")
    arrow(ax, 61, 112, 67, 120)
    # branch 2: spot embedding
    frame(ax, 67, 86, 121, 101, ORANGE, ec="#D9A47F")
    text(ax, 69.5, 96.8, "Spot embedding", weight="bold", ha="left")
    text(ax, 69.5, 91.2, "crop embeddings, morphology,\nimage composition", size=5.6, color=SOFT, ha="left")
    vend = vector(ax, 102.0, 92.2, 8, "#C0602A")
    text(ax, vend + 1.2, 93.0, r"$h^{0}$", size=7, ha="left")
    arrow(ax, 61, 93.5, 67, 93.5)
    # branch 3: neighbourhood set
    frame(ax, 67, 52.5, 121, 81, ORANGE, ec="#D9A47F")
    text(ax, 69.5, 76.5, "Neighbourhood set", weight="bold", ha="left")
    text(ax, 69.5, 70.6, "each neighbour: embedding\nand relative position", size=5.6, color=SOFT, ha="left")
    pts = lattice(ax, 80.5, 60.5, 2.5, "#C0602A", "#9EC9E2")
    for (px, py) in pts[1:]:
        ax.plot([80.5, px], [60.5, py], color="#7FA9C4", lw=0.5, zorder=4)
    text(ax, 103.5, 63.5, "mean and max\n(order-free)", size=5.6, color=SOFT)
    vector(ax, 96.5, 56.0, 8, "#2B7BBA")
    arrow(ax, 61, 80, 67, 72)
    arrow(ax, 94, 86, 94, 81, lw=0.7)
    text(ax, 95.5, 83.5, "same stem for neighbours", size=5.2, color=SOFT, ha="left")
    BOXES.append(("lattice", 72.0, 52.5, 89.0, 68.5))
    # fusion network
    frame(ax, 127, 72, 147, 124, ORANGE, ec="#D9A47F")
    icon(ax, "network", 129.5, 98, 15, 17)
    text(ax, 137, 90.0, "Fusion\nnetwork", weight="bold")
    text(ax, 137, 79.5, "concatenate,\ntwo layers", size=5.6, color=SOFT)
    arrow(ax, 121, 123.5, 127, 112)
    arrow(ax, 121, 93.5, 127, 93.5)
    arrow(ax, 121, 66.5, 127, 78)
    # outputs
    for k, (lab, h, y) in enumerate([("PT injury", None, 120.0), ("Inflammation", 0.83, 105.0),
                                     ("PT transport", 0.58, 90.0)]):
        icon(ax, "map", 151.5, y, 26, 10.5, shift=h)
        text(ax, 164.5, y + 12.8, lab, size=6.0, weight="bold")
        arrow(ax, 147, 98, 151.3, y + 5.0, lw=0.7)
    text(ax, 164.5, 140.0, "Programme maps", size=6.4, weight="bold")
    frame(ax, 151, 68.5, 177, 83, "white", ec="#9E9E9E", ls=(0, (3, 2)))
    ax.add_patch(Wedge((157.0, 75.8), 3.6, 0, 140, fc="#9EC9E2", ec="white", lw=0.4, zorder=5))
    ax.add_patch(Wedge((157.0, 75.8), 3.6, 140, 250, fc="#F2B880", ec="white", lw=0.4, zorder=5))
    ax.add_patch(Wedge((157.0, 75.8), 3.6, 250, 360, fc="#B9DDB0", ec="white", lw=0.4, zorder=5))
    text(ax, 169.0, 75.8, "Cell-class\ncomposition\n(training only)", size=5.4, color=SOFT)
    arrow(ax, 147, 78, 151, 76, lw=0.7, ls=(0, (3, 2)))
    # supervision and inference
    icon(ax, "rna", 129, 54.5, 11, 9)
    text(ax, 141.5, 59.0, "Training: same-section\nVisium programme scores", size=5.6, ha="left", color=SOFT)
    frame(ax, 4, 54, 35, 67, BLUE, ec="#9DB9D3")
    text(ax, 19.5, 60.5, "At inference: H&E image\nand spot coordinates only", size=5.6)
    # legend
    for i, (fc, ec, lab) in enumerate([(GREY, "#9E9E9E", "pretrained"), (ORANGE, "#D9A47F", "trained")]):
        lx = 4 + i * 20
        frame(ax, lx, 140.5, lx + 4, 143.5, fc, ec=ec, r=0.6)
        text(ax, lx + 5, 142.0, lab, size=5.6, ha="left")


# ------------------------------------------------------------------ panel b
def panel_b(ax, nums):
    frame(ax, 1.0, 13.0, 179.0, 45.5, "white", ec="#CFCFCF")
    steps = [
        ("cells", "Design rationale", f"{nums['pt_targets']:,} PT cells\n{nums['atlas_donors']} donors\n"
                                      "single-cell atlas"),
        ("cohort", "Development", f"{nums['kpmp_participants']} KPMP participants\n{nums['kpmp_spots']:,} "
                                  "spots\nheld-out participants"),
        (None, "Design tests", "same encoder:\nscale, pooling, context\n3 published methods"),
        ("hospital", "Second centre", f"{nums['ext_donors']} donors\n{nums['ext_spots']:,} spots\n"
                                      "model unchanged"),
    ]
    cw = 178.0 / 4
    for i, (ic, head, body) in enumerate(steps):
        x0 = 1.0 + i * cw
        cx = x0 + cw / 2
        if ic:
            icon(ax, ic, x0 + 3.0, 15.5, 17.0, 23.0)
        else:
            bars = [0.0, 0.009, 0.033, 0.036, 0.040]
            cols = ["#9E9E9E", "#9EC9E2", "#F2B880", "#E8590C", "#C0602A"]
            for j, (b, c) in enumerate(zip(bars, cols)):
                ax.add_patch(Rectangle((x0 + 5.0 + j * 3.0, 18.0), 2.3, 2.0 + b * 420, fc=c, ec="none", zorder=4))
            ax.plot([x0 + 4.0, x0 + 20.5], [18.0, 18.0], color=INK, lw=0.6, zorder=5)
        text(ax, x0 + 21.5 + (cw - 23.0) / 2, 37.0, head, size=6.6, weight="bold")
        text(ax, x0 + 21.5 + (cw - 23.0) / 2, 26.0, body, size=5.7, color=SOFT)
        if i < 3:
            arrow(ax, x0 + cw - 1.6, 27.0, x0 + cw + 1.6, 27.0, lw=1.2)


# ------------------------------------------------------------------ checks and output
def overlaps(fig, ax) -> list[str]:
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    boxes = []
    for t in TEXTS:
        bb = t.get_window_extent(rnd)
        (a, b), (c, d) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        boxes.append((t.get_text().replace("\n", " "), a, b, c, d))
    bad = []
    for i, p in enumerate(boxes):
        for q in boxes[i + 1:]:
            if min(p[3], q[3]) - max(p[1], q[1]) > 0.1 and min(p[4], q[4]) - max(p[2], q[2]) > 0.1:
                bad.append(f"text/text: {p[0]!r} x {q[0]!r}")
        for name, x0, y0, x1, y1 in BOXES:
            if min(p[3], x1) - max(p[1], x0) > 0.3 and min(p[4], y1) - max(p[2], y0) > 0.3:
                bad.append(f"text/icon: {p[0]!r} x {name}")
        if p[1] < 1.5 or p[3] > W - 1.5 or p[2] < Y0 + 1.5 or p[4] > H - 0.5:
            bad.append(f"outside the figure: {p[0]!r}")
    return bad


def main() -> None:
    atlas = pd.read_csv(R / "01_atlas_extract" / "atlas_samples.tsv", sep="\t")
    tg = pd.read_csv(R / "02_atlas_features" / "targets_per_section.tsv", sep="\t")
    kp = pd.read_csv(R / "11b_kpmp_targets" / "spot_summary.tsv", sep="\t")
    kp = kp[kp.status == "ok"]
    ex = pd.read_csv(R / "13_gse211785_prep" / "sections.tsv", sep="\t")
    used = atlas[atlas["sample"].isin(tg["sample"])]
    nums = dict(atlas_sections=len(used), atlas_donors=used.donor.nunique(), pt_targets=int(tg.n_targets.sum()),
                kpmp_participants=len(kp), kpmp_spots=int(kp.n_spots.sum()), ext_donors=len(ex),
                ext_spots=int(ex.n_spots.sum()))
    fig = plt.figure(figsize=(W / 25.4, (H - Y0) / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_a(ax, nums)
    panel_b(ax, nums)
    for letter, y in (("a", H - 1.0), ("b", 47.0)):
        ax.text(0.0, y, letter, fontsize=9, fontweight="bold", ha="left", va="top" if letter == "a" else "bottom")
    bad = overlaps(fig, ax)
    for b in bad:
        print("OVERLAP", b, file=sys.stderr)
    save(fig, "Fig1", {"numbers": pd.DataFrame([nums])}, check=False)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
