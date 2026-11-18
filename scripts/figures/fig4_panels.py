"""Panels of Figure 4 drawn as bars, schematics and a bubble chart (helpers for Fig4_context.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

from figstyle import PAL, ROOT

R = ROOT / "results"
PROGS = [("pt_injury", "PT injury", PAL["anatomy"]), ("inflammation", "Inflammation", PAL["group"]),
         ("pt_transport", "PT transport", PAL["ref"])]
STEPS = [("deepsets", "+ Neighbour-\nhood set", "set"), ("tok_mean", "Token\nmean", "mean"),
         ("tok_abmil", "Token\nattention", "att"), ("tok_abmil_ctx", "NephroSet", "nep")]
HEAT = np.exp(-((np.mgrid[0:6, 0:6][0] - 2.5) ** 2 + (np.mgrid[0:6, 0:6][1] - 2.5) ** 2) / 3.0)


def _contrast(d, a, b, k):
    c = pd.read_csv(R / d / "contrasts.tsv", sep="\t")
    return c[(c.a == a) & (c.b == b) & (c.programme == k) & (c.metric == "rho")].iloc[0]


def _icon(ax, x, kind, size=0.105, y=-0.29):
    """Mini schematic of a model below the axis, centred at data x (own square inset axes)."""
    x0, x1 = ax.get_xlim()
    fx = (x - x0) / (x1 - x0)
    w = ax.get_window_extent()
    hs = size * w.width / w.height
    ia = ax.inset_axes([fx - size / 2, y - hs / 2, size, hs])
    ia.set_xlim(-1, 1)
    ia.set_ylim(-1, 1)
    ia.set_aspect("equal")
    ia.axis("off")
    if kind == "set":
        for k, a in enumerate([None] + list(np.arange(6) * np.pi / 3)):
            cx, cy = (0, 0) if a is None else (0.62 * np.cos(a), 0.62 * np.sin(a))
            ia.add_patch(Circle((cx, cy), 0.3, fc="#C0602A" if a is None else "#9EC9E2", ec="white", lw=0.4))
        return
    n, half = 6, (0.62 if kind == "nep" else 0.9)
    c = 2 * half / n
    for r_ in range(n):
        for q in range(n):
            al = 0.45 if kind == "mean" else 0.15 + 0.8 * HEAT[r_, q]
            ia.add_patch(Rectangle((-half + q * c, -half + r_ * c), c, c, fc="#E8590C", alpha=al, ec="white", lw=0.25))
    if kind == "nep":
        for a in np.arange(6) * np.pi / 3 + np.pi / 6:
            ia.add_patch(Circle((0.86 * np.cos(a), 0.86 * np.sin(a)), 0.13, fc="#9EC9E2", ec="white", lw=0.3))


def ladder_panel(ax):
    """Gain of each component over the spot-only network, three programmes, with model schematics."""
    rows = []
    bw = 0.25
    for i, (m, lab, kind) in enumerate(STEPS):
        for j, (k, _, col) in enumerate(PROGS):
            q = _contrast("16_evaluate_cv", m, "mlp", k)
            x = i + (j - 1) * bw
            ax.bar(x, q["diff"], bw * 0.92, color=col, alpha=1.0 if m == "tok_abmil_ctx" else 0.78,
                   ec="k" if m == "tok_abmil_ctx" else "none", lw=0.5)
            ax.plot([x, x], [q.lo, q.hi], color="0.15", lw=0.6)
            rows.append(dict(model=m, programme=k, diff=q["diff"], lo=q.lo, hi=q.hi, frac_positive=q.frac_positive))
    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels([lab for _, lab, _ in STEPS], fontsize=5.6)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.axhline(0, color="0.4", lw=0.5)
    ax.set_xlim(-0.55, len(STEPS) - 0.45)
    for i, (_, _, kind) in enumerate(STEPS):
        _icon(ax, i, kind)
    ax.set_ylim(-0.004, 0.052)
    ax.set_ylabel("Δ ρ vs spot-only network\n(95% CI, 201 participants)")
    ax.set_title("What each component adds", pad=3)
    for (_, lab, col), y in zip(PROGS, (0.049, 0.0445, 0.040)):
        ax.add_patch(Rectangle((-0.48, y - 0.0012), 0.09, 0.0024, color=col))
        ax.text(-0.35, y, lab, fontsize=5.4, va="center")
    return pd.DataFrame(rows)


SCALES = [("deepsets", "mlp", "Between\nspots"), ("tok_abmil", "mlp", "Within the spot\n(8 µm tokens)"),
          ("tok_hyper_anat", "tok_abmil", "Anatomical\nhyperedges")]


def scale_panel(ax):
    """Gain from context at each scale, development cohort and second centre."""
    rows = []
    bw = 0.19
    series = [("16_evaluate_cv", "KPMP", "pt_injury"), ("16_evaluate_cv", "KPMP", "pt_transport"),
              ("18_evaluate_external", "Second centre", "pt_injury"),
              ("18_evaluate_external", "Second centre", "pt_transport")]
    for i, (a, b, _) in enumerate(SCALES):
        for j, (d, coh, k) in enumerate(series):
            q = _contrast(d, a, b, k)
            col = dict((p, c) for p, _, c in PROGS)[k]
            x = i + (j - 1.5) * bw
            ext = coh != "KPMP"
            ax.bar(x, q["diff"], bw * 0.9, color="white" if ext else col, ec=col, lw=0.8, hatch="////" if ext else None)
            ax.plot([x, x], [q.lo, q.hi], color="0.15", lw=0.6)
            rows.append(dict(cohort=coh, comparison=f"{a} vs {b}", programme=k, diff=q["diff"], lo=q.lo, hi=q.hi))
    ax.axhline(0, color="0.4", lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([t for *_, t in SCALES], fontsize=5.6)
    ax.set_ylabel("Δ ρ (95% CI)")
    ax.set_title("Gain from context at each scale", pad=3)
    hl = [Patch(fc=PAL["anatomy"], label="PT injury"), Patch(fc=PAL["ref"], label="PT transport"),
          Patch(fc="0.55", label="KPMP (held out)"), Patch(fc="white", ec="0.4", hatch="////", label="Second centre")]
    ax.legend(handles=hl, loc="upper left", fontsize=5.2, ncol=2, frameon=False, handlelength=1.2,
              columnspacing=0.8)
    lo = min(r["lo"] for r in rows)
    hi = max(r["hi"] for r in rows)
    ax.set_ylim(lo - 0.01, hi + 0.08)
    return pd.DataFrame(rows)


REG = [("disc", "Inside the\ncapture spot (32)", PAL["anatomy"]), ("ring", "Around it\n(164)", "#F2B880"),
       ("outer32", "Outermost\n(32)", PAL["ref"])]


def region_panel(ax, sch):
    """Token map of the crop and the gain from tokens read in each region."""
    r, c = np.divmod(np.arange(196), 14)
    d = np.hypot((r - 6.5) * 8, (c - 6.5) * 8)
    outer = d >= np.sort(d)[-32]
    for i in range(196):
        col = REG[0][2] if d[i] <= 27.5 else (REG[2][2] if outer[i] else REG[1][2])
        sch.add_patch(Rectangle((c[i] * 8 - 56, 56 - (r[i] + 1) * 8), 7.4, 7.4, color=col, lw=0))
    sch.add_patch(Circle((0, 0), 27.5, fill=False, lw=0.9, ec="k"))
    sch.set_xlim(-57, 57)
    sch.set_ylim(-57, 57)
    sch.set_aspect("equal")
    sch.axis("off")
    sch.set_title("112 µm crop, 14 × 14 tokens", fontsize=5.6, pad=2)
    rows = []
    y = np.arange(len(REG))[::-1]
    for (reg, lab, col), yy in zip(REG, y):
        for j, (pref, alpha, hatch) in enumerate([("tok_mean", 0.45, None), ("tok_abmil", 1.0, None)]):
            m = f"{pref}_{reg}" if not (pref == "tok_mean" and reg == "outer32") else None
            if m is None:
                continue
            q = _contrast("16_evaluate_cv", m, "mlp", "pt_injury")
            yb = yy + (0.18 if j == 0 else -0.18)
            ax.barh(yb, q["diff"], 0.34, color=col, alpha=alpha)
            ax.plot([q.lo, q.hi], [yb, yb], color="0.15", lw=0.6)
            rows.append(dict(region=reg, summary=pref, diff=q["diff"], lo=q.lo, hi=q.hi))
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in REG], fontsize=5.6)
    ax.axvline(0, color="0.4", lw=0.5)
    ax.set_xlabel("Δ ρ for PT injury vs spot only")
    ax.set_title("Where the gain is read", pad=3)
    ax.legend(handles=[Patch(fc="0.5", alpha=0.45, label="Token mean"), Patch(fc="0.5", label="Token attention")],
              loc="lower right", fontsize=5.2, frameon=False, handlelength=1.2)
    return pd.DataFrame(rows)


BUBBLE = [("mlp", "Spot only"), ("deepsets", "Neighbourhood set"), ("tok_mean", "Token mean"),
          ("tok_abmil", "Token attention"), ("tok_gat", "Token graph"), ("tok_hyper_anat", "Anatomical token\nhypergraph"),
          ("tok_hyper_anat_ctx", "Anatomical token\nhypergraph + set"), ("tok_abmil_ctx", "NephroSet")]
BOFF = {"mlp": (8, 0, "left"), "deepsets": (8, 0, "left"), "tok_mean": (-8, 3, "right"),
        "tok_abmil_ctx": (-9, 8, "right")}
LEAD = {"tok_abmil": (2.6, 0.2505), "tok_hyper_anat": (9.5, 0.2415), "tok_hyper_anat_ctx": (1.3, 0.2735),
        "tok_gat": (13.0, 0.2700)}  # label positions (data units) joined to their bubbles by a leader line


def bubble_panel(ax):
    """Injury agreement against training compute; bubble area, inference time per section."""
    cost = pd.read_csv(R / "21b_model_cost" / "model_cost.tsv", sep="\t").set_index("model")
    summ = pd.read_csv(R / "16_evaluate_cv" / "summary.tsv", sep="\t").set_index(["model", "programme", "metric"])
    rows = []
    for m, lab in BUBBLE:
        x, ms = cost.loc[m, "train_min_per_model"], cost.loc[m, "inference_s_per_section"] * 1000
        y = summ.loc[(m, "pt_injury", "rho"), "mean"]
        nep = m == "tok_abmil_ctx"
        fam = PAL["anatomy"] if nep else ("#5B5B5B" if m.startswith("tok") else "#A9A9A9")
        ax.scatter(x, y, s=12 + 7 * ms, color=fam, alpha=0.9 if nep else 0.55, ec="k" if nep else "white", lw=0.6,
                   zorder=4 if nep else 3)
        if m in LEAD:
            ax.annotate(lab, (x, y), xytext=LEAD[m],
                        textcoords="data", fontsize=5.2, ha="left" if LEAD[m][0] > x else "right", va="center",
                        color="0.2", arrowprops=dict(arrowstyle="-", lw=0.4, color="0.45", shrinkA=1, shrinkB=3))
        else:
            dx, dy, ha = BOFF[m]
            ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=5.2, ha=ha, va="center",
                        color=fam if nep else "0.2", fontweight="bold" if nep else "normal")
        rows.append(dict(model=m, train_min=x, inference_ms=ms, rho=y, parameters_m=cost.loc[m, "parameters"] / 1e6))
    ax.set_xscale("log")
    ax.set_xlim(0.04, 60)
    ax.set_ylim(0.212, 0.278)
    ax.set_xlabel("Estimated training compute (min per model)")
    ax.set_ylabel("PT injury ρ (held out)")
    ax.set_title("Accuracy and cost", pad=3)
    for ms, xx in ((1, 9.0), (10, 17.0), (30, 32.0)):
        ax.scatter(xx, 0.2185, s=12 + 7 * ms, color="0.75", ec="white", lw=0.5)
        ax.text(xx, 0.2135, f"{ms}", fontsize=4.8, ha="center", va="center")
    ax.text(17.0, 0.2245, "inference, ms per section", fontsize=4.8, ha="center")
    return pd.DataFrame(rows)
