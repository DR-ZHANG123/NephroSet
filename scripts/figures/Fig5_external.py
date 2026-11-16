"""Figure 5: estimates at a second centre."""
from __future__ import annotations

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figstyle import MODEL_LABEL, PAL, ROOT, W2, label, save

R = ROOT / "results"
PROG_LAB = {"pt_injury": "PT injury", "inflammation": "Inflammation", "pt_transport": "PT transport"}
SHOW = ["gbm_measured_comp", "gbm_comp_morph", "hist2st", "bleep", "istar", "ridge_image", "mlp", "deepsets",
        "tok_abmil_ctx"]
COL = {"gbm_measured_comp": PAL["composition"], "gbm_comp_morph": "#b07aa1", "ridge_image": PAL["image"],
       "mlp": PAL["pairwise"], "deepsets": PAL["group"], "tok_abmil_ctx": PAL["anatomy"],
       "hist2st": "#AA4499", "bleep": "#882255", "istar": "#CC79A7"}


def main() -> None:
    sm = pd.read_csv(R / "18_evaluate_external" / "section_metrics.tsv", sep="\t")
    ceil = pd.read_csv(R / "18b_external_reliability" / "ceiling.tsv", sep="\t")
    regx = pd.read_csv(R / "18c_external_region_coverage" / "region_summary.tsv", sep="\t")
    regk = pd.read_csv(R / "16b_reliability_region" / "summary.tsv", sep="\t")
    spread = pd.read_csv(R / "20_subgroups" / "agreement_vs_spread.tsv", sep="\t")
    ex = pd.read_csv(R / "19_examples" / "external_spots.tsv", sep="\t")
    img = mpimg.imread(R / "19_examples" / "external_he.png")

    fig = plt.figure(figsize=(W2, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.3, left=0.08, right=0.985, top=0.95, bottom=0.09)
    top = gs[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.08)
    bot = gs[1].subgridspec(1, 3, width_ratios=[1.3, 0.8, 1.0], wspace=0.45)
    for c in ["y_pt_injury", "pred_pt_injury"]:
        ex[f"rank_{c}"] = ex[c].rank(pct=True) * 100
    v = (0, 100)
    for i, (col, title) in enumerate([(None, f"H&E (donor {ex.section.iloc[0]})"), ("y_pt_injury", "Measured PT injury"),
                                      ("pred_pt_injury", "Predicted from H&E (NephroSet)")]):
        ax = fig.add_subplot(top[i])
        ax.imshow(img, alpha=1.0 if col is None else 0.35)
        if col is not None:
            sc = ax.scatter(ex.tx, ex.ty, c=ex[f"rank_{col}"], s=1.0, cmap="magma", vmin=v[0], vmax=v[1], lw=0)
        else:
            upx = ex.um_per_thumb_px.iloc[0]
            L = 1000 / upx
            h, w = img.shape[:2]
            ax.plot([w - L - 15, w - 15], [h - 18, h - 18], c="k", lw=1.2)
            ax.text(w - L / 2 - 15, h - 26, "1 mm", ha="center", va="bottom", fontsize=6)
            label(ax, "a")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, pad=2)
        for s in ax.spines.values():
            s.set_visible(False)
    cax = fig.add_subplot(top[3])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Within-section percentile", fontsize=6)
    cax.tick_params(labelsize=5.5)

    ax = fig.add_subplot(bot[0])
    rows = []
    rng = np.random.default_rng(1)
    for i, k in enumerate(PROG_LAB):
        for j, m in enumerate(SHOW):
            q = sm[(sm.model == m) & (sm.programme == k)]
            x = i + (j - (len(SHOW) - 1) / 2) * (0.86 / len(SHOW))
            ax.scatter(x + rng.uniform(-0.02, 0.02, len(q)), q.rho, s=3, color=COL[m], alpha=0.45, lw=0)
            ax.plot([x - 0.042, x + 0.042], [q.rho.mean()] * 2, color=COL[m], lw=1.6)
            rows += [dict(programme=k, model=m, section=s, rho=r) for s, r in zip(q.section, q.rho)]
        c = ceil[ceil.programme == k].rel_spot.iloc[0]
        ln = ax.hlines(c, i - 0.45, i + 0.45, colors="0.3", linestyles=(0, (2, 1.5)), lw=0.8)
        ln._is_reference = True
    ax.axhline(0, c="0.75", lw=0.5)
    ax.lines[-1]._is_reference = True
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(PROG_LAB.values()))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Within-section Spearman ρ\n(13 external donors)")
    ax.set_title("Spot level, second centre", pad=3)
    ax.set_ylim(-0.25, 1.45)
    ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    hl = [Line2D([], [], color=COL[m], lw=1.6, label=MODEL_LABEL[m]) for m in SHOW]
    hl.append(Line2D([], [], ls=(0, (2, 1.5)), color="0.3", lw=0.8, label="Reliability ceiling"))
    ax.legend(handles=hl, loc="upper left", fontsize=5.0, ncol=2, handlelength=1.2, columnspacing=0.6,
              borderaxespad=0.2)
    label(ax, "b")

    ax = fig.add_subplot(bot[1])
    rr = []
    for i, k in enumerate(PROG_LAB):
        for j, (coh, tab, col) in enumerate([("KPMP (held out)", regk, PAL["anatomy"]), ("Second centre", regx, PAL["ref"])]):
            q = tab[(tab.model == "tok_abmil_ctx") & (tab.programme == k) & (tab.metric == "rho_region")].iloc[0]
            x = i + (j - 0.5) * 0.3
            ax.bar(x, q["mean"], width=0.28, color=col, alpha=0.85)
            ax.errorbar(x, q["mean"], yerr=[[q["mean"] - q.lo], [q.hi - q["mean"]]], fmt="none", ecolor="k", lw=0.7)
            rr.append(dict(programme=k, cohort=coh, mean=q["mean"], lo=q.lo, hi=q.hi))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["PT\ninjury", "Inflam-\nmation", "PT\ntransport"])
    ax.set_ylabel("Regional Spearman ρ (~400 µm)")
    ax.set_title("NephroSet, regions", pad=3)
    ax.set_ylim(0, 1.0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PAL["anatomy"], alpha=0.85, label="KPMP (held out)"),
                       plt.Rectangle((0, 0), 1, 1, color=PAL["ref"], alpha=0.85, label="Second centre")],
              loc="upper left", fontsize=5.2, borderaxespad=0.2)
    label(ax, "c")

    ax = fig.add_subplot(bot[2])
    q = spread[spread.programme == "pt_injury"]
    for coh, col, sz in [("KPMP", "0.6", 5), ("External", PAL["ref"], 12)]:
        z = q[q.cohort == coh]
        ax.scatter(z.sd_measured, z.rho, s=sz, color=col, lw=0, label="KPMP (held out)" if coh == "KPMP" else "Second centre",
                   zorder=3 if coh == "External" else 2)
    ax.axhline(0, c="0.75", lw=0.5)
    ax.lines[-1]._is_reference = True
    ax.set_xlabel("Within-section SD of measured PT injury")
    ax.set_ylabel("Within-section Spearman ρ")
    ax.set_title("Agreement and injury spread", pad=3)
    ax.set_ylim(-0.25, 0.85)
    ax.legend(loc="upper left", fontsize=5.2, borderaxespad=0.2)
    label(ax, "d")
    save(fig, "Fig5", {"a_example_spots": ex[["barcode", "tx", "ty", "y_pt_injury", "pred_pt_injury", "rank_y_pt_injury", "rank_pred_pt_injury"]],
                       "b_donors": pd.DataFrame(rows), "b_ceiling": ceil, "c_regions": pd.DataFrame(rr),
                       "d_spread": q})


if __name__ == "__main__":
    main()
