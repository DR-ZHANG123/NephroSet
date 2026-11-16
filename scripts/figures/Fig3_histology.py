"""Figure 3: H&E estimates PT programmes in held-out participants, beyond tissue composition."""
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
CAT = {"Diabetic Kidney Disease": "DKD", "Hypertensive Kidney Disease": "HKD", "Healthy Reference": "Reference",
       "Acute Tubular Injury": "ATI"}


def spot_map(ax, t, col, img, title, vmin, vmax, cmap="magma"):
    ax.imshow(img, alpha=0.35)
    sc = ax.scatter(t.tx, t.ty, c=t[col], s=2.2, cmap=cmap, vmin=vmin, vmax=vmax, lw=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    return sc


def dots(ax, summ, metric, ceiling=None, ylab=""):
    progs = list(PROG_LAB)
    rows = []
    for i, k in enumerate(progs):
        for j, m in enumerate(SHOW):
            q = summ[(summ.model == m) & (summ.programme == k) & (summ.metric == metric)]
            if q.empty:
                continue
            q = q.iloc[0]
            x = i + (j - (len(SHOW) - 1) / 2) * (0.84 / len(SHOW))
            ax.errorbar(x, q["mean"], yerr=[[q["mean"] - q.lo], [q.hi - q["mean"]]], fmt="o", ms=2.8,
                        color=COL[m], lw=0.8)
            rows.append(dict(programme=k, model=m, metric=metric, mean=q["mean"], lo=q.lo, hi=q.hi))
        if ceiling is not None:
            c = ceiling[ceiling.programme == k]
            if not c.empty:
                ln = ax.hlines(c["mean"].iloc[0], i - 0.42, i + 0.42, colors="0.3", linestyles=(0, (2, 1.5)), lw=0.8)
                ln._is_reference = True
    ax.set_xticks(range(len(progs)))
    ax.set_xticklabels([PROG_LAB[k] for k in progs])
    ax.set_ylabel(ylab)
    ax.set_xlim(-0.5, len(progs) - 0.5)
    return rows


def main() -> None:
    summ = pd.read_csv(R / "16_evaluate_cv" / "summary.tsv", sep="\t")
    reg = pd.read_csv(R / "16b_reliability_region" / "summary.tsv", sep="\t")
    pm = pd.read_csv(R / "16_evaluate_cv" / "participant_metrics.tsv", sep="\t")
    idx = pd.read_csv(R / "10_kpmp_index" / "sections.tsv", sep="\t")
    ex = pd.read_csv(R / "19_examples" / "kpmp_spots.tsv", sep="\t")
    img = mpimg.imread(R / "19_examples" / "kpmp_he.png")

    fig = plt.figure(figsize=(W2, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.28, left=0.08, right=0.985, top=0.95, bottom=0.15)
    top = gs[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.08], wspace=0.08)
    bot = gs[1].subgridspec(1, 3, width_ratios=[1.15, 1.0, 0.95], wspace=0.42)
    ax = fig.add_subplot(top[0])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    upx = ex.um_per_thumb_px.iloc[0]
    L = 500 / upx
    h, w = img.shape[:2]
    ax.plot([w - L - 15, w - 15], [h - 20, h - 20], c="k", lw=1.2)
    ax.text(w - L / 2 - 15, h - 28, "500 µm", ha="center", va="bottom", fontsize=6)
    ax.set_title("H&E (held-out participant)", pad=2)
    label(ax, "a")
    for c in ["y_pt_injury", "pred_pt_injury"]:
        ex[f"rank_{c}"] = ex[c].rank(pct=True) * 100
    spot_map(fig.add_subplot(top[1]), ex, "rank_y_pt_injury", img, "Measured PT injury", 0, 100)
    sc = spot_map(fig.add_subplot(top[2]), ex, "rank_pred_pt_injury", img,
                  f"Predicted from H&E ({MODEL_LABEL['tok_abmil_ctx']})", 0, 100)
    cax = fig.add_subplot(top[3])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Within-section percentile", fontsize=6)
    cax.tick_params(labelsize=5.5)

    ceil = reg[(reg.model == "ceiling")]
    ax = fig.add_subplot(bot[0])
    r1 = dots(ax, summ, "rho", ceil[ceil.metric == "rel_spot"], "Within-section Spearman ρ\n(mean over participants)")
    ax.set_title("Spot level", pad=3)
    ax.set_ylim(0, 0.9)
    hl = [Line2D([], [], marker="o", ls="", ms=3, color=COL[m], label=MODEL_LABEL[m]) for m in SHOW]
    hl.append(Line2D([], [], ls=(0, (2, 1.5)), color="0.3", lw=0.8, label="Reliability ceiling"))
    label(ax, "b")
    ax = fig.add_subplot(bot[1])
    r2 = dots(ax, summ, "rho_beyond_comp", None, "Spearman ρ after removing\nmeasured composition")
    ax.set_title("Beyond composition", pad=3)
    ax.set_ylim(0, 0.9)
    fig.legend(handles=hl, loc="lower center", bbox_to_anchor=(0.36, 0.0), fontsize=5.2, handletextpad=0.2, ncol=4, columnspacing=0.9, frameon=False)
    label(ax, "c")
    ax = fig.add_subplot(bot[2])
    q = pm[(pm.model == "tok_abmil_ctx") & (pm.programme == "pt_injury")].merge(
        idx[["participant", "primary_adjudicated_category", "enrollment_category"]].drop_duplicates("participant"),
        on="participant")
    q["group"] = q.primary_adjudicated_category.fillna(q.enrollment_category).map(CAT).fillna("Other")
    gcol = {"DKD": PAL["anatomy"], "HKD": PAL["pairwise"], "Reference": PAL["ref"], "ATI": PAL["group"],
            "Other": "0.7"}
    for g, c in gcol.items():
        z = q[q.group == g]
        ax.scatter(z.mean_y, z.mean_pred, s=7, color=c, lw=0, label=f"{g} ({len(z)})", zorder=3 if g != "Other" else 2)
    r = np.corrcoef(q.mean_y, q.mean_pred)[0, 1]
    ax.set_xlabel("Measured mean PT injury")
    ax.set_ylabel("Predicted mean PT injury")
    ax.set_title(f"Participant burden (r = {r:.2f})", pad=3)
    ax.set_ylim(q.mean_pred.min() - 0.03, q.mean_pred.max() + 0.25 * (q.mean_pred.max() - q.mean_pred.min()))
    ax.legend(loc="upper left", fontsize=5.2, handletextpad=0.1, borderaxespad=0.2, markerscale=1.2, ncol=3,
              columnspacing=0.5)
    label(ax, "d")
    save(fig, "Fig3", {"a_example_spots": ex[["barcode", "tx", "ty", "y_pt_injury", "pred_pt_injury", "rank_y_pt_injury", "rank_pred_pt_injury"]],
                       "bc_summary": pd.DataFrame(r1 + r2), "b_ceiling": ceil,
                       "d_participants": q[["participant", "group", "mean_y", "mean_pred"]]})


if __name__ == "__main__":
    main()
