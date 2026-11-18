"""Supplementary figures S1, S2 and S4 and Table S1 (LaTeX rows), from result tables only."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figstyle import MODEL_LABEL, PAL, ROOT, W2, label, save

R = ROOT / "results"
TG = {"yres_pt_injury": "PT injury (cell)", "yunit_pt_injury": "PT injury (unit)",
      "yres_inflammation": "Inflammation (cell)", "yunit_inflammation": "Inflammation (unit)",
      "yres_pt_transport": "PT transport (cell)", "yunit_pt_transport": "PT transport (unit)",
      "is_iPT": "Injured PT label (AUROC)"}
LV = ["A", "AB", "ABC", "ABD", "ABDu", "ABDh", "ABCD"]
LVL = {"A": "A", "AB": "A+B", "ABC": "A+B+C", "ABD": "A+B+D", "ABDu": "A+B+unit", "ABDh": "A+B+halo", "ABCD": "A+B+C+D"}
PROG = {"pt_injury": "PT injury", "inflammation": "Inflammation", "pt_transport": "PT transport"}
ORDER = ["gbm_measured_comp", "gbm_comp_morph", "hist2st", "bleep", "istar", "knn_encoder", "tok_pred_mean", "ridge_image", "mlp", "gcn", "gat", "clique_gcn", "hgnn_ring",
         "hgnn_anat", "deepsets", "set_ring", "set_anat", "set_anat_rewired", "set_anat_noroles", "set_anat_nogeom",
         "tok_mean", "tok_abmil", "tok_gat", "tok_hyper_window", "tok_hyper_rolebag", "tok_hyper_rewired",
         "tok_hyper_profile", "tok_hyper_anat", "tok_hyper_anat_ctx", "tok_abmil_ctx"]


def fig_s1():
    perf = pd.read_csv(R / "03_atlas_increment" / "model_performance.tsv", sep="\t")
    fig, axes = plt.subplots(1, 7, figsize=(W2, 2.6), sharey=False)
    rows = []
    for ax, (t, lab) in zip(axes, TG.items()):
        for null, mk, col in [(False, "o", PAL["anatomy"]), (True, "s", PAL["null"])]:
            q = perf[(perf.programme == t) & (perf.null == null)].set_index("model").reindex(LV)
            ax.plot(q.donor_mean_rho.values, range(len(LV))[::-1], marker=mk, ls="", ms=2.8, color=col,
                    mfc=col if not null else "white")
            rows += [dict(target=t, level=m, permuted=null, value=v) for m, v in zip(LV, q.donor_mean_rho)]
        ax.set_yticks(range(len(LV))[::-1])
        ax.set_yticklabels([LVL[x] for x in LV] if ax is axes[0] else [])
        ax.set_title(lab.replace(" (", "\n("), fontsize=6, pad=2)
        ax.tick_params(axis="x", labelsize=5, rotation=45)
        v = perf[perf.programme == t].donor_mean_rho
        ax.set_xlim(v.min() - 0.01, v.max() + 0.01)
    axes[0].set_ylabel("Descriptor set")
    fig.supxlabel("Donor-mean within-section Spearman ρ (AUROC for the label)", fontsize=6.5, y=0.0)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color=PAL["anatomy"], label="Observed labels"),
                        Line2D([], [], marker="s", ls="", ms=3, mfc="white", color=PAL["null"],
                               label="Interstitial labels permuted within 100 µm")],
               loc="lower right", fontsize=5.5, ncol=2, frameon=False, bbox_to_anchor=(0.99, -0.02))
    fig.subplots_adjust(wspace=0.35, left=0.09, right=0.99, bottom=0.3, top=0.85)
    save(fig, "FigS1", {"levels": pd.DataFrame(rows)})


def fig_s2():
    cv = pd.read_csv(R / "16_evaluate_cv" / "summary.tsv", sep="\t")
    ex = pd.read_csv(R / "18_evaluate_external" / "summary.tsv", sep="\t")
    cols = [(cv, "rho", "KPMP\nall"), (cv, "rho_beyond_comp", "KPMP\nadj."), (ex, "rho", "2nd\ncentre\nall"),
            (ex, "rho_beyond_comp", "2nd\ncentre\nadj.")]
    fig = plt.figure(figsize=(W2, 3.6))
    gsx = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06], left=0.22, right=0.92, bottom=0.14, top=0.93, wspace=0.08)
    axes = [fig.add_subplot(gsx[i]) for i in range(3)]
    rows = []
    for ax, (k, lab) in zip(axes, PROG.items()):
        M = np.array([[t[(t.model == m) & (t.programme == k) & (t.metric == met)]["mean"].iloc[0] for t, met, _ in cols]
                      for m in ORDER])
        im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=max(0.65, M.max()))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j] + 0.0:.2f}".replace("-0.00", "0.00").replace("-", "\u2212"), ha="center", va="center", fontsize=4.6,
                        color="white" if M[i, j] < 0.4 else "black")._allow_on_data = True
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c[2] for c in cols], fontsize=5.3)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([MODEL_LABEL[m] for m in ORDER] if ax is axes[0] else [], fontsize=5.5)
        ax.set_title(lab, pad=3)
        for s in ax.spines.values():
            s.set_visible(False)
        rows += [dict(model=m, programme=k, cohort_metric=c[2], value=M[i, j]) for i, m in enumerate(ORDER)
                 for j, c in enumerate(cols)]
    cb = fig.colorbar(im, cax=fig.add_subplot(gsx[3]))
    cb.set_label("Within-section Spearman ρ", fontsize=6)
    save(fig, "FigS2", {"all_models": pd.DataFrame(rows)})


def fig_s3():
    cv = pd.read_csv(R / "16_evaluate_cv" / "summary.tsv", sep="\t")
    ex = pd.read_csv(R / "18_evaluate_external" / "summary.tsv", sep="\t")
    ms = ["ridge_image", "mlp", "gcn", "deepsets", "set_ring", "set_anat"]
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.4))
    rows = []
    for ax, (tab, coh) in zip(axes, [(cv, "KPMP (held out)"), (ex, "Second centre")]):
        for i, m in enumerate(ms):
            for j, (suf, col) in enumerate([("", PAL["anatomy"]), ("_sn", PAL["ref"])]):
                q = tab[(tab.model == m + suf) & (tab.programme == "pt_injury") & (tab.metric == "rho")].iloc[0]
                x = i + (j - 0.5) * 0.32
                ax.errorbar(x, q["mean"], yerr=[[q["mean"] - q.lo], [q.hi - q["mean"]]], fmt="o", ms=2.8, color=col, lw=0.8)
                rows.append(dict(cohort=coh, model=m, standardized=bool(suf), mean=q["mean"], lo=q.lo, hi=q.hi))
        ax.set_xticks(range(len(ms)))
        short = {"ridge_image": "Linear\nprobe", "mlp": "Spot\nonly", "gcn": "Lattice\ngraph", "deepsets": "Neigh-\nbour-\nhood set",
                 "set_ring": "Local set\nhyper-\ngraph", "set_anat": "Anat. set\nhyper-\ngraph"}
        ax.set_xticklabels([short[m] for m in ms], fontsize=5.2)
        ax.set_title(coh, pad=3)
        ax.set_ylabel("Within-section ρ, PT injury")
        ax.set_ylim(0, 0.32)
        label(ax, "a" if ax is axes[0] else "b")
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color=PAL["anatomy"], label="As trained"),
                            Line2D([], [], marker="o", ls="", ms=3, color=PAL["ref"], label="Standardized within section")],
                   loc="upper right", fontsize=5.2, borderaxespad=0.2)
    fig.subplots_adjust(wspace=0.35, bottom=0.25, left=0.1, right=0.98, top=0.88)
    save(fig, "FigS3", {"section_standardized": pd.DataFrame(rows)})


def table_s1():
    cov = pd.read_csv(R / "01_atlas_extract" / "panel_coverage.tsv", sep="\t")
    pv = cov.pivot_table(index=["programme", "gene"], columns="platform", values="frac_cells_detected").reset_index()
    pv = pv.sort_values(["programme", "gene"])
    name = {"pt_injury": "PT injury", "inflammation": "Inflammation", "pt_transport": "PT transport"}
    lines = []
    for _, r in pv.iterrows():
        c = lambda v: f"{100 * v:.1f}" if v >= 0.01 else f"{100 * v:.1f}$^{{*}}$"
        lines.append(f"{name[r.programme]} & \\textit{{{r.gene}}} & {c(r.CosMx)} & {c(r.Xenium)} \\\\")
    (ROOT / "figures" / "source_data" / "TableS1_rows.tex").write_text("\n".join(lines) + "\n")
    pv.to_csv(ROOT / "figures" / "source_data" / "TableS1.tsv", sep="\t", index=False)


if __name__ == "__main__":
    fig_s1()
    fig_s2()
    fig_s3()
    table_s1()
