"""Figure 4: context between spots versus structure within the spot (paired by participant)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figstyle import MODEL_LABEL, PAL, ROOT, W2, label, save

R = ROOT / "results"
PROG_LAB = {"pt_injury": "PT injury", "inflammation": "Inflammation", "pt_transport": "PT transport"}
PCOL = {"pt_injury": PAL["anatomy"], "inflammation": PAL["group"], "pt_transport": PAL["ref"]}
BETWEEN = [("gcn", "Lattice graph (GCN)"), ("gat", "Lattice graph (GAT)"), ("hgnn_ring", "Local hypergraph"),
           ("hgnn_anat", "Unit hypergraph"), ("clique_gcn", "Clique expansion"), ("set_ring", "Local set hypergraph"),
           ("set_anat", "Anatomical set hypergraph"), ("deepsets", "Neighbourhood set")]
WITHIN = [("tok_mean", "Token mean"), ("tok_abmil", "Token attention"), ("tok_gat", "Token graph (GAT)"),
          ("tok_hyper_anat", "Anatomical token hypergraph"), ("tok_abmil_ctx", "NephroSet")]
ABL = [("tok_hyper_rewired", "Random tokens, same size"), ("tok_hyper_rolebag", "Compartment, no contiguity"),
       ("tok_hyper_window", "3 × 3 windows"), ("tok_hyper_profile", "Profiles without halo"),
       ("tok_abmil", "No hyperedges (attention)")]


def paired(pm, a, b, k, metric, rng, n=2000):
    x = pm[(pm.model == a) & (pm.programme == k)].set_index("participant")[metric]
    y = pm[(pm.model == b) & (pm.programme == k)].set_index("participant")[metric]
    d = (x - y).dropna().values
    bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (d > 0).mean()


def forest(ax, pm, items, ref, rng, metric="rho"):
    rows, yt, yl = [], [], []
    y = 0
    for m, lab in items:
        for j, k in enumerate(PROG_LAB):
            mu, lo, hi, fp = paired(pm, m, ref, k, metric, rng)
            ax.errorbar(mu, -(y + j * 0.28), xerr=[[mu - lo], [hi - mu]], fmt="o", ms=2.6, color=PCOL[k], lw=0.8)
            rows.append(dict(model=m, reference=ref, programme=k, metric=metric, diff=mu, lo=lo, hi=hi,
                             frac_participants_positive=fp))
        yt.append(-(y + 0.28))
        yl.append(lab)
        y += 1.3
    ln = ax.axvline(0, c="0.6", lw=0.5)
    ln._is_reference = True
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    return rows


SCALE = [("deepsets", "mlp", "Between\nspots"),
         ("tok_abmil", "mlp", "Token\nresolution"),
         ("tok_hyper_anat", "tok_abmil", "Anatomical\nprior")]
LADDER = ["ridge_image", "mlp", "deepsets", "tok_abmil", "tok_abmil_ctx"]
LAD_LAB = {"ridge_image": "Linear\nprobe", "mlp": "Spot\nonly", "deepsets": "Neighbour-\nhood set",
           "tok_abmil": "Token\nattention", "tok_abmil_ctx": "Nephro-\nSet"}
COH = [("KPMP", "16_evaluate_cv", PAL["anatomy"], "o"), ("Second centre", "18_evaluate_external", PAL["ref"], "s")]


def scale_panel(ax):
    rows = []
    for c, (coh, d, col, mk) in enumerate(COH):
        con = pd.read_csv(R / d / "contrasts.tsv", sep="\t")
        for i, (a, b, _) in enumerate(SCALE):
            for j, k in enumerate(["pt_injury", "pt_transport"]):
                q = con[(con.a == a) & (con.b == b) & (con.programme == k) & (con.metric == "rho")].iloc[0]
                x = i + (j - 0.5) * 0.42 + (c - 0.5) * 0.17
                ax.errorbar(x, q["diff"], yerr=[[q["diff"] - q.lo], [q.hi - q["diff"]]], fmt=mk, ms=2.8, lw=0.8,
                            color=col, mfc=col if k == "pt_injury" else "white", mew=0.8)
                rows.append(dict(cohort=coh, comparison=f"{a} vs {b}", programme=k, diff=q["diff"], lo=q.lo, hi=q.hi,
                                 n=q.n, frac_positive=q.frac_positive))
    ln = ax.axhline(0, c="0.6", lw=0.5)
    ln._is_reference = True
    ax.set_xticks(range(3))
    ax.set_xticklabels([t for *_, t in SCALE])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Δ ρ (95% CI)")
    ax.set_title("Gain from context at each scale", pad=3)
    hl = [Line2D([], [], marker=mk, ls="", ms=3, color=col, label=coh) for coh, _, col, mk in COH]
    hl += [Line2D([], [], marker="o", ls="", ms=3, color="0.3", label="PT injury"),
           Line2D([], [], marker="o", ls="", ms=3, color="0.3", mfc="white", label="PT transport")]
    lo = min(r["lo"] for r in rows)
    hi = max(r["hi"] for r in rows)
    ax.set_ylim(lo - 0.01, hi + 0.075)
    ax.legend(handles=hl, loc="upper left", fontsize=5.2, ncol=2, columnspacing=0.9, handletextpad=0.2,
              borderaxespad=0.2)
    return pd.DataFrame(rows)


def ladder_panel(ax):
    rows = []
    for c, (coh, d, col, mk) in enumerate(COH):
        sm = pd.read_csv(R / d / "summary.tsv", sep="\t")
        for j, k in enumerate(["pt_injury", "pt_transport"]):
            ys = []
            for i, m in enumerate(LADDER):
                q = sm[(sm.model == m) & (sm.programme == k) & (sm.metric == "rho")].iloc[0]
                ys.append(q["mean"])
                rows.append(dict(cohort=coh, programme=k, model=m, rho=q["mean"], lo=q.lo, hi=q.hi))
            ax.plot(range(len(LADDER)), ys, marker=mk, ms=2.8, lw=0.9, color=col,
                    ls="-" if k == "pt_injury" else (0, (2, 1.5)), mfc=col if k == "pt_injury" else "white", mew=0.8)
    ax.set_xticks(range(len(LADDER)))
    ax.set_xticklabels([LAD_LAB[m] for m in LADDER], fontsize=5.4)
    ax.set_xlim(-0.4, 5.3)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Within-section Spearman ρ")
    ax.set_title("Absolute agreement", pad=3)
    for k, y in [("PT\ntransport", 0.57), ("PT\ninjury", 0.2)]:
        ax.text(4.25, y, k, fontsize=5.4, ha="left", va="center")
    return pd.DataFrame(rows)


REGION = [("tok_mean_disc", "Inside, mean (32)"), ("tok_abmil_disc", "Inside, attention (32)"),
          ("tok_mean_ring", "Around, mean (164)"), ("tok_abmil_ring", "Around, attention (164)"),
          ("tok_abmil_outer32", "Outermost, attention (32)"), ("tok_mean", "All, mean (196)"),
          ("tok_abmil", "All, attention (196)")]


def crop_schematic(ax):
    """112 um crop, 14 x 14 tokens of 8 um, and the 55 um capture spot."""
    r, c = np.divmod(np.arange(196), 14)
    d = np.hypot((r - 6.5) * 8, (c - 6.5) * 8)
    outer = d >= np.sort(d)[-32]
    for i in range(196):
        col = PAL["anatomy"] if d[i] <= 27.5 else (PAL["ref"] if outer[i] else "0.88")
        ax.add_patch(plt.Rectangle((c[i] * 8 - 56, 56 - (r[i] + 1) * 8), 7.4, 7.4, color=col, lw=0))
    ax.add_patch(plt.Circle((0, 0), 27.5, fill=False, lw=0.9, ec="k"))
    ax.set_xlim(-57, 57)
    ax.set_ylim(-57, 57)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("112 µm crop,\n55 µm spot", pad=2, fontsize=5.6)
    return pd.DataFrame(dict(token=np.arange(196), row=r, col=c, dist_um=d, inside=d <= 27.5, outermost32=outer))


COST = [("mlp", "Spot only"), ("deepsets", "Neighbourhood set"), ("tok_mean", "Token mean"),
        ("tok_abmil", "Token attention"), ("tok_gat", "Token graph"), ("tok_hyper_anat", "Anatomical token\nhypergraph"),
        ("tok_hyper_anat_ctx", "Anatomical token\nhypergraph + set"), ("tok_abmil_ctx", "NephroSet")]
OFFS = {"mlp": (6, 0, "left"), "deepsets": (6, 0, "left"), "tok_mean": (-6, 0, "right"),
        "tok_abmil": (5, -8, "left"), "tok_gat": (6, 0, "left"), "tok_hyper_anat": (4, -18, "left"),
        "tok_hyper_anat_ctx": (0, 11, "center"), "tok_abmil_ctx": (-6, 6, "right")}


def cost_rows(cost, summ):
    rows = []
    for m, lab in COST:
        r = summ.loc[(m, "pt_injury", "rho")]
        rows.append(dict(model=m, label=lab.replace("\n", " "), short=lab, train_min=cost.loc[m, "train_min_per_model"],
                         inference_ms=cost.loc[m, "inference_s_per_section"] * 1000, rho=r["mean"], lo=r["lo"],
                         hi=r["hi"]))
    return pd.DataFrame(rows)


def colour(m):
    return PAL["anatomy"] if m == "tok_abmil_ctx" else ("0.25" if m.startswith("tok") else "0.6")


def cost_panel(ax, d):
    """Injury agreement against estimated downstream training compute; NephroSet highlighted."""
    for r in d.itertuples():
        nep = r.model == "tok_abmil_ctx"
        ax.scatter(r.train_min, r.rho, s=34 if nep else 18, color=colour(r.model), ec="k" if nep else "none", lw=0.5,
                   zorder=4)
        dx, dy, ha = OFFS[r.model]
        ax.annotate(r.short, (r.train_min, r.rho), xytext=(dx, dy), textcoords="offset points", fontsize=5.2, ha=ha,
                    va="center", color=colour(r.model) if nep else "0.2", fontweight="bold" if nep else "normal")
    ax.axhline(d.set_index("model").loc["tok_abmil_ctx", "rho"], color=PAL["anatomy"], lw=0.6, ls=(0, (3, 2)),
               zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.04, 25)
    ax.set_ylim(0.212, 0.278)
    ax.set_xlabel("Estimated downstream training compute (min per model)")
    ax.set_ylabel("PT injury ρ (KPMP, held out)")
    ax.set_title("Accuracy and training cost", pad=3)


def inference_panel(ax, d):
    d = d.iloc[::-1]
    y = np.arange(len(d))
    ax.barh(y, d.inference_ms, color=[colour(m) for m in d.model], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("\n", " ") for s in d.short], fontsize=5.4)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    for yy, v in zip(y, d.inference_ms):
        ax.text(v * 1.15, yy, f"{v:.1f}" if v >= 1 else f"{v:.2f}", fontsize=5.0, va="center", ha="left")
    ax.set_xlabel("Downstream inference per section (ms)")
    ax.set_title("Inference time", pad=3)


def main() -> None:
    pm = pd.read_csv(R / "16_evaluate_cv" / "participant_metrics.tsv", sep="\t")
    rng = np.random.default_rng(2026)
    fig = plt.figure(figsize=(W2, 7.9))
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 0.85, 0.75], hspace=0.62, left=0.14, right=0.96, top=0.935,
                             bottom=0.07)
    gs = outer[0].subgridspec(1, 3, wspace=1.25)
    out = {}
    for j, (items, ref, xlab, title, xl, key) in enumerate([
            (BETWEEN, "mlp", "Δ ρ vs spot only", "Context between spots", (-0.03, 0.05), "a_between"),
            (WITHIN, "mlp", "Δ ρ vs spot only", "Token resolution", (-0.03, 0.05), "b_within"),
            (ABL, "tok_hyper_anat", "Δ ρ vs anatomical\ntoken hypergraph", "Other hyperedges", (-0.015, 0.016),
             "c_hyperedges")]):
        ax = fig.add_subplot(gs[j])
        out[key] = pd.DataFrame(forest(ax, pm, items, ref, rng))
        ax.set_xlabel(xlab + ("\n(95% CI, 201 participants)" if j < 2 else "\n(95% CI)"))
        ax.set_title(title, pad=3)
        ax.set_xlim(*xl)
        ax.tick_params(axis="y", labelsize=5.6)
        if j == 0:
            hl = [Line2D([], [], marker="o", ls="", ms=3, color=PCOL[k], label=PROG_LAB[k]) for k in PROG_LAB]
            fig.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.56, 0.995), ncol=3, fontsize=5.6,
                       handletextpad=0.1, columnspacing=1.2, frameon=False)
        label(ax, "abc"[j])
    bot = outer[1].subgridspec(1, 3, width_ratios=[1.1, 0.3, 0.62], wspace=0.95)
    ax = fig.add_subplot(bot[0])
    out["d_scale"] = scale_panel(ax)
    label(ax, "d")
    ax = fig.add_subplot(bot[1])
    out["e_crop"] = crop_schematic(ax)
    label(ax, "e")
    ax = fig.add_subplot(bot[2])
    out["f_region"] = pd.DataFrame(forest(ax, pm, REGION, "mlp", rng))
    ax.set_xlim(-0.015, 0.05)
    ax.set_xlabel("Δ ρ vs spot only\n(95% CI)")
    ax.set_title("Where the gain lies", pad=3)
    ax.tick_params(axis="y", labelsize=5.4)
    label(ax, "f")
    cost = pd.read_csv(R / "21b_model_cost" / "model_cost.tsv", sep="\t").set_index("model")
    summ = pd.read_csv(R / "16_evaluate_cv" / "summary.tsv", sep="\t").set_index(["model", "programme", "metric"])
    d = cost_rows(cost, summ)
    out["gh_cost"] = d.drop(columns="short")
    row3 = outer[2].subgridspec(1, 2, width_ratios=[1.25, 1], wspace=0.75)
    ax = fig.add_subplot(row3[0])
    cost_panel(ax, d)
    label(ax, "g")
    ax = fig.add_subplot(row3[1])
    inference_panel(ax, d)
    label(ax, "h")
    save(fig, "Fig4", out)


if __name__ == "__main__":
    main()
