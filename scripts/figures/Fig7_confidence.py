"""Figure 6: where the estimates hold — diagnosis, tissue source and calibrated intervals."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figstyle import PAL, ROOT, W2, label, save

R = ROOT / "results"
PROG_LAB = {"pt_injury": "PT injury", "inflammation": "Inflammation", "pt_transport": "PT transport"}
DX_ORDER = ["Diabetic kidney disease", "Hypertensive kidney disease", "Acute tubular injury",
            "Acute interstitial nephritis", "Diabetes without kidney disease", "Healthy reference",
            "Other or undetermined"]
SRC_ORDER = ["Needle biopsy", "Donor kidney biopsy", "Nephrectomy or intra-operative"]


def hbars(ax, d, order, cmap, overall=None):
    """Horizontal bars coloured by value, 95% interval whiskers, participants per group inside the bar."""
    rows = []
    vals = [d[d.group == g].rho.iloc[0] for g in order]
    lo_v, hi_v = min(vals), max(vals)
    for i, g in enumerate(order):
        q = d[d.group == g].iloc[0]
        c = cmap(0.35 + 0.6 * (q.rho - lo_v) / max(hi_v - lo_v, 1e-9))
        ax.barh(-i, q.rho, 0.72, color=c, ec="none")
        ax.plot([q.lo, q.hi], [-i, -i], color="0.15", lw=0.7)
        ax.text(q.hi + 0.008, -i, f"n = {int(q.n)}", fontsize=4.9, va="center", ha="left", color="0.3",
                bbox=dict(fc="white", ec="none", pad=0.4), zorder=6)
        rows.append(q.to_dict())
    if overall is not None:
        ln = ax.axvline(overall, color="0.3", lw=0.6, ls=(0, (2, 1.5)))
        ln._is_reference = True
        ax.text(overall + 0.006, 0.62, "all participants", fontsize=4.9, ha="left", va="bottom", color="0.3")
    ax.set_yticks(-np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-len(order) + 0.4, 1.1 if overall is not None else 0.6)
    return rows


def main() -> None:
    sg = pd.read_csv(R / "20_subgroups" / "subgroups.tsv", sep="\t")
    cov = pd.read_csv(R / "18c_external_region_coverage" / "coverage.tsv", sep="\t")
    cvp = pd.read_csv(R / "17_confidence" / "conformal_cv_participants.tsv", sep="\t")
    exs = pd.read_csv(R / "18c_external_region_coverage" / "coverage_sections.tsv", sep="\t")
    cur = pd.read_csv(R / "17_confidence" / "retained_curves.tsv", sep="\t")
    fig = plt.figure(figsize=(W2, 3.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 0.85, 0.85], height_ratios=[1.6, 1], hspace=0.75, wspace=0.55,
                          left=0.24, right=0.985, top=0.92, bottom=0.12)
    d = sg[(sg.grouping == "diagnosis") & (sg.programme == "pt_injury")]
    ax = fig.add_subplot(gs[0, 0])
    r1 = hbars(ax, d, DX_ORDER, plt.cm.Oranges, overall=0.261)
    ax.set_xlabel("Within-section ρ, PT injury")
    ax.set_title("By diagnosis (KPMP, held out)", pad=3)
    ax.set_xlim(0, 0.46)
    label(ax, "a")
    d = sg[(sg.grouping == "source") & (sg.programme == "pt_injury")]
    ax = fig.add_subplot(gs[1, 0])
    r2 = hbars(ax, d, SRC_ORDER, plt.cm.Blues)
    ax.set_xlabel("Within-section ρ, PT injury")
    ax.set_title("By tissue source", pad=3)
    ax.set_xlim(0, 0.46)
    label(ax, "b")

    ax = fig.add_subplot(gs[:, 1])
    rows = []
    for i, (k, lab) in enumerate(PROG_LAB.items()):
        q = cov[cov.programme == k].iloc[0]
        kp = cvp[cvp.programme == k].coverage
        ax.bar(i - 0.17, kp.mean(), width=0.32, color=PAL["anatomy"], alpha=0.85)
        ax.bar(i + 0.17, q.external_coverage, width=0.32, color=PAL["ref"], alpha=0.85)
        ax.errorbar(i + 0.17, q.external_coverage, yerr=[[q.external_coverage - q.lo], [q.hi - q.external_coverage]],
                    fmt="none", ecolor="k", lw=0.7)
        ev = exs[exs.programme == k].coverage.values
        jit = np.linspace(-0.09, 0.09, len(ev))
        ax.scatter(i + 0.17 + jit, ev, s=3, color="k", alpha=0.6, lw=0, zorder=3)
        q = q.copy()
        q["kpmp_coverage_participant_mean"] = kp.mean()
        rows.append(q.to_dict())
    ln = ax.axhline(0.9, c="k", lw=0.7, ls=(0, (3, 2)))
    ln._is_reference = True
    ax.set_xticks(range(3))
    ax.set_xticklabels(["PT\ninjury", "Inflam-\nmation", "PT\ntransport"])
    ax.set_ylabel("Coverage per participant")
    ax.set_ylim(0.4, 1.08)
    ax.set_title("Calibrated 90% intervals", pad=3)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PAL["anatomy"], alpha=0.85, label="KPMP (held out)"),
                       plt.Rectangle((0, 0), 1, 1, color=PAL["ref"], alpha=0.85, label="Second centre")],
              loc="upper right", fontsize=5.2, borderaxespad=0.2, ncol=1)
    label(ax, "c")

    ax = fig.add_subplot(gs[:, 2])
    q = cur[cur.programme == "pt_injury"]
    for sc, col, lab in [("sd_pt_injury", PAL["anatomy"], "Ensemble disagreement"), ("novelty", PAL["group"], "Image novelty")]:
        z = q[q.score == sc].sort_values("keep_frac", ascending=False)
        base = z.mae[z.keep_frac == 1.0].iloc[0]
        ax.plot(z.keep_frac * 100, 100 * (z.mae / base - 1), marker="o", ms=2.5, lw=1, color=col, label=lab)
    q = q.assign(mae_change_pct=q.groupby("score").mae.transform(lambda v: 100 * (v / v[q.loc[v.index, "keep_frac"] == 1.0].iloc[0] - 1)))
    ln = ax.axhline(0, c="0.6", lw=0.6)
    ln._is_reference = True
    ax.invert_xaxis()
    ax.set_xlabel("Spots retained (%)")
    ax.set_ylabel("Change in mean absolute error,\nPT injury (%)")
    ax.set_title("Selective prediction", pad=3)
    ax.set_ylim(-2.0, 0.3)
    ax.legend(loc="lower left", fontsize=5.2, borderaxespad=0.2)
    label(ax, "d")
    save(fig, "Fig7", {"a_diagnosis": pd.DataFrame(r1), "b_source": pd.DataFrame(r2), "c_coverage": pd.DataFrame(rows),
                       "d_retained": q})


if __name__ == "__main__":
    main()
