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


def hforest(ax, d, order, col):
    rows = []
    for i, g in enumerate(order):
        q = d[d.group == g].iloc[0]
        ax.errorbar(q.rho, -i, xerr=[[q.rho - q.lo], [q.hi - q.rho]], fmt="o", ms=3, color=col, lw=0.8)
        rows.append(q.to_dict())
    ax.set_yticks(-np.arange(len(order)))
    ax.set_yticklabels([f"{g} ({int(d[d.group == g].n.iloc[0])})" for g in order])
    return rows


def main() -> None:
    sg = pd.read_csv(R / "20_subgroups" / "subgroups.tsv", sep="\t")
    cov = pd.read_csv(R / "18c_external_region_coverage" / "coverage.tsv", sep="\t")
    cur = pd.read_csv(R / "17_confidence" / "retained_curves.tsv", sep="\t")
    fig = plt.figure(figsize=(W2, 3.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 0.85, 0.85], height_ratios=[1.6, 1], hspace=0.75, wspace=0.55,
                          left=0.24, right=0.985, top=0.92, bottom=0.12)
    d = sg[(sg.grouping == "diagnosis") & (sg.programme == "pt_injury")]
    ax = fig.add_subplot(gs[0, 0])
    r1 = hforest(ax, d, DX_ORDER, PAL["anatomy"])
    ax.set_xlabel("Within-section ρ, PT injury")
    ax.set_title("By diagnosis (KPMP, held out)", pad=3)
    ax.set_xlim(0, 0.4)
    label(ax, "a")
    d = sg[(sg.grouping == "source") & (sg.programme == "pt_injury")]
    ax = fig.add_subplot(gs[1, 0])
    r2 = hforest(ax, d, SRC_ORDER, PAL["ref"])
    ax.set_xlabel("Within-section ρ, PT injury")
    ax.set_title("By tissue source", pad=3)
    ax.set_xlim(0, 0.4)
    label(ax, "b")

    ax = fig.add_subplot(gs[:, 1])
    rows = []
    for i, (k, lab) in enumerate(PROG_LAB.items()):
        q = cov[cov.programme == k].iloc[0]
        ax.bar(i - 0.17, q.kpmp_coverage, width=0.32, color=PAL["anatomy"], alpha=0.85)
        ax.bar(i + 0.17, q.external_coverage, width=0.32, color=PAL["ref"], alpha=0.85)
        ax.errorbar(i + 0.17, q.external_coverage, yerr=[[q.external_coverage - q.lo], [q.hi - q.external_coverage]],
                    fmt="none", ecolor="k", lw=0.7)
        rows.append(q.to_dict())
    ln = ax.axhline(0.9, c="k", lw=0.7, ls=(0, (3, 2)))
    ln._is_reference = True
    ax.set_xticks(range(3))
    ax.set_xticklabels(["PT\ninjury", "Inflam-\nmation", "PT\ntransport"])
    ax.set_ylabel("Share of spots inside the interval")
    ax.set_ylim(0.4, 1.08)
    ax.set_title("Conformal 90% intervals", pad=3)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PAL["anatomy"], alpha=0.85, label="KPMP (held out)"),
                       plt.Rectangle((0, 0), 1, 1, color=PAL["ref"], alpha=0.85, label="Second centre")],
              loc="upper right", fontsize=5.2, borderaxespad=0.2, ncol=1)
    label(ax, "c")

    ax = fig.add_subplot(gs[:, 2])
    q = cur[cur.programme == "pt_injury"]
    for sc, col, lab in [("sd_pt_injury", PAL["anatomy"], "Ensemble disagreement"), ("novelty", PAL["group"], "Image novelty")]:
        z = q[q.score == sc].sort_values("keep_frac", ascending=False)
        ax.plot(z.keep_frac * 100, z.mae, marker="o", ms=2.5, lw=1, color=col, label=lab)
    ax.invert_xaxis()
    ax.set_xlabel("Spots retained (%)")
    ax.set_ylabel("Mean absolute error, PT injury")
    ax.set_title("Selective prediction", pad=3)
    lo, hi = q.mae.min(), q.mae.max()
    ax.set_ylim(lo - 0.0008, hi + 0.0016)
    ax.legend(loc="upper right", fontsize=5.2, borderaxespad=0.2)
    label(ax, "d")
    save(fig, "Fig6", {"a_diagnosis": pd.DataFrame(r1), "b_source": pd.DataFrame(r2), "c_coverage": pd.DataFrame(rows),
                       "d_retained": q})


if __name__ == "__main__":
    main()
