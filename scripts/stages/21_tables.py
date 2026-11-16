"""Main-text Table 2 and the Additional file 2 workbook, written from result tables only."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
S = stage_dir("21_tables")
MODELS = [("gbm_measured_comp", r"Measured composition\textsuperscript{a}"), ("gbm_comp_morph", "Image composition and morphology"),
          ("hist2st", "Hist2ST"), ("bleep", "BLEEP"), ("istar", "iStar"),
          ("ridge_image", "Linear probe"), ("mlp", "Spot only"), ("gcn", "Lattice graph (GCN)"),
          ("set_anat", "Anatomical spot hypergraph"), ("deepsets", "Neighbourhood set"),
          ("tok_abmil", "Token attention"), ("tok_hyper_anat", "Anatomical token hypergraph"),
          ("tok_abmil_ctx", r"\method{}")]


def f(x, lo=None, hi=None):
    g = lambda v: f"{v:.2f}".replace("-", "$-$")
    return g(x) if lo is None else f"{g(x)} ({g(lo)}--{g(hi)})"


def main() -> None:
    cv = pd.read_csv(p("results/16_evaluate_cv/summary.tsv"), sep="\t")
    ex = pd.read_csv(p("results/18_evaluate_external/summary.tsv"), sep="\t")
    rows, tex = [], []
    for m, lab in MODELS:
        r = {"model": m}
        cells = [lab]
        for k in ["pt_injury", "pt_transport"]:
            for tab, coh in [(cv, "KPMP"), (ex, "External")]:
                for met in ["rho", "rho_beyond_comp"]:
                    q = tab[(tab.model == m) & (tab.programme == k) & (tab.metric == met)].iloc[0]
                    r[f"{k}_{coh}_{met}"] = q["mean"]
                    r[f"{k}_{coh}_{met}_lo"], r[f"{k}_{coh}_{met}_hi"] = q.lo, q.hi
                    cells.append(f(q["mean"]))
        rows.append(r)
        tex.append(" & ".join(cells) + r" \\")
    pd.DataFrame(rows).to_csv(S / "table2.tsv", sep="\t", index=False)
    (S / "table2_rows.tex").write_text("\n".join(tex) + "\n")
    # Additional file 2 workbook
    sheets = {
        "Table S2 participants": pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
        .merge(pd.read_csv(p("results/11b_kpmp_targets/spot_summary.tsv"), sep="\t"), on="section", how="left",
               suffixes=("", "_spots")),
        "Table S3 single-cell": pd.read_csv(p("results/03_atlas_increment/model_performance.tsv"), sep="\t"),
        "Table S3b increments": pd.read_csv(p("results/03_atlas_increment/increments.tsv"), sep="\t"),
        "Table S4 CV summary": cv,
        "Table S5 CV contrasts": pd.read_csv(p("results/16_evaluate_cv/contrasts.tsv"), sep="\t"),
        "Table S5b scale": pd.read_csv(p("results/16c_scale_diagnostics/spot_mixing.tsv"), sep="\t"),
        "Table S5c token roles": pd.read_csv(p("results/16c_scale_diagnostics/token_roles_heldout.tsv"), sep="\t")
        .groupby(["cls"]).r.agg(["mean", "std", "count"]).reset_index(),
        "Table S6 external": ex,
        "Table S6b external contrasts": pd.read_csv(p("results/18_evaluate_external/contrasts.tsv"), sep="\t"),
        "Table S7 ceilings regions": pd.concat([
            pd.read_csv(p("results/16b_reliability_region/summary.tsv"), sep="\t").assign(cohort="KPMP"),
            pd.read_csv(p("results/18b_external_reliability/ceiling.tsv"), sep="\t").assign(cohort="External ceiling"),
            pd.read_csv(p("results/18c_external_region_coverage/region_summary.tsv"), sep="\t").assign(cohort="External")]),
        "Table S8 subgroups coverage": pd.concat([
            pd.read_csv(p("results/20_subgroups/subgroups.tsv"), sep="\t"),
            pd.read_csv(p("results/18c_external_region_coverage/coverage.tsv"), sep="\t").assign(grouping="coverage"),
            pd.read_csv(p("results/17_confidence/conformal_cv.tsv"), sep="\t").assign(grouping="coverage KPMP folds")]),
    }
    keep = ["participant", "section", "status", "n_spots_tissue", "frac_spots_on_tissue", "um_per_px",
            "enrollment_category", "primary_adjudicated_category", "sample_type", "tissue_source", "age_binned", "sex",
            "baseline_egfr", "diabetes_history", "hypertension_history", "n_spots_spots", "median_umi"]
    sheets["Table S2 participants"] = sheets["Table S2 participants"][[c for c in keep if c in sheets["Table S2 participants"]]]
    out = S / "Additional_file_2.xlsx"
    with pd.ExcelWriter(out) as w:
        readme = pd.DataFrame({"sheet": list(sheets), "content": [
            "KPMP sections: registration check, spot counts and open-access clinical categories",
            "Single-cell analysis: donor-mean agreement for each descriptor set, observed and permuted labels",
            "Single-cell analysis: increments with donor bootstrap intervals",
            "KPMP cross-validation: participant-mean agreement for every model, programme and metric",
            "KPMP cross-validation: paired model differences by participant",
            "Mixing of cell classes within Visium spots and size of spot-level anatomical units, per section",
            "Token roles: within-section correlation of spot-mean token roles with measured composition, held-out participants",
            "External cohort: agreement for every model, programme and metric",
            "External cohort: paired model differences by donor",
            "Reliability ceilings and regional agreement, both cohorts",
            "Agreement by diagnosis and tissue source; interval coverage"]})
        readme.to_excel(w, sheet_name="README", index=False)
        for k, v in sheets.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    print(pd.DataFrame(rows).round(3).T)
    write_provenance("21_tables", cfg, [], [S / "table2.tsv", out])


if __name__ == "__main__":
    main()
