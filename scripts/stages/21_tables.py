"""Main-text Table 2 and the Additional file 2 workbook, written from result tables only."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
S = stage_dir("21_tables")
MODELS = [("gbm_measured_comp", r"Measured composition\textsuperscript{a}"), ("gbm_comp_morph", "Image composition and morphology"),
          ("hist2st", "Hist2ST"), ("bleep", "BLEEP"), ("istar", "iStar"),
          ("knn_encoder", r"Retrieval, same encoder\textsuperscript{b}"),
          ("tok_pred_mean", r"Token-level prediction, same encoder\textsuperscript{c}"),
          ("ridge_image", "Linear probe"), ("mlp", "Spot only"), ("gcn", "Lattice graph (GCN)"),
          ("set_anat", "Anatomical spot hypergraph"), ("deepsets", "Neighbourhood set"),
          ("tok_abmil", "Token attention"), ("tok_hyper_anat", "Anatomical token hypergraph"),
          ("tok_abmil_ctx", r"\method{}")]


def f(x, lo=None, hi=None):
    g = lambda v: f"{v:.2f}".replace("-0.00", "0.00").replace("-", "$-$")
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
        "Table S5d token regions tissue": pd.read_csv(p("results/16h_region_tissue/contrasts.tsv"), sep="\t")
        .assign(mean_tissue_tokens_inside=lambda d: pd.read_csv(p("results/16h_region_tissue/tissue_tokens.tsv"), sep="\t")
                .mean_tissue_tokens_inside.mean(),
                mean_tissue_tokens_outermost=lambda d: pd.read_csv(p("results/16h_region_tissue/tissue_tokens.tsv"), sep="\t")
                .mean_tissue_tokens_outermost.mean()),
        "Table S6 external": ex,
        "Table S6b external contrasts": pd.read_csv(p("results/18_evaluate_external/contrasts.tsv"), sep="\t"),
        "Table S6c external tissue": pd.read_csv(p("results/18e_external_tissue/summary.tsv"), sep="\t")
        .merge(pd.read_csv(p("results/18e_external_tissue/spots_kept.tsv"), sep="\t"), on="filter"),
        "Table S7 reliability regions": pd.concat([
            pd.read_csv(p("results/16b_reliability_region/summary.tsv"), sep="\t").assign(cohort="KPMP"),
            pd.read_csv(p("results/18b_external_reliability/ceiling.tsv"), sep="\t").assign(cohort="External ceiling"),
            pd.read_csv(p("results/18c_external_region_coverage/region_summary.tsv"), sep="\t").assign(cohort="External")]),
        "Table S8 subgroups coverage": pd.concat([
            pd.read_csv(p("results/20_subgroups/subgroups.tsv"), sep="\t"),
            pd.read_csv(p("results/18c_external_region_coverage/coverage.tsv"), sep="\t").assign(grouping="coverage"),
            pd.read_csv(p("results/17_confidence/conformal_cv.tsv"), sep="\t").assign(grouping="coverage KPMP folds"),
            pd.read_csv(p("results/18c_external_region_coverage/coverage_sections.tsv"), sep="\t")
            .assign(grouping="coverage per external donor")]),
        "Table S9 composition models": pd.read_csv(p("results/16f_adjustment_sensitivity/summary.tsv"), sep="\t"),
        "Table S9b composition contrasts": pd.read_csv(p("results/16f_adjustment_sensitivity/contrasts.tsv"), sep="\t"),
        "Table S10 reliability": pd.read_csv(p("results/16g_reliability_sensitivity/summary.tsv"), sep="\t"),
        "Table S11 external donors": pd.read_csv(p("results/18d_external_donors/donor_metrics.tsv"), sep="\t"),
        "Table S11b seeds": pd.read_csv(p("results/18d_external_donors/seed_stability.tsv"), sep="\t"),
    }
    cost = p("results/21b_model_cost/model_cost.tsv")
    if cost.exists():
        sheets["Table S12 model cost"] = pd.read_csv(cost, sep="\t")
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
            "Token regions: tissue tokens inside the capture spot and at the crop edge, and agreement of the region models in spots where both regions are on tissue",
            "External cohort: agreement for every model, programme and metric",
            "External cohort: paired model differences by donor",
            "External cohort: NephroSet and spot-only network after removing spots or sections with little tissue in the image, and without the donors shared with the single-cell atlas",
            "Split-half reliability and regional agreement, both cohorts",
            "Agreement by diagnosis and tissue source; interval coverage",
            "Agreement beyond composition under alternative composition models and in PT-rich spots",
            "Paired differences of NephroSet from reference models under each composition model",
            "Split-half reliability of measured scores under alternative splits and estimators",
            "Second centre: agreement of each model for every donor, with measured spread and diagnosis",
            "Second centre: donor-mean agreement of single seeds and of seed ensembles, including a repeated training run of NephroSet",
            "Trainable parameters, inference time and training cost of each model"][:len(sheets)]})
        # number the sheets in the order in which the tables are first cited in the text
        renum = {4: 6, 5: 7, 6: 9, 7: 10, 8: 12, 9: 5, 10: 4, 11: 11, 12: 8}

        def new_key(k):
            m = re.match(r"Table S(\d+)(.*)", k)
            return f"Table S{renum.get(int(m.group(1)), int(m.group(1)))}{m.group(2)}"
        readme["sheet"] = [new_key(k) for k in readme.sheet]
        sheets = {new_key(k): v for k, v in sheets.items()}
        num = lambda k: (int(re.match(r"Table S(\d+)", k).group(1)), k)
        order = sorted(sheets, key=num)
        readme = readme.set_index("sheet").loc[order].reset_index()
        sheets = {k: sheets[k] for k in order}
        readme.to_excel(w, sheet_name="README", index=False)
        for k, v in sheets.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    print(pd.DataFrame(rows).round(3).T)
    write_provenance("21_tables", cfg, [], [S / "table2.tsv", out])


if __name__ == "__main__":
    main()
