#!/usr/bin/env bash
# NephroSet: all analyses from public data. Usage: bash run_all.sh [--resume|--force] [--skip=a,b]
set -euo pipefail
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)"
conda activate nephroset
MODE=resume; SKIP=""
for a in "$@"; do case $a in --force) MODE=force;; --resume) MODE=resume;; --skip=*) SKIP=${a#--skip=};; esac; done
mkdir -p logs
run() {  # name, output-that-marks-done, command...
  local name=$1 done=$2; shift 2
  [[ ",$SKIP," == *",$name,"* ]] && { echo "skip $name"; return; }
  if [[ $MODE == resume && -e $done ]]; then echo "done $name"; return; fi
  echo "run  $name"; "$@" 2>&1 | tee "logs/$name.log"
}
NN="mlp deepsets gcn gat clique_gcn hgnn_ring hgnn_anat set_ring set_anat set_anat_rewired set_anat_noroles set_anat_nogeom"
# data
run 00_download results/00_download/PROVENANCE.json python scripts/stages/00_download.py
# stage 1: single-cell spatial organisation (CosMx / Xenium)
run 01_atlas_extract results/01_atlas_extract/PROVENANCE.json python scripts/stages/01_atlas_extract.py
run 02_atlas_features results/02_atlas_features/PROVENANCE.json python scripts/stages/02_atlas_features.py
run 03_atlas_increment results/03_atlas_increment/PROVENANCE.json python scripts/stages/03_atlas_increment.py
run 04_atlas_examples results/04_atlas_examples/PROVENANCE.json python scripts/stages/04_atlas_examples.py
# stage 2: H&E -> spot programmes (KPMP development, GSE211785 external)
run 10_kpmp_index results/10_kpmp_index/PROVENANCE.json python scripts/stages/10_kpmp_index.py
run 11a_reference results/11a_reference/PROVENANCE.json python scripts/stages/11a_reference.py
run 11b_kpmp_targets results/11b_kpmp_targets/PROVENANCE.json python scripts/stages/11b_kpmp_targets.py
run 13a_export_counts data/interim/gse211785_counts/counts.mtx \
    ${RSCRIPT:-Rscript} scripts/stages/13a_export_counts.R \
    data/raw/GSE211785/GSE211785_EXPORT_ST_counts.rds.gz data/interim/gse211785_counts
run 13_gse211785_prep results/13_gse211785_prep/PROVENANCE.json python scripts/stages/13_gse211785_prep.py
run 12_image_features_kpmp results/12_image_features_kpmp/PROVENANCE.json python scripts/stages/12_image_features.py kpmp
run 12_image_features_gse211785 results/12_image_features_gse211785/PROVENANCE.json python scripts/stages/12_image_features.py gse211785
run 14a_baselines results/14_cv/ridge_image/PROVENANCE.json python scripts/stages/14a_baselines.py
for m in $NN; do run 14_cv_$m results/14_cv/$m/PROVENANCE.json python scripts/stages/14_train_cv.py $m; done
for m in ridge_image gbm_comp_morph gbm_measured_comp $NN; do
  run 15_external_$m results/15_external/$m/PROVENANCE.json python scripts/stages/15_external.py $m; done
# image features standardized within each section (transfer analysis)
run 14a_baselines_sn results/14_cv/ridge_image_sn/PROVENANCE.json python scripts/stages/14a_baselines.py --secnorm
for m in mlp gcn deepsets set_ring set_anat; do
  run 14_cv_${m}_sn results/14_cv/${m}_sn/PROVENANCE.json python scripts/stages/14_train_cv.py --secnorm $m; done
for m in ridge_image gbm_comp_morph mlp gcn deepsets set_ring set_anat; do
  run 15_external_${m}_sn results/15_external/${m}_sn/PROVENANCE.json python scripts/stages/15_external.py --secnorm $m; done
# structure within the spot: 8 um image tokens
TOK="tok_mean tok_abmil tok_gat tok_hyper_window tok_hyper_rolebag tok_hyper_rewired tok_hyper_profile tok_hyper_anat tok_abmil_ctx tok_hyper_anat_ctx tok_pred_mean tok_mean_disc tok_mean_ring tok_abmil_disc tok_abmil_ring tok_abmil_outer32 tok_abmil_ctx_disc"
run 12b_token_features_kpmp results/12b_token_features_kpmp/PROVENANCE.json python scripts/stages/12b_token_features.py kpmp
run 12b_token_features_gse211785 results/12b_token_features_gse211785/PROVENANCE.json python scripts/stages/12b_token_features.py gse211785
# token projection fitted within each cross-validation fold on its training participants
run 12c_token_fold_projection results/12c_token_fold_projection/PROVENANCE.json python scripts/stages/12c_token_fold_projection.py
for m in $TOK; do run 14_cv_$m results/14_cv/$m/PROVENANCE.json python scripts/stages/14b_train_tokens.py $m; done
for m in $TOK; do
  run 15_external_${m}_s10 results/15_external/${m}_s10/PROVENANCE.json python scripts/stages/15b_external_tokens.py $m --seeds10; done
# external predictions with ten seeds: the first three seeds (as in cross-validation) form the primary
# ensemble; all ten are used to assess stability over training seeds
for m in mlp deepsets gcn gat clique_gcn hgnn_ring hgnn_anat set_ring set_anat set_anat_rewired set_anat_noroles set_anat_nogeom; do
  run 15_external_${m}_s10 results/15_external/${m}_s10/PROVENANCE.json python scripts/stages/15_external.py $m --seeds10; done
for m in ridge_image gbm_comp_morph mlp gcn deepsets set_ring set_anat; do
  run 15_external_${m}_sn_s10 results/15_external/${m}_sn_s10/PROVENANCE.json python scripts/stages/15_external.py --secnorm $m --seeds10; done
# published methods retrained in the same folds on the same targets (bash scripts/third_party/fetch.sh first)
python scripts/stages/14c_istar.py --extract kpmp
python scripts/stages/14c_istar.py --extract gse211785
run 14_cv_istar results/14_cv/istar/PROVENANCE.json python scripts/stages/14c_istar.py
run 15_external_istar_s10 results/15_external/istar_s10/PROVENANCE.json python scripts/stages/15c_istar_external.py --seeds10
run 14_cv_bleep results/14_cv/bleep/PROVENANCE.json python scripts/stages/14e_bleep.py
run 15_external_bleep_s10 results/15_external/bleep_s10/PROVENANCE.json python scripts/stages/15e_bleep_external.py --seeds10
python scripts/stages/14d_hist2st.py --prep
run 14d_hist2st results/14d_hist2st/PROVENANCE.json python scripts/stages/14d_hist2st.py
run 15d_hist2st_external results/15d_hist2st_external/PROVENANCE.json python scripts/stages/15d_hist2st_external.py
# same-encoder retrieval reference (prediction rule of BLEEP on the shared encoder)
run 14f_retrieval results/14f_retrieval/PROVENANCE.json python scripts/stages/14f_retrieval.py
# token attention of NephroSet in held-out participants
run 16d_token_attention results/16d_token_attention/PROVENANCE.json python scripts/stages/16d_token_attention.py
run 16_evaluate_cv results/16_evaluate_cv/PROVENANCE.json python scripts/stages/16_evaluate_cv.py
run 16b_reliability_region results/16b_reliability_region/PROVENANCE.json python scripts/stages/16b_reliability_region.py
run 16c_scale_diagnostics results/16c_scale_diagnostics/PROVENANCE.json python scripts/stages/16c_scale_diagnostics.py
run 16e_composition_alt results/16e_composition_alt/PROVENANCE.json python scripts/stages/16e_composition_alt.py
run 16f_adjustment_sensitivity results/16f_adjustment_sensitivity/PROVENANCE.json python scripts/stages/16f_adjustment_sensitivity.py
run 16g_reliability_sensitivity results/16g_reliability_sensitivity/PROVENANCE.json python scripts/stages/16g_reliability_sensitivity.py
run 17_confidence results/17_confidence/PROVENANCE.json python scripts/stages/17_confidence.py
run 18_evaluate_external results/18_evaluate_external/PROVENANCE.json python scripts/stages/18_evaluate_external.py
run 18b_external_reliability results/18b_external_reliability/PROVENANCE.json python scripts/stages/18b_external_reliability.py
run 18c_external_region_coverage results/18c_external_region_coverage/PROVENANCE.json python scripts/stages/18c_external_region_coverage.py
run 18d_external_donors results/18d_external_donors/PROVENANCE.json python scripts/stages/18d_external_donors.py
run 18e_external_tissue results/18e_external_tissue/PROVENANCE.json python scripts/stages/18e_external_tissue.py
run 19_examples results/19_examples/PROVENANCE.json python scripts/stages/19_examples.py
run 20_subgroups results/20_subgroups/PROVENANCE.json python scripts/stages/20_subgroups.py
run 21_tables results/21_tables/PROVENANCE.json python scripts/stages/21_tables.py
run 21b_model_cost results/21b_model_cost/PROVENANCE.json python scripts/stages/21b_model_cost.py
# figures from result tables, then manuscript build with checks
(cd scripts/figures && for f in Fig*.py; do python "$f"; done)
python scripts/build_manuscript.py
