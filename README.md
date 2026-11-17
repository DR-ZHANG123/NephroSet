# NephroSet

Code, parameters, result tables and figure source data for the article
"NephroSet: integrating fine-scale morphology and neighbourhood context to estimate proximal tubule molecular
programmes from kidney histology".

NephroSet estimates three proximal tubule (PT) programmes (injury, inflammation, transport) for every Visium spot
from the haematoxylin and eosin (H&E) image alone. It reads the 8 µm Phikon-v2 image tokens of the 112 µm crop
centred on each spot with gated attention pooling, and the neighbouring spots as an unordered set. Models restricted
to the tokens inside the 55 µm capture spot, around it or at the edge of the crop (`tok_*_disc`, `tok_*_ring`,
`tok_abmil_outer32`) locate where the information lies.

## Layout
- `config/run.yaml`: every analysis parameter (programme gene sets, folds, seeds, model settings).
- `scripts/stages/`: numbered analysis stages. Each writes its tables to `results/<stage>/`.
  - `00`–`04`: download and single-cell spatial organisation (CosMx and Xenium).
  - `10`–`13`: KPMP and GSE211785 Visium data, targets, composition and image features
    (`12c` fits the token projection within each cross-validation fold on its training participants).
  - `14`–`15`: participant-held-out cross-validation and the external cohort for all spot and token models
    (`14b`/`15b` token models including NephroSet = `tok_abmil_ctx`; `14c`–`14e`/`15c`–`15e` iStar, Hist2ST and
    BLEEP retrained on the same participants and targets; `14f` and `tok_pred_mean` apply the prediction rules
    of BLEEP and iStar to the shared image encoder).
  - `16`–`21`: evaluation, split-half reliability (with alternative definitions), alternative composition
    adjustments, token attention, confidence, results by external donor and seed, external tissue coverage and
    donor overlap (`18e`), tissue coverage of token regions (`16h`), subgroups, tables and model cost.
  - External results use the ensemble of the same three seeds as cross-validation; ten seeds are trained to
    assess stability. Training on GPUs is not bit-reproducible, so repeated runs give slightly different models.
  - Prediction intervals are calibrated on validation participants that were not used to fit the model weights.
- `scripts/lib/`: data handling, models, evaluation and the published-method adapters.
- `scripts/figures/`: plotting only; reads `figures/source_data/`.
- `results/`, `figures/source_data/`: the tables behind every number and figure in the article.

## Reproduce
```bash
conda env create -f environment.yml          # environment "nephroset"
bash scripts/third_party/fetch.sh            # iStar, BLEEP and Hist2ST at the commits used, and HIPT weights
bash run_all.sh --resume                     # skips stages whose outputs exist
```
Stage `13a` converts the GSE211785 count object with R (Seurat); set `RSCRIPT` to an Rscript with Seurat installed.
Raw data are public and are downloaded by `00_download.py`: KPMP open-access Visium data
(https://atlas.kpmp.org/repository), GEO GSE211785, the CosMx and Xenium kidney atlas (Zenodo 10.5281/zenodo.19868428)
and the KPMP single-nucleus reference (CZ CELLxGENE Discover). Model training used one NVIDIA L40S GPU per run.

## Licence
MIT (see `LICENSE`). The published methods fetched by `scripts/third_party/fetch.sh` keep their own licences.
