"""Hist2ST (Zeng et al. 2022) under the participant-held-out cross-validation of 14_train_cv.py.

Same sections, outer folds, validation participants (default_rng(seed + fold)) and training seeds as
14_train_cv.py. The official network (final variant) regresses the three programme scores directly; the gene-panel
variant (300 genes, original MSE + ZINB, scores from predicted expression with the function
defining the targets) collapsed and is kept only as code path MODE="genes". Scores are mapped to the target scale on the
validation participants. Early stopping on within-section Spearman of the injury score of the
validation participants. Details and deviations: lib/baseline_hist2st.py.

Usage: python 14d_hist2st.py [--prep] [--resume] [--worker K --nworkers N] [--assemble]
Writes data/cache/cv_predictions/hist2st_fold{fi}.parquet.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import baseline_hist2st as B  # noqa: E402
from lib import data as D  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
MODE = "scores"  # direct regression of the programme scores; see lib/baseline_hist2st.py
T = cfg["train"]
OUTC = p(cfg["paths"]["cache"]) / "cv_predictions"
WORK = B.CACHE / "cv"
S = stage_dir("14d_hist2st")
MAX_EPOCHS, VAL_EVERY, PATIENCE, MIN_EPOCHS = 60, 2, 8, 20   # validation every 2 epochs as the original Lightning trainer


def log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def panel_for(name: str, data: dict, tr: list[str]) -> list[str]:
    f = WORK / f"panel_{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    pn = B.select_panel(data, tr, cfg["seed"])
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(pn))
    tmp.replace(f)
    return pn


def assemble(sections: pd.DataFrame) -> None:
    folds = D.participant_folds(sections, T["n_folds"], cfg["seed"])
    for fi in range(len(folds)):
        fs = [WORK / f"fold{fi}_seed{s}.parquet" for s in cfg["seeds"]]
        if all(f.exists() for f in fs) and not (OUTC / f"hist2st_fold{fi}.parquet").exists():
            pd.concat([pd.read_parquet(f) for f in fs]).to_parquet(OUTC / f"hist2st_fold{fi}.parquet")
            log(f"assembled fold {fi}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sections = B.sections_kpmp()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.prep:
        B.prepare("kpmp", list(sections.section))
        log("prepared")
        return
    if a.assemble:
        assemble(sections)
        return
    folds = D.participant_folds(sections, T["n_folds"], cfg["seed"])
    tasks = [(fi, sd) for fi in range(len(folds)) for sd in cfg["seeds"]][a.worker::a.nworkers]
    tasks = [t for t in tasks if not (a.resume and (WORK / f"fold{t[0]}_seed{t[1]}.parquet").exists())]
    if not tasks:
        assemble(sections)
        return
    data = {s: B.load_section("kpmp", s, True) for s in sections.section}
    log(f"loaded {len(data)} sections; tasks {tasks}")
    for fi, seed in tasks:
        tr_all, te = folds[fi]
        rng = np.random.default_rng(cfg["seed"] + fi)
        tr, va = D.split_validation(sections, tr_all, T["val_frac"], rng)
        panel = panel_for(f"fold{fi}", data, tr) if MODE == "genes" else []
        wd = WORK / f"fold{fi}_seed{seed}"
        if not a.resume and wd.exists():
            shutil.rmtree(wd)
        log(f"fold {fi} seed {seed}: train {len(tr)} val {len(va)} test {len(te)} sections, {len(panel)} genes")
        t0 = time.time()
        tn = B.Trainer(data, tr, va, panel, seed, wd, a.device, MAX_EPOCHS, VAL_EVERY, PATIENCE,
                       log=lambda m: log(f"[f{fi} s{seed}] {m}"), mode=MODE, min_epochs=MIN_EPOCHS)
        model, best, ep = tn.fit()
        vs = pd.concat([tn.predict_scores(model, s) for s in va])
        vt = pd.concat([data[s]["table"] for s in va])
        cal = B.calibrate(vs, vt)
        out = []
        for s in te:
            df = B.apply_cal(tn.predict_scores(model, s), cal)
            df["section"], df["seed"], df["fold"] = s, seed, fi
            out.append(df)
        pr = pd.concat(out)
        pr.index.name = "barcode"
        pr.to_parquet(WORK / f"fold{fi}_seed{seed}.parquet")
        row = dict(model="hist2st", fold=fi, seed=seed, val_rho=best, epochs=ep, n_genes=len(panel),
                   minutes=(time.time() - t0) / 60, calibration=json.dumps(cal))
        lf = S / "training_log.tsv"
        pd.DataFrame([row]).to_csv(lf, sep="\t", index=False, mode="a", header=not lf.exists())
        log(f"done fold {fi} seed {seed} val {best:.4f} epochs {ep} ({row['minutes']:.0f} min)")
        (wd / "ckpt.pt").unlink(missing_ok=True)
        del model, tn
        torch.cuda.empty_cache()
        assemble(sections)
    if all((OUTC / f"hist2st_fold{fi}.parquet").exists() for fi in range(len(folds))):
        write_provenance("14d_hist2st", cfg, [], [S / "training_log.tsv"], extra={"model": "hist2st", **{
            k: str(v) for k, v in B.HP.items()}, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "mode": MODE})


if __name__ == "__main__":
    main()
