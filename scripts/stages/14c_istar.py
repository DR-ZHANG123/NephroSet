"""iStar baseline (official HIPT features + ForwardSumModel) under participant-held-out CV on KPMP.

Usage:
  python 14c_istar.py --extract <kpmp|gse211785> [--part i/n]   # superpixel features per section
  python 14c_istar.py [--resume] [--phikon]                       # cross-validation
  (--phikon: encoder-matched control 'istar_phikon' on the Phikon-v2 tokens of 12b)
Folds, validation participants and seeds are those of 14_train_cv.py. See lib/baseline_istar.py
for what is taken unchanged from the official code and what is adapted.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import baseline_istar as BI  # noqa: E402
from lib import data as D  # noqa: E402
from lib.repro import load_config, p, stage_dir, write_provenance  # noqa: E402

cfg = load_config()
T = cfg["train"]
DEV = "cuda:0"  # select the physical GPU with CUDA_VISIBLE_DEVICES
OUTC = p(cfg["paths"]["cache"]) / "cv_predictions"
N_STATES = 5  # iStar default (impute.py --n-states)
# --phikon: encoder-matched control, iStar model/training on the Phikon-v2 tokens of 12b
PHIKON = "--phikon" in sys.argv
NAME = "istar_phikon" if PHIKON else "istar"


def load_x(cohort, sections, tables):
    if PHIKON:
        return BI.load_token_features(p(cfg["paths"]["cache"]) / f"tok_{cohort}", sections, tables)
    return BI.load_features(feat_dir(cohort), sections, tables)


def sections_kpmp() -> pd.DataFrame:
    idx = pd.read_csv(p("results/10_kpmp_index/sections.tsv"), sep="\t")
    enc = set(pd.read_csv(p("results/12_image_features_kpmp/encoded_sections.tsv"), sep="\t").section)
    sm = pd.read_csv(p("results/11b_kpmp_targets/spot_summary.tsv"), sep="\t")
    keep = set(sm.section[sm.status == "ok"]) & enc
    return idx[idx.section.isin(keep)].reset_index(drop=True)


def ext_sections() -> list[str]:
    return pd.read_csv(p("results/13_gse211785_prep/sections.tsv"), sep="\t").section.tolist()


def feat_dir(cohort: str) -> Path:
    d = p(cfg["paths"]["cache"]) / f"istar_{cohort}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prep(cohort, sid):
    import cv2
    cv2.setNumThreads(2)
    t = pd.read_parquet(p(cfg["paths"]["interim"]) / f"{cohort}_spots" / f"{sid}.parquet")
    img, locs = BI.load_section_image(p(cfg["paths"]["interim"]) / f"{cohort}_sections" / sid, t)
    return sid, img, locs, t.index.values


def _prep_star(a):
    return _prep(*a)


def extract(cohort: str, part: str) -> None:
    import multiprocessing as mp
    secs = list(sections_kpmp().section) if cohort == "kpmp" else ext_sections()
    i, n = map(int, part.split("/"))
    secs = secs[i::n]
    out = feat_dir(cohort)
    todo = [s for s in secs if not (out / f"{s}.npz").exists()]
    print(cohort, "part", part, len(todo), "sections to encode", flush=True)
    # image loading and rescaling (CPU) run ahead in worker processes while the GPU encodes
    with mp.get_context("spawn").Pool(3) as pool:
        for sid, img, locs, bc in pool.imap(_prep_star, [(cohort, s) for s in todo]):
            t0 = time.time()
            for attempt in range(1000):
                try:
                    embs = BI.hist_embeddings(img, "cuda")
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("OOM, retrying in 120 s", sid, flush=True)
                    time.sleep(120)
            x = BI.spot_patches(embs, locs)
            assert np.isfinite(x).all(), sid
            np.savez(out / f"{sid}.npz", x=x.astype(np.float16), barcode=bc)
            print(sid, img.shape, x.shape, f"{time.time() - t0:.0f}s", time.strftime("%H:%M"), flush=True)
            del embs, x
            torch.cuda.empty_cache()


def main() -> None:
    if "--extract" in sys.argv:
        cohort = sys.argv[sys.argv.index("--extract") + 1]
        part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "0/1"
        extract(cohort, part)
        return
    S = stage_dir(f"14_cv/{NAME}")
    sections = sections_kpmp()
    spots = p(cfg["paths"]["interim"]) / "kpmp_spots"
    tables = {s: pd.read_parquet(spots / f"{s}.parquet") for s in sections.section}
    Y = {s: t[D.PROGS].values.astype(np.float32) for s, t in tables.items()}
    X = load_x("kpmp", sections.section, tables)
    log, rows = [], []
    logf = S / "training_log.tsv"
    if logf.exists() and "--resume" in sys.argv:
        log = pd.read_csv(logf, sep="\t").to_dict("records")
    for fi, (tr_all, te) in enumerate(D.participant_folds(sections, T["n_folds"], cfg["seed"])):
        rng = np.random.default_rng(cfg["seed"] + fi)
        tr, va = D.split_validation(sections, tr_all, T["val_frac"], rng)
        rows.append(dict(model=NAME, fold=fi, n_train=len(tr), n_val=len(va), n_test=len(te)))
        if "--resume" in sys.argv and (OUTC / f"{NAME}_fold{fi}.parquet").exists():
            continue
        log = [r for r in log if r["fold"] != fi]
        preds = []
        for seed in cfg["seeds"]:
            out, lg = BI.fit_predict(X, Y, tr, va, te, seed, T, DEV, N_STATES, tag=f"fold {fi}")
            log += [dict(model=NAME, fold=fi, **r) for r in lg]
            for s in te:
                df = pd.DataFrame(out[s], columns=[f"pred_{c[2:]}" for c in D.PROGS], index=tables[s].index)
                df["section"], df["seed"], df["fold"] = s, seed, fi
                preds.append(df)
            pd.DataFrame(log).to_csv(logf, sep="\t", index=False)
        pd.concat(preds).to_parquet(OUTC / f"{NAME}_fold{fi}.parquet")
        print(NAME, "fold", fi, "done", time.strftime("%H:%M"), flush=True)
    pd.DataFrame(rows).to_csv(S / "folds.tsv", sep="\t", index=False)
    pd.DataFrame(log).to_csv(logf, sep="\t", index=False)
    write_provenance(f"14_cv/{NAME}", cfg, [], [S / "folds.tsv"],
                     extra={"model": NAME, "n_states": N_STATES,
                            "features": "Phikon-v2 tokens (12b)" if PHIKON else "HIPT (iStar official)"})


if __name__ == "__main__":
    main()
